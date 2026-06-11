"""Line-following controller for the hexapod robot.

Uses the onboard camera to detect a coloured floor line and steer the
robot to keep the line centred in the frame.

Supported line colours: red, white, blue.

Algorithm
---------
1. Grab a BGR frame from the robot camera.
2. Convert to HSV and threshold for the target colour.
3. Find the largest contour in the mask and compute its centroid.
4. Compute a normalised lateral error:
       error = (centroid_x - frame_cx) / frame_cx   ∈ [-1, 1]
5. PD controller drives omega:
       omega = -(Kp * error + Kd * d_error) * max_duty
6. Forward speed is reduced proportionally on tight turns:
       vx = base_speed * (1 - turn_reduction * |error|)
7. If no line is detected for `lost_timeout` seconds the robot stops.
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .robot_client import RobotClient


# ---------------------------------------------------------------------------
# HSV colour ranges  (H: 0-179, S: 0-255, V: 0-255 in OpenCV)
# ---------------------------------------------------------------------------

_COLOUR_RANGES: Dict[str, list] = {
    "red": [
        # Red wraps around the hue wheel — use two bands
        (np.array([0,   100,  70]), np.array([10,  255, 255])),
        (np.array([160, 100,  70]), np.array([179, 255, 255])),
    ],
    "white": [
        (np.array([0,   0,   180]), np.array([179, 50,  255])),
    ],
    "blue": [
        (np.array([90,  80,   50]), np.array([130, 255, 255])),
    ],
}

SUPPORTED_COLOURS = list(_COLOUR_RANGES.keys())


def _build_mask(hsv: np.ndarray, colour: str) -> np.ndarray:
    """Return a binary mask for the given colour in an HSV image."""
    ranges = _COLOUR_RANGES[colour]
    mask = cv2.inRange(hsv, ranges[0][0], ranges[0][1])
    for lo, hi in ranges[1:]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))

    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def _find_line_direction(
    mask: np.ndarray, frame_cx: float
) -> Optional[Tuple[float, float, int, int]]:
    """Fit a line to the largest colour blob and return orientation info.

    Returns:
        (lateral_error, heading_norm, cx, cy)  or  None if no line found.

        lateral_error  — normalised [-1, 1]: positive = line is right of centre.
                         Robot must steer right to re-centre.
        heading_norm   — normalised [-1, 1]: positive = line tilts right in frame
                         (robot is turned left relative to line direction).
                         Robot must turn right to align.
        cx, cy         — centroid of the detected blob in the mask image.
    """
    import math

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 500:
        return None

    # Centroid gives lateral position
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # fitLine gives the direction vector of the line's long axis
    line = cv2.fitLine(largest, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    vx, vy = float(line[0]), float(line[1])

    # Make sure the direction vector always points "downward" in the image
    # (toward the robot) so the angle sign is consistent
    if vy < 0:
        vx, vy = -vx, -vy

    # Heading angle from vertical (robot's forward axis in the image)
    # 0° = line runs straight ahead (robot aligned)
    # Positive = line tilts right = robot is turned left relative to line
    heading_deg = math.degrees(math.atan2(vx, vy))
    heading_norm = heading_deg / 90.0          # normalise to [-1, 1]
    heading_norm = max(-1.0, min(1.0, heading_norm))

    lateral_error = (cx - frame_cx) / frame_cx  # normalised [-1, 1]

    return lateral_error, heading_norm, cx, cy


class LineFollower:
    """Drives the hexapod along a coloured floor line.

    When a controller is supplied, moving any stick overrides the
    line follower so the operator can steer manually.  Releasing the
    sticks hands control back to the algorithm automatically.

    Args:
        client:               Connected RobotClient instance.
        colour:               Line colour — "red", "white", or "blue".
        base_speed:           Forward duty when centred on the line.
        max_duty:             Clamp for all duty outputs.
        kp:                   Proportional gain for the steering PD controller.
        kd:                   Derivative  gain for the steering PD controller.
        turn_reduction:       How much to slow forward speed on sharp turns (0-1).
        roi_top:              Top of the region-of-interest as a fraction of frame
                              height (0 = top, 1 = bottom).
        lost_timeout:         Seconds without a detected line before stopping.
        loop_hz:              Target control-loop frequency.
        show_preview:         Display a debug window (requires a local display).
        controller:           Optional PS5Controller for manual override.
        manual_threshold:     Minimum stick magnitude to trigger manual override.
        obstacle_stop:        Stop when an obstacle is detected inside the line ROI.
        obs_color_thresh:     BGR colour distance from the floor sample to count a
                              pixel as non-floor / obstacle.
        obs_presence_thresh:  Fraction of the ROI that must contain non-floor,
                              non-line pixels before triggering a stop.
    """

    def __init__(
        self,
        client: RobotClient,
        colour: str = "red",
        base_speed: float = 30.0,
        max_duty: float = 80.0,
        kp: float = 60.0,
        kd: float = 8.0,
        turn_reduction: float = 0.6,
        roi_top: float = 0.4,
        lost_timeout: float = 2.0,
        loop_hz: float = 10.0,
        show_preview: bool = False,
        kp_heading: float = 30.0,
        controller=None,
        manual_threshold: float = 5.0,
        memory_duration: float = 0.6,
        obstacle_stop: bool = False,
        obs_color_thresh: float = 40.0,
        obs_presence_thresh: float = 0.25,
    ):
        if colour not in _COLOUR_RANGES:
            raise ValueError(
                "colour must be one of {}; got {!r}".format(SUPPORTED_COLOURS, colour)
            )

        self.client        = client
        self.colour        = colour
        self.base_speed    = base_speed
        self.max_duty      = max_duty
        self.kp            = kp          # lateral (position) gain
        self.kp_heading    = kp_heading  # heading (angle) gain
        self.kd            = kd
        self.turn_reduction = turn_reduction
        self.roi_top       = roi_top
        self.lost_timeout  = lost_timeout
        self._period       = 1.0 / max(loop_hz, 1.0)
        self.show_preview  = show_preview

        self.controller       = controller
        self.manual_threshold = manual_threshold
        self.memory_duration  = memory_duration

        self._obstacle_stop       = obstacle_stop
        self._obs_color_thresh    = obs_color_thresh
        self._obs_presence_thresh = obs_presence_thresh
        self._obstacle_active   = False

        self._prev_error   = 0.0
        self._last_seen    = time.monotonic()
        self._running      = False
        self._last_vx      = 0.0
        self._last_omega   = 0.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main control loop.  Blocks until stopped or KeyboardInterrupt."""
        self._running = True
        self._prev_error = 0.0
        self._last_seen = time.monotonic()

        print("[LineFollower] Starting  colour={} speed={} kp={} kd={}".format(
            self.colour, self.base_speed, self.kp, self.kd
        ))

        try:
            while self._running:
                # Options button on PS5 exits
                if self.controller and self.controller.events.get("stop_session"):
                    print("\n[LineFollower] PS5 Options pressed — stopping.")
                    break
                t0 = time.monotonic()
                self._step()
                elapsed = time.monotonic() - t0
                sleep = self._period - elapsed
                if sleep > 0:
                    time.sleep(sleep)
        except KeyboardInterrupt:
            print("\n[LineFollower] Interrupted.")
        finally:
            self._stop_robot()
            if self.show_preview:
                cv2.destroyAllWindows()
            print("[LineFollower] Stopped.")

    def stop(self) -> None:
        """Signal the run loop to exit on the next iteration."""
        self._running = False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _step(self) -> None:
        # --- PS5 manual override ---
        if self.controller is not None:
            vx, vy, omega = self.controller.get_action()
            if abs(vx) > self.manual_threshold or \
               abs(vy) > self.manual_threshold or \
               abs(omega) > self.manual_threshold:
                try:
                    self.client.send_velocity(vx=vx, vy=vy, omega=omega)
                except Exception:
                    pass
                self._prev_error = 0.0
                self._last_seen = time.monotonic()
                print("\r[LineFollower] MANUAL  vx={:4.1f} vy={:4.1f} ω={:+5.1f}   ".format(
                    vx, vy, omega), end="", flush=True)
                return

        try:
            frame_bgr, _, _ = self.client.get_frame()
        except Exception as exc:
            print("[LineFollower] Frame error: {}".format(exc))
            return

        h, w = frame_bgr.shape[:2]
        frame_cx = w / 2.0

        # --- Obstacle detection (runs on the full frame before line logic) ---
        if self._obstacle_stop:
            if self._detect_obstacle(frame_bgr):
                self._stop_robot()
                if not self._obstacle_active:
                    self._obstacle_active = True
                    try:
                        self.client.beep(freq=1900, duration=0.2)
                    except Exception:
                        pass
                self._prev_error = 0.0
                print("\r[LineFollower] OBSTACLE — stopped.                    ",
                      end="", flush=True)
                return
            self._obstacle_active = False

        # Crop to the lower portion of the frame (look-ahead ROI)
        roi_y = int(h * self.roi_top)
        roi = frame_bgr[roi_y:, :]

        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = _build_mask(hsv, self.colour)
        result = _find_line_direction(mask, frame_cx)

        now = time.monotonic()

        if result is None:
            lost_for = now - self._last_seen
            if lost_for > self.lost_timeout:
                self._stop_robot()
                self._last_vx = 0.0
                self._last_omega = 0.0
                print("\r[LineFollower] Line lost — stopped.         ", end="", flush=True)
            elif lost_for < self.memory_duration and \
                    (abs(self._last_vx) > 0 or abs(self._last_omega) > 0):
                try:
                    self.client.send_velocity(vx=self._last_vx, vy=0.0,
                                              omega=self._last_omega)
                except Exception:
                    pass
                print("\r[LineFollower] Coasting  vx={:.1f} ω={:+.1f}          ".format(
                    self._last_vx, self._last_omega), end="", flush=True)
            else:
                print("\r[LineFollower] Searching...                 ", end="", flush=True)
            self._prev_error = 0.0

            if self.show_preview:
                self._draw_preview(frame_bgr, roi_y, mask, None, 0.0, 0.0, 0.0)
            return

        self._last_seen = now
        lateral_error, heading_norm, cx, cy = result

        # Derivative on lateral error for damping
        d_lateral = lateral_error - self._prev_error
        self._prev_error = lateral_error

        # Steering combines:
        #   lateral correction — move robot back to centre of the line
        #   heading correction — align robot's direction with the line's angle
        # Both push omega in the same direction when the robot drifts off-line.
        omega = -(self.kp * lateral_error
                  + self.kd * d_lateral
                  + self.kp_heading * heading_norm)
        omega = float(np.clip(omega, -self.max_duty, self.max_duty))

        # Slow down when misaligned
        misalignment = abs(lateral_error) + abs(heading_norm)
        vx = self.base_speed * (1.0 - self.turn_reduction * min(misalignment, 1.0))
        vx = float(np.clip(vx, 0.0, self.max_duty))

        try:
            self.client.send_velocity(vx=vx, vy=0.0, omega=omega)
            self._last_vx = vx
            self._last_omega = omega
        except Exception as exc:
            print("[LineFollower] Send error: {}".format(exc))
            return

        print(
            "\r[LineFollower] lat={:+.2f}  hdg={:+.2f}  vx={:4.1f}  ω={:+5.1f}   ".format(
                lateral_error, heading_norm, vx, omega
            ),
            end="", flush=True,
        )

        if self.show_preview:
            self._draw_preview(frame_bgr, roi_y, mask, (cx, cy + roi_y),
                               lateral_error, vx, omega)

    def _detect_obstacle(self, frame_bgr: np.ndarray) -> bool:
        """Return True if an obstacle is present inside the line ROI.

        Samples the floor colour from the bottom strip (always bare floor),
        then counts pixels inside the ROI that are neither floor-coloured nor
        line-coloured.  Using both exclusions prevents the line tape itself
        from triggering a false stop.
        """
        h, w = frame_bgr.shape[:2]

        # Floor colour reference: bottom 15% of frame, centre 50% width
        floor_strip = frame_bgr[int(h * 0.85):, int(w * 0.25):int(w * 0.75)]
        if floor_strip.size == 0:
            return False
        floor_color = np.median(
            floor_strip.reshape(-1, 3), axis=0
        ).astype(np.float32)

        # Detection zone: the line ROI (above floor sample), centre 60% width
        obs_top = int(h * self.roi_top)
        obs_bot = int(h * 0.85)
        if obs_bot <= obs_top:
            return False
        zone_bgr = frame_bgr[obs_top:obs_bot, int(w * 0.2):int(w * 0.8)]
        if zone_bgr.size == 0:
            return False

        # Non-floor mask: pixels whose BGR distance from the floor exceeds threshold
        diff = zone_bgr.astype(np.float32) - floor_color
        dist = np.sqrt((diff ** 2).sum(axis=2))
        non_floor = dist > self._obs_color_thresh

        # Non-line mask: pixels that do NOT match the tracked line colour
        zone_hsv = cv2.cvtColor(zone_bgr, cv2.COLOR_BGR2HSV)
        line_mask = _build_mask(zone_hsv, self.colour)
        non_line = line_mask == 0

        # Obstacle: non-floor AND non-line (excludes floor and the tape itself)
        return float((non_floor & non_line).mean()) > self._obs_presence_thresh

    def _stop_robot(self) -> None:
        try:
            self.client.stop()
        except Exception:
            pass

    def _draw_preview(
        self,
        frame_bgr: np.ndarray,
        roi_y: int,
        mask: np.ndarray,
        centroid: Optional[Tuple[int, int]],
        error: float,
        vx: float,
        omega: float,
    ) -> None:
        """Overlay debug info and show the frame in a local window."""
        vis = frame_bgr.copy()
        h, w = vis.shape[:2]

        # ROI line
        cv2.line(vis, (0, roi_y), (w, roi_y), (0, 255, 255), 1)
        # Frame centre
        cv2.line(vis, (w // 2, roi_y), (w // 2, h), (255, 255, 0), 1)

        # Obstacle detection zone (orange box, inside ROI)
        if self._obstacle_stop:
            obs_top = roi_y
            obs_bot = int(h * 0.85)
            ox1, ox2 = int(w * 0.2), int(w * 0.8)
            color = (0, 0, 255) if self._obstacle_active else (0, 140, 255)
            cv2.rectangle(vis, (ox1, obs_top), (ox2, obs_bot), color, 2)
            if self._obstacle_active:
                cv2.putText(vis, "OBSTACLE", (ox1 + 4, obs_top + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Mask overlay (coloured tint)
        colour_map = {"red": (0, 0, 180), "white": (200, 200, 200), "blue": (180, 0, 0)}
        tint = np.zeros_like(vis)
        tint[roi_y:][mask > 0] = colour_map.get(self.colour, (0, 255, 0))
        vis = cv2.addWeighted(vis, 0.8, tint, 0.4, 0)

        if centroid is not None:
            cv2.circle(vis, centroid, 8, (0, 255, 0), -1)

        # HUD
        cv2.putText(vis, "err={:+.2f} vx={:.0f} w={:+.0f}".format(error, vx, omega),
                    (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(vis, "colour: {}".format(self.colour),
                    (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("LineFollower", vis)
        cv2.waitKey(1)
