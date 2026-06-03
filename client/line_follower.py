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


def _find_centroid(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    """Return (cx, cy) of the largest contour, or None if nothing found."""
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 500:   # ignore tiny blobs
        return None

    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy


class LineFollower:
    """Drives the hexapod along a coloured floor line.

    Args:
        client:         Connected RobotClient instance.
        colour:         Line colour — "red", "white", or "blue".
        base_speed:     Forward duty when centred on the line.
        max_duty:       Clamp for all duty outputs.
        kp:             Proportional gain for the steering PD controller.
        kd:             Derivative  gain for the steering PD controller.
        turn_reduction: How much to slow forward speed on sharp turns (0-1).
        roi_top:        Top of the region-of-interest as a fraction of frame
                        height (0 = top, 1 = bottom).  Keeping this in the
                        lower half helps the robot react early.
        lost_timeout:   Seconds without a detected line before stopping.
        loop_hz:        Target control-loop frequency.
        show_preview:   Display a debug window (requires a local display).
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
    ):
        if colour not in _COLOUR_RANGES:
            raise ValueError(
                "colour must be one of {}; got {!r}".format(SUPPORTED_COLOURS, colour)
            )

        self.client        = client
        self.colour        = colour
        self.base_speed    = base_speed
        self.max_duty      = max_duty
        self.kp            = kp
        self.kd            = kd
        self.turn_reduction = turn_reduction
        self.roi_top       = roi_top
        self.lost_timeout  = lost_timeout
        self._period       = 1.0 / max(loop_hz, 1.0)
        self.show_preview  = show_preview

        self._prev_error   = 0.0
        self._last_seen    = time.monotonic()
        self._running      = False

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
        try:
            frame_bgr, _, _ = self.client.get_frame()
        except Exception as exc:
            print("[LineFollower] Frame error: {}".format(exc))
            return

        h, w = frame_bgr.shape[:2]
        frame_cx = w / 2.0

        # Crop to the lower portion of the frame (look-ahead ROI)
        roi_y = int(h * self.roi_top)
        roi = frame_bgr[roi_y:, :]

        hsv  = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = _build_mask(hsv, self.colour)
        centroid = _find_centroid(mask)

        now = time.monotonic()

        if centroid is None:
            # Line lost
            if now - self._last_seen > self.lost_timeout:
                self._stop_robot()
                print("\r[LineFollower] Line lost — stopped.         ", end="", flush=True)
            else:
                print("\r[LineFollower] Searching...                 ", end="", flush=True)
            self._prev_error = 0.0

            if self.show_preview:
                self._draw_preview(frame_bgr, roi_y, mask, None, 0.0, 0.0, 0.0)
            return

        self._last_seen = now
        cx, cy = centroid

        # Normalised lateral error: negative = line to the left, positive = right
        error    = (cx - frame_cx) / frame_cx
        d_error  = error - self._prev_error
        self._prev_error = error

        # Steering: positive omega = turn left
        omega = -(self.kp * error + self.kd * d_error)
        omega = float(np.clip(omega, -self.max_duty, self.max_duty))

        # Slow down proportionally when turning hard
        vx = self.base_speed * (1.0 - self.turn_reduction * abs(error))
        vx = float(np.clip(vx, 0.0, self.max_duty))

        try:
            self.client.send_velocity(vx=vx, vy=0.0, omega=omega)
        except Exception as exc:
            print("[LineFollower] Send error: {}".format(exc))
            return

        print(
            "\r[LineFollower] err={:+.2f}  vx={:4.1f}  ω={:+5.1f}   ".format(
                error, vx, omega
            ),
            end="", flush=True,
        )

        if self.show_preview:
            self._draw_preview(frame_bgr, roi_y, mask, (cx, cy + roi_y), error, vx, omega)

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
