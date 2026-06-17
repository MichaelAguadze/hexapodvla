"""Object-detection-based robot driver for the hexapod.

Behavior priority (highest → lowest):
  1. PS5 manual override  (if a controller is attached)
  2. Reactive behaviors   (react_map: class_label → action)
  3. Generic avoidance    (steer around anything in the danger corridor)
  4. Free forward travel

Supported reaction actions
--------------------------
  "approach"  – drive toward the target; stop when its bbox fills
                 `approach_stop_area` fraction of the frame.
  "follow"    – keep the target at a fixed distance (forward if too far,
                 backward if too close).
  "stop"      – hold still while the target is in view.
  "avoid"     – treat this class exactly like an unrecognised obstacle.
  "ignore"    – never react to this class (skip avoidance too).

react_map label convention
--------------------------
  Keys are YOLO class-name strings (e.g. "sports ball", "person").
  In the CLI, underscores in label keys are converted to spaces so
  "sports_ball" → "sports ball".
"""

from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from .robot_client import RobotClient
from .object_detector import Detection, ObjectDetector


_ACTION_COLORS: Dict[str, Tuple[int, int, int]] = {
    "approach": (0, 200, 0),
    "follow":   (0, 180, 255),
    "stop":     (0, 0, 220),
    "avoid":    (0, 140, 255),
    "obstacle": (0, 140, 255),
    "ignore":   (160, 160, 160),
}
_DEFAULT_COLOR = (200, 200, 200)


class DetectionDriver:
    """Drives the hexapod based on YOLO object detections.

    Args:
        client:               Connected RobotClient.
        detector:             Loaded ObjectDetector.
        react_map:            Mapping of YOLO class label → action string.
                              e.g. {"sports ball": "approach", "person": "follow"}
        forward_speed:        Duty cycle for free forward travel.
        max_duty:             Maximum duty cycle for any axis.
        kp_steer:             Steering gain: omega = ±kp_steer × lateral_offset × max_duty.
                              lateral_offset is in [-0.5, 0.5] (fraction of half-frame width).
        danger_corridor:      Fraction of frame width treated as the forward path [0, 1].
                              Only objects whose center_x falls in the centre ±corridor/2
                              trigger general avoidance.
        min_obstacle_area:    Smallest area_fraction to count as an obstacle.
        approach_stop_area:   Stop approaching when target bbox exceeds this area fraction.
        approach_slow_area:   Start slowing approach when target exceeds this area fraction.
        follow_target_area:   Target area fraction for follow mode (maintain this distance).
        avoid_stop_area:      Stop forward motion when obstacle exceeds this area fraction.
        loop_hz:              Target control-loop frequency.
        show_preview:         Open an OpenCV debug window.
        controller:           Optional PS5Controller for manual override.
        manual_threshold:     Minimum stick magnitude to trigger manual override.
    """

    def __init__(
        self,
        client: RobotClient,
        detector: ObjectDetector,
        react_map: Optional[Dict[str, str]] = None,
        forward_speed: float = 30.0,
        max_duty: float = 80.0,
        kp_steer: float = 1.5,
        danger_corridor: float = 0.60,
        min_obstacle_area: float = 0.01,
        approach_stop_area: float = 0.12,
        approach_slow_area: float = 0.04,
        follow_target_area: float = 0.08,
        avoid_stop_area: float = 0.10,
        loop_hz: float = 10.0,
        show_preview: bool = False,
        controller=None,
        manual_threshold: float = 5.0,
    ) -> None:
        self.client         = client
        self.detector       = detector
        self.react_map      = react_map or {}
        self.forward_speed  = forward_speed
        self.max_duty       = max_duty
        self._kp            = kp_steer
        self._danger_lo     = 0.5 - danger_corridor / 2.0
        self._danger_hi     = 0.5 + danger_corridor / 2.0
        self._min_area      = min_obstacle_area
        self._app_stop      = approach_stop_area
        self._app_slow      = approach_slow_area
        self._follow_area   = follow_target_area
        self._avoid_stop    = avoid_stop_area
        self._period        = 1.0 / max(loop_hz, 1.0)
        self.show_preview   = show_preview
        self.controller     = controller
        self._manual_thr    = manual_threshold
        self._running       = False
        self._state_label   = "INIT"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Main control loop. Blocks until stopped or KeyboardInterrupt."""
        self._running = True
        print("[DetectionDriver] Starting.  react_map={}".format(
            {k: v for k, v in self.react_map.items()} or "none (avoidance only)"
        ))
        try:
            while self._running:
                if self.controller and self.controller.events.get("stop_session"):
                    print("\n[DetectionDriver] PS5 Options pressed — stopping.")
                    break
                t0 = time.monotonic()
                self._step()
                elapsed = time.monotonic() - t0
                sleep = self._period - elapsed
                if sleep > 0:
                    time.sleep(sleep)
        except KeyboardInterrupt:
            print("\n[DetectionDriver] Interrupted.")
        finally:
            self._stop_robot()
            if self.show_preview:
                cv2.destroyAllWindows()
            print("[DetectionDriver] Stopped.")

    def stop(self) -> None:
        self._running = False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _step(self) -> None:
        # Manual override
        if self.controller is not None:
            vx, vy, omega = self.controller.get_action()
            if (abs(vx) > self._manual_thr or abs(vy) > self._manual_thr
                    or abs(omega) > self._manual_thr):
                try:
                    self.client.send_velocity(vx=vx, vy=vy, omega=omega)
                except Exception:
                    pass
                self._state_label = "MANUAL"
                print("\r[DetectionDriver] MANUAL  vx={:4.1f} vy={:4.1f} ω={:+5.1f}   ".format(
                    vx, vy, omega), end="", flush=True)
                return

        try:
            frame, _, _ = self.client.get_frame()
        except Exception as exc:
            print("[DetectionDriver] Frame error: {}".format(exc))
            return

        detections = self.detector.detect(frame)
        vx, vy, omega, self._state_label = self._decide(detections)

        try:
            self.client.send_velocity(vx=vx, vy=vy, omega=omega)
        except Exception as exc:
            print("[DetectionDriver] Send error: {}".format(exc))
            return

        print(
            "\r[DetectionDriver] {:14s}  vx={:5.1f}  ω={:+6.1f}  dets={}   ".format(
                self._state_label, vx, omega, len(detections)
            ),
            end="", flush=True,
        )

        if self.show_preview:
            self._draw_preview(frame, detections, vx, omega)

    def _decide(
        self, detections: list
    ) -> Tuple[float, float, float, str]:
        """Return (vx, vy, omega, state_label) for the current detections."""

        # --- Priority 1: react_map matches ---
        react_target: Optional[Detection] = None
        react_action: Optional[str] = None

        for det in sorted(detections, key=lambda d: -d.area_fraction):
            action = self.react_map.get(det.label)
            if action == "ignore":
                continue
            if action in ("approach", "follow", "stop") and react_target is None:
                react_target = det
                react_action = action
                break  # highest-area target wins

        if react_target is not None:
            if react_action == "approach":
                vx, vy, omega = self._approach(react_target)
                arrived = react_target.area_fraction >= self._app_stop
                label = "ARRIVED" if arrived else "APPROACH:{}".format(react_target.label)
                return vx, vy, omega, label
            if react_action == "follow":
                vx, vy, omega = self._follow(react_target)
                return vx, vy, omega, "FOLLOW:{}".format(react_target.label)
            if react_action == "stop":
                return 0.0, 0.0, 0.0, "STOP:{}".format(react_target.label)

        # --- Priority 2: generic avoidance ---
        obstacles = [
            d for d in detections
            if self._danger_lo <= d.center_x <= self._danger_hi
            and d.area_fraction >= self._min_area
            and self.react_map.get(d.label) != "ignore"
        ]
        if obstacles:
            biggest = max(obstacles, key=lambda d: d.area_fraction)
            vx, vy, omega = self._avoid(biggest)
            return vx, vy, omega, "AVOID:{}".format(biggest.label)

        # --- Priority 3: free forward ---
        return self.forward_speed, 0.0, 0.0, "FORWARD"

    # ------------------------------------------------------------------
    # Motion primitives
    # ------------------------------------------------------------------

    def _steer_toward(self, center_x: float) -> float:
        """Omega to steer toward (center) the given normalised x position."""
        lateral = center_x - 0.5  # [-0.5, +0.5]; positive = target is right
        # Negative: target right → turn right (negative omega, per line-follower convention)
        omega = -self._kp * lateral * self.max_duty * 2.0
        return float(np.clip(omega, -self.max_duty, self.max_duty))

    def _steer_away(self, center_x: float) -> float:
        """Omega to steer away from the given normalised x position."""
        lateral = center_x - 0.5
        # Positive: obstacle right → turn left (positive omega)
        omega = self._kp * lateral * self.max_duty * 2.0
        return float(np.clip(omega, -self.max_duty, self.max_duty))

    def _approach(self, target: Detection) -> Tuple[float, float, float]:
        omega = self._steer_toward(target.center_x)
        if target.area_fraction >= self._app_stop:
            return 0.0, 0.0, omega
        if target.area_fraction >= self._app_slow:
            t = ((target.area_fraction - self._app_slow)
                 / max(self._app_stop - self._app_slow, 1e-6))
            vx = self.forward_speed * (1.0 - 0.8 * t)
        else:
            vx = self.forward_speed
        return float(np.clip(vx, 0.0, self.max_duty)), 0.0, omega

    def _follow(self, target: Detection) -> Tuple[float, float, float]:
        omega = self._steer_toward(target.center_x)
        area_error = self._follow_area - target.area_fraction
        # Positive error → too far → drive forward; negative → too close → back up
        t = float(np.clip(area_error / max(self._follow_area, 1e-6), -1.0, 1.0))
        vx = self.forward_speed * t
        vx = float(np.clip(vx, -self.max_duty * 0.5, self.max_duty))
        return vx, 0.0, omega

    def _avoid(self, obstacle: Detection) -> Tuple[float, float, float]:
        omega = self._steer_away(obstacle.center_x)
        if obstacle.area_fraction >= self._avoid_stop:
            return 0.0, 0.0, omega
        t = min(obstacle.area_fraction / self._avoid_stop, 1.0)
        vx = self.forward_speed * (1.0 - t)
        return float(np.clip(vx, 0.0, self.max_duty)), 0.0, omega

    def _stop_robot(self) -> None:
        try:
            self.client.stop()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _draw_preview(
        self,
        frame: np.ndarray,
        detections: list,
        vx: float,
        omega: float,
    ) -> None:
        vis = frame.copy()
        h, w = vis.shape[:2]

        # Danger corridor guide lines
        lx = int(self._danger_lo * w)
        rx = int(self._danger_hi * w)
        cv2.line(vis, (lx, 0), (lx, h), (80, 80, 80), 1)
        cv2.line(vis, (rx, 0), (rx, h), (80, 80, 80), 1)

        for det in detections:
            action = self.react_map.get(det.label)
            in_danger = (self._danger_lo <= det.center_x <= self._danger_hi
                         and det.area_fraction >= self._min_area)

            if action == "ignore":
                color = _ACTION_COLORS["ignore"]
            elif action in _ACTION_COLORS:
                color = _ACTION_COLORS[action]
            elif in_danger:
                color = _ACTION_COLORS["obstacle"]
            else:
                color = _DEFAULT_COLOR

            cv2.rectangle(vis, (det.x1, det.y1), (det.x2, det.y2), color, 2)
            label_text = "{} {:.0f}%".format(det.label, det.confidence * 100)
            cv2.putText(vis, label_text, (det.x1, max(det.y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # HUD
        cv2.putText(
            vis,
            "{} | vx={:.0f} w={:+.0f}".format(self._state_label, vx, omega),
            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
        cv2.imshow("DetectionDriver", vis)
        cv2.waitKey(1)
