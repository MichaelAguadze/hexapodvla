#!/usr/bin/env python3
"""Motor controller wrapper for the Leo Rover (ROS 1).

Replaces the hexapod `ros_robot_controller_sdk` backend with ROS 1 topics:
  - /cmd_vel                          (geometry_msgs/Twist)  — body velocity
  - /firmware/wheel_*/cmd_velocity    (std_msgs/Float32)     — per-wheel (rad/s)
  - /firmware/battery                 (std_msgs/Float32)     — battery voltage (V)

Duty values (-max_duty … +max_duty) are scaled to SI units before publishing.
All hexapod-specific features (servos, buzzer, RGB LEDs) become no-ops so
server.py requires no changes.

Run server.py inside a sourced ROS 1 environment:
    source /opt/ros/noetic/setup.bash
    python3 server.py
"""

from __future__ import annotations

import threading
import time

# ROS 1 packages — available on the Leo Rover but not in the laptop venv.
# IDE "unresolved import" warnings here are expected and can be ignored.
import rospy  # type: ignore[import-untyped]
from geometry_msgs.msg import Twist  # type: ignore[import-untyped]
from std_msgs.msg import Float32  # type: ignore[import-untyped]


# ---------------------------------------------------------------------------
# Scaling constants
# Leo Rover max linear speed ≈ 0.4 m/s; max angular speed ≈ 1.0 rad/s.
# Duty values are normalised against max_duty (default 80) to these limits.
# ---------------------------------------------------------------------------
_MAX_LINEAR_MS   = 0.4   # m/s  at duty == max_duty
_MAX_ANGULAR_RDS = 1.0   # rad/s at duty == max_duty
_WHEEL_RADIUS_M  = 0.065 # metres — used to convert linear m/s → wheel rad/s


def mecanum_ik(vx: float, vy: float, omega: float) -> list[list]:
    """Return per-wheel duty values for logging compatibility.

    Leo Rover firmware handles the actual IK via /cmd_vel, so this is only
    used to produce a wheel-duty log entry that matches the hexapod format.
    """
    v1 = vx - vy - omega
    v2 = vx + vy + omega
    v3 = vx + vy - omega
    v4 = vx - vy + omega
    return [[1, -v1], [2, v2], [3, -v3], [4, v4]]


# Wheel index → Leo Rover topic suffix (matches hexapod wheel ID order)
_WHEEL_TOPICS = {1: "FL", 2: "FR", 3: "RL", 4: "RR"}


class MotorController:
    """Thread-safe motor controller for Leo Rover via ROS 1 (rospy)."""

    def __init__(self, max_duty: float = 80.0):
        self.max_duty = max_duty
        self._lock = threading.Lock()
        self._last_command_time = time.monotonic()
        self._battery_mv: int | None = None

        rospy.init_node("turbovla_motor_controller", anonymous=False,
                        disable_signals=True)

        # Body velocity publisher
        self._cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        # Per-wheel velocity publishers (rad/s)
        self._wheel_pubs = {
            idx: rospy.Publisher(
                f"/firmware/wheel_{name}/cmd_velocity", Float32, queue_size=10
            )
            for idx, name in _WHEEL_TOPICS.items()
        }

        # Battery subscriber — Leo Rover publishes Volts as Float32
        rospy.Subscriber("/firmware/battery", Float32, self._battery_cb)

        # Spin in a background daemon thread so callbacks are processed
        self._spin_thread = threading.Thread(target=rospy.spin, daemon=True)
        self._spin_thread.start()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _battery_cb(self, msg: Float32) -> None:
        self._battery_mv = int(msg.data * 1000)

    def _clamp(self, value: float) -> float:
        return max(-self.max_duty, min(self.max_duty, value))

    def _duty_to_linear(self, duty: float) -> float:
        return duty / self.max_duty * _MAX_LINEAR_MS

    def _duty_to_angular(self, duty: float) -> float:
        return duty / self.max_duty * _MAX_ANGULAR_RDS

    def _duty_to_wheel_rads(self, duty: float) -> float:
        return self._duty_to_linear(duty) / _WHEEL_RADIUS_M

    # ------------------------------------------------------------------
    # Public motor API (identical signature to hexapod version)
    # ------------------------------------------------------------------

    def set_velocity(self, vx: float, vy: float, omega: float) -> list[list]:
        """Send a body velocity command.

        Args:
            vx:    forward/backward duty (-max_duty … +max_duty)
            vy:    strafe left/right duty (Leo Rover ignores this — diff drive)
            omega: rotation duty

        Returns:
            Per-wheel duty list for logging (same format as hexapod).
        """
        vx    = self._clamp(vx)
        vy    = self._clamp(vy)
        omega = self._clamp(omega)

        # Leo Rover is differential drive — vy (strafe) is impossible, so fold it
        # into angular.z rotation.  omega (explicit rotate) takes priority; vy
        # is only used when omega is zero so the two commands don't stack.
        effective_omega = omega if omega != 0.0 else vy

        twist = Twist()
        twist.linear.x  = self._duty_to_linear(vx)
        twist.angular.z = self._duty_to_angular(effective_omega)

        with self._lock:
            self._cmd_vel_pub.publish(twist)
            self._last_command_time = time.monotonic()

        return mecanum_ik(vx, vy, omega)

    def set_raw_wheels(self, wheels: list[list]) -> None:
        """Send per-wheel velocity commands.

        Args:
            wheels: [[1, duty1], [2, duty2], [3, duty3], [4, duty4]]
                    Wheel IDs: 1=FL, 2=FR, 3=RL, 4=RR
        """
        with self._lock:
            for wheel_id, duty in wheels:
                duty = self._clamp(duty)
                msg = Float32()
                msg.data = float(self._duty_to_wheel_rads(duty))
                pub = self._wheel_pubs.get(wheel_id)
                if pub:
                    pub.publish(msg)
            self._last_command_time = time.monotonic()

    def stop(self) -> None:
        """Emergency stop — publish zero Twist."""
        with self._lock:
            self._cmd_vel_pub.publish(Twist())
            self._last_command_time = time.monotonic()

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def get_battery_mv(self) -> int | None:
        """Battery voltage in millivolts (from /firmware/battery topic)."""
        return self._battery_mv

    def get_imu(self) -> tuple | None:
        """IMU data is available via /firmware/imu — not polled here."""
        return None

    # ------------------------------------------------------------------
    # No-ops: hardware not present on Leo Rover
    # ------------------------------------------------------------------

    def center_servos(self) -> None:
        """No-op: Leo Rover has no pan-tilt servos."""

    def set_servos(self, *_) -> None:
        """No-op: Leo Rover has no pan-tilt servos."""

    def beep(self, *_) -> None:
        """No-op: Leo Rover has no buzzer."""

    def set_rgb(self, *_) -> None:
        """No-op: Leo Rover has no onboard RGB LEDs."""

    # ------------------------------------------------------------------
    # Watchdog support
    # ------------------------------------------------------------------

    @property
    def seconds_since_last_command(self) -> float:
        """Seconds elapsed since the last motor command."""
        return time.monotonic() - self._last_command_time

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Signal ROS 1 node shutdown."""
        rospy.signal_shutdown("turbovla server shutting down")
