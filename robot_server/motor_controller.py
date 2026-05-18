#!/usr/bin/env python3
"""Motor controller for the XiaoRGEEK hexapod robot (ROS 1).

Publishes to topics consumed by the hexapod_controller C++ node:
  - /cmd_vel       (geometry_msgs/Twist)        — locomotion
  - /state         (std_msgs/Bool)               — stand (True) / sit (False)
  - /head_scalar   (geometry_msgs/AccelStamped)  — camera pan / tilt
  - /body_scalar   (geometry_msgs/AccelStamped)  — body orientation override

Subscribes to:
  - /imu/data      (sensor_msgs/Imu)             — IMU readings

Axis-swap note
--------------
The hexapod_controller cmd_velCallback negates and swaps axes before passing
them into the gait engine:
  gait forward ← -angular_z_in  (scaled by maxRadiansPerSec → maxMeterPerSec)
  gait turning ← -linear_x_in   (scaled by maxMeterPerSec  → maxRadiansPerSec)
  gait strafe  ←  linear_y_in   (passed through unchanged)

set_velocity() compensates for this so that vx>0 always means forward,
omega>0 always means left turn, and vy>0 always means strafe left.

Run server.py inside a sourced ROS 1 environment:
    source /opt/ros/noetic/setup.bash
    python3 server.py
"""

from __future__ import annotations

import math
import threading
import time

import rospy  # type: ignore[import-untyped]
from geometry_msgs.msg import AccelStamped, Twist  # type: ignore[import-untyped]
from sensor_msgs.msg import Imu  # type: ignore[import-untyped]
from std_msgs.msg import Bool  # type: ignore[import-untyped]

try:
    from hexapod_msgs.msg import Sounds as _HexSounds  # type: ignore[import-untyped]
    _HEXAPOD_MSGS = True
except ImportError:
    _HEXAPOD_MSGS = False


# ---------------------------------------------------------------------------
# Scaling constants
# These should match MAX_METERS_PER_SEC and MAX_RADIANS_PER_SEC in the
# hexapod ROS parameter server (loaded from the hexapod YAML config).
# Duty values are normalised against max_duty before scaling to SI units.
# ---------------------------------------------------------------------------
_MAX_LINEAR_MS   = 0.3   # m/s   — typical hexapod forward speed at max_duty
_MAX_ANGULAR_RDS = 1.3   # rad/s — typical hexapod yaw rate at max_duty


class MotorController:
    """Thread-safe motor controller for the XiaoRGEEK hexapod via ROS 1."""

    def __init__(self, max_duty: float = 80.0, auto_stand: bool = True):
        self.max_duty = max_duty
        self._lock = threading.Lock()
        self._last_command_time = time.monotonic()
        self._imu_data: tuple[float, float, float] | None = None

        rospy.init_node("turbovla_motor_controller", anonymous=False,
                        disable_signals=True)

        self._cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self._state_pub   = rospy.Publisher("/state", Bool, queue_size=10)
        self._head_pub    = rospy.Publisher("/head_scalar", AccelStamped, queue_size=10)
        self._body_pub    = rospy.Publisher("/body_scalar", AccelStamped, queue_size=10)

        if _HEXAPOD_MSGS:
            self._sounds_pub = rospy.Publisher("/sounds", _HexSounds, queue_size=10)
        else:
            self._sounds_pub = None

        rospy.Subscriber("/imu/data", Imu, self._imu_cb)

        self._spin_thread = threading.Thread(target=rospy.spin, daemon=True)
        self._spin_thread.start()

        # Give publishers time to register with the master before first command.
        rospy.sleep(0.5)

        if auto_stand:
            self.stand()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _imu_cb(self, msg: Imu) -> None:
        q = msg.orientation
        sinr = 2.0 * (q.w * q.x + q.y * q.z)
        cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        roll = math.atan2(sinr, cosr)

        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        pitch = (math.copysign(math.pi / 2.0, sinp)
                 if abs(sinp) >= 1.0 else math.asin(sinp))

        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny, cosy)

        self._imu_data = (roll, pitch, yaw)

    def _clamp(self, value: float) -> float:
        return max(-self.max_duty, min(self.max_duty, value))

    def _duty_to_linear(self, duty: float) -> float:
        return duty / self.max_duty * _MAX_LINEAR_MS

    def _duty_to_angular(self, duty: float) -> float:
        return duty / self.max_duty * _MAX_ANGULAR_RDS

    # ------------------------------------------------------------------
    # Hexapod-specific stand / sit
    # ------------------------------------------------------------------

    def stand(self) -> None:
        """Publish True on /state to trigger the hexapod stand-up sequence."""
        with self._lock:
            self._state_pub.publish(Bool(data=True))

    def sit(self) -> None:
        """Publish False on /state to trigger the hexapod sit-down sequence."""
        with self._lock:
            self._state_pub.publish(Bool(data=False))

    # ------------------------------------------------------------------
    # Public motor API (identical signature to the Leo Rover version)
    # ------------------------------------------------------------------

    def set_velocity(self, vx: float, vy: float, omega: float) -> list[list]:
        """Send a body velocity command to the hexapod.

        Args:
            vx:    forward/backward duty (-max_duty … +max_duty)
            vy:    strafe left/right duty (hexapod supports lateral motion)
            omega: rotation duty (positive = counter-clockwise / left turn)

        Returns:
            Per-leg duty list for logging (6 legs, simplified).
        """
        vx    = self._clamp(vx)
        vy    = self._clamp(vy)
        omega = self._clamp(omega)

        # The hexapod_controller cmd_velCallback negates and swaps linear.x ↔
        # angular.z before passing them to the gait engine.  We pre-compensate
        # so that the caller's semantics (vx=forward, omega=left-turn) are
        # preserved: publish -omega on linear.x and -vx on angular.z.
        twist = Twist()
        twist.linear.x  = -self._duty_to_linear(omega)   # → gait turning
        twist.linear.y  =  self._duty_to_linear(vy)       # → gait strafe
        twist.angular.z = -self._duty_to_angular(vx)      # → gait forward

        with self._lock:
            self._cmd_vel_pub.publish(twist)
            self._last_command_time = time.monotonic()

        # Return 6-leg pseudo-duty list for logging compatibility.
        norm = vx / self.max_duty if self.max_duty else 0.0
        return [[i + 1, norm] for i in range(6)]

    def set_raw_wheels(self, wheels: list[list]) -> None:
        """No-op: hexapod has no individually-driven wheels."""

    def stop(self) -> None:
        """Emergency stop — publish zero Twist."""
        with self._lock:
            self._cmd_vel_pub.publish(Twist())
            self._last_command_time = time.monotonic()

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def get_battery_mv(self) -> int | None:
        """Battery voltage in millivolts. Not published by the hexapod stack."""
        return None

    def get_imu(self) -> tuple[float, float, float] | None:
        """Latest IMU reading as (roll, pitch, yaw) in radians, or None."""
        return self._imu_data

    # ------------------------------------------------------------------
    # Servos / head pan-tilt
    # ------------------------------------------------------------------

    def center_servos(self) -> None:
        """Center the camera pan-tilt by publishing zero AccelStamped."""
        msg = AccelStamped()
        msg.header.stamp = rospy.Time.now()
        with self._lock:
            self._head_pub.publish(msg)

    def set_servos(self, pan: float = 0.0, tilt: float = 0.0, *_) -> None:
        """Command the camera pan/tilt servos via /head_scalar.

        Args:
            pan:  horizontal angle duty (-max_duty … +max_duty).
                  Maps to accel.angular.z; scaled to [-1, 1] for HEAD_MAX_YAW.
            tilt: vertical angle duty (-max_duty … +max_duty).
                  Maps to accel.angular.y; scaled to [-1, 1] for HEAD_MAX_PITCH.
        """
        pan  = self._clamp(pan)
        tilt = self._clamp(tilt)

        msg = AccelStamped()
        msg.header.stamp    = rospy.Time.now()
        msg.accel.angular.z = pan  / self.max_duty   # normalised [-1, 1]
        msg.accel.angular.y = tilt / self.max_duty

        with self._lock:
            self._head_pub.publish(msg)

    def beep(self, *_) -> None:
        """Trigger a hexapod sound via /sounds (requires hexapod_msgs)."""
        if self._sounds_pub is None:
            return
        msg = _HexSounds()
        msg.stand = True
        with self._lock:
            self._sounds_pub.publish(msg)

    def set_rgb(self, *_) -> None:
        """No-op: hexapod has no onboard RGB LEDs."""

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
        """Sit the hexapod down, then signal ROS 1 node shutdown."""
        self.sit()
        rospy.sleep(0.5)
        rospy.signal_shutdown("turbovla server shutting down")
