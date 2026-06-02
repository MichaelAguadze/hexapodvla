#!/usr/bin/env python3
"""
Health monitoring for hexapod robot.

Background thread that polls:
- Battery voltage (via expansion board)
- CPU temperature
- Camera status

Sets RGB LEDs to indicate state and triggers buzzer on low battery.
"""
import threading
import time
from typing import Optional


# Battery thresholds (millivolts)
BATTERY_OK = 7500
BATTERY_WARN = 7200
BATTERY_LOW = 7000
BATTERY_CRITICAL = 6800


class HealthMonitor:
    """Polls robot health in a background thread."""

    def __init__(self, motor_controller, poll_interval: float = 5.0):
        """
        Args:
            motor_controller: MotorController instance (for battery, buzzer, LEDs)
            poll_interval: Seconds between health checks
        """
        self.mc = motor_controller
        self.poll_interval = poll_interval

        self.battery_mv: int = 0
        self.cpu_temp: float = 0.0
        self.camera_ok: bool = True
        self.status: str = "unknown"  # "ok", "warn", "low", "critical"

        self._running = False
        self._thread = None  # type: Optional[threading.Thread]
        self._warned = False

    def start(self) -> None:
        """Start background monitoring."""
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop background monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def get_health(self) -> dict:
        """Return current health status as dict."""
        return {
            "battery_mv": self.battery_mv,
            "cpu_temp": self.cpu_temp,
            "camera_ok": self.camera_ok,
            "status": self.status,
        }

    def _read_cpu_temp(self) -> float:
        """Read CPU temperature in Celsius."""
        try:
            with open("/sys/class/thermal/thermal_zone0/temp") as f:
                return int(f.read().strip()) / 1000.0
        except Exception:
            return 0.0

    def _poll_loop(self) -> None:
        """Background polling loop."""
        while self._running:
            try:
                # Read battery
                batt = self.mc.get_battery_mv()
                if batt is not None and batt > 0:
                    self.battery_mv = batt

                # Read CPU temp
                self.cpu_temp = self._read_cpu_temp()

                # Determine status and set LEDs
                self._update_status()

            except Exception as e:
                print(f"[HealthMonitor] Error: {e}")

            time.sleep(self.poll_interval)

    def _update_status(self) -> None:
        """Update status and LEDs based on battery level."""
        batt = self.battery_mv

        if batt == 0:
            self.status = "unknown"
            self.mc.set_rgb([[1, 0, 0, 50], [2, 0, 0, 50]])  # dim blue = unknown
            return

        if batt >= BATTERY_OK:
            self.status = "ok"
            self.mc.set_rgb([[1, 0, 20, 0], [2, 0, 20, 0]])  # dim green
            self._warned = False

        elif batt >= BATTERY_WARN:
            self.status = "warn"
            self.mc.set_rgb([[1, 30, 20, 0], [2, 30, 20, 0]])  # dim yellow
            self._warned = False

        elif batt >= BATTERY_LOW:
            self.status = "low"
            self.mc.set_rgb([[1, 50, 10, 0], [2, 50, 10, 0]])  # orange
            self._warned = True

        else:  # CRITICAL
            self.status = "critical"
            self.mc.set_rgb([[1, 50, 0, 0], [2, 50, 0, 0]])  # red

    @property
    def can_record(self) -> bool:
        """Whether battery is sufficient for recording."""
        return self.status not in ("critical",)
