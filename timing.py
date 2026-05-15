"""FPS timing utilities for the recording pipeline.

Ported from LeRobot's precise_sleep with Windows spin-wait support.
"""
import time
import platform
from collections import deque


IS_WINDOWS = platform.system() == "Windows"
IS_MACOS = platform.system() == "Darwin"


def precise_sleep(seconds: float, spin_threshold: float = 0.010,
                  sleep_margin: float = 0.005) -> None:
    """High-precision sleep using hybrid sleep + spin-wait.

    On Windows/macOS, time.sleep() has ~15ms jitter. This function
    uses time.sleep() for the bulk then spin-waits for the final ms.

    Args:
        seconds: Time to sleep in seconds.
        spin_threshold: Switch to spin-wait when remaining time is below this.
        sleep_margin: Safety margin subtracted from sleep time.
    """
    if seconds <= 0:
        return

    if not (IS_WINDOWS or IS_MACOS):
        # Linux has good enough sleep precision
        time.sleep(seconds)
        return

    end = time.perf_counter() + seconds
    while True:
        remaining = end - time.perf_counter()
        if remaining <= 0:
            break
        if remaining > spin_threshold:
            time.sleep(remaining - sleep_margin)
        # else: spin-wait (busy loop for final milliseconds)


class FPSRegulator:
    """Maintains a target FPS by sleeping between loop iterations."""

    def __init__(self, target_fps: float = 10.0, history_size: int = 100):
        self.target_dt = 1.0 / target_fps
        self.target_fps = target_fps
        self._last_tick: float | None = None
        self._dt_history: deque[float] = deque(maxlen=history_size)

    def tick(self) -> float:
        """Call at the START of each loop iteration.

        Sleeps to maintain target FPS since last tick.
        Returns actual dt since last tick (seconds).
        """
        now = time.perf_counter()

        if self._last_tick is not None:
            elapsed = now - self._last_tick
            sleep_time = self.target_dt - elapsed
            if sleep_time > 0:
                precise_sleep(sleep_time)
            now = time.perf_counter()
            actual_dt = now - self._last_tick
            self._dt_history.append(actual_dt)
        else:
            actual_dt = 0.0

        self._last_tick = now
        return actual_dt

    @property
    def actual_fps(self) -> float:
        """Rolling average FPS."""
        if not self._dt_history:
            return 0.0
        avg_dt = sum(self._dt_history) / len(self._dt_history)
        return 1.0 / avg_dt if avg_dt > 0 else 0.0

    @property
    def is_lagging(self) -> bool:
        """True if actual FPS is more than 20% below target."""
        return self.actual_fps < self.target_fps * 0.8

    def reset(self) -> None:
        """Reset timing state (call when starting a new episode)."""
        self._last_tick = None
        self._dt_history.clear()
