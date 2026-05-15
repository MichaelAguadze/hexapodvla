"""Non-blocking keyboard teleop controller using pynput.

Tracks held keys for smooth velocity control.
Also handles episode management keys (arrows, escape).
"""
import threading
import numpy as np

try:
    from pynput import keyboard
except ImportError:
    raise ImportError("pynput is required: pip install pynput")


class TeleopController:
    """Non-blocking keyboard teleop with held-key tracking."""

    def __init__(self, speed: float = 50.0, max_speed: float = 100.0,
                 min_speed: float = 10.0, speed_step: float = 10.0):
        self.speed = speed
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.speed_step = speed_step

        # Currently held keys
        self._held: set[str] = set()
        self._lock = threading.Lock()
        self._listener: keyboard.Listener | None = None

        # Episode control events (polled by recording session)
        self.events = {
            "accept_episode": False,   # Right arrow
            "discard_episode": False,  # Left arrow
            "stop_session": False,     # Escape
            "enter_pressed": False,    # Enter key
        }

    def start(self) -> None:
        """Start listening for keyboard events."""
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.daemon = True
        self._listener.start()

    def stop(self) -> None:
        """Stop keyboard listener."""
        if self._listener:
            self._listener.stop()
            self._listener = None
        with self._lock:
            self._held.clear()

    def _key_to_str(self, key) -> str:
        """Normalize key to string."""
        if isinstance(key, keyboard.KeyCode):
            return key.char.lower() if key.char else ""
        elif key == keyboard.Key.up:
            return "up"
        elif key == keyboard.Key.down:
            return "down"
        elif key == keyboard.Key.left:
            return "left"
        elif key == keyboard.Key.right:
            return "right"
        elif key == keyboard.Key.space:
            return "space"
        elif key == keyboard.Key.esc:
            return "esc"
        elif key == keyboard.Key.enter:
            return "enter"
        else:
            return str(key)

    def _on_press(self, key) -> None:
        k = self._key_to_str(key)
        with self._lock:
            self._held.add(k)

        # Episode control events
        if k == "right":
            self.events["accept_episode"] = True
        elif k == "left":
            self.events["discard_episode"] = True
        elif k == "esc":
            self.events["stop_session"] = True
        elif k == "enter":
            self.events["enter_pressed"] = True

        # Speed adjustment
        if k in ("+", "="):
            self.speed = min(self.max_speed, self.speed + self.speed_step)
        elif k in ("-", "_"):
            self.speed = max(self.min_speed, self.speed - self.speed_step)

    def _on_release(self, key) -> None:
        k = self._key_to_str(key)
        with self._lock:
            self._held.discard(k)

    def clear_events(self) -> None:
        """Reset all episode control events."""
        for k in self.events:
            self.events[k] = False

    def wait_for_enter(self) -> None:
        """Block until Enter is pressed."""
        self.events["enter_pressed"] = False
        while not self.events["enter_pressed"] and not self.events["stop_session"]:
            import time
            time.sleep(0.05)
        self.events["enter_pressed"] = False

    def get_action(self) -> tuple[float, float, float]:
        """Get current velocity command as duty-cycle values.

        Returns:
            (vx, vy, omega) in duty cycle units [-speed, speed]
        """
        with self._lock:
            held = set(self._held)

        vx, vy, omega = 0.0, 0.0, 0.0

        if "w" in held:
            vx = self.speed
        elif "s" in held:
            vx = -self.speed

        if "a" in held:
            vy = self.speed
        elif "d" in held:
            vy = -self.speed

        if "q" in held:
            omega = self.speed
        elif "e" in held:
            omega = -self.speed

        if "space" in held:
            vx, vy, omega = 0.0, 0.0, 0.0

        return vx, vy, omega

    def get_normalized_action(self, duty_range: float = 80.0) -> np.ndarray:
        """Get current velocity normalized to [-1, 1].

        Args:
            duty_range: The max duty cycle value to normalize against.

        Returns:
            np.array([vx, vy, omega], dtype=float32) in [-1, 1]
        """
        vx, vy, omega = self.get_action()
        return np.array([vx / duty_range, vy / duty_range, omega / duty_range],
                        dtype=np.float32)
