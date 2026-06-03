"""PS5 DualSense gamepad controller for hexapod teleop.

Drop-in replacement for TeleopController with identical public interface.

Stick / button mapping (DualSense via USB or Bluetooth):
    Left  stick Y  → vx  (forward / backward)
    Left  stick X  → vy  (strafe left / right)
    Right stick X  → omega (rotate left / right)
    Cross (X) held → emergency stop (zero all axes)
    R1             → speed up
    L1             → speed down
    Triangle       → accept_episode event
    Square         → discard_episode event
    Options        → stop_session event
    Touchpad click → enter_pressed event

Axis indices follow the SDL2 / pygame mapping for DualSense on macOS and
Linux.  If your axis assignments differ, pass custom indices to __init__.
"""

import threading
import time

import numpy as np

try:
    import pygame
except ImportError:
    raise ImportError("pygame is required: pip install pygame")

# Default DualSense axis indices (SDL2 / pygame)
_AXIS_LX = 0   # left stick horizontal
_AXIS_LY = 1   # left stick vertical   (up = negative)
_AXIS_RX = 2   # right stick horizontal

# Default DualSense button indices (SDL2 / pygame)
_BTN_CROSS    = 0
_BTN_CIRCLE   = 1
_BTN_SQUARE   = 2
_BTN_TRIANGLE = 3
_BTN_L1       = 4
_BTN_R1       = 5
_BTN_L2       = 6
_BTN_R2       = 7
_BTN_CREATE   = 8
_BTN_OPTIONS  = 9
_BTN_L3       = 10
_BTN_R3       = 11
_BTN_PS       = 12
_BTN_TOUCHPAD = 13

_DEADZONE = 0.08  # ignore stick values below this magnitude


class PS5Controller:
    """Non-blocking PS5 DualSense gamepad controller.

    Identical public interface to TeleopController so it can be swapped in
    anywhere a TeleopController is used.
    """

    def __init__(
        self,
        speed: float = 50.0,
        max_speed: float = 100.0,
        min_speed: float = 10.0,
        speed_step: float = 10.0,
        joystick_index: int = 0,
        deadzone: float = _DEADZONE,
        axis_lx: int = _AXIS_LX,
        axis_ly: int = _AXIS_LY,
        axis_rx: int = _AXIS_RX,
    ):
        self.speed = speed
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.speed_step = speed_step
        self.joystick_index = joystick_index
        self.deadzone = deadzone

        self._axis_lx = axis_lx
        self._axis_ly = axis_ly
        self._axis_rx = axis_rx

        self._joystick = None
        self._running = False
        self._thread = None

        self.events = {
            "accept_episode":  False,
            "discard_episode": False,
            "stop_session":    False,
            "enter_pressed":   False,
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Connect to the PS5 controller and start the event pump."""
        pygame.init()
        pygame.joystick.init()

        count = pygame.joystick.get_count()
        if count == 0:
            raise RuntimeError(
                "No gamepad detected. Connect the PS5 controller via USB or Bluetooth."
            )

        self._joystick = pygame.joystick.Joystick(self.joystick_index)
        self._joystick.init()
        print("[PS5] Connected: {}  ({} axes, {} buttons)".format(
            self._joystick.get_name(),
            self._joystick.get_numaxes(),
            self._joystick.get_numbuttons(),
        ))

        self._running = True
        self._thread = threading.Thread(target=self._event_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the event pump and release pygame resources."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        try:
            pygame.joystick.quit()
            pygame.quit()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal event pump
    # ------------------------------------------------------------------

    def _event_loop(self) -> None:
        """Background thread: pump pygame events to detect button presses."""
        while self._running:
            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN:
                    self._on_button_down(event.button)
                elif event.type == pygame.JOYDEVICEREMOVED:
                    print("[PS5] Controller disconnected!")
            time.sleep(0.01)

    def _on_button_down(self, button: int) -> None:
        if button == _BTN_TRIANGLE:
            self.events["accept_episode"] = True
        elif button == _BTN_SQUARE:
            self.events["discard_episode"] = True
        elif button == _BTN_OPTIONS:
            self.events["stop_session"] = True
        elif button == _BTN_TOUCHPAD:
            self.events["enter_pressed"] = True
        elif button == _BTN_R1:
            self.speed = min(self.max_speed, self.speed + self.speed_step)
            print("\r[PS5] Speed: {:.0f}  ".format(self.speed), end="", flush=True)
        elif button == _BTN_L1:
            self.speed = max(self.min_speed, self.speed - self.speed_step)
            print("\r[PS5] Speed: {:.0f}  ".format(self.speed), end="", flush=True)

    # ------------------------------------------------------------------
    # Axis helpers
    # ------------------------------------------------------------------

    def _apply_deadzone(self, value: float) -> float:
        """Zero values inside the deadzone; rescale the live range to [0, 1]."""
        if abs(value) < self.deadzone:
            return 0.0
        sign = 1.0 if value > 0 else -1.0
        return sign * (abs(value) - self.deadzone) / (1.0 - self.deadzone)

    def _axis(self, index: int) -> float:
        if self._joystick is None:
            return 0.0
        return self._apply_deadzone(self._joystick.get_axis(index))

    # ------------------------------------------------------------------
    # Public API (mirrors TeleopController)
    # ------------------------------------------------------------------

    def get_action(self) -> tuple:
        """Return current velocity command as (vx, vy, omega) duty values.

        Cross (X) held overrides everything with a full stop.
        """
        if self._joystick is None:
            return 0.0, 0.0, 0.0

        if self._joystick.get_button(_BTN_CROSS):
            return 0.0, 0.0, 0.0

        lx = self._axis(self._axis_lx)
        ly = self._axis(self._axis_ly)
        rx = self._axis(self._axis_rx)

        vx    = -ly * self.speed   # stick up (negative Y) → forward
        vy    = -lx * self.speed   # stick left (negative X) → strafe left
        omega = -rx * self.speed   # right stick left (negative X) → rotate left

        return vx, vy, omega

    def get_normalized_action(self, duty_range: float = 80.0) -> np.ndarray:
        """Return current velocity normalized to [-1, 1].

        Args:
            duty_range: Max duty cycle value to normalise against.

        Returns:
            np.array([vx, vy, omega], dtype=float32) in [-1, 1]
        """
        vx, vy, omega = self.get_action()
        return np.array(
            [vx / duty_range, vy / duty_range, omega / duty_range],
            dtype=np.float32,
        )

    def clear_events(self) -> None:
        """Reset all episode control events."""
        for k in self.events:
            self.events[k] = False

    def wait_for_enter(self) -> None:
        """Block until the touchpad is clicked (or stop_session fires)."""
        self.events["enter_pressed"] = False
        while not self.events["enter_pressed"] and not self.events["stop_session"]:
            time.sleep(0.05)
        self.events["enter_pressed"] = False

    def is_connected(self) -> bool:
        """Return True if a joystick is initialised."""
        return self._joystick is not None
