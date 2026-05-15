"""HTTP client for communicating with the TurboPi robot server."""
import time
import io

import numpy as np
import requests
from PIL import Image


class RobotClient:
    """Connects to the robot server over HTTP."""

    def __init__(self, robot_url: str = "http://192.168.149.1:8081",
                 timeout: float = 2.0, max_retries: int = 3):
        self.robot_url = robot_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        # Keep-alive for lower latency
        self.session.headers.update({"Connection": "keep-alive"})

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        """HTTP request with retry logic."""
        kwargs.setdefault("timeout", self.timeout)
        for attempt in range(self.max_retries):
            try:
                resp = self.session.request(method, f"{self.robot_url}{path}", **kwargs)
                return resp
            except (requests.ConnectionError, requests.Timeout):
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(0.3 * (attempt + 1))
        raise requests.ConnectionError(f"Failed after {self.max_retries} retries")

    def get_frame(self) -> tuple[np.ndarray, float, int]:
        """Fetch a single camera frame with timestamp.

        Returns:
            (image_bgr, timestamp, frame_index)
            image_bgr: numpy array (480, 640, 3) uint8 BGR
            timestamp: monotonic time from robot
            frame_index: incrementing frame counter from robot
        """
        resp = self._request("GET", "/snapshot")
        if resp.status_code != 200:
            raise RuntimeError(f"Snapshot failed: HTTP {resp.status_code}")

        timestamp = float(resp.headers.get("X-Timestamp", 0))
        frame_index = int(resp.headers.get("X-Frame-Index", 0))

        # Decode JPEG to numpy
        img = Image.open(io.BytesIO(resp.content))
        frame = np.array(img)

        # PIL gives RGB, convert to BGR for OpenCV compatibility
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame = frame[:, :, ::-1].copy()

        return frame, timestamp, frame_index

    def get_frame_rgb(self) -> tuple[np.ndarray, float, int]:
        """Fetch a single camera frame in RGB format (for dataset storage).

        Returns:
            (image_rgb, timestamp, frame_index)
        """
        resp = self._request("GET", "/snapshot")
        if resp.status_code != 200:
            raise RuntimeError(f"Snapshot failed: HTTP {resp.status_code}")

        timestamp = float(resp.headers.get("X-Timestamp", 0))
        frame_index = int(resp.headers.get("X-Frame-Index", 0))

        img = Image.open(io.BytesIO(resp.content))
        frame = np.array(img)  # RGB

        return frame, timestamp, frame_index

    def send_motor(self, wheels: list[list]) -> bool:
        """Send raw wheel duties.

        Args:
            wheels: [[1,d1],[2,d2],[3,d3],[4,d4]]
        """
        resp = self._request("POST", "/motor", json={"wheels": wheels})
        return resp.status_code == 200

    def send_velocity(self, vx: float, vy: float, omega: float) -> bool:
        """Send body velocity (server does IK).

        Args:
            vx: forward/backward duty
            vy: strafe left/right duty
            omega: rotation duty
        """
        resp = self._request("POST", "/velocity",
                             json={"vx": vx, "vy": vy, "omega": omega})
        return resp.status_code == 200

    def stop(self) -> bool:
        """Emergency stop all motors."""
        try:
            resp = self._request("POST", "/stop", timeout=1.0)
            return resp.status_code == 200
        except Exception:
            return False

    def get_health(self) -> dict:
        """Get robot health status."""
        resp = self._request("GET", "/health")
        return resp.json()

    def beep(self, freq: int = 1900, duration: float = 0.1) -> None:
        """Play buzzer tone on robot."""
        self._request("POST", "/buzzer",
                      json={"freq": freq, "duration": duration})

    def set_servos(self, positions: list[list]) -> None:
        """Set servo positions."""
        self._request("POST", "/servo", json={"servos": positions})

    def is_connected(self) -> bool:
        """Check if robot server is reachable."""
        try:
            resp = self._request("GET", "/health", timeout=1.0)
            return resp.status_code == 200
        except Exception:
            return False

    @property
    def stream_url(self) -> str:
        """URL for MJPEG live stream."""
        return f"{self.robot_url}/stream"

    @property
    def snapshot_url(self) -> str:
        """URL for single snapshot."""
        return f"{self.robot_url}/snapshot"
