#!/usr/bin/env python3
"""
hexapod Robot Server — lightweight HTTP API for remote control.

Runs on the Raspberry Pi 5. Exposes camera, motors, and health over HTTP.
The recording client on the laptop connects to this server.

Endpoints:
    GET  /stream    — MJPEG live stream (for dashboard)
    GET  /snapshot  — Single JPEG frame with X-Timestamp header
    POST /motor     — Send wheel duties: {"wheels": [[1,d1],[2,d2],[3,d3],[4,d4]]}
    POST /velocity  — Send body velocity: {"vx": 0, "vy": 0, "omega": 0}
    POST /stop      — Emergency stop all motors
    GET  /health    — Battery, CPU temp, camera status
    POST /servo     — Set servos: {"servos": [[1,1500],[2,1500]]}
    POST /buzzer    — Play tone: {"freq": 1900, "duration": 0.1}
    GET  /          — Simple status page

Usage:
    python3 server.py [--port 8080]
"""
import os
import sys
import time
import json
import signal
import argparse
import threading
from io import BytesIO
from typing import Optional, Tuple

import cv2
from flask import Flask, Response, request, jsonify

# Import our modules from the same directory
from motor_controller import MotorController
from health_monitor import HealthMonitor


# --- Camera Capture Thread ---

class CameraCapture:
    """Background thread that captures camera frames."""

    def __init__(self, device: int = 0, width: int = 640, height: int = 480,
                 fps: int = 30, jpeg_quality: int = 70):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.jpeg_quality = jpeg_quality

        self._frame = None  # type: Optional[bytes]
        self._raw_frame = None
        self._timestamp = 0.0  # type: float
        self._frame_index = 0  # type: int
        self._lock = threading.Lock()
        self._running = False
        self._thread = None  # type: Optional[threading.Thread]
        self._cap = None  # type: Optional[object]
        self._use_openni = False

    def start(self) -> bool:
        """Start camera capture. Returns True if camera opened successfully."""
        # Try standard V4L2 first
        os.system("sudo fuser -k /dev/video{} 2>/dev/null".format(self.device))
        time.sleep(0.5)

        self._cap = cv2.VideoCapture(self.device)
        self._use_openni = False

        if not self._cap.isOpened():
            print("[Camera] V4L2 failed, trying OpenNI2 (Orbbec)...")
            self._cap = cv2.VideoCapture(self.device, cv2.CAP_OPENNI2)
            self._use_openni = True

        if not self._cap.isOpened():
            print("[Camera] ERROR: Cannot open camera via V4L2 or OpenNI2")
            return False

        if not self._use_openni:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
            actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print("[Camera] Opened /dev/video{} at {}x{}".format(self.device, actual_w, actual_h))
        else:
            print("[Camera] Opened via OpenNI2 (Orbbec color stream)")

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self) -> None:
        """Stop camera capture."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()

    def _capture_loop(self) -> None:
        """Continuously capture frames."""
        # OpenNI2 color channel index (5 = CAP_OPENNI_BGR_IMAGE)
        _OPENNI_BGR = getattr(cv2, 'CAP_OPENNI_BGR_IMAGE', 5)
        while self._running:
            if self._use_openni:
                if not self._cap.grab():
                    time.sleep(0.01)
                    continue
                ret, frame = self._cap.retrieve(_OPENNI_BGR)
            else:
                ret, frame = self._cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            # Encode to JPEG
            ok, jpeg = cv2.imencode(
                ".jpg",
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
            )
            if not ok:
                time.sleep(0.01)
                continue

            with self._lock:
                self._frame = jpeg.tobytes()
                self._raw_frame = frame
                self._timestamp = time.monotonic()
                self._frame_index += 1

        print("[Camera] Capture loop ended")

    def get_jpeg(self) -> Tuple[Optional[bytes], float, int]:
        """Get latest JPEG frame, timestamp, and frame index."""
        with self._lock:
            return self._frame, self._timestamp, self._frame_index

    def get_raw(self):
        """Get latest raw BGR numpy frame."""
        with self._lock:
            return self._raw_frame

    @property
    def is_alive(self) -> bool:
        """Whether camera is capturing frames."""
        if not self._running:
            return False
        with self._lock:
            if self._timestamp == 0:
                return False
            return (time.monotonic() - self._timestamp) < 2.0


# --- ROS Camera Capture (Orbbec via ros_astra_camera) ---

class ROSCameraCapture:
    """Camera capture by subscribing to /camera/color/image_raw ROS topic.

    Used as a fallback when V4L2 and OpenNI2 both fail (e.g. Jetson + Orbbec).
    Requires that a ROS node has already been initialised (MotorController does
    this on startup).
    """

    def __init__(self, topic="/camera/color/image_raw", jpeg_quality=70):
        self.topic = topic
        self.jpeg_quality = jpeg_quality

        self._frame = None       # type: Optional[bytes]
        self._raw_frame = None
        self._timestamp = 0.0
        self._frame_index = 0
        self._lock = threading.Lock()
        self._running = False
        self._sub = None

    def start(self) -> bool:
        try:
            import rospy
            from sensor_msgs.msg import Image as RosImage
        except ImportError:
            print("[Camera] rospy not available — cannot use ROS topic.")
            return False

        try:
            self._sub = rospy.Subscriber(
                self.topic, RosImage, self._image_cb, queue_size=1,
            )
            self._running = True
            print("[Camera] Subscribed to ROS topic: {}".format(self.topic))
            return True
        except Exception as exc:
            print("[Camera] ROS subscription failed: {}".format(exc))
            return False

    def stop(self) -> None:
        self._running = False
        if self._sub:
            self._sub.unregister()

    def _decode_ros_image(self, msg) -> Optional[object]:
        """Convert sensor_msgs/Image to a BGR numpy array."""
        import numpy as np
        # Try cv_bridge first
        try:
            from cv_bridge import CvBridge
            return CvBridge().imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            pass
        # Manual fallback
        try:
            data = np.frombuffer(msg.data, dtype=np.uint8)
            img = data.reshape((msg.height, msg.width, -1))
            if msg.encoding in ("rgb8", "RGB8"):
                return img[:, :, ::-1].copy()
            return img
        except Exception:
            return None

    def _image_cb(self, msg) -> None:
        if not self._running:
            return
        frame_bgr = self._decode_ros_image(msg)
        if frame_bgr is None:
            return
        ok, jpeg = cv2.imencode(
            ".jpg", frame_bgr,
            [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not ok:
            return
        with self._lock:
            self._frame = jpeg.tobytes()
            self._raw_frame = frame_bgr
            self._timestamp = time.monotonic()
            self._frame_index += 1

    def get_jpeg(self) -> Tuple[Optional[bytes], float, int]:
        with self._lock:
            return self._frame, self._timestamp, self._frame_index

    def get_raw(self):
        with self._lock:
            return self._raw_frame

    @property
    def is_alive(self) -> bool:
        if not self._running:
            return False
        with self._lock:
            if self._timestamp == 0:
                return False
            return (time.monotonic() - self._timestamp) < 2.0


# --- Motor Watchdog ---

class MotorWatchdog:
    """Stops motors if no command received within timeout."""

    def __init__(self, motor_controller: MotorController, timeout: float = 0.5):
        self.mc = motor_controller
        self.timeout = timeout
        self._running = False
        self._thread = None  # type: Optional[threading.Thread]

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _watchdog_loop(self) -> None:
        while self._running:
            if self.mc.seconds_since_last_command > self.timeout:
                self.mc.stop()
            time.sleep(0.05)  # Check every 50ms


# --- Flask App ---

def create_app(mc: MotorController, camera: CameraCapture,
               health: HealthMonitor) -> Flask:
    """Create Flask app with all endpoints."""
    app = Flask(__name__)

    @app.after_request
    def add_cors(response):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response

    @app.route('/')
    def index():
        h = health.get_health()
        return jsonify({
            "name": "hexapod Robot Server",
            "camera": camera.is_alive,
            "health": h,
            "endpoints": ["/stream", "/snapshot", "/motor", "/velocity",
                          "/stop", "/health", "/servo", "/buzzer"],
        })

    @app.route('/stream')
    def stream():
        """MJPEG live stream for dashboard viewing."""
        def generate():
            while True:
                jpeg, _, _ = camera.get_jpeg()
                if jpeg is None:
                    time.sleep(0.033)
                    continue
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' +
                       jpeg + b'\r\n')
                time.sleep(0.033)  # ~30fps max

        return Response(generate(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/snapshot')
    def snapshot():
        """Single JPEG frame with timing headers for recording."""
        jpeg, timestamp, frame_idx = camera.get_jpeg()
        if jpeg is None:
            return Response("No frame available", status=503)

        response = Response(jpeg, mimetype='image/jpeg')
        response.headers['X-Timestamp'] = f"{timestamp:.6f}"
        response.headers['X-Frame-Index'] = str(frame_idx)
        response.headers['Cache-Control'] = 'no-cache, no-store'
        return response

    @app.route('/motor', methods=['POST', 'OPTIONS'])
    def motor():
        """Send raw wheel duties: {"wheels": [[1,d1],[2,d2],[3,d3],[4,d4]]}"""
        if request.method == 'OPTIONS':
            return '', 204

        data = request.get_json(silent=True)
        if not data or 'wheels' not in data:
            return jsonify({"ok": False, "error": "Missing 'wheels' field"}), 400

        try:
            mc.set_raw_wheels(data['wheels'])
            return jsonify({"ok": True})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route('/velocity', methods=['POST', 'OPTIONS'])
    def velocity():
        """Send body velocity: {"vx": 0, "vy": 0, "omega": 0}"""
        if request.method == 'OPTIONS':
            return '', 204

        data = request.get_json(silent=True)
        if not data:
            return jsonify({"ok": False, "error": "Missing JSON body"}), 400

        vx = float(data.get('vx', 0))
        vy = float(data.get('vy', 0))
        omega = float(data.get('omega', 0))

        try:
            wheels = mc.set_velocity(vx, vy, omega)
            return jsonify({"ok": True, "wheels": wheels})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route('/stop', methods=['POST', 'OPTIONS'])
    def stop():
        """Emergency stop all motors."""
        if request.method == 'OPTIONS':
            return '', 204
        mc.stop()
        return jsonify({"ok": True})

    @app.route('/health')
    def health_endpoint():
        """System health: battery, CPU temp, camera status."""
        h = health.get_health()
        h['camera_ok'] = camera.is_alive
        h['uptime_s'] = time.monotonic()
        return jsonify(h)

    @app.route('/servo', methods=['POST', 'OPTIONS'])
    def servo():
        """Set servo positions: {"servos": [[1,1500],[2,1500]]}"""
        if request.method == 'OPTIONS':
            return '', 204

        data = request.get_json(silent=True)
        if not data or 'servos' not in data:
            return jsonify({"ok": False, "error": "Missing 'servos' field"}), 400

        try:
            mc.set_servos(data['servos'])
            return jsonify({"ok": True})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    @app.route('/buzzer', methods=['POST', 'OPTIONS'])
    def buzzer():
        """Play buzzer tone: {"freq": 1900, "duration": 0.1}"""
        if request.method == 'OPTIONS':
            return '', 204

        data = request.get_json(silent=True) or {}
        freq = int(data.get('freq', 1900))
        duration = float(data.get('duration', 0.1))

        try:
            mc.beep(freq, duration)
            return jsonify({"ok": True})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500

    return app


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Leorover Robot Server")
    parser.add_argument('--port', type=int, default=8080, help='HTTP port')
    parser.add_argument('--camera', type=int, default=0, help='Camera device index')
    parser.add_argument('--watchdog-timeout', type=float, default=0.5,
                        help='Seconds before watchdog stops motors')
    parser.add_argument('--max-duty', type=float, default=80.0,
                        help='Maximum motor duty cycle')
    parser.add_argument('--jpeg-quality', type=int, default=70,
                        help='JPEG compression quality (0-100)')
    args = parser.parse_args()

    # Kill anything on our port
    os.system(f"sudo fuser -k {args.port}/tcp 2>/dev/null")
    time.sleep(0.3)

    print("=" * 50)
    print("  hexapod Robot Server")
    print("=" * 50)

    # Initialize motor controller
    print("[Init] Motor controller...")
    mc = MotorController(max_duty=args.max_duty)

    # Initialize camera — try V4L2, then OpenNI2, then ROS topic
    print("[Init] Camera...")
    camera = CameraCapture(device=args.camera, jpeg_quality=args.jpeg_quality)
    if not camera.start():
        print("[Camera] Trying ROS topic /camera/color/image_raw...")
        camera = ROSCameraCapture(jpeg_quality=args.jpeg_quality)
        if not camera.start():
            print("[WARNING] Camera unavailable. /stream and /snapshot will return 503.")

    # Wait up to 2s for first frame
    if camera._running:
        deadline = time.monotonic() + 2.0
        while not camera.is_alive and time.monotonic() < deadline:
            time.sleep(0.1)
        if camera.is_alive:
            print("[Camera] First frame received.")
        else:
            print("[WARNING] Camera started but no frames yet.")

    # Initialize health monitor
    print("[Init] Health monitor...")
    health = HealthMonitor(mc, poll_interval=5.0)
    health.start()

    # Initialize motor watchdog
    print("[Init] Motor watchdog (timeout={:.1f}s)...".format(args.watchdog_timeout))
    watchdog = MotorWatchdog(mc, timeout=args.watchdog_timeout)
    watchdog.start()

    # Create Flask app
    app = create_app(mc, camera, health)

    # Graceful shutdown
    def shutdown(*_):
        print("\n[Shutdown] Stopping motors...")
        mc.stop()
        mc.set_rgb([[1, 0, 0, 0], [2, 0, 0, 0]])  # LEDs off
        print("[Shutdown] Stopping camera...")
        camera.stop()
        print("[Shutdown] Stopping watchdog...")
        watchdog.stop()
        print("[Shutdown] Stopping health monitor...")
        health.stop()
        print("[Shutdown] Done.")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Start server
    import socket
    hostname = socket.gethostname()
    ip_addrs = []
    try:
        import netifaces
        for iface in netifaces.interfaces():
            addrs = netifaces.ifaddresses(iface)
            if 2 in addrs:  # AF_INET
                for addr in addrs[2]:
                    ip = addr.get('addr', '')
                    if ip and ip != '127.0.0.1':
                        ip_addrs.append(ip)
    except ImportError:
        pass

    print()
    print("┌──────────────────────────────────────────┐")
    print("│  Leorover Robot Server                    │")
    print("├──────────────────────────────────────────┤")
    for ip in ip_addrs:
        print(f"│  http://{ip}:{args.port:<5}                  │")
    print("│                                          │")
    print("│  Endpoints:                              │")
    print("│    /stream   — MJPEG live stream         │")
    print("│    /snapshot — Single frame + timestamp  │")
    print("│    /motor    — Send wheel duties          │")
    print("│    /velocity — Send body velocity        │")
    print("│    /stop     — Emergency stop            │")
    print("│    /health   — Battery + system health   │")
    print("│    /servo    — Pan-tilt control          │")
    print("│    /buzzer   — Audio feedback            │")
    print("│                                          │")
    print("│  Ctrl+C to stop                          │")
    print("└──────────────────────────────────────────┘")
    print()

    # Run Flask (single-threaded is fine for our use case,
    # but threaded=True allows concurrent snapshot + motor requests)
    app.run(host='0.0.0.0', port=args.port, threaded=True, use_reloader=False)


if __name__ == '__main__':
    main()
