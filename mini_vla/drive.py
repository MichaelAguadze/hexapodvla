"""Drive hexapod using a trained Mini VLA model with live task switching."""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as TF

from client.robot_client import RobotClient

from .model import load_checkpoint


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Drive hexapod with Mini VLA")
    p.add_argument("--robot-ip", default="192.168.149.1")
    p.add_argument("--robot-port", type=int, default=8080)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--task", default=None,
                   help="Initial task (optional, defaults to first task)")
    p.add_argument("--loop-hz", type=float, default=10.0)
    p.add_argument("--smoothing", type=float, default=0.5,
                   help="EMA factor for previous action (0=no smoothing)")
    p.add_argument("--max-duty", type=float, default=80.0,
                   help="Max duty used during data normalization")
    p.add_argument("--device", default="auto")
    return p


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def frame_to_tensor(
    frame: np.ndarray, *, image_width: int, image_height: int
) -> torch.Tensor:
    image = Image.fromarray(frame).convert("RGB")
    image = image.resize((image_width, image_height), Image.Resampling.BILINEAR)
    return TF.to_tensor(image)


class TaskSwitcher:
    """Listen for keyboard input to switch tasks live."""

    def __init__(self, task_to_idx: dict[str, int], initial_task: str, device: torch.device):
        self.task_to_idx = task_to_idx
        self.tasks = sorted(task_to_idx.keys(), key=lambda t: task_to_idx[t])
        self.device = device
        self._current_idx = task_to_idx[initial_task]
        self._lock = threading.Lock()
        self._stop = threading.Event()

    @property
    def current_task(self) -> str:
        with self._lock:
            return self.tasks[self._current_idx]

    @property
    def current_tensor(self) -> torch.Tensor:
        with self._lock:
            return torch.tensor([self._current_idx], dtype=torch.long, device=self.device)

    def set_task(self, idx: int) -> None:
        with self._lock:
            self._current_idx = idx % len(self.tasks)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _listen(self) -> None:
        """Read single keystrokes: 1-9 select task, q quits."""
        try:
            import tty
            import termios
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            try:
                while not self._stop.is_set():
                    if sys.stdin.readable():
                        ch = sys.stdin.read(1)
                        if ch == "q":
                            self._stop.set()
                            break
                        if ch.isdigit():
                            num = int(ch) - 1
                            if 0 <= num < len(self.tasks):
                                self.set_task(num)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        except Exception:
            pass

    @property
    def should_stop(self) -> bool:
        return self._stop.is_set()


def main() -> None:
    args = build_parser().parse_args()
    device = resolve_device(args.device)

    model, payload = load_checkpoint(Path(args.checkpoint), map_location=device)
    model = model.to(device)
    model.eval()

    task_to_idx = payload["task_to_idx"]
    tasks_sorted = sorted(task_to_idx.keys(), key=lambda t: task_to_idx[t])

    if args.task is not None:
        if args.task not in task_to_idx:
            raise ValueError(
                f"Unknown task: '{args.task}'\nAvailable: {tasks_sorted}"
            )
        initial_task = args.task
    else:
        initial_task = tasks_sorted[0]

    image_w = model.config.image_width
    image_h = model.config.image_height
    period_s = 1.0 / max(args.loop_hz, 1.0)
    smooth_alpha = float(np.clip(args.smoothing, 0.0, 0.99))

    robot_url = f"http://{args.robot_ip}:{args.robot_port}"
    client = RobotClient(robot_url=robot_url, timeout=1.0, max_retries=2)

    if not client.is_connected():
        raise RuntimeError(f"Cannot reach robot at {robot_url}")

    try:
        health = client.get_health()
    except Exception:
        health = {}

    switcher = TaskSwitcher(task_to_idx, initial_task, device)

    print()
    print("=" * 50)
    print("  hexapod Mini VLA Drive")
    print("=" * 50)
    print(f"  Robot:   {robot_url}")
    print(f"  Device:  {device}")
    print(f"  Epoch:   {payload.get('epoch')}")
    print(f"  Battery: {health.get('battery_mv', '?')}mV")
    print(f"  Camera:  {'OK' if health.get('camera_ok') else 'FAIL'}")
    print()
    print("  Tasks (press number to switch):")
    for i, task in enumerate(tasks_sorted):
        marker = " *" if task == initial_task else ""
        print(f"    [{i + 1}] {task}{marker}")
    print()
    print("  q = quit")
    print()

    previous_action = np.zeros(3, dtype=np.float32)
    switcher.start()

    def safe_stop(*_args):
        switcher.stop()
        try:
            client.stop()
        except Exception:
            pass
        raise SystemExit(0)

    signal.signal(signal.SIGINT, safe_stop)
    signal.signal(signal.SIGTERM, safe_stop)

    try:
        while not switcher.should_stop:
            loop_start = time.monotonic()

            frame, _, _ = client.get_frame_rgb()
            image_tensor = frame_to_tensor(
                frame, image_width=image_w, image_height=image_h
            ).unsqueeze(0).to(device)

            task_tensor = switcher.current_tensor
            current_task = switcher.current_task

            with torch.no_grad():
                pred = (
                    model(image_tensor, task_tensor)
                    .squeeze(0)
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

            pred_clipped = np.clip(pred, -1.0, 1.0)
            smoothed = smooth_alpha * previous_action + (1.0 - smooth_alpha) * pred_clipped
            previous_action = smoothed

            command = smoothed * args.max_duty

            try:
                client.send_velocity(
                    float(command[0]), float(command[1]), float(command[2])
                )
            except Exception as exc:
                print(f"\n  [WARN] Send failed: {exc}")
                client.stop()
                break

            short_task = current_task[:30]
            print(
                f"\r  [{short_task:<30}]  "
                f"raw=[{pred[0]:.3f},{pred[1]:.3f},{pred[2]:.3f}]  "
                f"cmd=[{command[0]:.1f},{command[1]:.1f},{command[2]:.1f}]   ",
                end="",
                flush=True,
            )

            elapsed = time.monotonic() - loop_start
            remaining = period_s - elapsed
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        pass
    finally:
        print()
        switcher.stop()
        try:
            client.stop()
        except Exception:
            pass
        print("  VLA drive stopped.")


if __name__ == "__main__":
    main()
