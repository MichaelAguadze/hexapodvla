"""Teleop-only entry point for driving the TurboPi without recording."""

from __future__ import annotations

import argparse
import sys
import time

from .robot_client import RobotClient
from .teleop_controller import TeleopController


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for teleop-only mode."""
    parser = argparse.ArgumentParser(description="TurboPi teleop-only client")
    parser.add_argument("--robot-ip", default="192.168.149.1")
    parser.add_argument("--robot-port", type=int, default=8081)
    parser.add_argument("--speed", type=float, default=50.0,
                        help="Initial teleop speed in duty units")
    parser.add_argument("--max-duty", type=float, default=80.0,
                        help="Maximum teleop speed in duty units")
    parser.add_argument("--loop-hz", type=float, default=10.0,
                        help="How often to send velocity commands")
    return parser


def _status_label(vx: float, vy: float, omega: float) -> str:
    """Return a short label for the current command."""
    if vx > 0:
        return "FWD"
    if vx < 0:
        return "BWD"
    if vy > 0:
        return "LEFT"
    if vy < 0:
        return "RIGHT"
    if omega > 0:
        return "ROT_L"
    if omega < 0:
        return "ROT_R"
    return "STOP"


def _flush_stdin() -> None:
    """Flush buffered console input so held keys do not spill into the shell."""
    try:
        import msvcrt

        while msvcrt.kbhit():
            msvcrt.getch()
    except ImportError:
        try:
            import termios

            termios.tcflush(sys.stdin, termios.TCIFLUSH)
        except Exception:
            pass


def main() -> None:
    """Run teleop-only mode."""
    args = build_parser().parse_args()
    robot_url = f"http://{args.robot_ip}:{args.robot_port}"
    period_s = 1.0 / max(args.loop_hz, 1.0)

    client = RobotClient(
        robot_url=robot_url,
        timeout=1.0,
        max_retries=2,
    )
    teleop = TeleopController(
        speed=args.speed,
        max_speed=args.max_duty,
    )

    print()
    print("=" * 50)
    print("  TurboPi Teleop")
    print("=" * 50)
    print(f"  Robot: {robot_url}")
    print()

    print("  Checking robot connection...")
    if not client.is_connected():
        print(f"  ERROR: Cannot reach robot at {robot_url}")
        print("  Make sure the robot server is running.")
        return

    try:
        health = client.get_health()
    except Exception:
        health = {}

    print(
        f"  Connected! Battery: {health.get('battery_mv', '?')}mV, "
        f"Camera: {'OK' if health.get('camera_ok') else 'FAIL'}"
    )
    print()
    print("  Controls:")
    print("    WASD  = translate")
    print("    Q/E   = rotate")
    print("    +/-   = speed up/down")
    print("    Space = stop movement")
    print("    Esc   = exit teleop")
    print()

    teleop.start()
    last_health_check = 0.0

    try:
        while not teleop.events["stop_session"]:
            vx, vy, omega = teleop.get_action()
            status = _status_label(vx, vy, omega)

            try:
                sent = client.send_velocity(vx, vy, omega)
            except Exception as exc:
                sent = False
                error = str(exc)
            else:
                error = ""

            now = time.monotonic()
            if now - last_health_check >= 10.0:
                last_health_check = now
                try:
                    health = client.get_health()
                    battery = health.get("battery_mv", 0)
                    if battery and battery < 7200:
                        print(f"\n  [WARN] Low battery: {battery}mV")
                except Exception:
                    pass

            line = f"\r  [{status:<6}] speed={teleop.speed:.0f}%"
            if not sent:
                line += f"  WARN: command not delivered{': ' + error if error else ''}"
            print(f"{line}   ", end="", flush=True)
            time.sleep(period_s)

    except KeyboardInterrupt:
        print("\n\n  Ctrl+C - stopping teleop...")
    finally:
        print()
        try:
            client.stop()
        except Exception:
            pass
        teleop.stop()
        _flush_stdin()
        print("  Teleop stopped.")
        print()


if __name__ == "__main__":
    main()
