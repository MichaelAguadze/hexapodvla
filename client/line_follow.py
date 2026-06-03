"""CLI entry point for line-following mode."""

from __future__ import annotations

import argparse

from .robot_client import RobotClient
from .line_follower import LineFollower, SUPPORTED_COLOURS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="hexapod line-follower")
    parser.add_argument("--robot-ip",   default="192.168.149.1")
    parser.add_argument("--robot-port", type=int, default=8081)
    parser.add_argument(
        "--colour", choices=SUPPORTED_COLOURS, default="red",
        help="Line colour to follow (default: red)",
    )
    parser.add_argument("--speed",  type=float, default=30.0,
                        help="Forward speed duty when centred (default 30)")
    parser.add_argument("--max-duty", type=float, default=80.0)
    parser.add_argument("--kp",     type=float, default=60.0,
                        help="Proportional steering gain")
    parser.add_argument("--kd",     type=float, default=8.0,
                        help="Derivative steering gain")
    parser.add_argument("--roi-top", type=float, default=0.4,
                        help="Top of look-ahead ROI as fraction of frame height (0-1)")
    parser.add_argument("--lost-timeout", type=float, default=2.0,
                        help="Seconds without line before stopping")
    parser.add_argument("--loop-hz", type=float, default=10.0)
    parser.add_argument("--preview", action="store_true",
                        help="Show OpenCV debug window (requires local display)")
    parser.add_argument("--controller", choices=["ps5"], default=None,
                        help="Enable PS5 manual override (sticks override the line follower)")
    parser.add_argument("--joystick-index", type=int, default=0,
                        help="Pygame joystick index (for --controller ps5)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    robot_url = "http://{}:{}".format(args.robot_ip, args.robot_port)

    print()
    print("=" * 50)
    print("  hexapod Line Follower")
    print("=" * 50)
    print("  Robot:      {}".format(robot_url))
    print("  Colour:     {}".format(args.colour))
    print("  Speed:      {}".format(args.speed))
    print("  Kp/Kd:      {} / {}".format(args.kp, args.kd))
    print("  Controller: {}".format(args.controller or "none"))
    print()

    client = RobotClient(robot_url=robot_url, timeout=1.0, max_retries=2)

    print("  Checking connection...")
    if not client.is_connected():
        print("  ERROR: Cannot reach robot at {}".format(robot_url))
        print("  Make sure the robot server is running.")
        return

    controller = None
    if args.controller == "ps5":
        from .ps5_controller import PS5Controller
        controller = PS5Controller(
            speed=args.max_duty,
            max_speed=args.max_duty,
            joystick_index=args.joystick_index,
        )
        controller.start()
        print("  PS5 controller connected — move sticks to override line follower.")
        print("  Press Options to stop.")
    else:
        print("  Connected. Starting line follower — Ctrl+C to stop.")
    print()

    follower = LineFollower(
        client=client,
        colour=args.colour,
        base_speed=args.speed,
        max_duty=args.max_duty,
        kp=args.kp,
        kd=args.kd,
        roi_top=args.roi_top,
        lost_timeout=args.lost_timeout,
        loop_hz=args.loop_hz,
        show_preview=args.preview,
        controller=controller,
    )

    try:
        follower.run()
    finally:
        if controller:
            controller.stop()


if __name__ == "__main__":
    main()
