"""CLI entry point for YOLO object-detection driving mode."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="hexapod detection driver — avoids random objects and reacts to named ones"
    )
    parser.add_argument("--robot-ip",   default="192.168.149.1")
    parser.add_argument("--robot-port", type=int, default=8081)

    # YOLO settings
    yolo = parser.add_argument_group("YOLO model")
    yolo.add_argument("--model",      default="yolov8n.pt",
                      help="ultralytics model file (default: yolov8n.pt, downloaded on first run)")
    yolo.add_argument("--confidence", type=float, default=0.40,
                      help="minimum detection confidence [0-1] (default 0.40)")
    yolo.add_argument("--device",     default="cpu",
                      help="inference device: cpu | cuda | mps (default cpu)")

    # Motion
    motion = parser.add_argument_group("motion")
    motion.add_argument("--speed",    type=float, default=30.0,
                        help="forward duty cycle during free travel (default 30)")
    motion.add_argument("--max-duty", type=float, default=80.0)
    motion.add_argument("--kp-steer", type=float, default=1.5,
                        help="steering gain (default 1.5)")
    motion.add_argument("--loop-hz",  type=float, default=10.0)

    # Avoidance
    avoid = parser.add_argument_group("avoidance")
    avoid.add_argument("--danger-corridor", type=float, default=0.60,
                       help="fraction of frame width treated as the forward path (default 0.60)")
    avoid.add_argument("--min-obstacle-area", type=float, default=0.01,
                       help="minimum bbox area fraction to count as an obstacle (default 0.01)")
    avoid.add_argument("--avoid-stop-area", type=float, default=0.10,
                       help="stop forward motion when obstacle exceeds this area fraction (default 0.10)")

    # Reactions
    react = parser.add_argument_group("reactions")
    react.add_argument(
        "--react", nargs="*", default=[],
        metavar="LABEL:ACTION",
        help=(
            "Map a YOLO class label to an action.  Repeat for multiple entries.\n"
            "  Actions:  approach | follow | stop | avoid | ignore\n"
            "  Spaces in label names can be written as underscores.\n"
            "  Example:  --react sports_ball:approach person:follow stop_sign:stop\n"
            "  Default:  sports_ball:approach"
        ),
    )
    react.add_argument("--approach-stop-area", type=float, default=0.12,
                       help="stop approaching when target bbox exceeds this fraction (default 0.12)")
    react.add_argument("--approach-slow-area", type=float, default=0.04,
                       help="start slowing approach at this fraction (default 0.04)")
    react.add_argument("--follow-target-area", type=float, default=0.08,
                       help="target area fraction for follow mode (default 0.08)")

    # UI / controller
    parser.add_argument("--preview", action="store_true",
                        help="show OpenCV debug window (requires a local display)")
    parser.add_argument("--controller", choices=["ps5"], default=None)
    parser.add_argument("--joystick-index", type=int, default=0)
    return parser


def _parse_react_map(entries: list) -> dict:
    """Parse ["label:action", ...] into {label: action}.

    Underscores in label names are converted to spaces so that
    multi-word YOLO class names can be written without quoting, e.g.
    sports_ball → "sports ball".
    """
    if not entries:
        # Sensible default: approach a sports ball
        return {"sports ball": "approach"}

    valid_actions = {"approach", "follow", "stop", "avoid", "ignore"}
    react_map = {}
    for entry in entries:
        if ":" not in entry:
            raise ValueError(
                "react entry {!r} must be in LABEL:ACTION format".format(entry)
            )
        label_raw, action = entry.rsplit(":", 1)
        label = label_raw.replace("_", " ").strip()
        action = action.strip().lower()
        if action not in valid_actions:
            raise ValueError(
                "unknown action {!r}; choose from {}".format(action, valid_actions)
            )
        react_map[label] = action
    return react_map


def main() -> None:
    args = build_parser().parse_args()
    robot_url = "http://{}:{}".format(args.robot_ip, args.robot_port)

    try:
        react_map = _parse_react_map(args.react)
    except ValueError as exc:
        print("ERROR:", exc)
        return

    print()
    print("=" * 55)
    print("  hexapod Detection Driver")
    print("=" * 55)
    print("  Robot:       {}".format(robot_url))
    print("  Model:       {}".format(args.model))
    print("  Confidence:  {:.0%}".format(args.confidence))
    print("  Device:      {}".format(args.device))
    print("  Speed:       {}".format(args.speed))
    print("  Reactions:   {}".format(
        ", ".join("{} → {}".format(k, v) for k, v in react_map.items()) or "none"
    ))
    print("  Controller:  {}".format(args.controller or "none"))
    print()

    from .robot_client import RobotClient
    from .object_detector import ObjectDetector
    from .detection_driver import DetectionDriver

    client = RobotClient(robot_url=robot_url, timeout=1.0, max_retries=2)
    print("  Checking connection...")
    if not client.is_connected():
        print("  ERROR: Cannot reach robot at {}".format(robot_url))
        return

    print("  Loading YOLO model...")
    detector = ObjectDetector(
        model_name=args.model,
        confidence=args.confidence,
        device=args.device,
    )
    print("  Model loaded.")

    controller = None
    if args.controller == "ps5":
        from .ps5_controller import PS5Controller
        controller = PS5Controller(
            speed=args.max_duty,
            max_speed=args.max_duty,
            joystick_index=args.joystick_index,
        )
        controller.start()
        print("  PS5 controller connected — move sticks to override.")

    print()
    print("  Running — Ctrl+C to stop.")
    print()

    driver = DetectionDriver(
        client=client,
        detector=detector,
        react_map=react_map,
        forward_speed=args.speed,
        max_duty=args.max_duty,
        kp_steer=args.kp_steer,
        danger_corridor=args.danger_corridor,
        min_obstacle_area=args.min_obstacle_area,
        approach_stop_area=args.approach_stop_area,
        approach_slow_area=args.approach_slow_area,
        follow_target_area=args.follow_target_area,
        avoid_stop_area=args.avoid_stop_area,
        loop_hz=args.loop_hz,
        show_preview=args.preview,
        controller=controller,
    )

    try:
        driver.run()
    finally:
        if controller:
            controller.stop()


if __name__ == "__main__":
    main()
