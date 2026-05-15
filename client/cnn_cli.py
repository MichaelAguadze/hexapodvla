"""CNN launcher branch and generic dataset recorder entry points."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path


def _run_cnn_language_placeholder() -> None:
    """Show the language-intent placeholder screen."""
    print()
    print("=" * 50)
    print("  CNN With Language Intent")
    print("=" * 50)
    print("\n  This path is coming in the future.")
    print("  For now, use CNN-based -> without language intent -> dataset recording.\n")


def _run_cnn_dataset_recording(args: Namespace) -> None:
    """Run the no-language CNN dataset recorder."""
    from config import RecordingConfig

    from .cnn_loop_session import CNNLoopSession

    episode_time = args.episode_time if args.episode_time != 30.0 else 60.0
    config = RecordingConfig(
        robot_ip=args.robot_ip,
        robot_port=args.robot_port,
        dataset_name=args.cnn_dataset,
        repo_id=args.repo_id,
        fps=args.fps,
        num_episodes=args.episodes,
        episode_time_s=episode_time,
        teleop_speed=args.speed,
        data_dir=Path(args.data_dir),
    )
    session = CNNLoopSession(config)
    session.run()


def run_from_args(args: Namespace, prompt_menu) -> None:
    """Handle the CNN launcher subtree."""
    if args.cnn_intent is None:
        selection = prompt_menu(
            "CNN Intent Options",
            ["with language intent", "without language intent"],
        )
        if selection is None:
            return
        args.cnn_intent = "language" if selection == 0 else "no-language"

    if args.cnn_intent == "language":
        _run_cnn_language_placeholder()
        return

    if args.cnn_task is None:
        selection = prompt_menu(
            "CNN No-Language Options",
            ["dataset recording"],
        )
        if selection is None:
            return
        args.cnn_task = "dataset-recording"

    if args.cnn_task in {"dataset-recording", "circular-loop"}:
        _run_cnn_dataset_recording(args)
