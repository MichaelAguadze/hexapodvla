#!/usr/bin/env python3
"""Inspect recorded hexapod episodes for action/state correctness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd


AXES = ("vx", "vy", "omega")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Inspect recorded hexapod episodes")
    parser.add_argument("--episodes-dir", default="data/hexapod_nav/episodes")
    parser.add_argument(
        "--session",
        default=None,
        help="Session folder name or explicit path. Defaults to the latest session.",
    )
    parser.add_argument(
        "--episode",
        default=None,
        help="Episode folder name such as episode_000000. Defaults to all episodes in the session.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="How many rows of the per-frame table to print per episode.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="Minimum normalized magnitude to count a component as active.",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional output CSV path for the flattened inspection table.",
    )
    return parser


def resolve_session_dir(episodes_dir: Path, session_arg: str | None) -> Path:
    """Resolve the target session directory."""
    if session_arg:
        candidate = Path(session_arg)
        if candidate.exists():
            return candidate
        candidate = episodes_dir / session_arg
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"Session not found: {session_arg}")

    direct_episodes = sorted(path for path in episodes_dir.glob("episode_*") if path.is_dir())
    if direct_episodes:
        return episodes_dir

    sessions = sorted(path for path in episodes_dir.iterdir() if path.is_dir())
    if not sessions:
        raise FileNotFoundError(f"No session folders found in {episodes_dir}")
    return sessions[-1]


def resolve_episode_dirs(session_dir: Path, episode_arg: str | None) -> list[Path]:
    """Resolve the target episode directories."""
    if episode_arg:
        candidate = session_dir / episode_arg
        if candidate.is_dir():
            return [candidate]
        raise FileNotFoundError(f"Episode folder not found: {candidate}")

    episode_dirs = sorted(path for path in session_dir.glob("episode_*") if path.is_dir())
    if not episode_dirs:
        raise FileNotFoundError(f"No episode folders found in {session_dir}")
    return episode_dirs


def decode_frame_count(video_path: Path) -> int:
    """Count decoded frames in a video."""
    with av.open(str(video_path)) as container:
        return sum(1 for _ in container.decode(video=0))


def flatten_episode(df: pd.DataFrame) -> pd.DataFrame:
    """Expand vector columns into scalar columns for easier inspection."""
    rows = []
    for _, row in df.iterrows():
        action = np.asarray(row["action"], dtype=np.float32).reshape(-1)
        state = np.asarray(row["observation.state"], dtype=np.float32).reshape(-1)
        rows.append(
            {
                "frame_index": int(row["frame_index"]),
                "timestamp": float(row["timestamp"]),
                "task": str(row["task"]),
                "task_index": int(row["task_index"]),
                "state_vx": float(state[0]),
                "state_vy": float(state[1]),
                "state_omega": float(state[2]),
                "action_vx": float(action[0]),
                "action_vy": float(action[1]),
                "action_omega": float(action[2]),
            }
        )
    return pd.DataFrame(rows)


def summarize_components(flat_df: pd.DataFrame, threshold: float) -> dict[str, int]:
    """Summarize which action components are active."""
    summary = {
        "forward_frames": int((flat_df["action_vx"] > threshold).sum()),
        "backward_frames": int((flat_df["action_vx"] < -threshold).sum()),
        "left_frames": int((flat_df["action_vy"] > threshold).sum()),
        "right_frames": int((flat_df["action_vy"] < -threshold).sum()),
        "rotate_left_frames": int((flat_df["action_omega"] > threshold).sum()),
        "rotate_right_frames": int((flat_df["action_omega"] < -threshold).sum()),
    }

    action_mag = flat_df[["action_vx", "action_vy", "action_omega"]].abs().max(axis=1)
    summary["stop_frames"] = int((action_mag <= threshold).sum())
    return summary


def compute_shift_checks(flat_df: pd.DataFrame, threshold: float) -> dict[str, float]:
    """Measure whether state is shifted relative to action or leaking same-step labels."""
    if len(flat_df) <= 1:
        return {
            "shift_match_ratio": 1.0,
            "same_step_match_ratio": 1.0,
        }

    action = flat_df[["action_vx", "action_vy", "action_omega"]].to_numpy(dtype=np.float32)
    state = flat_df[["state_vx", "state_vy", "state_omega"]].to_numpy(dtype=np.float32)

    shifted_diff = np.abs(state[1:] - action[:-1]).max(axis=1)
    same_step_diff = np.abs(state - action).max(axis=1)

    return {
        "shift_match_ratio": float((shifted_diff <= threshold).mean()),
        "same_step_match_ratio": float((same_step_diff <= threshold).mean()),
    }


def print_episode_report(episode_dir: Path, flat_df: pd.DataFrame, video_frames: int, threshold: float, limit: int) -> None:
    """Print a human-readable report for an episode."""
    summary = summarize_components(flat_df, threshold)
    checks = compute_shift_checks(flat_df, threshold)

    print()
    print("=" * 72)
    print(f"Episode: {episode_dir}")
    print("=" * 72)
    print(f"rows: {len(flat_df)}")
    print(f"video_frames: {video_frames}")
    print(f"task: {flat_df['task'].iloc[0]}")
    print(f"timestamp_range: {flat_df['timestamp'].iloc[0]:.3f} -> {flat_df['timestamp'].iloc[-1]:.3f}")
    print(f"shift_match_ratio: {checks['shift_match_ratio']:.3f}")
    print(f"same_step_match_ratio: {checks['same_step_match_ratio']:.3f}")
    print()
    print("action summary:")
    print(json.dumps(summary, indent=2))
    print()
    print("per-frame values:")
    preview = flat_df.head(limit)
    print(preview.to_string(index=False, float_format=lambda x: f"{x: .3f}"))
    if len(flat_df) > limit:
        print()
        print(f"... truncated to first {limit} rows")


def main() -> None:
    """Run the inspection."""
    args = build_parser().parse_args()
    episodes_dir = Path(args.episodes_dir)
    session_dir = resolve_session_dir(episodes_dir, args.session)
    episode_dirs = resolve_episode_dirs(session_dir, args.episode)

    csv_frames: list[pd.DataFrame] = []

    print(f"session_dir: {session_dir}")
    for episode_dir in episode_dirs:
        parquet_path = episode_dir / "data.parquet"
        video_path = episode_dir / "video.mp4"
        if not parquet_path.exists() or not video_path.exists():
            raise FileNotFoundError(f"Missing data.parquet or video.mp4 in {episode_dir}")

        df = pd.read_parquet(parquet_path)
        flat_df = flatten_episode(df)
        flat_df.insert(0, "episode_dir", str(episode_dir))
        csv_frames.append(flat_df)

        video_frames = decode_frame_count(video_path)
        print_episode_report(episode_dir, flat_df.drop(columns=["episode_dir"]), video_frames, args.threshold, args.limit)

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(csv_frames, ignore_index=True).to_csv(csv_path, index=False)
        print()
        print(f"wrote csv: {csv_path}")


if __name__ == "__main__":
    main()
