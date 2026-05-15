# Data Collection Guide

This repo records teleoperated episodes from a laptop while the TurboPi runs the robot server.

If you are building the separate CNN workflow, see [CNN Dataset And Training Guide](cnn.md). That path uses its own dataset root and recording rules.

## What The Current Code Does

The committed code supports:

- keyboard teleoperation from the laptop
- teleop-only driving without recording
- launcher-based routing into `CNN-based` or `VLA-based` recording
- live frame capture from the robot over HTTP
- raw session backups
- accepted-episode exports as MP4 plus Parquet
- CNN training, evaluation, and laptop-first inference
- LeRobot export from accepted episodes into `data/<dataset_name>/lerobot/`

The committed code does not currently include:

- a dashboard app
- a VLA training stack
- a dataset upload script beyond the optional `--push-to-hub` flag in the exporter

## Start A Recording Session

If you only want to drive the robot and not save data yet:

```bash
python -m client.teleop --robot-ip <ROBOT_IP>
```

That is useful for:

- checking that the server is running
- checking Wi-Fi latency
- confirming the robot moves the way you expect

When you are ready to record:

1. Make sure the robot server is running.
2. Activate your laptop virtual environment.
3. Start the launcher:

```bash
python -m client.cli --robot-ip <ROBOT_IP>
```

For AP-mode testing:

```bash
python -m client
```

After launch:

- choose `VLA-based` for the current task list workflow
- choose `CNN-based` for the CNN dataset workflow

## Controls

- `W`, `A`, `S`, `D`: drive forward, left, backward, right
- `Q`, `E`: rotate
- `+`, `-`: increase or decrease teleop speed
- right arrow: start recording after positioning, or accept the current episode
- left arrow: discard the current episode
- `Esc`: stop the full session

## Default Tasks

By default, the task list is:

- `go to the left of the box`
- `go to the right of the box`
- `go forward to the box`
- `go behind the box`

You can change these in `tasks.py`.

The CNN path does not use this task list. It asks for `clockwise` or `counterclockwise` instead and treats one full lap of your chosen track as one accepted episode.

## What Gets Saved

The client creates a fresh timestamped session folder every time you run it.

Example layout:

```text
data/turbopi_nav/
|-- raw/
|   `-- session_20260401_101500/
|       |-- session_info.json
|       |-- telemetry.jsonl
|       `-- video.mp4
`-- episodes/
    `-- session_20260401_101500/
        |-- session_info.json
        |-- tasks.json
        `-- episode_000000/
            |-- data.parquet
            `-- video.mp4
```

Why there are two outputs:

- `raw/` keeps a continuous backup of the full session
- `episodes/` keeps only accepted episodes in a cleaner structure

## Why The Saved State Matters

For training, `observation.state` should not be the same thing as the current target action.

This recorder now saves:

- `observation.state`: previous normalized action
- `action`: current normalized teleop command

That avoids leaking the answer into the model input when you train an action head.

## No Overwrite Behavior

Each run creates a new folder named like `session_YYYYMMDD_HHMMSS`.

That means:

- old recordings stay untouched
- you can compare sessions later
- students do not lose data by starting a new run

## Export To LeRobot

Accepted episodes can be converted into a LeRobot-compatible dataset:

```bash
pip install -r requirements-export.txt

python scripts/export_lerobot.py \
  --episodes-dir data/turbopi_nav/episodes \
  --output-dir data/turbopi_nav/lerobot \
  --repo-id <HF_DATASET_REPO>
```

Important exporter behavior:

- it scans every accepted `episode_*` folder under `episodes/`
- it can also scan just one `session_YYYYMMDD_HHMMSS/` folder if you only want to export that run
- it verifies that `data.parquet` rows match decoded video frames
- it rebuilds `observation.state` from the previous action by default
- it writes a standard LeRobot dataset with `observation.images.front`, `action`, and `observation.state`
- it may pack multiple exported episodes into one chunked dataset video file, which is normal for LeRobot

What this means in practice:

- the exported video duration is `total_frames / fps`
- the packed dataset video is not the same thing as your original wall-clock recording session
- short trash episodes can still appear in export unless you remove those `episode_*` folders first
- episode boundaries are preserved in LeRobot metadata even when frames are packed into one video chunk

If you want different behavior:

- `--state-source shifted_action`: recommended default and the safest choice for older data
- `--state-source recorded`: trust the saved `observation.state`
- `--state-source zeros`: use a zero vector for every frame
- `--state-source none`: export without `observation.state`

Example: export one cleaned session only

```bash
python scripts/export_lerobot.py \
  --episodes-dir data/turbopi_nav/episodes/session_20260402_114217 \
  --output-dir data/turbopi_nav/lerobot_session_20260402_114217 \
  --state-source shifted_action \
  --overwrite
```

## Train The CNN Baseline

CNN data is intentionally separate from VLA data:

```text
data/turbopi_cnn/
```

Install the CNN extras:

```bash
pip install -r requirements-cnn.txt
```

Train:

```bash
python -m cnn_policy.train \
  --episodes-dir data/turbopi_cnn/episodes \
  --run-dir runs/cnn_v1
```

Evaluate:

```bash
python -m cnn_policy.eval \
  --episodes-dir data/turbopi_cnn/episodes \
  --checkpoint <RUN_DIR>/checkpoints/best.pt
```

Drive:

```bash
python -m cnn_policy.drive \
  --robot-ip <ROBOT_IP> \
  --checkpoint <RUN_DIR>/checkpoints/best.pt
```

## Inspect Recorded Actions

If you record two test episodes such as "go left" and "go right", you can inspect exactly what was saved:

```bash
python scripts/inspect_episode.py --episodes-dir data/turbopi_nav/episodes
```

Useful things this reports:

- `action_vx`: forward and backward component
- `action_vy`: left and right component
- `action_omega`: rotation component
- `state_*`: previous-step state that will be fed to training
- counts for left, right, rotate-left, rotate-right, and stop frames
- whether `state` looks like a shifted version of `action`

If you want a CSV for manual inspection:

```bash
python scripts/inspect_episode.py \
  --episodes-dir data/turbopi_nav/episodes \
  --csv data/turbopi_nav/inspection.csv
```

## Practical Tips

- Start with a lower teleop speed until the controls feel natural.
- Charge the battery before long recording sessions.
- Run the client on the local laptop session, not inside a remote SSH shell, because `pynput` listens for local keyboard events.
- If a Wi-Fi hiccup causes motor commands to fail, the recorder now skips those frames instead of silently saving bad action labels.
- Validation-only folders such as `data/workflow_validation/` are disposable and can be deleted after you confirm the pipeline works.
