# CNN Dataset And Training Guide

This guide describes the CNN-based, no-language workflow that sits beside the existing VLA recorder. For the visual map, open [CNN V1 Overview](../design/cnn_v1/overview.html).

## What This Mode Is

The CNN mode is for image-only path following. It is a good fit for taped tracks such as:

- circles
- squares
- rounded rectangles

In this flow:

- `CNN-based` means image-only control, no task text, no language intent
- `VLA-based` keeps the current task-driven recording flow
- `dataset recording` is the CNN mode that records one full lap per accepted episode

## Launcher Flow

```text
python -m client.cli --robot-ip <ROBOT_IP>
  -> CNN-based
  -> without language intent
  -> dataset recording
```

The `with language intent` path is reserved for future work.

## Recording Setup

Use a high-contrast taped path and keep the floor lighting stable.

Good setup rules:

- pick one tape color and keep it consistent
- make the path wide enough that the robot can correct smoothly
- avoid shiny floors and strong shadows if possible
- keep the robot camera pointed at the path and close enough that the tape is easy to see

## What One Episode Means

For CNN mode, one episode equals one complete lap of your chosen track.

Episode flow:

1. Select the drive direction.
2. Drive into the path.
3. Press accept when the lap is complete.
4. Save that lap as one episode.

Do not mix multiple laps into one episode unless you intentionally want that for a later experiment.

## Direction Labels

Every CNN episode should be labeled with one of these directions:

- `clockwise`
- `counterclockwise`

Keep both directions in the dataset. That usually makes the policy easier to debug and more stable in closed loop.

## Recommended Dataset Size

Start small and prove the path works first, then scale up.

Suggested collection target:

- smoke test: `20` accepted laps per direction
- first serious dataset: `50` accepted laps per direction
- strong v1 goal: `100` accepted laps total

For square tracks, expect to add more episodes if the corners are still weak.

## Data Layout

CNN data should live in its own dataset root, separate from the VLA recordings.

Example layout:

```text
data/turbopi_cnn/
|-- raw/
|   `-- session_YYYYMMDD_HHMMSS/
|       |-- session_info.json
|       |-- telemetry.jsonl
|       `-- video.mp4
`-- episodes/
    `-- session_YYYYMMDD_HHMMSS/
        |-- session_info.json
        |-- tasks.json
        `-- episode_000000/
            |-- data.parquet
            `-- video.mp4
```

If you already recorded under the older `data/turbopi_cnn_loop/` path, that older layout still works.

## What The CNN Sees

The first baseline uses:

- 3 recent RGB frames
- full-frame resize to `160x120`
- a stacked input tensor shaped `9 x 120 x 160`
- normalized pixels in `[0, 1]`

The network predicts:

```text
[vx, vy, omega]
```

Those values stay normalized and get converted back to robot command units only at inference time.

## CNN Architecture

The first baseline is intentionally small:

```text
Input: 9 x 120 x 160
Conv 9 -> 32, k=5, s=2, p=2 + BN + ReLU
Conv 32 -> 64, k=3, s=2, p=1 + BN + ReLU
Conv 64 -> 128, k=3, s=2, p=1 + BN + ReLU
Conv 128 -> 128, k=3, s=2, p=1 + BN + ReLU
Global Average Pool
Linear 128 -> 64 + ReLU + Dropout
Linear 64 -> 32 + ReLU
Linear 32 -> 3 + Tanh
```

Why this shape:

- it is small enough to run fast on a laptop in a tight control loop
- it keeps the model easy to debug
- it is strong enough for taped path following without jumping to a heavyweight backbone

## Install The CNN Extras

On the laptop:

```bash
pip install -r requirements-cnn.txt
```

## Training Command

Train from the accepted CNN episode root:

```bash
python -m cnn_policy.train --episodes-dir data/turbopi_cnn/episodes --run-dir runs/cnn_v1
```

The trainer prints a concrete child run such as `runs/cnn_v1/run_YYYYMMDD_HHMMSS`. Use that full folder as `<RUN_DIR>` for evaluation and driving.

Training defaults:

- Huber loss on normalized actions
- session-level train/val split
- light brightness, contrast, hue, blur, and small geometry augmentation
- each launch creates a new timestamped run folder under the base `--run-dir`
- each epoch saves its own folder under `checkpoints/epoch_XXX/` with both `last.pt` and `best.pt`

## Evaluation Command

```bash
python -m cnn_policy.eval --episodes-dir data/turbopi_cnn/episodes --checkpoint <RUN_DIR>/checkpoints/best.pt
```

## Inference Command

The laptop-first inference command is:

```bash
python -m cnn_policy.drive --robot-ip <ROBOT_IP> --checkpoint <FILE>
```

## Upload One Session To Hugging Face

If you want to share one selected recorded session directly on Hugging Face, run:

```bash
python scripts/upload_hf_session.py
```

What it does:

- scans the CNN episodes root and lists the discovered sessions
- shows each session name plus exact episode and frame counts
- lets you select one session in a small picker window
- creates a dataset repo named after that selected session
- uploads the accepted session folder and, optionally, the matching raw backup

If you prefer the terminal instead of the picker:

```bash
python scripts/upload_hf_session.py --no-gui
```

To test the upload flow without pushing anything:

```bash
python scripts/upload_hf_session.py --dry-run --no-gui
```

Inference loop:

1. fetch the latest camera frame from `/snapshot`
2. maintain a rolling 3-frame buffer
3. resize to `160x120`
4. run the CNN
5. smooth the normalized prediction, then scale it by safe `vx`, `vy`, and `omega` caps
6. send `/velocity` to the robot

## Common Issues

- If the robot drifts away from the path, you probably need more clean laps and more recovery examples.
- If motion feels jittery, keep the model small and add a little more runtime smoothing before making the network deeper.
- If the path is hard to see, improve lighting before adding more data.
- If the model learns one direction but fails on the other, rebalance clockwise and counterclockwise samples.
- If the robot keeps stopping early, check Wi-Fi and the robot watchdog before blaming the model.

## Quick Rule

One lap equals one episode.
If a run is bad, discard it rather than saving a partial lap into the clean dataset.
