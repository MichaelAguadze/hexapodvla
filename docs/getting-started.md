# Getting Started

This guide gets a student from "fresh clone" to "recording data" and then to either "trained CNN policy" or "exported LeRobot dataset."

If you are working on the separate CNN workflow, use [CNN Dataset And Training Guide](cnn.md) and [CNN V1 Overview](../design/cnn_v1/overview.html) alongside this page.

## What You Need

- A TurboPi Advanced Kit with the vendor image installed
- A Windows, macOS, or Linux laptop with Python 3.10 or newer
- SSH access to the robot
- A Wi-Fi network that both the robot and the laptop can join

## Install Laptop Dependencies

Windows PowerShell:

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-laptop.txt
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-laptop.txt
```

The laptop requirements cover the recording client and local dataset writing.

## Install Robot Dependencies

On the TurboPi:

```bash
python3 -m pip install -r requirements-robot.txt
```

Important:

- `ros_robot_controller_sdk` is not installed from `pip`
- `cv2` is usually already present on the TurboPi image
- this repo assumes you are using the normal TurboPi software image, not a clean Raspberry Pi install

If you prefer to do that from your laptop, use:

```bash
bash scripts/deploy_server.sh deps
```

## First Run Order

1. Follow [Wi-Fi and SSH Guide](wifi-and-ssh.md) to move the robot onto the same Wi-Fi as your laptop.
2. Install the robot-side Python packages if needed.
3. Start the robot server.
4. Optionally test teleop-only mode.
5. Start the laptop launcher.
6. Choose `VLA-based` or `CNN-based`.
7. Record accepted episodes.
8. Train the CNN model or export VLA episodes to LeRobot.

## Start the Robot Server

If the repo is already on the robot:

```bash
python3 robot_server/server.py --port 8080
```

If the repo only exists on your laptop, use the deploy helper from a Bash shell:

```bash
bash scripts/deploy_server.sh start
```

Windows users can run that from Git Bash or WSL.

## Start the Laptop Client

If you want to confirm the robot responds before recording:

```bash
python -m client.teleop --robot-ip <ROBOT_IP>
```

Then start the launcher:

Shared-Wi-Fi mode:

```bash
python -m client.cli --robot-ip <ROBOT_IP>
```

Hotspot-only quick test:

```bash
python -m client
```

`python -m client` uses the default AP-mode robot IP `192.168.149.1`. Once you move to shared Wi-Fi, pass `--robot-ip <ROBOT_IP>` explicitly.

The launcher then asks which workflow you want:

- `CNN-based`: image-only CNN dataset recording
- `VLA-based`: the original task-based dataset flow

## Where Recordings Are Saved

The client writes new timestamped folders under:

```text
data/<dataset_name>/
|-- raw/
`-- episodes/
```

Example:

```text
data/turbopi_nav/
|-- raw/session_20260401_101500/
`-- episodes/session_20260401_101500/
```

Each run gets its own `session_YYYYMMDD_HHMMSS` folder, so old recordings stay untouched.

## Train The CNN Policy

Install the training extras on the laptop:

```bash
pip install -r requirements-cnn.txt
```

Train from the CNN episode root:

```bash
python -m cnn_policy.train \
  --episodes-dir data/turbopi_cnn/episodes \
  --run-dir runs/cnn_v1
```

The `--run-dir` value is a base directory. Each training launch creates a fresh timestamped child run so older checkpoints stay untouched.

Evaluate a trained checkpoint:

```bash
python -m cnn_policy.eval \
  --episodes-dir data/turbopi_cnn/episodes \
  --checkpoint <RUN_DIR>/checkpoints/best.pt
```

Drive the robot using the trained checkpoint:

```bash
python -m cnn_policy.drive \
  --robot-ip <ROBOT_IP> \
  --checkpoint <RUN_DIR>/checkpoints/best.pt
```

## Export To LeRobot

Install the export extras on the laptop:

```bash
pip install -r requirements-export.txt
```

Then run:

```bash
python scripts/export_lerobot.py \
  --episodes-dir data/turbopi_nav/episodes \
  --output-dir data/turbopi_nav/lerobot \
  --repo-id <HF_DATASET_REPO>
```

What the exporter does:

- scans all accepted `episode_*` folders under `episodes/`
- decodes each saved `video.mp4`
- reads each `data.parquet`
- rebuilds a LeRobot dataset under `data/<dataset_name>/lerobot/`

The default `--state-source shifted_action` is important. It reconstructs `observation.state` from the previous action so older recordings are still safe to use for training.

Helpful notes:

- `--episodes-dir` can be the full `data/turbopi_nav/episodes` folder or one specific `session_YYYYMMDD_HHMMSS` folder.
- The exporter includes every accepted `episode_*` folder under that path.
- If one episode is clearly bad, delete that episode folder before exporting.
- LeRobot often stores exported frames as chunked dataset videos rather than one MP4 per episode.
- If the packed video looks shorter than your full teleop session, that is expected. The duration is based only on accepted frames at the exported FPS.

## Useful Flags

Shorter collection run:

```bash
python -m client.cli --robot-ip <ROBOT_IP> --episodes 10 --episode-time 20
```

Different VLA dataset name:

```bash
python -m client.cli --robot-ip <ROBOT_IP> --dataset classroom_nav
```

Different CNN dataset name:

```bash
python -m client.cli --robot-ip <ROBOT_IP> --cnn-dataset classroom_cnn
```

Show export options:

```bash
python scripts/export_lerobot.py --help
```

## Change the Default Tasks

The built-in task list lives in `tasks.py`. Edit `DEFAULT_TASKS` to match your classroom task setup before you record.
