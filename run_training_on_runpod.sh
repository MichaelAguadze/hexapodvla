#!/bin/bash
# Run this from the project root:
#   bash run_training_on_runpod.sh

set -e

POD_SSH="hg475buqtz5l6c-64410cf5@ssh.runpod.io"
SSH_KEY="~/.ssh/id_ed25519"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Step 1: Bundle code + data ==="
cd "$PROJECT_DIR"
tar czf /tmp/mini_vla_bundle.tar.gz mini_vla/ data/hexapod_nav/episodes/session_20260407_105651/

echo "=== Step 2: Upload to RunPod ==="
scp -i $SSH_KEY /tmp/mini_vla_bundle.tar.gz $POD_SSH:/workspace/

echo "=== Step 3: Extract + install deps + train ==="
ssh -i $SSH_KEY $POD_SSH bash -s <<'REMOTE'
set -e
cd /workspace
tar xzf mini_vla_bundle.tar.gz
pip install av pandas pyarrow Pillow tqdm --quiet
echo "=== Starting training ==="
python -m mini_vla.train \
  --session-dir data/hexapod_nav/episodes/session_20260407_105651 \
  --run-dir /workspace/runs/mini_vla \
  --epochs 50 \
  --batch-size 64 \
  --lr 3e-4 \
  --device cuda
echo "=== Training complete ==="
ls -la /workspace/runs/mini_vla/*/checkpoints/
REMOTE

echo "=== Step 4: Download best checkpoint ==="
RUN_DIR=$(ssh -i $SSH_KEY $POD_SSH "ls -d /workspace/runs/mini_vla/run_* | tail -1")
scp -i $SSH_KEY "$POD_SSH:$RUN_DIR/checkpoints/best.pt" "$PROJECT_DIR/runs/mini_vla_best.pt"
scp -i $SSH_KEY "$POD_SSH:$RUN_DIR/training_summary.json" "$PROJECT_DIR/runs/mini_vla_summary.json"

echo ""
echo "=== Done! ==="
echo "Checkpoint: $PROJECT_DIR/runs/mini_vla_best.pt"
echo "Summary:    $PROJECT_DIR/runs/mini_vla_summary.json"
echo ""
echo "Don't forget to stop the pod when done:"
echo "  runpodctl pod stop hg475buqtz5l6c"
