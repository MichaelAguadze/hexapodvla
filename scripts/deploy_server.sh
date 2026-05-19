#!/bin/bash
set -euo pipefail

# Deploy the robot server to a hexapod and optionally install lightweight deps.
#
# Usage:
#   bash scripts/deploy_server.sh [start|deploy|deps|stop|logs]
#
# Environment:
#   ROBOT_IP    Robot IP address. Defaults to 192.168.149.1 for AP mode.
#   ROBOT_USER  SSH username. Defaults to pi.
#   ROBOT_PORT  Server port. Defaults to 8080.

ROBOT_IP="${ROBOT_IP:-192.168.88.100}"
ROBOT_USER="${ROBOT_USER:-xrrobot}"
ROBOT_PORT="${ROBOT_PORT:-8080}"
REMOTE_DIR="/home/${ROBOT_USER}/robot_server"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCAL_DIR="${REPO_ROOT}/robot_server"
ROBOT_REQUIREMENTS="${REPO_ROOT}/requirements-robot.txt"

copy_server_files() {
    ssh "${ROBOT_USER}@${ROBOT_IP}" "mkdir -p ${REMOTE_DIR}"
    scp \
        "${LOCAL_DIR}/server.py" \
        "${LOCAL_DIR}/motor_controller.py" \
        "${LOCAL_DIR}/health_monitor.py" \
        "${ROBOT_USER}@${ROBOT_IP}:${REMOTE_DIR}/"
}

case "${1:-start}" in
    deploy)
        echo "=== Deploying robot server to ${ROBOT_USER}@${ROBOT_IP} ==="
        copy_server_files
        echo "=== Deployed to ${REMOTE_DIR} ==="
        ;;

    deps)
        echo "=== Copying requirements to ${ROBOT_USER}@${ROBOT_IP} ==="
        ssh "${ROBOT_USER}@${ROBOT_IP}" "mkdir -p ${REMOTE_DIR}"
        scp "${ROBOT_REQUIREMENTS}" "${ROBOT_USER}@${ROBOT_IP}:${REMOTE_DIR}/requirements-robot.txt"
        echo "=== Installing robot Python dependencies ==="
        ssh "${ROBOT_USER}@${ROBOT_IP}" \
            "mkdir -p ${REMOTE_DIR} && cd ${REMOTE_DIR} && python3 -m pip install -r requirements-robot.txt"
        echo "=== Done. Note: cv2 and ros_robot_controller_sdk should come from the hexapod image. ==="
        ;;

    start)
        echo "=== Deploying and starting robot server ==="
        copy_server_files

        echo "=== Starting server on robot ==="
        ssh "${ROBOT_USER}@${ROBOT_IP}" \
            "pkill -f 'python3.*server.py' 2>/dev/null || true; sleep 0.5"
        ssh "${ROBOT_USER}@${ROBOT_IP}" \
            "cd ${REMOTE_DIR} && nohup python3 server.py --port ${ROBOT_PORT} > /tmp/robot_server.log 2>&1 &"
        sleep 2

        if curl -s --connect-timeout 3 "http://${ROBOT_IP}:${ROBOT_PORT}/" >/dev/null 2>&1; then
            echo "=== Server running at http://${ROBOT_IP}:${ROBOT_PORT} ==="
        else
            echo "=== WARNING: Server may not have started. Last log lines: ==="
            ssh "${ROBOT_USER}@${ROBOT_IP}" "tail -20 /tmp/robot_server.log"
        fi
        ;;

    stop)
        echo "=== Stopping robot server ==="
        ssh "${ROBOT_USER}@${ROBOT_IP}" "pkill -f 'python3.*server.py' 2>/dev/null || true"
        echo "=== Stopped ==="
        ;;

    logs)
        ssh "${ROBOT_USER}@${ROBOT_IP}" "tail -50 /tmp/robot_server.log"
        ;;

    *)
        echo "Usage: bash scripts/deploy_server.sh [deploy|deps|start|stop|logs]"
        exit 1
        ;;
esac
