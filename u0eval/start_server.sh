#!/usr/bin/env bash
set -uo pipefail
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

ARM=${1:?arm: u0|wam|fallback}
LOGDIR=/hy-tmp/logs/u0eval
mkdir -p "$LOGDIR"
WS=/hy-tmp/underwater_wam

case "$ARM" in
  u0)
    PORT=${2:-8001}
    source /hy-tmp/envs/gr00t/bin/activate
    cd /hy-tmp/u0model
    exec python scripts/inference_service_u0.py --server --http-server \
      --model-path /hy-tmp/models/u0_final --data-config u0_bot \
      --embodiment-tag new_embodiment --denoising-steps 4 \
      --host 0.0.0.0 --port "$PORT" \
      >"$LOGDIR/u0_server.log" 2>&1
    ;;
  gr00t|gr00t_zs|gr00t2)
    PORT=${2:-8004}
    source /hy-tmp/envs/gr00t/bin/activate
    cd /hy-tmp/u0model
    exec python scripts/inference_service_u0.py --server --http-server \
      --model-path "${GR00T_CKPT:-/hy-tmp/baselines/ft/gr00t_n15_lora_b16_40k}" --data-config u0_bot \
      --embodiment-tag new_embodiment --denoising-steps 4 \
      --host 0.0.0.0 --port "$PORT" \
      >"$LOGDIR/${1}_server_$PORT.log" 2>&1
    ;;
  pi05)
    PORT=${2:-8005}
    export HF_HUB_OFFLINE=1 HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
    exec /hy-tmp/envs/miniconda3/envs/lerobot/bin/python "$WS/u0eval/lerobot_policy_server.py" \
      --port "$PORT" --ckpt "${PI05_CKPT:-/hy-tmp/baselines/ft/pi05_usim/checkpoints/last/pretrained_model}" \
      >"$LOGDIR/pi05_server_$PORT.log" 2>&1
    ;;
  u0rep)
    PORT=${2:-8006}
    source /hy-tmp/envs/gr00t/bin/activate
    cd /hy-tmp/u0model
    exec python scripts/inference_service_u0.py --server --http-server \
      --model-path /hy-tmp/models/u0_final --data-config u0_bot \
      --embodiment-tag new_embodiment --denoising-steps 4 \
      --host 0.0.0.0 --port "$PORT" \
      >"$LOGDIR/u0rep_server_$PORT.log" 2>&1
    ;;
  pi05_openpi)
    PORT=${2:-8016}
    cd /hy-tmp/baselines/pi05/u0-openpi || exit 1
    export DATA_BASE_DIR=/hy-tmp/data MODEL_BASE_DIR=/hy-tmp/baselines/pi05 HF_HUB_OFFLINE=1
    export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.4}
    export LD_LIBRARY_PATH=/hy-tmp/envs/miniconda3/envs/lerobot/lib:${LD_LIBRARY_PATH:-}
    exec .venv/bin/python scripts/inference_service_openpi.py --config pi05_u0bot \
      --checkpoint_dir "${PI05_OPENPI_CKPT:-/hy-tmp/baselines/pi05/u0_pi05}" --host 0.0.0.0 --port "$PORT" \
      >"$LOGDIR/pi05_openpi_server_$PORT.log" 2>&1
    ;;
  openvla)
    PORT=${2:-8010}
    export HF_HUB_OFFLINE=1
    exec /hy-tmp/envs/openvla/bin/python "$WS/u0eval/openvla_policy_server.py" \
      --port "$PORT" --ckpt "${OPENVLA_CKPT:-/hy-tmp/baselines/ft/openvla_usim/merged}" \
      --unnorm-key "${OPENVLA_UNNORM_KEY:-usim}" \
      >"$LOGDIR/openvla_server_$PORT.log" 2>&1
    ;;
  fastwam)
    PORT=${2:-8009}
    export HF_HUB_OFFLINE=1 HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
    exec /hy-tmp/envs/miniconda3/envs/lerobot/bin/python "$WS/u0eval/lerobot_policy_server.py" \
      --port "$PORT" --ckpt "${FASTWAM_CKPT:-/hy-tmp/baselines/ft/fastwam_usim/checkpoints/last/pretrained_model}" \
      >"$LOGDIR/fastwam_server_$PORT.log" 2>&1
    ;;
  pi05_zs|xvla_zs|smolvla_zs|fastwam_zs)
    PORT=${2:-8009}
    export HF_HUB_OFFLINE=1 HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 LEROBOT_ZERO_SHOT=1
    exec /hy-tmp/envs/miniconda3/envs/lerobot/bin/python "$WS/u0eval/lerobot_policy_server.py" \
      --port "$PORT" --ckpt "${LEROBOT_CKPT:?LEROBOT_CKPT=<pretrained checkpoint dir>}" \
      >"$LOGDIR/${1}_server_$PORT.log" 2>&1
    ;;
  xvla)
    PORT=${2:-8007}
    export HF_HUB_OFFLINE=1 HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
    exec /hy-tmp/envs/miniconda3/envs/lerobot/bin/python "$WS/u0eval/lerobot_policy_server.py" \
      --port "$PORT" --ckpt "${XVLA_CKPT:-/hy-tmp/baselines/ft/xvla_usim/checkpoints/last/pretrained_model}" \
      >"$LOGDIR/xvla_server_$PORT.log" 2>&1
    ;;
  smolvla)
    PORT=${2:-8008}
    export HF_HUB_OFFLINE=1 HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
    exec /hy-tmp/envs/miniconda3/envs/lerobot/bin/python "$WS/u0eval/lerobot_policy_server.py" \
      --port "$PORT" --ckpt "${SMOLVLA_CKPT:-/hy-tmp/baselines/ft/smolvla_usim/checkpoints/last/pretrained_model}" \
      >"$LOGDIR/smolvla_server_$PORT.log" 2>&1
    ;;
  wam)
    PORT=${2:-8000}
    exec /usr/local/bin/python3 "$WS/u0eval/wam_policy_server.py" --port "$PORT" \
      --eval-root "${EVAL_ROOT:-/hy-tmp/u0env/dataset/eval}" \
      --ckpt "${WAM_CKPT:-/hy-tmp/models/uwam/best_scenes.pt}" \
      --vel-ens "${WAM_VEL_ENS:-/hy-tmp/models/uwam/vel_ens_scenes.pt}" \
      --ou /hy-tmp/data/ou_explore \
      --vmax-scale "${WAM_VMAX_SCALE:-1.0}" \
      --hold-anchor mixer --dump-dir "${WAM_DUMP_DIR-/hy-tmp/data/planner_task}" \
      --n-samples "${WAM_N_SAMPLES:-0}" --cem-iters "${WAM_CEM_ITERS:-0}" \
      --action-source "${WAM_ACTION_SOURCE:-mpc}" \
      --direct-ckpt "${WAM_DIRECT_CKPT:-/hy-tmp/models/uwam/direct_head.pt}" \
      --grasp-planner "${WAM_GRASP_PLANNER:-primitive}" \
      --grasp-ckpt "${WAM_GRASP_CKPT:-/hy-tmp/models/uwam/grasp_core.pt}" \
      --arm-kin "${WAM_ARM_KIN:-/hy-tmp/models/uwam/arm_kin.pt}" \
      --wrist-pose-ckpt "${WAM_WRIST_POSE:-}" \
      >"$LOGDIR/wam_server${PORT/#8000/}.log" 2>&1
    ;;
  wam_percept)
    PORT=${2:-8003}
    exec /usr/local/bin/python3 "$WS/u0eval/wam_policy_server.py" --port "$PORT" \
      --eval-root "${EVAL_ROOT:-/hy-tmp/u0env/dataset/eval}" \
      --vmax-scale "${WAM_VMAX_SCALE:-1.0}" \
      --hold-anchor mixer --goal-source percept --dump-dir "" \
      --vel-ens "${WAM_VEL_ENS:-/hy-tmp/models/uwam/vel_ens_scenes.pt}" \
      --percept-e2e "${WAM_PERCEPT_E2E:-}" \
      >"$LOGDIR/wam_percept_server${PORT/#8003/}.log" 2>&1
    ;;
  fallback)
    PORT=${2:-8002}
    U0URL=${3:-http://127.0.0.1:8001/act}
    exec /usr/local/bin/python3 "$WS/u0eval/fallback_policy_server.py" --port "$PORT" \
      --eval-root "${EVAL_ROOT:-/hy-tmp/u0env/dataset/eval}" \
      --u0-url "$U0URL" \
      --ckpt "${WAM_CKPT:-/hy-tmp/models/uwam/best_scenes.pt}" \
      --vel-ens "${WAM_VEL_ENS:-/hy-tmp/models/uwam/vel_ens_scenes.pt}" \
      --ou /hy-tmp/data/ou_explore \
      --vmax-scale "${WAM_VMAX_SCALE:-1.0}" \
      >"$LOGDIR/fallback_server.log" 2>&1
    ;;
  *)
    echo "unknown arm: $ARM"; exit 1 ;;
esac
