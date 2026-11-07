#!/usr/bin/env bash
set -uo pipefail
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy


TASK=${1:?task_code}
ARM=${2:?arm_tag}
NEP=${3:?n_episodes}
PORT=${4:?vla_port}
DROP=${5:--1.0}
DMODE=${6:-zero}

U0ENV=${U0ENV:-/hy-tmp/u0env}
DISPLAY_NUM=${DISPLAY_NUM:-:1}
COND=full
if [[ "$DROP" != "-1.0" && "$DROP" != "-1" ]]; then COND="drop${DROP}s_${DMODE}"; fi
if [ -n "${COND_TAG:-}" ]; then COND="${COND_TAG}"; fi
CONFIG=${CONFIG:-exp_setting.csv}
DEST="$U0ENV/dataset/eval_runs/${ARM}_${COND}/$TASK"

XDISP=${DISPLAY_NUM#:} XGPU=${X_GPU:-0} bash /hy-tmp/underwater_wam/scripts/start_nvidia_x.sh || true
export DISPLAY=$DISPLAY_NUM
set +u
export PATH=/hy-tmp/envs/ros_env/bin:$PATH
export CONDA_PREFIX=/hy-tmp/envs/ros_env
export LD_LIBRARY_PATH=$U0ENV/build/stonefish_install/lib:/hy-tmp/envs/ros_env/lib:${LD_LIBRARY_PATH:-}
source /hy-tmp/envs/ros_env/setup.bash
source $U0ENV/ros_ws/devel/setup.bash
set -u

mine() {
  local base=${INSTANCE_ROSPORT:-11311}
  local uri; uri=$(tr '\0' '\n' < /proc/$1/environ 2>/dev/null | sed -n 's/^ROS_MASTER_URI=http:\/\/[^:]*:\([0-9]*\).*/\1/p')
  [ -n "$uri" ] && [ "$uri" -ge "$base" ] && [ "$uri" -lt $((base + 100)) ]
}
for pat in 'ros_gr00t_bridge' 'vla_data_collector' 'eval_tracking' 'eval_grasping' \
           'eval_follow' 'eval_transporting' 'mapper_setup' 'alpha5_' 'move_group' \
           'rostopic' '/parsed_simulator ' 'roslaunch' 'rosmaster' 'rosout'; do
  for p in $(ps -eo pid,cmd | awk -v pat="$pat" 'index($0, pat) && !/awk/{print $1}'); do
    mine "$p" && kill -9 "$p" 2>/dev/null || true
  done
done
sleep 3

cd "$U0ENV"
rm -rf "dataset/eval/$TASK"
echo "=== [$ARM/$COND] $TASK x $NEP episodes  (port $PORT, drop=$DROP/$DMODE)  $(date +%H:%M:%S)"
bash batch_run_ext.sh -m eval -t "$TASK" --vlaport "$PORT" -c "$CONFIG" \
  --eval-num "$NEP" --drop-dvl-at "$DROP" --dvl-drop-mode "$DMODE" \
  ${INSTANCE_ROSPORT:+--rosport "$INSTANCE_ROSPORT"}
rc=$?

mkdir -p "$DEST"
if [ -d "dataset/eval/$TASK" ]; then
  rm -rf "$DEST/logs" "$DEST/episode0" "$DEST/results.csv"
  cp -r "dataset/eval/$TASK/logs" "$DEST/" 2>/dev/null || true
  cp "dataset/eval/$TASK/results.csv" "$DEST/" 2>/dev/null || true
  [ -f "$U0ENV/KEEP_ALL_EPISODES" ] && KEEP_ALL_EPISODES=1
  if [ "${KEEP_ALL_EPISODES:-0}" = 1 ]; then
    for e in "dataset/eval/$TASK"/episode*/; do [ -d "$e" ] && cp -r "$e" "$DEST/"; done
  else
    [ -d "dataset/eval/$TASK/episode0" ] && cp -r "dataset/eval/$TASK/episode0" "$DEST/"
    find "dataset/eval/$TASK" -mindepth 1 -maxdepth 1 -type d -name 'episode[1-9]*' -exec rm -rf {} +
  fi
  echo "=== archived to $DEST (results + logs + episode0)"
  echo "--- results.csv ---"
  cat "$DEST/results.csv" 2>/dev/null || echo "(no results.csv)"
fi
exit $rc
