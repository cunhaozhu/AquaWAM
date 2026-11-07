#!/usr/bin/env bash
INST=${1:?A|B|C|D}; STALL_MIN=${2:-20}
case $INST in
  A) ROOT=/hy-tmp/u0env;   ROS0=11311; PORTS="800[0-9]" ;;
  B) ROOT=/hy-tmp/u0env_b; ROS0=12311; PORTS="801[0-9]" ;;
  C) ROOT=/hy-tmp/u0env_c; ROS0=13311; PORTS="802[0-9]" ;;
  D) ROOT=/hy-tmp/u0env_d; ROS0=14311; PORTS="803[0-9]" ;;
  *) exit 1 ;;
esac
while true; do
  if [ "$STALL_MIN" != 0 ]; then
    sleep 300
    pgrep -f "run_eval_task.sh .* $PORTS " >/dev/null || continue
    since=$(date -d "-${STALL_MIN} minutes" '+%Y-%m-%d %H:%M:%S')
    newest=$(find $ROOT/dataset/eval -maxdepth 3 \( -name 'episode_*_data.csv' -o -name results.csv -o -name '*.log' \) -newermt "$since" 2>/dev/null | head -1)
    [ -n "$newest" ] && continue
  fi
  n=0
  for p in $(ps -eo pid,cmd | awk '/roslaunch|rosmaster|parsed_simulator|gr00t_bridge|data_collector|eval_tracking|eval_grasping|eval_transporting|eval_follow|rosout|robot_state_publisher|move_group/ && !/awk|batch_run|run_eval_task|stall_watchdog/{print $1}'); do
    uri=$(tr '\0' '\n' < /proc/$p/environ 2>/dev/null | sed -n 's/^ROS_MASTER_URI=http:\/\/[^:]*:\([0-9]*\).*/\1/p')
    if [ -n "$uri" ] && [ "$uri" -ge $ROS0 ] && [ "$uri" -lt $((ROS0 + 100)) ]; then kill -9 "$p" 2>/dev/null; n=$((n + 1)); fi
  done
  echo "[$(date +%m-%d\ %H:%M:%S)] [$INST] no progress for ${STALL_MIN} min -> killed $n ROS processes of the hung episode"
  [ "$STALL_MIN" = 0 ] && exit 0
done
