#!/usr/bin/env bash
set -uo pipefail
NEP=${1:-5}
shift || true
TASKS=("$@")
if [ ${#TASKS[@]} -eq 0 ]; then
  TASKS=(goto_charge_station goto_water_tower scan_ship_ancient scan_ship_modern
         inspect_pipeline_pool inspect_pipeline_sea follow_boat)
fi
WS=/hy-tmp/underwater_wam
LOG=/hy-tmp/logs/u0eval
mkdir -p "$LOG"

drop_for() {
  case "$1" in
    goto_*)            echo 8   ;;
    scan_*|inspect_*)  echo 40  ;;
    follow_*)          echo 20  ;;
    pick_*|transfer_*) echo 30  ;;
    *)                 echo 20  ;;
  esac
}
port_for() {
  case "$1" in
    u0) echo 8001 ;;
    wam) echo 8000 ;;
    fallback) echo 8002 ;;
  esac
}

for arm in u0 wam fallback; do
  PORT=$(port_for "$arm")
  if ! curl -s -m 5 "http://127.0.0.1:$PORT/health" >/dev/null; then
    echo "!!! $arm server on $PORT not healthy -- aborting"; exit 1
  fi
done

for task in "${TASKS[@]}"; do
  DROP=$(drop_for "$task")
  for arm in u0 wam fallback; do
    PORT=$(port_for "$arm")
    echo "########## PILOT $task / $arm / full  $(date +%H:%M:%S)"
    bash "$WS/u0eval/run_eval_task.sh" "$task" "$arm" "$NEP" "$PORT" -1.0 zero \
      >"$LOG/pilot_${task}_${arm}_full.log" 2>&1 || echo "  (nonzero exit)"
    echo "########## PILOT $task / $arm / drop${DROP}s  $(date +%H:%M:%S)"
    bash "$WS/u0eval/run_eval_task.sh" "$task" "$arm" "$NEP" "$PORT" "$DROP" zero \
      >"$LOG/pilot_${task}_${arm}_drop.log" 2>&1 || echo "  (nonzero exit)"
  done
done
echo "PILOT_ALL_DONE $(date +%H:%M:%S)"
