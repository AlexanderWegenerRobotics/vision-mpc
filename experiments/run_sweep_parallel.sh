#!/usr/bin/env bash
set -euo pipefail

DIST_CFG="configs/disturbance_config.yaml"
N_SEEDS=20
SCENARIOS=("free" "keepout")
VARIANTS=("BASELINE" "CERTAINTY_EQUIV" "UNCERTAINTY_AWARE")
CELLS=(
  "clean         0.000  0.000  0.0   0"
  "noise_low     0.001  0.005  0.0   0"
  "noise_med     0.003  0.017  0.0   0"
  "noise_high    0.010  0.050  0.0   0"
  "drop_low      0.000  0.000  0.10  0"
  "drop_med      0.000  0.000  0.30  0"
  "drop_high     0.000  0.000  0.50  0"
  "latency_low   0.000  0.000  0.0   2"
  "latency_med   0.000  0.000  0.0   5"
  "latency_high  0.000  0.000  0.0  10"
)

write_cfg () {
  local scenario="$1" level="$2" sxy="$3" sth="$4" drop="$5" lat="$6"
  cat > "$DIST_CFG" <<EOF
scenario: "$scenario"
level_name: "$level"

disturbance:
  sigma_xy: $sxy
  sigma_theta: $sth
  drop_prob: $drop
  latency_frames: $lat
EOF
}

mkdir -p log/runs

for scenario in "${SCENARIOS[@]}"; do
  for cell in "${CELLS[@]}"; do
    read -r level sxy sth drop lat <<< "$cell"
    echo ""
    echo "=== $(date +%H:%M:%S)  scenario=$scenario  cell=$level ==="
    write_cfg "$scenario" "$level" "$sxy" "$sth" "$drop" "$lat"

    pids=()
    for variant in "${VARIANTS[@]}"; do
      logfile="log/runs/${variant}_${scenario}_${level}.log"
      echo "  launching $variant -> $logfile"
      python main.py --variant "$variant" --n-seeds "$N_SEEDS" \
        > "$logfile" 2>&1 &
      pids+=($!)
    done

    fail=0
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        echo "  [warn] pid $pid exited non-zero"
        fail=$((fail+1))
      fi
    done
    if [ "$fail" -gt 0 ]; then
      echo "  $fail variant(s) failed for cell $level — continuing"
    else
      echo "  cell $level done"
    fi
  done
done

echo ""
echo "=== sweep complete at $(date +%H:%M:%S) ==="