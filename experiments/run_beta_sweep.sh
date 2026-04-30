#!/usr/bin/env bash
set -euo pipefail

TASK_CFG="configs/task_config.yaml"
DIST_CFG="configs/disturbance_config.yaml"
STUDY_CFG="configs/study_config.yaml"
N_SEEDS=20
VARIANT="UNCERTAINTY_AWARE"

BETAS=(0.5 1.0 1.282 1.645 2.0 2.576)

write_dist_cfg () {
  cat > "$DIST_CFG" <<EOF
scenario: "keepout"
level_name: "noise_high"

disturbance:
  sigma_xy: 0.010
  sigma_theta: 0.050
  drop_prob: 0.0
  latency_frames: 0
EOF
}

write_beta () {
  local beta="$1"
  sed -i '' "s/^  cc_beta:.*$/  cc_beta: $beta/" "$TASK_CFG"
}

write_scenario_fixed () {
  # Overwrite study config with fixed start/goal and keepout wall
  cat > "$STUDY_CFG" <<EOF
keepout:
  enabled: true
  lbx: [-1.0e6, -1.0e6, -0.05]
  ubx: [ 1.0e6,  0.18,   0.05]

scenarios:
  free:
    start: { xy: [0.55, -0.2], theta: 0.0 }
    goal:  { xy: [0.6,  0.1], theta: 0.0 }
    keepout:
      enabled: false
      lbx: [-1.0e6, -1.0e6, -0.05]
      ubx: [ 1.0e6,  1.0e6,  0.05]

  keepout:
    start: { xy: [0.55, -0.2], theta: 0.2 }
    goal:  { xy: [0.6,   0.1], theta: 0.0 }
    keepout:
      enabled: true
      lbx: [-1.0e6, -1.0e6, -0.05]
      ubx: [ 1.0e6,  0.18,   0.05]
EOF
}

mkdir -p log/beta_sweep

write_dist_cfg
write_scenario_fixed

echo "=== beta sweep: keepout / noise_high / UNCERTAINTY_AWARE ==="
echo "=== fixed start [0.55, -0.2] -> goal [0.6, 0.1], wall y=0.35 ==="
echo ""

for beta in "${BETAS[@]}"; do
  echo "=== $(date +%H:%M:%S)  beta=$beta ==="
  write_beta "$beta"

  logfile="log/beta_sweep/beta_${beta}.log"
  if ! ACADOS_VARIANT_TAG="beta_${beta}" python main.py \
        --variant "$VARIANT" --n-seeds "$N_SEEDS" \
        > "$logfile" 2>&1; then
    echo "  [warn] beta=$beta exited non-zero"
  else
    echo "  beta=$beta done"
  fi
done

# Restore original beta
write_beta "1.645"
echo ""
echo "=== sweep complete at $(date +%H:%M:%S) — cc_beta restored to 1.645 ==="