#!/bin/bash
set -euo pipefail

## HP LR and WD sweep:
#MODELS=(base SENNrawx SENNfixed SENNfixed_concepttheta LogisticConcepts)
#LRS=(2e-4 2e-3 2e-2)
#WDS=(1e-5 1e-4 1e-3)
#RobLoss=(0.0)
#

# RobLoss sweep
MODELS=(SENNrawx)
LRS=(2e-3)
WDS=(1e-3)
RobLoss=(0.0 1e-8 3e-8 1e-7 3e-7 1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3)


for MODEL in "${MODELS[@]}"; do
  for LR in "${LRS[@]}"; do
    for WD in "${WDS[@]}"; do
      for RL in "${RobLoss[@]}"; do
        sbatch ./slurm_jobs/CV_jobHP.sh "$MODEL" "$LR" "$WD" "$RL"
      done
    done
  done
done