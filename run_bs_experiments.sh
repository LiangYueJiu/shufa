#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="../chinese_fonts"
EPOCHS=50
LR=0.0005

BATCH_SIZES=(16 32 64)

for BS in "${BATCH_SIZES[@]}"; do
  RUN_NAME="lr_${LR}_bs_${BS}"
  echo "=============================="
  echo "Starting experiment: ${RUN_NAME}"
  echo "=============================="

  python train_model.py \
    --data-dir "${DATA_DIR}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BS}" \
    --lr "${LR}" \
    --log-dir "output/logs/${RUN_NAME}" \
    --model-dir "output/models/${RUN_NAME}"

  echo "Finished experiment: ${RUN_NAME}"
done

echo "All experiments completed."
