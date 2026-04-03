#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="../chinese_fonts"
EPOCHS=50
BATCH_SIZE=32

LRS=(0.01 0.001 0.0001)

for LR in "${LRS[@]}"; do
  RUN_NAME="lr_${LR}_bs_${BATCH_SIZE}"
  echo "=============================="
  echo "Starting experiment: ${RUN_NAME}"
  echo "=============================="

  python train_model.py \
    --data-dir "${DATA_DIR}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --log-dir "output/logs/${RUN_NAME}" \
    --model-dir "output/models/${RUN_NAME}"

  echo "Finished experiment: ${RUN_NAME}"
done

echo "All experiments completed."
