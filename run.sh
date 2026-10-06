#!/bin/bash
# Clean PatchNMF training & evaluation pipeline script

option=${1:-1}
BASE_DIR="${2:-.}"
OUT_DIR="${3:-${BASE_DIR}/output_patchnmf}"

if [ "$option" -eq 1 ]; then
    echo "=== Training PatchNMF with Patient-Level Splitting & Tissue-Only Scaling ==="
    python train_patch.py \
        --base "$BASE_DIR" \
        --out "$OUT_DIR" \
        --normal-class after \
        --norm-mode global \
        --patch 16 --stride 8 \
        --block 16 \
        --k-list 20,30,40,50,60,75 \
        --max-iter 500 \
        --max-patches 250000 \
        --train-frac 0.70 --val-frac 0.15

elif [ "$option" -eq 2 ]; then
    echo "=== Evaluating PatchNMF Anomaly Detection (Ischemia vs Normal) ==="
    MODEL_PATH="${4:-${OUT_DIR}/models/patchnmf_k40.joblib}"
    EVAL_OUT="${OUT_DIR}/eval_k40"

    python evaluate_hybrid_patch_anomaly.py \
        --dataset-root "$BASE_DIR" \
        --train-normal Training/after \
        --test-normal Testing/after \
        --test-anom Testing/Ischemia \
        --model "$MODEL_PATH" \
        --out "$EVAL_OUT" \
        --norm-mode global \
        --block 16 \
        --score-mode quantile --score-quantile 0.995 \
        --w-res 1.0 --w-lat 0.5 --w-rar 0.5
fi
