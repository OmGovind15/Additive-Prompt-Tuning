#!/bin/bash

# Experiment settings
GPUID='0'
SEEDS=(1 2 3)
DELAY=10

# Master function to run a single experiment
run_experiment() {
    local DATASET=$1
    local CONFIG=$2
    local LEARNER_TYPE=$3
    local LEARNER_NAME=$4
    local LR=$5
    local SCHEDULE=$6
    local EMA_COEFF=0.7

    echo "========================================"
    echo "Starting $DATASET with $LEARNER_NAME"
    echo "========================================"

    for seed in "${SEEDS[@]}"
    do
        local LOG_DIR="logs"
        local OUTDIR="./checkpoints/${DATASET}-pa-apt-v2/seed${seed}"
        local LOG_FILE="${LOG_DIR}/${DATASET}-pa-apt-v2/seed${seed}.log"
        
        mkdir -p "${OUTDIR}"
        mkdir -p "${LOG_DIR}/${DATASET}-pa-apt-v2"

        echo ">>> Running $DATASET Seed $seed"
        
        # Use the base run_pa_apt.py, passing all V2 parameters via CLI
        nohup python -u run_pa_apt.py \
            --config "${CONFIG}" \
            --gpuid "${GPUID}" \
            --lr "${LR}" \
            --schedule "${SCHEDULE}" \
            --ema_coeff "${EMA_COEFF}" \
            --seed "${seed}" \
            --learner_type "${LEARNER_TYPE}" \
            --learner_name "${LEARNER_NAME}" \
            --log_dir "${OUTDIR}" > "${LOG_FILE}" 2>&1 &

        PID=$!
        wait $PID
        
        if [ $? -eq 0 ]; then
            echo "Seed $seed completed successfully."
        else
            echo "X Seed $seed failed. Check $LOG_FILE"
        fi
        
        # Add delay before next experiment
        echo "----------------------------------------"
        echo "Waiting for ${DELAY} seconds..."
        sleep "${DELAY}"
    done
}

# --- Experiment Definitions ---

# 1. CIFAR-100 PA-APT-V2
#run_experiment "cifar-100" \
 #              "configs/cifar-100_pa_apt_v2.yaml" \
  #             "pa_apt_v2" \
   #            "PA_APT_V2_Learner" \
    #           "0.004" \
     #          "30"

# 2. CUB-200 PA-APT-V2
run_experiment "CUB200" \
               "configs/cub200_pa_apt_v2.yaml" \
               "pa_apt_v2" \
               "PA_APT_V2_Learner" \
               "0.02" \
               "25"

echo "All PA-APT V2 experiments are complete."
