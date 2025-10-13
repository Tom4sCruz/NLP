#!/usr/bin/bash

# RUN inside Scripts/
# Usage:
#   bash testing_all.sh 0.8

# ===============================
# Experiment setup
# ===============================
if [ -z "$1" ]; then
  echo "Usage: bash testing_all.sh <split_ratio>"
  echo "Example: bash testing_all.sh 0.8"
  exit 1
fi

SPLIT="$1"  # Take split ratio from command line

TRAIN_PATH="../Datasets/Splits/training_set_${SPLIT}.csv"
TEST_PATH="../Datasets/Splits/testing_set_${SPLIT}.csv"
TEST_PATH_NO_CHEF="../Datasets/Splits/testing_set_${SPLIT}_no_chef_id.csv"

RESULT_RNN="../Results/results_rnn_testing_set_${SPLIT}_no_chef_id.txt"
RESULT_SCV="../Results/results_linearSVC_testing_set_${SPLIT}_no_chef_id.txt"

HIDDEN_DIMS=(128)
EPOCHS_LIST=(30)

# ===============================
# Run grid of experiments
# ===============================
for H in "${HIDDEN_DIMS[@]}"; do
  for E in "${EPOCHS_LIST[@]}"; do
        echo "[+] Running RNN with HIDDEN_DIM=$H | EPOCHS=$E | MAX_LEN=100 | SPLIT_RATIO=$SPLIT"
        
        python3 Models/rnn.py "$TRAIN_PATH" "$TEST_PATH_NO_CHEF" \
        --hidden_dim "$H" \
        --epochs "$E" \
        --max_len 100

        python3 Misc/obtain_metrics.py "$RESULT_RNN" "$TEST_PATH"
        
        echo "[+] Finished run: H=$H | E=$E | L=100 | SPLIT_RATIO=$SPLIT"
        echo "--------------------------------------------"
  done
done

python3 Models/linearSCV.py "$TRAIN_PATH" "$TEST_PATH_NO_CHEF"
echo "$TRAIN_PATH | $TEST_PATH_NO_CHEF | SPLIT_RATIO=$SPLIT" # > ../Results/Metrics/linearSVC_${SPLIT}_metrics
python3 Misc/obtain_metrics.py "$RESULT_SCV" "$TEST_PATH" # >> ../Results/Metrics/linearSVC_${SPLIT}_metrics

echo "[✅] All experiments completed!"
