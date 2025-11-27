#!/bin/bash
# Evaluate HMER model on all CROHME test sets (2014, 2016, 2019)

# Activate conda environment
# eval "$(conda shell.bash hook)"
# conda activate dacn

# --- ADDED: Ensure python can find local modules ---
export PYTHONPATH=$PYTHONPATH:.
# --------------------------------------------------
# Configuration
# CHECKPOINT="${1:-checkpoints/CROHME_best.pt}"
CHECKPOINT="${1:-checkpoints/CROHME_final.pt}"

DATA_DIR="data/CROHME"
DICT="${DATA_DIR}/dictionary.txt"
BEAM_SIZE="${2:-5}"
BATCH_SIZE="${3:-16}"

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT"
    echo "Usage: $0 <checkpoint_path> [beam_size] [batch_size]"
    exit 1
fi

# Check if dictionary exists
if [ ! -f "$DICT" ]; then
    echo "Error: Dictionary not found at $DICT"
    exit 1
fi

echo "=========================================="
echo "CROHME Evaluation Script"
echo "=========================================="
echo "Checkpoint:  $CHECKPOINT"
echo "Dictionary:  $DICT"
echo "Beam size:   $BEAM_SIZE"
echo "Batch size:  $BATCH_SIZE"
echo "=========================================="
echo ""

# Evaluate on 2014 test set
echo "Evaluating on CROHME 2014..."
python scripts/evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --test-path "${DATA_DIR}/2014" \
    --dict "$DICT" \
    --beam-size "$BEAM_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --save-results "results_2014.json"

echo ""
echo "=========================================="
echo ""

# Evaluate on 2016 test set
echo "Evaluating on CROHME 2016..."
python scripts/evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --test-path "${DATA_DIR}/2016" \
    --dict "$DICT" \
    --beam-size "$BEAM_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --save-results "results_2016.json"

echo ""
echo "=========================================="
echo ""

# Evaluate on 2019 test set
echo "Evaluating on CROHME 2019..."
python scripts/evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --test-path "${DATA_DIR}/2019" \
    --dict "$DICT" \
    --beam-size "$BEAM_SIZE" \
    --batch-size "$BATCH_SIZE" \
    --save-results "results_2019.json"

echo ""
echo "=========================================="
echo "All evaluations completed!"
echo "Results saved to:"
echo "  - results_2014.json"
echo "  - results_2016.json"
echo "  - results_2019.json"
echo "=========================================="
