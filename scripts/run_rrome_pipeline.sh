#!/bin/bash
# r-ROME full experiment pipeline
# Run with: nohup bash scripts/run_rrome_pipeline.sh > results/rrome_pipeline.log 2>&1 &
set -e
cd /home/jse44/modules/XAI/rome

# Prevent MKL threading deadlock and PyTorch oversubscription
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MKL_THREADING_LAYER=SEQUENTIAL

# -u = unbuffered stdout so we can monitor progress in the log
CONDA_RUN="conda run -n rome --no-capture-output"
PY="python -u"

echo "========================================"
echo "r-ROME Pipeline — started $(date)"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "========================================"

# ── Exp 1B ────────────────────────────────────
if [ -f results/exp1b_rrome/results.json ]; then
    echo "[exp1b] Already complete, skipping."
else
    echo "[exp1b] Running... $(date)"
    $CONDA_RUN $PY scripts/exp1b_rrome.py
    echo "[exp1b] Done. $(date)"
fi

# ── Exp 2 ─────────────────────────────────────
if [ -f results/exp2_rrome/edit_vectors.pt ]; then
    echo "[exp2] Already complete, skipping."
else
    echo "[exp2] Running... $(date)"
    $CONDA_RUN $PY scripts/exp2_rrome.py
    echo "[exp2] Done. $(date)"
fi

# ── Exp 3 ─────────────────────────────────────
if [ -f results/exp3_rrome/results.json ]; then
    echo "[exp3] Already complete, skipping."
else
    echo "[exp3] Running... $(date)"
    $CONDA_RUN $PY scripts/exp3_rrome.py
    echo "[exp3] Done. $(date)"
fi

# ── Exp 4 ─────────────────────────────────────
if [ -f results/exp4_rrome/results.json ]; then
    echo "[exp4] Already complete, skipping."
else
    echo "[exp4] Running... $(date)"
    $CONDA_RUN $PY scripts/exp4_rrome.py
    echo "[exp4] Done. $(date)"
fi

echo "========================================"
echo "r-ROME Pipeline — finished $(date)"
echo "========================================"