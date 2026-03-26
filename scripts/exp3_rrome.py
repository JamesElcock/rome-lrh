"""
Experiment 3 (r-ROME variant): Layer Propagation Analysis

Identical to exp3_layer_propagation.py except uses r-ROME edit vectors
from results/exp2_rrome/. Saves to results/exp3_rrome/.

No ROME/r-ROME execution needed — this experiment only applies pre-computed
u ⊗ v edits and measures perturbation propagation.
"""

import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(4)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path

# Import the original experiment module
import scripts.exp3_layer_propagation as exp3

# ── Override paths ────────────────────────────────────────────
exp3.RESULTS_DIR = Path("results/exp3_rrome")
exp3.EXP2_DIR = Path("results/exp2_rrome")  # use r-ROME edit vectors

# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING EXP 3 WITH r-ROME edit vectors")
    print("=" * 60)
    exp3.main()
