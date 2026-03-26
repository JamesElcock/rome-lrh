"""
Experiment 4 (r-ROME variant): Causal Decomposition of the Shared Edit Vector

Identical to exp4_causal_decomposition.py except uses r-ROME edit vectors
from results/exp2_rrome/. Saves to results/exp4_rrome/.

No ROME/r-ROME execution needed — this experiment decomposes pre-computed
v_mean vectors and tests each component as a standalone edit.
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
import scripts.exp4_causal_decomposition as exp4

# ── Override paths ────────────────────────────────────────────
exp4.RESULTS_DIR = Path("results/exp4_rrome")
exp4.EXP2_DIR = Path("results/exp2_rrome")  # use r-ROME edit vectors

# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING EXP 4 WITH r-ROME edit vectors")
    print("=" * 60)
    exp4.main()
