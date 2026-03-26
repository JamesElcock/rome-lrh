"""
Experiment 1B (r-ROME variant): Shared Component Analysis

Identical to exp1b_shared_component.py except uses r-ROME (fixed key averaging
in compute_v) instead of original ROME. Saves to results/exp1b_rrome/.
"""

import os
import sys

# Prevent thread oversubscription deadlocks (PyTorch + MKL/BLAS)
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(4)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from rrome import execute_rrome

# Import the original experiment module
import scripts.exp1b_shared_component as exp1b

# ── Override paths ────────────────────────────────────────────
exp1b.RESULTS_DIR = Path("results/exp1b_rrome")

# ── Override run_rome_edit to use r-ROME ──────────────────────
def run_rrome_edit(model, tok, request, hparams):
    """Run r-ROME, return (u, v). Model unchanged."""
    deltas = execute_rrome(model, tok, request, hparams)
    for key, (u, v) in deltas.items():
        return u.detach().cpu(), v.detach().cpu()

exp1b.run_rome_edit = run_rrome_edit

# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING EXP 1B WITH r-ROME (fixed key averaging)")
    print("=" * 60)
    exp1b.main()
