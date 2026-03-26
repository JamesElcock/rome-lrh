"""Quick test that the analysis + eval phases don't deadlock with thread limits."""
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

import torch
torch.set_num_threads(4)
torch.set_num_interop_threads(4)

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time

# ── Phase A analysis ops ──
print("[TEST] Phase A: numpy/sklearn ops (cosine, SVD, PCA)...")
t0 = time.time()

V = np.random.randn(30, 1600).astype(np.float64)
concept_dir = np.random.randn(1600).astype(np.float64)
concept_dir /= np.linalg.norm(concept_dir)

raw_mean = V.mean(0)
cos = float(np.dot(raw_mean, concept_dir) / (np.linalg.norm(raw_mean) * np.linalg.norm(concept_dir)))
print(f"  cosine: {cos:.4f}")

V_c = V - V.mean(0, keepdims=True)
U, S, Vh = np.linalg.svd(V_c, full_matrices=False)
print(f"  SVD: {U.shape}, {S.shape}, {Vh.shape}")

# Simulate sklearn logistic regression (used in concept direction extraction)
from sklearn.linear_model import LogisticRegressionCV
X = np.random.randn(100, 1600)
y = np.random.randint(0, 2, 100)
clf = LogisticRegressionCV(max_iter=200, cv=3)
clf.fit(X, y)
print(f"  LogisticRegressionCV fit done, score={clf.score(X, y):.2f}")

print(f"  Phase A: {time.time()-t0:.1f}s  OK")

# ── Phase B model eval ops ──
print("\n[TEST] Phase B: model load + eval...")
t0 = time.time()

from transformers import AutoModelForCausalLM, AutoTokenizer
from util import nethook
from rome.rome_hparams import ROMEHyperParams
from util.globals import HPARAMS_DIR

model = AutoModelForCausalLM.from_pretrained("gpt2-xl").cuda().eval()
tok = AutoTokenizer.from_pretrained("gpt2-xl")
tok.pad_token = tok.eos_token
hparams = ROMEHyperParams.from_json(HPARAMS_DIR / "ROME" / "gpt2-xl.json")

layer = 17
weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
w = nethook.get_parameter(model, weight_name)

u = torch.randn(6400, device=w.device)
v = torch.randn(1600, device=w.device)
upd = u.unsqueeze(1) @ v.unsqueeze(0)
if upd.shape != w.shape:
    upd = upd.T

with torch.no_grad():
    w[...] += upd

inputs = tok("The mother tongue of Danielle Darrieux is", return_tensors="pt").to(w.device)
with torch.no_grad():
    logits = model(**inputs).logits[0, -1]
pred = tok.decode([logits.argmax().item()])
print(f"  Eval predicted: '{pred}'")

with torch.no_grad():
    w[...] -= upd

print(f"  Phase B: {time.time()-t0:.1f}s  OK")

# ── Phase A+B interleaved (the actual failure pattern) ──
print("\n[TEST] Phase A+B interleaved: numpy after model is loaded...")
t0 = time.time()

V2 = np.random.randn(30, 1600).astype(np.float64)
U2, S2, Vh2 = np.linalg.svd(V2 - V2.mean(0, keepdims=True), full_matrices=False)
cos2 = [float(np.dot(V2[i], concept_dir) / (np.linalg.norm(V2[i]) + 1e-10)) for i in range(30)]
print(f"  SVD + 30 cosines done in {time.time()-t0:.2f}s  OK")

print("\n[TEST] ALL PASSED")
