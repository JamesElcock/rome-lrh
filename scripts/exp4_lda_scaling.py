"""
Experiment 4 extension: Scale the LDA component of v_mean beyond 1×.

Uses the same decomposition as Exp 4 (v_mean = v_concept + v_lda + v_residual)
but tests what happens when the LDA component is rescaled to 2×, 4×, 8× the
original v_mean norm. Also tests residual at same scales for comparison.

Saves to results/exp4/lda_scaling.json
"""

import json
import os
import sys
import numpy as np
import torch
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from rome.rome_hparams import ROMEHyperParams
from util import nethook
from util.globals import HPARAMS_DIR
import rome.rome_main as rome_main_module

# ============================================================
RELATIONS = ["P176", "P1412", "P37", "P27", "P413"]
EDIT_LAYER = 17
N_TEST_ENTITIES = 5
N_LDA_DIRS = 9
SEED = 42
SCALE_FACTORS = [1, 2, 3, 4, 8, 16, 32, 64]

RESULTS_DIR = Path("results/exp4")
EXP1_DIR = Path("results/exp1")
EXP2_DIR = Path("results/exp2")
OUTPUT_FILE = RESULTS_DIR / "lda_scaling_v4.json"


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def rescale(v, target_norm):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-10:
        return v
    return v * (target_norm / n)


def orthogonalize_lda_against_concept(lda_dirs, concept_dir):
    d = concept_dir / (np.linalg.norm(concept_dir) + 1e-10)
    residuals = []
    for i in range(lda_dirs.shape[1]):
        li = lda_dirs[:, i].copy()
        li -= np.dot(li, d) * d
        norm = np.linalg.norm(li)
        if norm > 1e-8:
            residuals.append(li)
    if not residuals:
        return np.zeros((len(d), 0))
    R = np.stack(residuals, axis=1)
    Q, _ = np.linalg.qr(R)
    k = min(len(residuals), Q.shape[1])
    return Q[:, :k]


def apply_edit_and_eval(model, tok, u, v, hparams, prompt_text, target_str):
    layer = hparams.layers[0]
    weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
    w = nethook.get_parameter(model, weight_name)

    u_t = torch.tensor(u, dtype=torch.float32).to(w.device)
    v_t = torch.tensor(v, dtype=torch.float32).to(w.device)
    upd = u_t.unsqueeze(1) @ v_t.unsqueeze(0)
    if upd.shape != w.shape:
        upd = upd.T

    with torch.no_grad():
        w[...] += upd

    target_tok = tok(f" {target_str.strip()}", return_tensors="pt")["input_ids"][0]
    first_target_tok = target_tok[0].item()

    inputs = tok(prompt_text, return_tensors="pt").to(w.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
    probs = torch.softmax(logits.float(), dim=0)
    pred_tok = logits.argmax().item()
    target_prob = probs[first_target_tok].item()

    with torch.no_grad():
        w[...] -= upd

    return {
        "efficacy": int(pred_tok == first_target_tok),
        "target_prob": float(target_prob),
        "pred_token": tok.decode([pred_tok]),
    }


def main():
    print("=" * 70)
    print("EXPERIMENT 4 EXTENSION: LDA COMPONENT SCALING")
    print(f"Scale factors: {SCALE_FACTORS}")
    print("=" * 70)

    # Load model
    print("\nLoading GPT-2 XL...")
    model = AutoModelForCausalLM.from_pretrained("gpt2-xl").cuda().eval()
    tok = AutoTokenizer.from_pretrained("gpt2-xl")
    tok.pad_token = tok.eos_token
    hparams = ROMEHyperParams.from_json(HPARAMS_DIR / "ROME" / "gpt2-xl.json")

    with open(EXP1_DIR / "context_templates.json") as f:
        templates = json.load(f)
    rome_main_module.CONTEXT_TEMPLATES_CACHE = templates

    # Load artifacts
    exp2_data = torch.load(EXP2_DIR / "edit_vectors.pt", map_location="cpu")
    exp2_meta = exp2_data["metadata"]
    exp2_V = exp2_data["v"].numpy()
    exp2_U = exp2_data["u"].numpy()

    cd_all = torch.load(EXP1_DIR / "concept_directions_layer17.pt", map_location="cpu")
    lda_all = np.load(EXP2_DIR / "lda_directions.npy")
    lda_dirs = lda_all[:, :N_LDA_DIRS].copy()

    with open(EXP1_DIR / "relation_targets.json") as f:
        relation_targets = json.load(f)

    # Define tasks
    tasks = []
    for rid in RELATIONS:
        for target in relation_targets[rid][:2]:
            tasks.append({"relation_id": rid, "target": target})
    print(f"\n{len(tasks)} concepts")

    rng = np.random.RandomState(SEED)
    all_results = []

    for task in tasks:
        rid = task["relation_id"]
        target = task["target"]
        key = f"{rid}_{target}"

        # Get edit vectors for this concept
        idxs = [i for i, m in enumerate(exp2_meta)
                if m["relation_id"] == rid and m["target_value"] == target]
        if len(idxs) < 3:
            continue

        V = exp2_V[idxs]
        U = exp2_U[idxs]
        v_mean = V.mean(axis=0)
        vm_norm = float(np.linalg.norm(v_mean))

        # Concept direction
        cd_key = f"{rid}_{target}_logistic"
        if cd_key not in cd_all:
            continue
        concept_dir = cd_all[cd_key].numpy().astype(np.float64)

        # Decompose
        lda_orth = orthogonalize_lda_against_concept(lda_dirs, concept_dir)
        v_concept = np.dot(v_mean, concept_dir) * concept_dir
        v_lda = lda_orth @ (lda_orth.T @ v_mean)
        v_residual = v_mean - v_concept - v_lda

        print(f"\n  {key}: ||v_mean||={vm_norm:.3f}  "
              f"||v_lda||={np.linalg.norm(v_lda):.3f}  "
              f"||v_res||={np.linalg.norm(v_residual):.3f}")

        # Select test entities
        n_avail = len(idxs)
        test_idxs = rng.choice(n_avail, min(N_TEST_ENTITIES, n_avail), replace=False)

        # Build conditions
        conditions = {}

        # Baselines
        conditions["full_v_mean"] = v_mean
        conditions["residual_1x"] = rescale(v_residual, vm_norm)

        # Components at various scales
        for sf in SCALE_FACTORS:
            conditions[f"vmean_{sf}x"] = rescale(v_mean, vm_norm * sf)
            conditions[f"concept_{sf}x"] = rescale(v_concept, vm_norm * sf)
            conditions[f"lda_{sf}x"] = rescale(v_lda, vm_norm * sf)
            conditions[f"residual_{sf}x"] = rescale(v_residual, vm_norm * sf)
            # Also: LDA at scale + residual at natural
            conditions[f"lda_{sf}x_plus_residual"] = rescale(v_lda, vm_norm * sf) + v_residual
            # And: concept + LDA at scale
            conditions[f"concept_plus_lda_{sf}x"] = v_concept + rescale(v_lda, vm_norm * sf)

        # Random control at max scale
        rand_v = rng.randn(1600).astype(np.float64)
        conditions[f"random_{max(SCALE_FACTORS)}x"] = rescale(rand_v, vm_norm * max(SCALE_FACTORS))

        # Evaluate
        concept_results = []
        for ti, eidx in enumerate(test_idxs):
            u_entity = U[eidx]
            v_entity = V[eidx]
            meta = exp2_meta[idxs[eidx]]
            prompt_text = meta["prompt"].replace("{}", meta["subject"])

            entity_res = {"subject": meta["subject"]}

            # Own v
            res = apply_edit_and_eval(model, tok, u_entity, v_entity,
                                     hparams, prompt_text, target)
            entity_res["own_v"] = res

            for cond_name, v_cond in conditions.items():
                res = apply_edit_and_eval(model, tok, u_entity, v_cond,
                                         hparams, prompt_text, target)
                entity_res[cond_name] = res

            concept_results.append(entity_res)

        # Summarize this concept
        cond_names = ["own_v", "full_v_mean"] + sorted(
            [c for c in conditions if c not in ("full_v_mean",)])
        print(f"    {'Condition':40s} {'Eff':>6s} {'Prob':>8s}")
        for c in cond_names:
            effs = [e[c]["efficacy"] for e in concept_results if c in e]
            probs = [e[c]["target_prob"] for e in concept_results if c in e]
            if effs:
                print(f"    {c:40s} {np.mean(effs):6.2f} {np.mean(probs):8.4f}")

        all_results.append({
            "concept": key,
            "v_mean_norm": vm_norm,
            "norm_v_lda": float(np.linalg.norm(v_lda)),
            "norm_v_residual": float(np.linalg.norm(v_residual)),
            "entity_results": concept_results,
        })

    # ── Grand average ──
    print("\n" + "=" * 70)
    print("GRAND AVERAGE")
    print("=" * 70)

    cond_names = ["own_v", "full_v_mean"]
    for sf in SCALE_FACTORS:
        cond_names.extend([f"vmean_{sf}x", f"concept_{sf}x", f"lda_{sf}x", f"residual_{sf}x",
                          f"lda_{sf}x_plus_residual", f"concept_plus_lda_{sf}x"])
    cond_names.append(f"random_{max(SCALE_FACTORS)}x")

    grand = {}
    print(f"\n  {'Condition':40s} {'Eff':>6s} {'Prob':>8s} {'N':>4s}")
    print("  " + "-" * 62)
    for c in cond_names:
        effs, probs = [], []
        for cr in all_results:
            for e in cr["entity_results"]:
                if c in e:
                    effs.append(e[c]["efficacy"])
                    probs.append(e[c]["target_prob"])
        if effs:
            grand[c] = {"efficacy": float(np.mean(effs)),
                        "target_prob": float(np.mean(probs)),
                        "n": len(effs)}
            print(f"  {c:40s} {np.mean(effs):6.3f} {np.mean(probs):8.4f} {len(effs):4d}")

    # Save
    output = {
        "config": {
            "scale_factors": SCALE_FACTORS,
            "relations": RELATIONS,
            "n_test_entities": N_TEST_ENTITIES,
        },
        "grand_average": grand,
        "per_concept": [{
            "concept": r["concept"],
            "v_mean_norm": r["v_mean_norm"],
            "norm_v_lda": r["norm_v_lda"],
            "norm_v_residual": r["norm_v_residual"],
            "entity_results": r["entity_results"],
        } for r in all_results],
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()