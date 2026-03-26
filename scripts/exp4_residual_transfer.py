"""
Experiment 4 extension: Cross-relation transfer of the dark residual.

Tests whether the dark residual from relation A can produce correct edits
when paired with u vectors from relation B.

Design:
- For each relation, compute v_residual (dark component of v_mean)
- For each test edit, pair the entity's own u with:
  1. Own relation's residual (within-relation, should work ~96% at 3x)
  2. Each other relation's residual (cross-relation transfer)
  3. Average residual across all relations (universal residual)
  4. Own v_mean (baseline)
  5. Random direction (control)

All residuals rescaled to 3× ||v_mean|| (optimal scale from sweep).

Saves to results/exp4/residual_transfer.json
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
N_LDA_DIRS = 9
N_TEST_ENTITIES = 5
SEED = 42
OPTIMAL_SCALE = 3.0

EXP1_DIR = Path("results/exp1")
EXP2_DIR = Path("results/exp2")
RESULTS_DIR = Path("results/exp4")
OUTPUT_FILE = RESULTS_DIR / "residual_transfer.json"


def rescale(v, target_norm):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-10:
        return v
    return v * (target_norm / n)


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


def orthogonalize_lda_against_concept(lda_dirs, concept_dir):
    d = concept_dir / (np.linalg.norm(concept_dir) + 1e-10)
    residuals = []
    for i in range(lda_dirs.shape[1]):
        li = lda_dirs[:, i].copy()
        li -= np.dot(li, d) * d
        if np.linalg.norm(li) > 1e-8:
            residuals.append(li)
    if not residuals:
        return np.zeros((len(d), 0))
    R = np.stack(residuals, axis=1)
    Q, _ = np.linalg.qr(R)
    return Q[:, :min(len(residuals), Q.shape[1])]


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    print("=" * 70)
    print("EXPERIMENT 4 EXTENSION: CROSS-RELATION RESIDUAL TRANSFER")
    print(f"Optimal scale: {OPTIMAL_SCALE}×")
    print("=" * 70)

    # Load model
    print("\nLoading GPT-2 XL...")
    model = AutoModelForCausalLM.from_pretrained("gpt2-xl").cuda().eval()
    tok = AutoTokenizer.from_pretrained("gpt2-xl")
    tok.pad_token = tok.eos_token
    hparams = ROMEHyperParams.from_json(HPARAMS_DIR / "ROME" / "gpt2-xl.json")

    with open(EXP1_DIR / "context_templates.json") as f:
        rome_main_module.CONTEXT_TEMPLATES_CACHE = json.load(f)

    # Load artifacts
    exp2_data = torch.load(EXP2_DIR / "edit_vectors.pt", map_location="cpu")
    exp2_meta = exp2_data["metadata"]
    exp2_V = exp2_data["v"].numpy()
    exp2_U = exp2_data["u"].numpy()

    cd_all = torch.load(EXP1_DIR / "concept_directions_layer17.pt", map_location="cpu")
    lda_dirs = np.load(EXP2_DIR / "lda_directions.npy")[:, :N_LDA_DIRS].copy()

    with open(EXP1_DIR / "relation_targets.json") as f:
        relation_targets = json.load(f)

    # ── Compute dark residual per relation ──
    print("\n[1] Computing dark residuals per relation...")

    residuals = {}  # relation -> residual direction (unit norm)
    vmeans = {}     # relation -> v_mean
    vmean_norms = {}

    for rid in RELATIONS:
        # Use first target concept for this relation's concept direction
        target = relation_targets[rid][0]
        cd_key = f"{rid}_{target}_logistic"
        if cd_key not in cd_all:
            print(f"  WARN: no concept dir for {cd_key}")
            continue
        concept_dir = cd_all[cd_key].numpy().astype(np.float64)

        # Get all edit vectors for this relation
        idxs = [i for i, m in enumerate(exp2_meta) if m["relation_id"] == rid]
        V = exp2_V[idxs]
        v_mean = V.mean(axis=0)
        vm_norm = float(np.linalg.norm(v_mean))

        # Decompose
        lda_orth = orthogonalize_lda_against_concept(lda_dirs, concept_dir)
        v_concept = np.dot(v_mean, concept_dir) * concept_dir
        v_lda = lda_orth @ (lda_orth.T @ v_mean)
        v_residual = v_mean - v_concept - v_lda

        residuals[rid] = v_residual / (np.linalg.norm(v_residual) + 1e-10)
        vmeans[rid] = v_mean
        vmean_norms[rid] = vm_norm

        print(f"  {rid}: ||v_mean||={vm_norm:.3f}  ||v_res||={np.linalg.norm(v_residual):.3f}  "
              f"energy_res={np.linalg.norm(v_residual)**2 / (vm_norm**2 + 1e-10):.3f}")

    # Pairwise cosine between residuals
    print("\n  Pairwise cosine between dark residuals:")
    print(f"  {'':8s}", "".join(f"{r:>8s}" for r in RELATIONS))
    for r1 in RELATIONS:
        row = f"  {r1:8s}"
        for r2 in RELATIONS:
            if r1 in residuals and r2 in residuals:
                c = cosine(residuals[r1], residuals[r2])
                row += f"{c:8.3f}"
            else:
                row += f"{'N/A':>8s}"
        print(row)

    # Average residual across all relations
    all_res = np.stack([residuals[r] for r in RELATIONS if r in residuals])
    avg_residual = all_res.mean(axis=0)
    avg_residual = avg_residual / (np.linalg.norm(avg_residual) + 1e-10)
    print(f"\n  Average residual cos with each relation's residual:")
    for rid in RELATIONS:
        if rid in residuals:
            print(f"    {rid}: {cosine(avg_residual, residuals[rid]):.4f}")

    # ── Run transfer tests ──
    print("\n" + "=" * 70)
    print("[2] TRANSFER TESTS")
    print("=" * 70)

    rng = np.random.RandomState(SEED)
    rand_dir = rng.randn(1600).astype(np.float64)
    rand_dir = rand_dir / np.linalg.norm(rand_dir)

    all_results = []

    for rid in RELATIONS:
        if rid not in residuals:
            continue

        # Get test entities for this relation (first target concept)
        target = relation_targets[rid][0]
        idxs = [i for i, m in enumerate(exp2_meta)
                if m["relation_id"] == rid and m["target_value"] == target]

        if len(idxs) < N_TEST_ENTITIES:
            continue

        test_idxs = rng.choice(len(idxs), N_TEST_ENTITIES, replace=False)
        vm_norm = vmean_norms[rid]

        print(f"\n  {rid} ({target}, {N_TEST_ENTITIES} entities):")

        for eidx in test_idxs:
            meta = exp2_meta[idxs[eidx]]
            u_entity = exp2_U[idxs[eidx]]
            prompt_text = meta["prompt"].replace("{}", meta["subject"])

            entity_res = {
                "relation": rid,
                "target": target,
                "subject": meta["subject"],
            }

            # 1. Own v_mean
            v_cond = rescale(vmeans[rid], vm_norm * OPTIMAL_SCALE)
            res = apply_edit_and_eval(model, tok, u_entity, v_cond, hparams, prompt_text, target)
            entity_res["own_vmean"] = res

            # 2. Own residual
            v_cond = rescale(residuals[rid], vm_norm * OPTIMAL_SCALE)
            res = apply_edit_and_eval(model, tok, u_entity, v_cond, hparams, prompt_text, target)
            entity_res["own_residual"] = res

            # 3. Each other relation's residual
            for donor_rid in RELATIONS:
                if donor_rid == rid or donor_rid not in residuals:
                    continue
                v_cond = rescale(residuals[donor_rid], vm_norm * OPTIMAL_SCALE)
                res = apply_edit_and_eval(model, tok, u_entity, v_cond, hparams, prompt_text, target)
                entity_res[f"residual_from_{donor_rid}"] = res

            # 4. Average residual
            v_cond = rescale(avg_residual, vm_norm * OPTIMAL_SCALE)
            res = apply_edit_and_eval(model, tok, u_entity, v_cond, hparams, prompt_text, target)
            entity_res["avg_residual"] = res

            # 5. Random
            v_cond = rescale(rand_dir, vm_norm * OPTIMAL_SCALE)
            res = apply_edit_and_eval(model, tok, u_entity, v_cond, hparams, prompt_text, target)
            entity_res["random"] = res

            all_results.append(entity_res)

    # ── Aggregate ──
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    # Collect by condition type
    own_vmean = [r["own_vmean"] for r in all_results]
    own_res = [r["own_residual"] for r in all_results]
    avg_res = [r["avg_residual"] for r in all_results]
    rand_res = [r["random"] for r in all_results]

    # Cross-relation: group by whether donor is same or different
    cross_results = defaultdict(list)
    for r in all_results:
        src_rid = r["relation"]
        for key, val in r.items():
            if key.startswith("residual_from_"):
                donor = key.replace("residual_from_", "")
                cross_results["cross_all"].append(val)
                cross_results[f"cross_from_{donor}"].append(val)

    print(f"\n  {'Condition':35s} {'Eff':>6s} {'Prob':>8s} {'N':>4s}")
    print("  " + "-" * 57)

    summary = {}
    for name, results_list in [
        ("Own v_mean (3×)", own_vmean),
        ("Own dark residual (3×)", own_res),
        ("Avg dark residual (3×)", avg_res),
        ("Cross-relation residual (3×)", cross_results["cross_all"]),
        ("Random (3×)", rand_res),
    ]:
        effs = [r["efficacy"] for r in results_list]
        probs = [r["target_prob"] for r in results_list]
        e, p = np.mean(effs), np.mean(probs)
        print(f"  {name:35s} {e:6.3f} {p:8.4f} {len(effs):4d}")
        summary[name] = {"efficacy": float(e), "target_prob": float(p), "n": len(effs)}

    # Per-donor breakdown
    print(f"\n  Cross-relation breakdown by donor:")
    print(f"  {'Donor':10s} {'Eff':>6s} {'Prob':>8s} {'N':>4s}")
    print("  " + "-" * 32)
    donor_summary = {}
    for donor_rid in RELATIONS:
        key = f"cross_from_{donor_rid}"
        if key in cross_results:
            effs = [r["efficacy"] for r in cross_results[key]]
            probs = [r["target_prob"] for r in cross_results[key]]
            e, p = np.mean(effs), np.mean(probs)
            print(f"  {donor_rid:10s} {e:6.3f} {p:8.4f} {len(effs):4d}")
            donor_summary[donor_rid] = {"efficacy": float(e), "target_prob": float(p), "n": len(effs)}

    # Per-relation breakdown (as recipient)
    print(f"\n  Per-relation breakdown (as recipient):")
    print(f"  {'Relation':10s} {'Own res':>8s} {'Cross':>8s} {'Avg res':>8s} {'v_mean':>8s}")
    print("  " + "-" * 48)
    for rid in RELATIONS:
        rid_results = [r for r in all_results if r["relation"] == rid]
        if not rid_results:
            continue
        own_e = np.mean([r["own_residual"]["efficacy"] for r in rid_results])
        cross_e_list = []
        for r in rid_results:
            for k, v in r.items():
                if k.startswith("residual_from_"):
                    cross_e_list.append(v["efficacy"])
        cross_e = np.mean(cross_e_list) if cross_e_list else 0
        avg_e = np.mean([r["avg_residual"]["efficacy"] for r in rid_results])
        vm_e = np.mean([r["own_vmean"]["efficacy"] for r in rid_results])
        print(f"  {rid:10s} {own_e:8.2f} {cross_e:8.2f} {avg_e:8.2f} {vm_e:8.2f}")

    # Save
    output = {
        "config": {
            "relations": RELATIONS,
            "optimal_scale": OPTIMAL_SCALE,
            "n_test_entities": N_TEST_ENTITIES,
        },
        "pairwise_residual_cosines": {
            f"{r1}_vs_{r2}": cosine(residuals[r1], residuals[r2])
            for r1 in RELATIONS for r2 in RELATIONS
            if r1 in residuals and r2 in residuals and r1 < r2
        },
        "summary": summary,
        "donor_breakdown": donor_summary,
        "per_edit": all_results,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()