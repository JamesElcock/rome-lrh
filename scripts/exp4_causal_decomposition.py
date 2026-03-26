"""
Experiment 4: Causal Decomposition of the Shared Edit Vector

Decomposes v_mean into three components — concept projection, LDA projection,
and residual — then tests each as a standalone edit to determine which
geometric structure carries the causal signal.

Components:
- v_concept: projection onto the logistic probe concept direction (1D)
- v_lda: projection onto the 9D LDA discriminant subspace (orthogonalized against concept)
- v_residual: everything else (v_mean - v_concept - v_lda)

Key question: does the edit mechanism live in concept-space, ROME's LDA-space,
or a "dark" residual subspace not captured by either?

Updates from Experiments 1-3:
- Concept alignment is weak (|cos| ≈ 0.11) and doesn't predict success (Exp 3)
- LDA captures ~22% of v variance (Exp 2 PERMANOVA)
- Perturbation amplifies 4-7x but stays orthogonal to concept-space (Exp 3)
- Exp 1B v_mean is normalized (norm=1) — must recompute from Exp 2 raw vectors

Saves results to results/exp4/
"""

import json
import os
import sys
import time
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
# Configuration
# ============================================================
RELATIONS = ["P176", "P1412", "P37", "P27", "P413"]
EDIT_LAYER = 17
N_TEST_ENTITIES = 5   # per concept
N_LDA_DIRS = 9        # 10-class LDA → 9 discriminant directions
SEED = 42

RESULTS_DIR = Path("results/exp4")
EXP1_DIR = Path("results/exp1")
EXP2_DIR = Path("results/exp2")


# ============================================================
# Utilities
# ============================================================
def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def rescale(v, target_norm):
    """Rescale vector to target norm, preserving direction."""
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-10:
        return v
    return v * (target_norm / n)


def apply_edit_and_eval(model, tok, u, v, hparams, prompt_text, target_str):
    """
    Apply rank-1 edit u ⊗ v, evaluate on single prompt, restore weights.
    Returns dict with efficacy and target_prob.
    """
    layer = hparams.layers[0]
    weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
    w = nethook.get_parameter(model, weight_name)

    u_t = torch.tensor(u, dtype=torch.float32).to(w.device)
    v_t = torch.tensor(v, dtype=torch.float32).to(w.device)
    upd = u_t.unsqueeze(1) @ v_t.unsqueeze(0)
    if upd.shape != w.shape:
        upd = upd.T

    # Apply
    with torch.no_grad():
        w[...] += upd

    # Evaluate
    target_tok = tok(f" {target_str.strip()}", return_tensors="pt")["input_ids"][0]
    first_target_tok = target_tok[0].item()

    inputs = tok(prompt_text, return_tensors="pt").to(w.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
    probs = torch.softmax(logits.float(), dim=0)
    pred_tok = logits.argmax().item()
    target_prob = probs[first_target_tok].item()

    # Restore
    with torch.no_grad():
        w[...] -= upd

    return {
        "efficacy": int(pred_tok == first_target_tok),
        "target_prob": float(target_prob),
        "pred_token": tok.decode([pred_tok]),
    }


def orthogonalize_lda_against_concept(lda_dirs, concept_dir):
    """
    Remove concept_dir component from each LDA direction, then QR for stability.
    Returns orthonormal basis (1600, k) orthogonal to concept_dir.
    """
    d = concept_dir / (np.linalg.norm(concept_dir) + 1e-10)
    residuals = []
    for i in range(lda_dirs.shape[1]):
        li = lda_dirs[:, i].copy()
        li -= np.dot(li, d) * d  # remove concept component
        norm = np.linalg.norm(li)
        if norm > 1e-8:
            residuals.append(li)

    if not residuals:
        return np.zeros((len(d), 0))

    R = np.stack(residuals, axis=1)  # (1600, k)
    Q, _ = np.linalg.qr(R)
    # Keep only columns with significant norm (numerical cleanup)
    k = min(len(residuals), Q.shape[1])
    return Q[:, :k]


# ============================================================
# Main
# ============================================================
def main():
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT 4: CAUSAL DECOMPOSITION")
    print("=" * 70)

    # ── Load model ────────────────────────────────────────────
    print("\n[1] Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained("gpt2-xl").cuda().eval()
    tok = AutoTokenizer.from_pretrained("gpt2-xl")
    tok.pad_token = tok.eos_token
    hparams = ROMEHyperParams.from_json(HPARAMS_DIR / "ROME" / "gpt2-xl.json")

    with open(EXP1_DIR / "context_templates.json") as f:
        templates = json.load(f)
    rome_main_module.CONTEXT_TEMPLATES_CACHE = templates

    # ── Load artifacts ────────────────────────────────────────
    print("\n[2] Loading artifacts...")

    # Edit vectors from Exp 2
    exp2_data = torch.load(EXP2_DIR / "edit_vectors.pt", map_location="cpu")
    exp2_meta = exp2_data["metadata"]
    exp2_V = exp2_data["v"].numpy()  # (500, 1600)
    exp2_U = exp2_data["u"].numpy()  # (500, 6400)

    # Concept directions at layer 17
    cd_all = torch.load(EXP1_DIR / "concept_directions_layer17.pt", map_location="cpu")

    # LDA directions (columns 0-8 are the 9 discriminant directions)
    lda_all = np.load(EXP2_DIR / "lda_directions.npy")
    lda_dirs = lda_all[:, :N_LDA_DIRS].copy()  # (1600, 9)

    # Relation targets
    with open(EXP1_DIR / "relation_targets.json") as f:
        relation_targets = json.load(f)

    # ── Define tasks ──────────────────────────────────────────
    tasks = []
    for rid in RELATIONS:
        for target in relation_targets[rid][:2]:
            tasks.append({"relation_id": rid, "target": target})
    print(f"   {len(tasks)} concepts")

    # ── Compute v_mean per concept from Exp 2 ────────────────
    print("\n[3] Computing v_mean per concept (from Exp 2 un-normalized vectors)...")

    concept_data = {}
    for task in tasks:
        rid = task["relation_id"]
        target = task["target"]
        key = f"{rid}_{target}"

        # Find matching edit vectors
        idxs = [i for i, m in enumerate(exp2_meta)
                if m["relation_id"] == rid and m["target_value"] == target]

        V = exp2_V[idxs]  # (n, 1600)
        U = exp2_U[idxs]  # (n, 6400)
        v_mean = V.mean(axis=0)

        concept_data[key] = {
            "relation_id": rid,
            "target": target,
            "v_mean": v_mean,
            "v_mean_norm": float(np.linalg.norm(v_mean)),
            "v_individual": V,
            "u_individual": U,
            "n_entities": len(idxs),
            "meta": [exp2_meta[i] for i in idxs],
        }

        # Wrong concept: other target from same relation
        other_targets = [t for r, t in [(tk["relation_id"], tk["target"]) for tk in tasks]
                        if r == rid and t != target]
        if other_targets:
            other_key = f"{rid}_{other_targets[0]}"
            other_idxs = [i for i, m in enumerate(exp2_meta)
                         if m["relation_id"] == rid and m["target_value"] == other_targets[0]]
            if other_idxs:
                concept_data[key]["v_mean_wrong"] = exp2_V[other_idxs].mean(axis=0)

        print(f"   {key}: n={len(idxs)}, ||v_mean||={np.linalg.norm(v_mean):.4f}, "
              f"mean ||v_i||={np.linalg.norm(V, axis=1).mean():.4f}")

    # ── Decompose v_mean for each concept ─────────────────────
    print("\n[4] Decomposing v_mean = v_concept + v_lda + v_residual...")

    decompositions = {}
    for key, cd_entry in concept_data.items():
        rid = cd_entry["relation_id"]
        target = cd_entry["target"]
        v_mean = cd_entry["v_mean"]
        vm_norm = np.linalg.norm(v_mean)

        # Concept direction (logistic probe at layer 17)
        cd_key = f"{rid}_{target}_logistic"
        if cd_key not in cd_all:
            print(f"   WARN: no concept direction for {cd_key}")
            continue
        concept_dir = cd_all[cd_key].numpy().astype(np.float64)

        # Orthogonalize LDA against concept direction
        lda_orth = orthogonalize_lda_against_concept(lda_dirs, concept_dir)

        # Decompose
        v_concept = np.dot(v_mean, concept_dir) * concept_dir
        v_lda = lda_orth @ (lda_orth.T @ v_mean)
        v_residual = v_mean - v_concept - v_lda

        # Verify reconstruction
        recon = v_concept + v_lda + v_residual
        recon_err = np.linalg.norm(recon - v_mean)

        # Energy fractions
        e_concept = np.linalg.norm(v_concept)**2 / (vm_norm**2 + 1e-10)
        e_lda = np.linalg.norm(v_lda)**2 / (vm_norm**2 + 1e-10)
        e_residual = np.linalg.norm(v_residual)**2 / (vm_norm**2 + 1e-10)

        decompositions[key] = {
            "v_concept": v_concept,
            "v_lda": v_lda,
            "v_residual": v_residual,
            "concept_dir": concept_dir,
            "lda_orth": lda_orth,
            "energy_concept": e_concept,
            "energy_lda": e_lda,
            "energy_residual": e_residual,
            "recon_error": float(recon_err),
            "cos_v_mean_concept": cosine(v_mean, concept_dir),
            "norm_v_concept": float(np.linalg.norm(v_concept)),
            "norm_v_lda": float(np.linalg.norm(v_lda)),
            "norm_v_residual": float(np.linalg.norm(v_residual)),
        }

        print(f"   {key}: energy concept={e_concept:.4f} lda={e_lda:.4f} "
              f"residual={e_residual:.4f} (sum={e_concept+e_lda+e_residual:.4f}) "
              f"recon_err={recon_err:.2e}")

    # ── Run causal tests ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("CAUSAL DECOMPOSITION TESTS")
    print("=" * 70)

    rng = np.random.RandomState(SEED)
    all_results = []

    for key, cd_entry in concept_data.items():
        if key not in decompositions:
            continue

        rid = cd_entry["relation_id"]
        target = cd_entry["target"]
        v_mean = cd_entry["v_mean"]
        vm_norm = cd_entry["v_mean_norm"]
        dec = decompositions[key]

        # Select test entities
        n_avail = cd_entry["n_entities"]
        test_idxs = rng.choice(n_avail, min(N_TEST_ENTITIES, n_avail), replace=False)

        print(f"\n   {key} (||v_mean||={vm_norm:.3f}):")

        # Build condition vectors
        conditions = {}

        # 1. Entity's own v (per-entity, added in loop below)
        # 2. Full v_mean
        conditions["full_v_mean"] = v_mean

        # 3-4. Concept component
        conditions["concept_natural"] = dec["v_concept"]
        conditions["concept_rescaled"] = rescale(dec["v_concept"], vm_norm)

        # 5-6. LDA component
        conditions["lda_natural"] = dec["v_lda"]
        conditions["lda_rescaled"] = rescale(dec["v_lda"], vm_norm)

        # 7-8. Residual component
        conditions["residual_natural"] = dec["v_residual"]
        conditions["residual_rescaled"] = rescale(dec["v_residual"], vm_norm)

        # 9. Concept + LDA combined
        conditions["concept_plus_lda"] = dec["v_concept"] + dec["v_lda"]

        # 10. Random direction
        rand_v = rng.randn(1600).astype(np.float64)
        conditions["random"] = rescale(rand_v, vm_norm)

        # 11. Wrong concept
        if "v_mean_wrong" in cd_entry:
            conditions["wrong_concept"] = rescale(cd_entry["v_mean_wrong"], vm_norm)

        # Run conditions for each test entity
        concept_results = []
        for ti, eidx in enumerate(test_idxs):
            u_entity = cd_entry["u_individual"][eidx]
            v_entity = cd_entry["v_individual"][eidx]
            meta = cd_entry["meta"][eidx]
            prompt_text = meta["prompt"].replace("{}", meta["subject"])

            entity_results = {
                "subject": meta["subject"],
                "case_id": meta["case_id"],
            }

            # Own v first
            res = apply_edit_and_eval(model, tok, u_entity, v_entity,
                                     hparams, prompt_text, target)
            entity_results["own_v"] = res

            # All shared conditions
            for cond_name, v_cond in conditions.items():
                res = apply_edit_and_eval(model, tok, u_entity, v_cond,
                                         hparams, prompt_text, target)
                entity_results[cond_name] = res

            concept_results.append(entity_results)

            if ti == 0:
                own_eff = entity_results["own_v"]["efficacy"]
                vm_eff = entity_results["full_v_mean"]["efficacy"]
                res_eff = entity_results["residual_natural"]["efficacy"]
                print(f"     entity 1 ({meta['subject'][:25]}): "
                      f"own={own_eff} v_mean={vm_eff} residual={res_eff}")

        all_results.append({
            "concept": key,
            "relation_id": rid,
            "target": target,
            "n_test": len(test_idxs),
            "v_mean_norm": vm_norm,
            "decomposition": {
                "energy_concept": dec["energy_concept"],
                "energy_lda": dec["energy_lda"],
                "energy_residual": dec["energy_residual"],
                "cos_v_mean_concept": dec["cos_v_mean_concept"],
                "norm_v_concept": dec["norm_v_concept"],
                "norm_v_lda": dec["norm_v_lda"],
                "norm_v_residual": dec["norm_v_residual"],
                "recon_error": dec["recon_error"],
            },
            "entity_results": concept_results,
        })

    # ── Aggregate and print results ───────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Collect all condition names
    all_conds = ["own_v", "full_v_mean",
                 "concept_natural", "concept_rescaled",
                 "lda_natural", "lda_rescaled",
                 "residual_natural", "residual_rescaled",
                 "concept_plus_lda", "random", "wrong_concept"]

    # Per-concept summary
    print("\n   Mean efficacy by condition:")
    header = f"   {'Concept':<22} " + " ".join(f"{c[:8]:>8}" for c in all_conds)
    print(header)
    print("   " + "-" * len(header))

    grand_totals = defaultdict(list)

    for cr in all_results:
        key = cr["concept"]
        entities = cr["entity_results"]
        vals = []
        for cond in all_conds:
            effs = [e[cond]["efficacy"] for e in entities if cond in e]
            mean_eff = float(np.mean(effs)) if effs else -1
            vals.append(mean_eff)
            if effs:
                grand_totals[cond].extend(effs)

        print(f"   {key:<22} " +
              " ".join(f"{v:8.2f}" if v >= 0 else "     N/A" for v in vals))

    # Grand average
    print(f"   {'GRAND AVERAGE':<22} " +
          " ".join(f"{np.mean(grand_totals[c]):8.3f}" if grand_totals[c] else "     N/A"
                   for c in all_conds))

    # Target probability summary
    print("\n   Mean target probability by condition:")
    print(f"   {'Concept':<22} " + " ".join(f"{c[:8]:>8}" for c in all_conds))
    print("   " + "-" * (22 + len(all_conds) * 9))

    grand_probs = defaultdict(list)
    for cr in all_results:
        key = cr["concept"]
        entities = cr["entity_results"]
        vals = []
        for cond in all_conds:
            probs = [e[cond]["target_prob"] for e in entities if cond in e]
            mean_p = float(np.mean(probs)) if probs else -1
            vals.append(mean_p)
            if probs:
                grand_probs[cond].extend(probs)
        print(f"   {key:<22} " +
              " ".join(f"{v:8.4f}" if v >= 0 else "     N/A" for v in vals))

    print(f"   {'GRAND AVERAGE':<22} " +
          " ".join(f"{np.mean(grand_probs[c]):8.4f}" if grand_probs[c] else "     N/A"
                   for c in all_conds))

    # Energy decomposition summary
    print("\n   Energy decomposition (fraction of ||v_mean||² in each component):")
    print(f"   {'Concept':<22} {'concept':>10} {'lda':>10} {'residual':>10} {'sum':>10}")
    for cr in all_results:
        d = cr["decomposition"]
        s = d["energy_concept"] + d["energy_lda"] + d["energy_residual"]
        print(f"   {cr['concept']:<22} {d['energy_concept']:10.4f} "
              f"{d['energy_lda']:10.4f} {d['energy_residual']:10.4f} {s:10.4f}")

    # Key comparisons
    print("\n   Key comparisons (grand average efficacy):")
    for a, b in [("residual_natural", "lda_natural"),
                 ("residual_natural", "concept_natural"),
                 ("residual_rescaled", "lda_rescaled"),
                 ("residual_rescaled", "concept_rescaled"),
                 ("concept_plus_lda", "residual_natural"),
                 ("full_v_mean", "residual_natural")]:
        ea = np.mean(grand_totals[a]) if grand_totals[a] else 0
        eb = np.mean(grand_totals[b]) if grand_totals[b] else 0
        print(f"   {a} ({ea:.3f}) vs {b} ({eb:.3f}): "
              f"Δ = {ea-eb:+.3f}")

    # ── Save results ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    results = {
        "config": {
            "relations": RELATIONS,
            "edit_layer": EDIT_LAYER,
            "n_test_entities": N_TEST_ENTITIES,
            "n_lda_dirs": N_LDA_DIRS,
            "n_concepts": len(tasks),
            "seed": SEED,
        },
        "per_concept": [{
            "concept": cr["concept"],
            "relation_id": cr["relation_id"],
            "target": cr["target"],
            "n_test": cr["n_test"],
            "v_mean_norm": cr["v_mean_norm"],
            "decomposition": cr["decomposition"],
            "mean_efficacy": {
                cond: float(np.mean([e[cond]["efficacy"] for e in cr["entity_results"]
                                     if cond in e]))
                for cond in all_conds
                if any(cond in e for e in cr["entity_results"])
            },
            "mean_target_prob": {
                cond: float(np.mean([e[cond]["target_prob"] for e in cr["entity_results"]
                                     if cond in e]))
                for cond in all_conds
                if any(cond in e for e in cr["entity_results"])
            },
            "entity_details": cr["entity_results"],
        } for cr in all_results],
        "grand_average": {
            "efficacy": {
                cond: float(np.mean(grand_totals[cond]))
                for cond in all_conds if grand_totals[cond]
            },
            "target_prob": {
                cond: float(np.mean(grand_probs[cond]))
                for cond in all_conds if grand_probs[cond]
            },
        },
    }

    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"   Saved to {RESULTS_DIR}/results.json")

    # ── Final summary ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("EXPERIMENT 4 SUMMARY")
    print("=" * 70)

    ga = results["grand_average"]["efficacy"]
    print(f"\n   Grand average efficacy across {len(all_results)} concepts:")
    for cond in all_conds:
        if cond in ga:
            bar = "█" * int(ga[cond] * 40)
            print(f"   {cond:<22} {ga[cond]:.3f}  {bar}")

    # The key finding
    res_nat = ga.get("residual_natural", 0)
    lda_nat = ga.get("lda_natural", 0)
    con_nat = ga.get("concept_natural", 0)
    res_rsc = ga.get("residual_rescaled", 0)
    lda_rsc = ga.get("lda_rescaled", 0)
    con_rsc = ga.get("concept_rescaled", 0)
    vm = ga.get("full_v_mean", 0)

    print(f"\n   Natural magnitude:  concept={con_nat:.3f}  lda={lda_nat:.3f}  "
          f"residual={res_nat:.3f}  v_mean={vm:.3f}")
    print(f"   Rescaled to ||v_mean||: concept={con_rsc:.3f}  lda={lda_rsc:.3f}  "
          f"residual={res_rsc:.3f}")

    if res_nat > max(lda_nat, con_nat) + 0.1:
        print(f"\n   → Residual dominates: the causal signal is in the 'dark' subspace")
    elif lda_rsc > con_rsc + 0.1:
        print(f"\n   → LDA geometry carries the signal (when magnitude-matched)")
    elif con_rsc > lda_rsc + 0.1:
        print(f"\n   → Concept direction carries the signal (when magnitude-matched)")
    else:
        print(f"\n   → No single component dominates — distributed mechanism")

    print(f"\n   Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
