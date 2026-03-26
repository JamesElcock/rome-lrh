"""
Experiment 8b: ROME–MEND alignment at layer 47.

Tests whether ROME's propagated perturbation at L47 aligns with MEND's
direct edit vectors at L47, for matching relation/target concepts.

Three analyses:
1. v_mean alignment: Compare ROME's v_mean (L17) with MEND's v_mean (L47)
   per concept — do they point in the same direction despite being in
   different layers?

2. Propagated alignment: Run ROME edits, measure the perturbation at L47
   using TraceDict (post-edit minus pre-edit activations), compare with
   MEND's v directions at L47.

3. Subspace overlap: Do ROME and MEND v vectors span similar subspaces
   within each concept? (Grassmann distance, principal angles)

Saves to results/exp8_mend/rome_mend_alignment.json
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

# ============================================================
RELATIONS = ["P176", "P1412", "P37", "P27", "P413"]
SEED = 42
N_PROBE_PROMPTS = 10  # prompts per concept for propagation measurement
BATCH_SIZE = 8

EXP1_DIR = Path("results/exp1")
EXP2_DIR = Path("results/exp2")
MEND_DIR = Path("results/exp8_mend")
OUTPUT_FILE = MEND_DIR / "rome_mend_alignment.json"


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def subspace_overlap(A, B, k=5):
    """Principal angles between column spaces of A and B.
    Returns mean cosine of principal angles (1 = identical subspaces)."""
    if A.shape[1] == 0 or B.shape[1] == 0:
        return 0.0
    Qa, _ = np.linalg.qr(A[:, :k])
    Qb, _ = np.linalg.qr(B[:, :k])
    M = Qa.T @ Qb
    svals = np.linalg.svd(M, compute_uv=False)
    # Clamp to [0,1] for numerical safety
    svals = np.clip(svals, 0, 1)
    return float(np.mean(svals))


def grassmann_distance(A, B, k=5):
    """Grassmann distance between k-dim subspaces."""
    if A.shape[1] == 0 or B.shape[1] == 0:
        return np.pi / 2
    Qa, _ = np.linalg.qr(A[:, :k])
    Qb, _ = np.linalg.qr(B[:, :k])
    M = Qa.T @ Qb
    svals = np.linalg.svd(M, compute_uv=False)
    svals = np.clip(svals, 0, 1)
    angles = np.arccos(svals)
    return float(np.sqrt(np.sum(angles ** 2)))


def main():
    print("=" * 70)
    print("EXPERIMENT 8b: ROME–MEND ALIGNMENT AT LAYER 47")
    print("=" * 70)

    # Load model
    print("\nLoading GPT-2 XL...")
    model = AutoModelForCausalLM.from_pretrained("gpt2-xl").cuda().eval()
    tok = AutoTokenizer.from_pretrained("gpt2-xl")
    tok.pad_token = tok.eos_token
    hparams = ROMEHyperParams.from_json(HPARAMS_DIR / "ROME" / "gpt2-xl.json")

    # Load ROME edit vectors (from Exp 2)
    print("Loading ROME edit vectors...")
    rome_data = torch.load(EXP2_DIR / "edit_vectors.pt", map_location="cpu")
    rome_meta = rome_data["metadata"]
    rome_V = rome_data["v"].numpy()  # (500, 1600) — v in residual stream at L17
    rome_U = rome_data["u"].numpy()  # (500, 6400) — u in MLP space at L17

    # Load MEND edit vectors
    print("Loading MEND edit vectors...")
    mend_data = torch.load(MEND_DIR / "edit_vectors.pt", map_location="cpu")
    mend_meta = mend_data["metadata"]
    mend_V = {k: v.numpy() for k, v in mend_data["v_by_layer"].items()}  # layer -> (200, 1600)

    # Load relation targets
    with open(EXP1_DIR / "relation_targets.json") as f:
        relation_targets = json.load(f)

    # ================================================================
    # ANALYSIS 1: v_mean alignment per concept
    # ================================================================
    print("\n" + "=" * 70)
    print("[1] v_mean ALIGNMENT (ROME L17 vs MEND L47)")
    print("=" * 70)

    results_vmean = []

    for rid in RELATIONS:
        targets = relation_targets.get(rid, [])
        for target in targets[:2]:  # First 2 targets per relation
            # ROME v vectors for this concept
            rome_idxs = [i for i, m in enumerate(rome_meta)
                         if m["relation_id"] == rid and m["target_value"] == target]
            # MEND v vectors for this concept
            mend_idxs = [i for i, m in enumerate(mend_meta)
                         if m["relation_id"] == rid and m["target_value"] == target]

            if len(rome_idxs) < 3 or len(mend_idxs) < 3:
                continue

            rome_v_mean = rome_V[rome_idxs].mean(axis=0)
            mend_v_mean_47 = mend_V[47][mend_idxs].mean(axis=0)
            mend_v_mean_46 = mend_V[46][mend_idxs].mean(axis=0)
            mend_v_mean_45 = mend_V[45][mend_idxs].mean(axis=0)

            cos_17_47 = cosine(rome_v_mean, mend_v_mean_47)
            cos_17_46 = cosine(rome_v_mean, mend_v_mean_46)
            cos_17_45 = cosine(rome_v_mean, mend_v_mean_45)

            res = {
                "concept": f"{rid}_{target}",
                "n_rome": len(rome_idxs),
                "n_mend": len(mend_idxs),
                "cos_rome_L17_vs_mend_L47": cos_17_47,
                "cos_rome_L17_vs_mend_L46": cos_17_46,
                "cos_rome_L17_vs_mend_L45": cos_17_45,
                "rome_vmean_norm": float(np.linalg.norm(rome_v_mean)),
                "mend_vmean_L47_norm": float(np.linalg.norm(mend_v_mean_47)),
            }
            results_vmean.append(res)

            print(f"\n  {rid}_{target} (ROME:{len(rome_idxs)}, MEND:{len(mend_idxs)}):")
            print(f"    cos(ROME_L17, MEND_L47) = {cos_17_47:.4f}")
            print(f"    cos(ROME_L17, MEND_L46) = {cos_17_46:.4f}")
            print(f"    cos(ROME_L17, MEND_L45) = {cos_17_45:.4f}")
            print(f"    ||ROME v_mean|| = {res['rome_vmean_norm']:.3f}  "
                  f"||MEND v_mean@47|| = {res['mend_vmean_L47_norm']:.3f}")

    if results_vmean:
        print(f"\n  AGGREGATE:")
        for key in ["cos_rome_L17_vs_mend_L47", "cos_rome_L17_vs_mend_L46", "cos_rome_L17_vs_mend_L45"]:
            vals = [r[key] for r in results_vmean]
            print(f"    {key}: mean={np.mean(vals):.4f}  "
                  f"std={np.std(vals):.4f}  |mean|={np.mean(np.abs(vals)):.4f}")

    # ================================================================
    # ANALYSIS 2: Propagated ROME perturbation at L47 vs MEND v at L47
    # ================================================================
    print("\n" + "=" * 70)
    print("[2] PROPAGATED ROME PERTURBATION vs MEND (at L47)")
    print("=" * 70)
    print("    For each concept, apply ROME edit at L17, measure Δh at L45-47,")
    print("    compare with MEND's v directions at the same layers.")

    import rome.rome_main as rome_main_module
    with open(EXP1_DIR / "context_templates.json") as f:
        rome_main_module.CONTEXT_TEMPLATES_CACHE = json.load(f)

    results_prop = []
    rng = np.random.RandomState(SEED)

    for rid in RELATIONS:
        targets = relation_targets.get(rid, [])
        for target in targets[:2]:
            rome_idxs = [i for i, m in enumerate(rome_meta)
                         if m["relation_id"] == rid and m["target_value"] == target]
            mend_idxs = [i for i, m in enumerate(mend_meta)
                         if m["relation_id"] == rid and m["target_value"] == target]

            if len(rome_idxs) < 3 or len(mend_idxs) < 3:
                continue

            # MEND v_mean at each layer for this concept
            mend_v_mean = {l: mend_V[l][mend_idxs].mean(axis=0) for l in [45, 46, 47]}

            # Select a few ROME entities to test propagation
            test_idxs = rng.choice(len(rome_idxs), min(N_PROBE_PROMPTS, len(rome_idxs)), replace=False)

            concept_deltas = {45: [], 46: [], 47: []}

            for eidx in test_idxs:
                idx = rome_idxs[eidx]
                meta = rome_meta[idx]
                u_vec = rome_U[idx]
                v_vec = rome_V[idx]
                prompt_text = meta["prompt"].replace("{}", meta["subject"])

                # Layers to trace
                trace_layers = [f"transformer.h.{l}" for l in [45, 46, 47]]

                # Pre-edit activations
                inputs = tok(prompt_text, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    with nethook.TraceDict(model, trace_layers) as tr_pre:
                        model(**inputs)
                last_pos = inputs["attention_mask"][0].sum().item() - 1
                h_pre = {l: tr_pre[f"transformer.h.{l}"].output[0][0, last_pos].detach().cpu().numpy()
                         for l in [45, 46, 47]}

                # Apply ROME edit
                layer = hparams.layers[0]
                weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
                w = nethook.get_parameter(model, weight_name)
                u_t = torch.tensor(u_vec, dtype=torch.float32).to(w.device)
                v_t = torch.tensor(v_vec, dtype=torch.float32).to(w.device)
                upd = u_t.unsqueeze(1) @ v_t.unsqueeze(0)
                if upd.shape != w.shape:
                    upd = upd.T
                with torch.no_grad():
                    w[...] += upd

                # Post-edit activations
                with torch.no_grad():
                    with nethook.TraceDict(model, trace_layers) as tr_post:
                        model(**inputs)
                h_post = {l: tr_post[f"transformer.h.{l}"].output[0][0, last_pos].detach().cpu().numpy()
                          for l in [45, 46, 47]}

                # Restore weights
                with torch.no_grad():
                    w[...] -= upd

                # Compute delta
                for l in [45, 46, 47]:
                    delta = h_post[l] - h_pre[l]
                    concept_deltas[l].append(delta)

            # Average propagated delta per layer
            mean_delta = {l: np.mean(concept_deltas[l], axis=0) for l in [45, 46, 47]}

            # Compare with MEND
            res = {"concept": f"{rid}_{target}", "n_rome_probes": len(test_idxs), "n_mend": len(mend_idxs)}

            for l in [45, 46, 47]:
                delta_norm = float(np.linalg.norm(mean_delta[l]))
                mend_norm = float(np.linalg.norm(mend_v_mean[l]))
                cos_val = cosine(mean_delta[l], mend_v_mean[l])

                res[f"delta_norm_L{l}"] = delta_norm
                res[f"mend_vmean_norm_L{l}"] = mend_norm
                res[f"cos_rome_delta_vs_mend_L{l}"] = cos_val

            results_prop.append(res)

            print(f"\n  {rid}_{target}:")
            for l in [45, 46, 47]:
                print(f"    L{l}: ||Δh||={res[f'delta_norm_L{l}']:.3f}  "
                      f"||MEND v||={res[f'mend_vmean_norm_L{l}']:.3f}  "
                      f"cos={res[f'cos_rome_delta_vs_mend_L{l}']:.4f}")

    if results_prop:
        print(f"\n  AGGREGATE:")
        for l in [45, 46, 47]:
            key = f"cos_rome_delta_vs_mend_L{l}"
            vals = [r[key] for r in results_prop]
            norms = [r[f"delta_norm_L{l}"] for r in results_prop]
            print(f"    L{l}: cos mean={np.mean(vals):.4f} std={np.std(vals):.4f} "
                  f"|cos| mean={np.mean(np.abs(vals)):.4f}  "
                  f"||Δh|| mean={np.mean(norms):.2f}")

    # ================================================================
    # ANALYSIS 3: Subspace overlap (ROME vs MEND per concept)
    # ================================================================
    print("\n" + "=" * 70)
    print("[3] SUBSPACE OVERLAP (ROME vs MEND per concept at L47)")
    print("=" * 70)

    results_subspace = []

    for rid in RELATIONS:
        targets = relation_targets.get(rid, [])
        for target in targets[:2]:
            rome_idxs = [i for i, m in enumerate(rome_meta)
                         if m["relation_id"] == rid and m["target_value"] == target]
            mend_idxs = [i for i, m in enumerate(mend_meta)
                         if m["relation_id"] == rid and m["target_value"] == target]

            if len(rome_idxs) < 3 or len(mend_idxs) < 3:
                continue

            # ROME v vectors as columns
            rome_V_concept = rome_V[rome_idxs].T  # (1600, n_rome)
            # MEND v vectors at L47 as columns
            mend_V_concept = mend_V[47][mend_idxs].T  # (1600, n_mend)

            k = min(5, rome_V_concept.shape[1], mend_V_concept.shape[1])
            overlap = subspace_overlap(rome_V_concept, mend_V_concept, k=k)
            grass_dist = grassmann_distance(rome_V_concept, mend_V_concept, k=k)

            # Also: pairwise cosines between individual ROME and MEND v vectors
            cos_matrix = np.array([
                [cosine(rome_V[ri], mend_V[47][mi])
                 for mi in mend_idxs]
                for ri in rome_idxs
            ])

            res = {
                "concept": f"{rid}_{target}",
                "n_rome": len(rome_idxs),
                "n_mend": len(mend_idxs),
                "subspace_k": k,
                "subspace_overlap": overlap,
                "grassmann_distance": grass_dist,
                "pairwise_cos_mean": float(np.mean(cos_matrix)),
                "pairwise_cos_abs_mean": float(np.mean(np.abs(cos_matrix))),
                "pairwise_cos_std": float(np.std(cos_matrix)),
                "pairwise_cos_max_abs": float(np.max(np.abs(cos_matrix))),
            }
            results_subspace.append(res)

            print(f"\n  {rid}_{target}: overlap={overlap:.4f}  "
                  f"Grassmann={grass_dist:.4f}  "
                  f"pairwise |cos| mean={res['pairwise_cos_abs_mean']:.4f}")

    if results_subspace:
        print(f"\n  AGGREGATE:")
        for key in ["subspace_overlap", "grassmann_distance",
                     "pairwise_cos_abs_mean", "pairwise_cos_max_abs"]:
            vals = [r[key] for r in results_subspace]
            print(f"    {key}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}")

    # ================================================================
    # ANALYSIS 4: Cross-method within-concept vs between-concept
    # ================================================================
    print("\n" + "=" * 70)
    print("[4] WITHIN vs BETWEEN CONCEPT (ROME×MEND cross-cosines)")
    print("=" * 70)

    within_cos = []
    between_cos = []

    concepts_data = {}
    for rid in RELATIONS:
        targets = relation_targets.get(rid, [])
        for target in targets[:2]:
            rome_idxs = [i for i, m in enumerate(rome_meta)
                         if m["relation_id"] == rid and m["target_value"] == target]
            mend_idxs = [i for i, m in enumerate(mend_meta)
                         if m["relation_id"] == rid and m["target_value"] == target]
            if len(rome_idxs) >= 3 and len(mend_idxs) >= 3:
                concepts_data[f"{rid}_{target}"] = (rome_idxs, mend_idxs)

    concept_keys = list(concepts_data.keys())

    for i, key_i in enumerate(concept_keys):
        rome_i, mend_i = concepts_data[key_i]
        # Within-concept: ROME from concept i vs MEND from concept i
        for ri in rome_i:
            for mi in mend_i:
                within_cos.append(cosine(rome_V[ri], mend_V[47][mi]))

        # Between-concept: ROME from concept i vs MEND from other concepts
        for j, key_j in enumerate(concept_keys):
            if i == j:
                continue
            _, mend_j = concepts_data[key_j]
            for ri in rome_i:
                for mj in mend_j:
                    between_cos.append(cosine(rome_V[ri], mend_V[47][mj]))

    within_cos = np.array(within_cos)
    between_cos = np.array(between_cos)

    print(f"  Within-concept  (ROME×MEND): mean={np.mean(within_cos):.4f}  "
          f"|mean|={np.mean(np.abs(within_cos)):.4f}  n={len(within_cos)}")
    print(f"  Between-concept (ROME×MEND): mean={np.mean(between_cos):.4f}  "
          f"|mean|={np.mean(np.abs(between_cos)):.4f}  n={len(between_cos)}")

    # d-prime
    dprime = float((np.mean(np.abs(within_cos)) - np.mean(np.abs(between_cos))) /
                   np.sqrt(0.5 * (np.std(within_cos) ** 2 + np.std(between_cos) ** 2) + 1e-10))
    print(f"  d' = {dprime:.4f}")

    # ================================================================
    # Save
    # ================================================================
    output = {
        "config": {
            "relations": RELATIONS,
            "n_probe_prompts": N_PROBE_PROMPTS,
        },
        "vmean_alignment": results_vmean,
        "propagated_alignment": results_prop,
        "subspace_overlap": results_subspace,
        "cross_method": {
            "within_cos_mean": float(np.mean(within_cos)),
            "within_cos_abs_mean": float(np.mean(np.abs(within_cos))),
            "between_cos_mean": float(np.mean(between_cos)),
            "between_cos_abs_mean": float(np.mean(np.abs(between_cos))),
            "dprime": dprime,
            "n_within": len(within_cos),
            "n_between": len(between_cos),
        },
        "aggregate": {
            "vmean": {
                key: {
                    "mean": float(np.mean([r[key] for r in results_vmean])),
                    "std": float(np.std([r[key] for r in results_vmean])),
                }
                for key in ["cos_rome_L17_vs_mend_L47", "cos_rome_L17_vs_mend_L46"]
            } if results_vmean else {},
            "propagated": {
                f"cos_rome_delta_vs_mend_L{l}": {
                    "mean": float(np.mean([r[f"cos_rome_delta_vs_mend_L{l}"] for r in results_prop])),
                    "std": float(np.std([r[f"cos_rome_delta_vs_mend_L{l}"] for r in results_prop])),
                }
                for l in [45, 46, 47]
            } if results_prop else {},
        },
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()