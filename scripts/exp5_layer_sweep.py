"""
Experiment 5: Layer Sweep — Does Edit Layer Determine Concept Alignment?

Tests whether editing at different layers (not just default layer 17) produces
different concept alignment and efficacy trade-offs.

Research question: Is there a layer where both edit efficacy AND concept
alignment are high? Or does ROME's success depend on operating orthogonally
to the model's concept geometry?

Phases:
  0 — Precompute mom2 statistics for each sweep layer (one-time, skippable)
  1 — Extract concept directions at each layer from cached exp1 activations
  2 — Run ROME edits at each layer (1800 total)
  3 — Compute shared alignment (v_mean analysis)
  4 — Aggregate and save

Saves results to results/exp5_layer_sweep/
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

# Thread safety — must be set before any numpy/torch import
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

import numpy as np
import torch

torch.set_num_threads(4)
torch.set_num_interop_threads(4)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from rome.rome_main import execute_rome, get_context_templates
from rome.rome_hparams import ROMEHyperParams
from rome.layer_stats import layer_stats
from util import nethook
from util.globals import HPARAMS_DIR, STATS_DIR
import rome.rome_main as rome_main_module

# ============================================================
# Configuration
# ============================================================
SWEEP_LAYERS = [5, 10, 15, 17, 20, 25, 30, 35, 40]
RELATIONS = ["P176", "P1412", "P37", "P27", "P413"]
N_TARGETS_PER_RELATION = 2
N_ENTITIES_PER_CONCEPT = 20
V_LOSS_LAYER = 47
SEED = 42
RESULTS_DIR = Path("results/exp5_layer_sweep")
EXP1_DIR = Path("results/exp1")


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


def select_entities(metadata, relation_id, target_value, n, seed=SEED):
    """Select n entities from the negative class for editing TO target_value."""
    rng = np.random.RandomState(seed)
    records = metadata[relation_id]
    negatives = [r for r in records if r["target_true"].strip() != target_value.strip()]
    if len(negatives) < n:
        print(f"  WARNING: Only {len(negatives)} negatives for {relation_id} '{target_value}'")
        n = len(negatives)
    selected = rng.choice(len(negatives), n, replace=False)
    return [negatives[i] for i in selected]


def apply_edit_and_eval(model, tok, u, v, hparams, layer, prompt_texts, target_str):
    """
    Apply rank-1 edit u ⊗ v at specified layer, evaluate, restore.
    Returns dict with efficacy and mean target probability.
    """
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

    results = []
    for prompt_text in prompt_texts:
        inputs = tok(prompt_text, return_tensors="pt").to(w.device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1]
        probs = torch.softmax(logits.float(), dim=0)
        pred_tok = logits.argmax().item()
        target_prob = probs[first_target_tok].item()
        results.append({
            "correct": pred_tok == first_target_tok,
            "target_prob": float(target_prob),
        })

    with torch.no_grad():
        w[...] -= upd

    n_correct = sum(r["correct"] for r in results)
    mean_prob = np.mean([r["target_prob"] for r in results])
    return {
        "efficacy": n_correct / len(results),
        "mean_target_prob": float(mean_prob),
    }


def pca_enhanced_mean(V, variance_threshold=0.90):
    """PCA-enhanced mean direction estimation."""
    raw_mean = V.mean(axis=0)
    raw_mean_norm = raw_mean / (np.linalg.norm(raw_mean) + 1e-10)

    V_c = V - V.mean(axis=0, keepdims=True)
    U, S, Vh = np.linalg.svd(V_c, full_matrices=False)
    cum_var = np.cumsum(S**2) / (S**2).sum()
    k = int(np.searchsorted(cum_var, variance_threshold) + 1)
    k = min(k, len(S))

    basis = Vh[:k]
    V_proj = V @ basis.T
    mean_proj = V_proj.mean(axis=0)
    pca_mean = mean_proj @ basis
    pca_mean_norm = pca_mean / (np.linalg.norm(pca_mean) + 1e-10)

    return raw_mean_norm, pca_mean_norm, k, float(cum_var[k - 1])


# ============================================================
# Phase 0: Precompute mom2 statistics
# ============================================================
def phase0_precompute(model, tok):
    print("\n" + "=" * 70)
    print("PHASE 0: Precompute mom2 statistics")
    print("=" * 70)

    for layer in SWEEP_LAYERS:
        layer_name = f"transformer.h.{layer}.mlp.c_proj"
        stat_path = (
            Path(STATS_DIR)
            / "gpt2-xl"
            / "wikipedia_stats"
            / f"{layer_name}_float32_mom2_100000.npz"
        )
        if stat_path.exists():
            print(f"  Layer {layer:2d}: cached ({stat_path.name})")
            continue

        print(f"  Layer {layer:2d}: computing (100K Wikipedia samples)...")
        t0 = time.time()
        layer_stats(
            model, tok, layer_name, STATS_DIR,
            "wikipedia", ["mom2"],
            sample_size=100000, precision="float32",
        )
        print(f"  Layer {layer:2d}: done in {time.time() - t0:.0f}s")


# ============================================================
# Phase 1: Extract concept directions at each layer
# ============================================================
def phase1_concept_directions():
    print("\n" + "=" * 70)
    print("PHASE 1: Extract concept directions at each sweep layer")
    print("=" * 70)

    # Load exp1 metadata
    metadata = json.load(open(EXP1_DIR / "record_metadata.json"))
    relation_targets = json.load(open(EXP1_DIR / "relation_targets.json"))

    # Build tasks
    tasks = []
    for rid in RELATIONS:
        targets = relation_targets.get(rid, [])
        for t in targets[:N_TARGETS_PER_RELATION]:
            tasks.append((rid, t))

    concept_dirs = {}  # {(rid, target, method, layer): np.array}
    probe_results = {}  # {(rid, target, layer): {"balanced_accuracy": float}}

    for rid in RELATIONS:
        act_path = EXP1_DIR / "activations" / f"{rid}.pt"
        if not act_path.exists():
            print(f"  WARNING: No activations for {rid}, skipping")
            continue

        acts = torch.load(act_path, map_location="cpu")  # {layer: tensor(n, 1600)}
        records = metadata[rid]

        targets = relation_targets.get(rid, [])[:N_TARGETS_PER_RELATION]

        for target in targets:
            # Build binary labels: 1 if target_true == target, 0 otherwise
            labels = np.array([
                1 if r["target_true"].strip() == target.strip() else 0
                for r in records
            ])
            n_pos = labels.sum()
            n_neg = len(labels) - n_pos
            if n_pos < 10 or n_neg < 10:
                print(f"  SKIP {rid} '{target}': too few samples ({n_pos}/{n_neg})")
                continue

            # Train/test split (same as exp1)
            rng = np.random.RandomState(SEED)
            idx = rng.permutation(len(labels))
            n_train = int(0.7 * len(labels))
            train_idx, test_idx = idx[:n_train], idx[n_train:]

            for layer in SWEEP_LAYERS:
                if layer not in acts:
                    continue

                X = acts[layer].numpy()
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = labels[train_idx], labels[test_idx]

                # Mean-diff direction
                pos_mean = X_train[y_train == 1].mean(0)
                neg_mean = X_train[y_train == 0].mean(0)
                md = pos_mean - neg_mean
                md_norm = np.linalg.norm(md)
                if md_norm > 1e-10:
                    md = md / md_norm
                concept_dirs[(rid, target, "mean_diff", layer)] = md

                # Logistic probe
                scaler = StandardScaler()
                Xtr = scaler.fit_transform(X_train)
                Xte = scaler.transform(X_test)

                clf = LogisticRegressionCV(
                    Cs=[1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
                    cv=5, max_iter=1000,
                    scoring="balanced_accuracy",
                    random_state=42,
                )
                clf.fit(Xtr, y_train)

                y_pred = clf.predict(Xte)
                bal_acc = float(balanced_accuracy_score(y_test, y_pred))

                w = clf.coef_[0] / scaler.scale_
                w = w / (np.linalg.norm(w) + 1e-10)
                concept_dirs[(rid, target, "logistic", layer)] = w

                probe_results[(rid, target, layer)] = {
                    "balanced_accuracy": bal_acc,
                }

            print(f"  {rid} '{target}': directions at {len(SWEEP_LAYERS)} layers "
                  f"(probe acc range: {min(probe_results[(rid, target, l)]['balanced_accuracy'] for l in SWEEP_LAYERS if (rid, target, l) in probe_results):.3f}"
                  f"–{max(probe_results[(rid, target, l)]['balanced_accuracy'] for l in SWEEP_LAYERS if (rid, target, l) in probe_results):.3f})")

    # Save
    dir_tensors = {}
    for (rid, target, method, layer), d in concept_dirs.items():
        dir_tensors[f"{rid}_{target}_{method}_{layer}"] = torch.from_numpy(d).float()
    torch.save(dir_tensors, RESULTS_DIR / "concept_directions_by_layer.pt")

    return tasks, concept_dirs, probe_results


# ============================================================
# Phase 2: Run ROME edits at each layer
# ============================================================
def phase2_edits(model, tok, hparams, tasks):
    print("\n" + "=" * 70)
    print("PHASE 2: Run ROME edits at each sweep layer")
    print("=" * 70)

    # Load metadata and entity selection
    metadata = json.load(open(EXP1_DIR / "record_metadata.json"))
    cf_data = json.load(open("data/counterfact.json"))
    cf_by_id = {r["case_id"]: r for r in cf_data}

    # Pre-select entities (same across all layers)
    entity_pool = {}  # {(rid, target): [entity_dicts]}
    for rid, target in tasks:
        entities = select_entities(metadata, rid, target, N_ENTITIES_PER_CONCEPT)
        entity_pool[(rid, target)] = entities

    # Storage
    all_vectors = {}  # {layer: {"u": [], "v": [], "meta": []}}
    all_eval = {}     # {layer: [{"efficacy": ..., "target_prob": ..., ...}]}

    for li, layer in enumerate(SWEEP_LAYERS):
        print(f"\n  ══ Layer {layer} ({li+1}/{len(SWEEP_LAYERS)}) ══")
        hparams_L = replace(hparams, layers=[layer])
        t0 = time.time()

        layer_u, layer_v, layer_meta, layer_eval = [], [], [], []
        n_edits = 0

        for ti, (rid, target) in enumerate(tasks):
            entities = entity_pool[(rid, target)]

            for ei, entity in enumerate(entities):
                request = {
                    "prompt": entity["prompt"],
                    "subject": entity["subject"],
                    "target_new": {"str": target},
                }

                try:
                    deltas = execute_rome(model, tok, request, hparams_L)
                    for key, (u, v) in deltas.items():
                        u_np = u.detach().cpu().numpy()
                        v_np = v.detach().cpu().numpy()
                except Exception as e:
                    print(f"    ERROR {rid}/{target}/{entity['subject']}: {e}")
                    continue

                # Evaluate efficacy
                prompt_text = entity["prompt"].replace("{}", entity["subject"])
                eval_prompts = [prompt_text]
                cf_record = cf_by_id.get(entity["case_id"], {})
                paraphrases = cf_record.get("paraphrase_prompts", [])
                eval_prompts.extend(paraphrases[:3])

                eval_result = apply_edit_and_eval(
                    model, tok, u_np, v_np, hparams_L, layer,
                    eval_prompts, target
                )

                layer_u.append(u_np)
                layer_v.append(v_np)
                layer_meta.append({
                    "relation_id": rid,
                    "target_value": target,
                    "subject": entity["subject"],
                    "case_id": entity["case_id"],
                    "prompt": entity["prompt"],
                })
                layer_eval.append({
                    **eval_result,
                    "relation_id": rid,
                    "target_value": target,
                    "subject": entity["subject"],
                })

                n_edits += 1
                if n_edits % 50 == 0:
                    print(f"    {n_edits} edits done...")

        all_vectors[layer] = {
            "u": np.stack(layer_u) if layer_u else np.empty((0, 6400)),
            "v": np.stack(layer_v) if layer_v else np.empty((0, 1600)),
            "meta": layer_meta,
        }
        all_eval[layer] = layer_eval

        mean_eff = np.mean([e["efficacy"] for e in layer_eval]) if layer_eval else 0
        mean_prob = np.mean([e["mean_target_prob"] for e in layer_eval]) if layer_eval else 0
        mean_vnorm = np.mean(np.linalg.norm(all_vectors[layer]["v"], axis=1)) if layer_v else 0
        print(f"  Layer {layer}: {n_edits} edits in {time.time()-t0:.0f}s | "
              f"efficacy={mean_eff:.3f} target_prob={mean_prob:.3f} ||v||={mean_vnorm:.1f}")

    # Save edit vectors
    save_data = {}
    for layer in SWEEP_LAYERS:
        d = all_vectors[layer]
        save_data[layer] = {
            "u": torch.from_numpy(d["u"]),
            "v": torch.from_numpy(d["v"]),
            "meta": d["meta"],
        }
    torch.save(save_data, RESULTS_DIR / "edit_vectors_by_layer.pt")

    return all_vectors, all_eval


# ============================================================
# Phase 3: Shared alignment analysis
# ============================================================
def phase3_alignment(tasks, all_vectors, all_eval, concept_dirs, probe_results):
    print("\n" + "=" * 70)
    print("PHASE 3: Shared alignment analysis")
    print("=" * 70)

    alignment_results = {}  # {layer: {concept_key: {...}}}

    for layer in SWEEP_LAYERS:
        layer_results = {}
        V_all = all_vectors[layer]["v"]
        meta = all_vectors[layer]["meta"]
        evals = all_eval[layer]

        for rid, target in tasks:
            key = f"{rid}_{target}"

            # Gather v vectors for this concept at this layer
            idxs = [i for i, m in enumerate(meta)
                    if m["relation_id"] == rid and m["target_value"] == target]
            if len(idxs) < 5:
                continue

            V = V_all[idxs]  # (n, 1600)
            concept_evals = [evals[i] for i in idxs]

            # Shared component
            raw_mean, pca_mean, n_pca, pca_var = pca_enhanced_mean(V)
            v_mean_unnorm = V.mean(axis=0)

            result = {
                "n_vectors": len(V),
                "v_norm_mean": float(np.linalg.norm(V, axis=1).mean()),
                "v_norm_std": float(np.linalg.norm(V, axis=1).std()),
                "efficacy_mean": float(np.mean([e["efficacy"] for e in concept_evals])),
                "target_prob_mean": float(np.mean([e["mean_target_prob"] for e in concept_evals])),
                "pca_n_components": n_pca,
                "pca_var_explained": pca_var,
            }

            # Concept alignment
            for method in ["mean_diff", "logistic"]:
                cd_key = (rid, target, method, layer)
                if cd_key not in concept_dirs:
                    continue
                cd = concept_dirs[cd_key]

                # Individual alignment
                ind_cos = [cosine(V[i], cd) for i in range(len(V))]
                result[f"{method}_individual_abs_mean"] = float(np.mean(np.abs(ind_cos)))
                result[f"{method}_individual_mean"] = float(np.mean(ind_cos))

                # Shared alignment (raw mean and PCA)
                result[f"{method}_shared_abs_raw"] = abs(cosine(raw_mean, cd))
                result[f"{method}_shared_abs_pca"] = abs(cosine(pca_mean, cd))

                # v_mean (unnormalized) alignment
                result[f"{method}_vmean_abs"] = abs(cosine(v_mean_unnorm, cd))

            # Consistency
            consistency = [cosine(V[i], raw_mean) for i in range(len(V))]
            result["consistency_mean"] = float(np.mean(consistency))
            result["consistency_std"] = float(np.std(consistency))

            # Shared fraction
            fractions = []
            for i in range(len(V)):
                proj = np.dot(V[i], raw_mean)
                fractions.append(proj**2 / (np.linalg.norm(V[i])**2 + 1e-10))
            result["shared_fraction_mean"] = float(np.mean(fractions))

            # Probe accuracy at this layer
            pr_key = (rid, target, layer)
            if pr_key in probe_results:
                result["probe_balanced_accuracy"] = probe_results[pr_key]["balanced_accuracy"]

            layer_results[key] = result

        alignment_results[layer] = layer_results

        # Print layer summary
        if layer_results:
            effs = [v["efficacy_mean"] for v in layer_results.values()]
            aligns_log = [v.get("logistic_shared_abs_raw", 0) for v in layer_results.values()]
            probes = [v.get("probe_balanced_accuracy", 0) for v in layer_results.values()]
            vnorms = [v["v_norm_mean"] for v in layer_results.values()]
            print(f"  Layer {layer:2d}: eff={np.mean(effs):.3f} "
                  f"|cos(v_mean,concept)|={np.mean(aligns_log):.3f} "
                  f"probe_acc={np.mean(probes):.3f} ||v||={np.mean(vnorms):.1f}")

    return alignment_results


# ============================================================
# Phase 4: Aggregate and save
# ============================================================
def phase4_aggregate(tasks, all_vectors, all_eval, alignment_results, concept_dirs, probe_results):
    print("\n" + "=" * 70)
    print("PHASE 4: Aggregate results")
    print("=" * 70)

    # Build the layer summary table
    layer_summary = {}
    for layer in SWEEP_LAYERS:
        evals = all_eval[layer]
        V = all_vectors[layer]["v"]
        ar = alignment_results.get(layer, {})

        effs = [e["efficacy"] for e in evals]
        probs = [e["mean_target_prob"] for e in evals]
        vnorms = np.linalg.norm(V, axis=1) if len(V) > 0 else [0]

        # Aggregate alignment across concepts
        ind_abs_log = [v.get("logistic_individual_abs_mean", 0) for v in ar.values()]
        shared_abs_log = [v.get("logistic_shared_abs_raw", 0) for v in ar.values()]
        ind_abs_md = [v.get("mean_diff_individual_abs_mean", 0) for v in ar.values()]
        shared_abs_md = [v.get("mean_diff_shared_abs_raw", 0) for v in ar.values()]
        probes = [v.get("probe_balanced_accuracy", 0) for v in ar.values()]
        consistency = [v.get("consistency_mean", 0) for v in ar.values()]
        shared_frac = [v.get("shared_fraction_mean", 0) for v in ar.values()]

        layer_summary[layer] = {
            "efficacy_mean": float(np.mean(effs)),
            "efficacy_std": float(np.std(effs)),
            "target_prob_mean": float(np.mean(probs)),
            "target_prob_std": float(np.std(probs)),
            "v_norm_mean": float(np.mean(vnorms)),
            "v_norm_std": float(np.std(vnorms)),
            "logistic_individual_abs_mean": float(np.mean(ind_abs_log)) if ind_abs_log else 0,
            "logistic_shared_abs_mean": float(np.mean(shared_abs_log)) if shared_abs_log else 0,
            "mean_diff_individual_abs_mean": float(np.mean(ind_abs_md)) if ind_abs_md else 0,
            "mean_diff_shared_abs_mean": float(np.mean(shared_abs_md)) if shared_abs_md else 0,
            "probe_accuracy_mean": float(np.mean(probes)) if probes else 0,
            "consistency_mean": float(np.mean(consistency)) if consistency else 0,
            "shared_fraction_mean": float(np.mean(shared_frac)) if shared_frac else 0,
            "n_edits": len(evals),
            "n_concepts": len(ar),
        }

    # Print the key table
    print("\n" + "=" * 70)
    print("LAYER SWEEP RESULTS")
    print("=" * 70)

    header = (f"  {'Layer':>5} {'Efficacy':>8} {'TgtProb':>8} "
              f"{'|cos_ind|':>9} {'|cos_shd|':>9} {'ProbeAcc':>8} "
              f"{'||v||':>8} {'Consist':>8} {'ShdFrac':>8}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    for layer in SWEEP_LAYERS:
        s = layer_summary[layer]
        print(f"  {layer:5d} {s['efficacy_mean']:8.3f} {s['target_prob_mean']:8.3f} "
              f"{s['logistic_individual_abs_mean']:9.3f} {s['logistic_shared_abs_mean']:9.3f} "
              f"{s['probe_accuracy_mean']:8.3f} "
              f"{s['v_norm_mean']:8.1f} {s['consistency_mean']:8.3f} "
              f"{s['shared_fraction_mean']:8.3f}")

    # Compute correlations across layers
    from scipy.stats import spearmanr

    layers_arr = np.array(SWEEP_LAYERS, dtype=float)
    eff_arr = np.array([layer_summary[l]["efficacy_mean"] for l in SWEEP_LAYERS])
    align_arr = np.array([layer_summary[l]["logistic_shared_abs_mean"] for l in SWEEP_LAYERS])
    probe_arr = np.array([layer_summary[l]["probe_accuracy_mean"] for l in SWEEP_LAYERS])
    vnorm_arr = np.array([layer_summary[l]["v_norm_mean"] for l in SWEEP_LAYERS])

    correlations = {}
    for name, x, y in [
        ("layer_vs_efficacy", layers_arr, eff_arr),
        ("layer_vs_alignment", layers_arr, align_arr),
        ("layer_vs_probe_acc", layers_arr, probe_arr),
        ("efficacy_vs_alignment", eff_arr, align_arr),
        ("probe_acc_vs_alignment", probe_arr, align_arr),
        ("layer_vs_vnorm", layers_arr, vnorm_arr),
    ]:
        rho, p = spearmanr(x, y)
        correlations[name] = {"rho": float(rho), "p": float(p)}

    print("\n  Cross-layer Spearman correlations:")
    for name, vals in correlations.items():
        print(f"    {name:<30s} ρ = {vals['rho']:+.3f}  (p = {vals['p']:.4f})")

    # Check for sweet spot
    print("\n  Sweet spot check (efficacy >= 0.60 AND |cos| >= 0.30):")
    found_sweet_spot = False
    for layer in SWEEP_LAYERS:
        s = layer_summary[layer]
        eff = s["efficacy_mean"]
        aln = s["logistic_shared_abs_mean"]
        if eff >= 0.60 and aln >= 0.30:
            print(f"    Layer {layer}: efficacy={eff:.3f}, alignment={aln:.3f} — SWEET SPOT")
            found_sweet_spot = True
    if not found_sweet_spot:
        print("    No layer meets both thresholds.")

    # Per-relation breakdown
    print("\n  Per-relation efficacy by layer:")
    print(f"  {'Relation':<8} " + " ".join(f"L{l:2d}" for l in SWEEP_LAYERS))
    for rid in RELATIONS:
        vals = []
        for layer in SWEEP_LAYERS:
            evals = all_eval[layer]
            rel_effs = [e["efficacy"] for e in evals if e["relation_id"] == rid]
            vals.append(np.mean(rel_effs) if rel_effs else 0)
        print(f"  {rid:<8} " + " ".join(f"{v:.2f}" for v in vals))

    print("\n  Per-relation alignment (logistic shared) by layer:")
    print(f"  {'Relation':<8} " + " ".join(f"L{l:2d}" for l in SWEEP_LAYERS))
    for rid in RELATIONS:
        vals = []
        for layer in SWEEP_LAYERS:
            ar = alignment_results.get(layer, {})
            rel_aligns = [v.get("logistic_shared_abs_raw", 0) for k, v in ar.items()
                         if k.startswith(rid)]
            vals.append(np.mean(rel_aligns) if rel_aligns else 0)
        print(f"  {rid:<8} " + " ".join(f"{v:.3f}" for v in vals))

    # ── Save ──
    results = {
        "config": {
            "sweep_layers": SWEEP_LAYERS,
            "relations": RELATIONS,
            "n_targets_per_relation": N_TARGETS_PER_RELATION,
            "n_entities_per_concept": N_ENTITIES_PER_CONCEPT,
            "v_loss_layer": V_LOSS_LAYER,
            "seed": SEED,
        },
        "layer_summary": {str(k): v for k, v in layer_summary.items()},
        "alignment_by_layer": {
            str(layer): {k: {kk: vv for kk, vv in v.items()}
                        for k, v in lr.items()}
            for layer, lr in alignment_results.items()
        },
        "correlations": correlations,
        "per_edit_eval": {
            str(layer): evals for layer, evals in all_eval.items()
        },
    }

    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save a concise summary table
    with open(RESULTS_DIR / "layer_summary.json", "w") as f:
        json.dump({str(k): v for k, v in layer_summary.items()}, f, indent=2)

    print(f"\n  Results saved to {RESULTS_DIR}/")

    return layer_summary, correlations


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Experiment 5: Layer Sweep")
    parser.add_argument("--precompute-only", action="store_true",
                        help="Only precompute mom2 stats, then exit")
    parser.add_argument("--skip-precompute", action="store_true",
                        help="Skip mom2 precomputation (assume cached)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    print("=" * 70)
    print("EXPERIMENT 5: LAYER SWEEP")
    print(f"Layers: {SWEEP_LAYERS}")
    print(f"Scale: {len(RELATIONS)} relations × {N_TARGETS_PER_RELATION} targets "
          f"× {N_ENTITIES_PER_CONCEPT} entities = "
          f"{len(RELATIONS) * N_TARGETS_PER_RELATION * N_ENTITIES_PER_CONCEPT} edits/layer")
    print(f"Total: {len(SWEEP_LAYERS)} layers × "
          f"{len(RELATIONS) * N_TARGETS_PER_RELATION * N_ENTITIES_PER_CONCEPT} = "
          f"{len(SWEEP_LAYERS) * len(RELATIONS) * N_TARGETS_PER_RELATION * N_ENTITIES_PER_CONCEPT} edits")
    print("=" * 70)

    # Load model
    print("\nLoading GPT-2 XL...")
    model = AutoModelForCausalLM.from_pretrained("gpt2-xl").cuda().eval()
    tok = AutoTokenizer.from_pretrained("gpt2-xl")
    tok.pad_token = tok.eos_token
    hparams = ROMEHyperParams.from_json(HPARAMS_DIR / "ROME" / "gpt2-xl.json")

    # Set context templates
    with open(EXP1_DIR / "context_templates.json") as f:
        templates = json.load(f)
    rome_main_module.CONTEXT_TEMPLATES_CACHE = templates
    print(f"Fixed {len(templates)} context templates from Experiment 1")

    # Phase 0
    if not args.skip_precompute:
        phase0_precompute(model, tok)
        if args.precompute_only:
            print(f"\nPrecomputation complete. Total time: {time.time()-t_start:.0f}s")
            return

    # Phase 1
    tasks, concept_dirs, probe_results = phase1_concept_directions()

    # Phase 2
    all_vectors, all_eval = phase2_edits(model, tok, hparams, tasks)

    # Phase 3
    alignment_results = phase3_alignment(tasks, all_vectors, all_eval, concept_dirs, probe_results)

    # Phase 4
    layer_summary, correlations = phase4_aggregate(
        tasks, all_vectors, all_eval, alignment_results, concept_dirs, probe_results
    )

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT 5 COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
