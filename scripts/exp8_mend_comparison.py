"""
Experiment 8: Learned vs Optimized Edit Directions (MEND Comparison)

MEND and ROME both produce rank-1 updates (u⊗v) to MLP projection matrices,
but via fundamentally different mechanisms:
  - ROME: per-edit constrained optimization (20 Adam steps)
  - MEND: learned meta-network (single forward pass through GradientTransform)

MEND edits layers 45-47 (where concept structure is strong), while ROME edits
layer 17 (where concept structure is weak). If MEND's v vectors are also
orthogonal to concept directions, the "dark subspace" finding is structural,
not algorithm-specific.

Four phases:
  Phase 0 — Extract MEND edit vectors (u, v) for 200 entities
  Phase 1 — Concept alignment analysis (v vs concept directions)
  Phase 2 — Edit geometry clustering (PERMANOVA, LDA, cosine)
  Phase 3 — Layer propagation (Δh alignment across layers)

Saves results to results/exp8_mend/
"""

import json
import logging
import os
import sys
import time
import numpy as np
import torch
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util import nethook

SEED = 42
RESULTS_DIR = Path("results/exp8_mend")
EXP1_DIR = Path("results/exp1")
EXP2_DIR = Path("results/exp2")
EXP3_DIR = Path("results/exp3")
EXP5_DIR = Path("results/exp5_layer_sweep")

RELATIONS = ["P176", "P1412", "P37", "P27", "P413"]
MEND_LAYERS = [45, 46, 47]
ALL_LAYERS = list(range(48))
PERMANOVA_N_PERMS = 10000
LDA_N_FOLDS = 5

# Logging
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
_fh = logging.FileHandler(RESULTS_DIR / "exp8.log", mode="w")
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logging.basicConfig(level=logging.INFO, handlers=[_fh, _sh])
log = logging.getLogger(__name__)


# ============================================================
# Helpers
# ============================================================

def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def permanova(D_sq, labels, n_perms=10000, seed=42):
    """PERMANOVA (Anderson 2001) on a squared distance matrix."""
    rng = np.random.RandomState(seed)
    n = len(labels)
    unique = np.unique(labels)
    k = len(unique)
    if k < 2 or n < k + 1:
        return 0.0, 1.0, 0.0

    def compute_stats(labs):
        ss_t = D_sq.sum() / (2 * n)
        ss_w = 0.0
        for g in unique:
            mask = (labs == g)
            n_g = mask.sum()
            if n_g > 1:
                ss_w += D_sq[np.ix_(mask, mask)].sum() / (2 * n_g)
        ss_b = ss_t - ss_w
        f = (ss_b / max(k - 1, 1)) / (ss_w / max(n - k, 1) + 1e-10)
        r2 = ss_b / (ss_t + 1e-10)
        return f, r2

    f_obs, r2_obs = compute_stats(labels)
    count = 0
    t0 = time.time()
    for _ in range(n_perms):
        f_perm, _ = compute_stats(rng.permutation(labels))
        if f_perm >= f_obs:
            count += 1
    log.info(f"  PERMANOVA: {n_perms} perms in {time.time()-t0:.1f}s (k={k}, n={n})")

    p_value = (count + 1) / (n_perms + 1)
    return float(f_obs), float(p_value), float(r2_obs)


def sq_euclidean_dist_matrix(X):
    X = np.asarray(X, dtype=np.float64)
    norms_sq = (X ** 2).sum(axis=1)
    D_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * X @ X.T
    D_sq = np.maximum(D_sq, 0)
    return D_sq


# ============================================================
# Phase 0: MEND Edit Vector Extraction
# ============================================================

def phase_0(model, tok, mend_executor, mend_hparams):
    """Run MEND edits and extract u/v factors directly from the meta-network."""
    log.info("=" * 70)
    log.info("PHASE 0: MEND Edit Vector Extraction")
    log.info("=" * 70)
    t0 = time.time()

    # Load entity list from Exp 5
    log.info("Loading entity list from Exp 5...")
    exp5_data = torch.load(EXP5_DIR / "edit_vectors_by_layer.pt", map_location="cpu")
    exp5_meta = exp5_data[17]["meta"]  # 200 entities with prompts
    log.info(f"  {len(exp5_meta)} entities loaded")

    # Storage: MEND returns explicit u/v factors per parameter.
    # For GPT-2 XL: delta = u.T @ v (for c_proj), scaled by edit_lr.
    # u is the transformed input activation, v is the transformed gradient.
    # For c_proj: u ∈ R^{batch×d_inner}, v ∈ R^{batch×d_model}
    # We extract v (R^1600) as the concept-space direction, analogous to ROME's v.
    v_by_layer = defaultdict(list)   # layer -> list of v tensors (R^1600)
    u_by_layer = defaultdict(list)   # layer -> list of u tensors (R^6400)
    metadata = []
    efficacies = []

    log.info(f"Running 200 MEND edits...")
    for i, entity in enumerate(exp5_meta):
        request = {
            "prompt": entity["prompt"],
            "subject": entity["subject"],
            "target_new": {"str": entity["target_value"]},
            "target_true": {"str": ""},  # Not needed for MEND
        }

        weights_copy = None
        try:
            # Run MEND edit (modifies model in-place, returns orig weights)
            edited_model, weights_copy = mend_executor.apply_to_model(
                model, tok, [request], mend_hparams,
                copy=False, return_orig_weights=True,
            )

            # Check efficacy on the edited model
            prompt_text = entity["prompt"].replace("{}", entity["subject"])
            inputs = tok(prompt_text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                logits = model(**inputs).logits
            target_str = entity["target_value"]
            target_tok_id = tok(f" {target_str.strip()}", return_tensors="pt")["input_ids"][0][0].item()
            pred = logits[0, -1].argmax().item()
            target_prob = torch.softmax(logits[0, -1].float(), dim=0)[target_tok_id].item()
            efficacy = int(pred == target_tok_id)

            # Extract u/v factors: compute delta = W_edited - W_original for each param,
            # then since delta = u.T @ v * lr (rank-1), recover u/v via SVD (rank-1 exact).
            for param_name, orig_w in weights_copy.items():
                current_w = nethook.get_parameter(model, param_name).data
                delta = (current_w - orig_w).detach().cpu().float()

                layer_num = None
                for l in MEND_LAYERS:
                    if f"transformer.h.{l}" in param_name:
                        layer_num = l
                        break
                if layer_num is None:
                    continue

                if "c_proj" in param_name:
                    # delta = u.T @ v * lr (rank-1 by construction)
                    # GPT-2 uses Conv1D so c_proj weight is (d_inner, d_model) = (6400, 1600)
                    # SVD recovers the exact rank-1 factors
                    U_svd, S_svd, Vt_svd = torch.linalg.svd(delta, full_matrices=False)
                    # Verify rank-1: S[0] should dominate
                    rank1_frac = (S_svd[0] / (S_svd.sum() + 1e-10)).item()
                    if rank1_frac < 0.99:
                        log.warning(f"  Edit {i}, layer {layer_num} c_proj: rank-1 fraction = {rank1_frac:.4f}")
                    v_vec = Vt_svd[0]   # R^1600 — the direction in residual stream
                    u_vec = U_svd[:, 0]  # R^6400 — the direction in MLP key space
                    sigma = S_svd[0].item()
                    # Store v scaled by singular value (preserves magnitude info)
                    v_by_layer[layer_num].append(v_vec * sigma)
                    u_by_layer[layer_num].append(u_vec)

            # Restore original weights
            with torch.no_grad():
                for param_name, orig_w in weights_copy.items():
                    nethook.get_parameter(model, param_name).data.copy_(orig_w)

        except Exception as e:
            log.info(f"  Edit {i} failed: {e}")
            # Restore weights if partially modified
            if weights_copy:
                with torch.no_grad():
                    for pn, ow in weights_copy.items():
                        nethook.get_parameter(model, pn).data.copy_(ow)
            efficacy = 0
            target_prob = 0.0
            for l in MEND_LAYERS:
                v_by_layer[l].append(torch.zeros(1600))
                u_by_layer[l].append(torch.zeros(6400))

        metadata.append({
            "relation_id": entity["relation_id"],
            "target_value": entity["target_value"],
            "subject": entity["subject"],
            "case_id": entity["case_id"],
            "prompt": entity["prompt"],
            "efficacy": efficacy,
            "target_prob": float(target_prob),
        })
        efficacies.append(efficacy)

        if (i + 1) % 20 == 0:
            log.info(f"  {i+1}/200 edits done, efficacy so far: {np.mean(efficacies):.3f}")

    # Stack tensors
    v_tensors = {l: torch.stack(vecs) for l, vecs in v_by_layer.items()}
    u_tensors = {l: torch.stack(vecs) for l, vecs in u_by_layer.items()}

    log.info(f"\nMEND edit summary:")
    log.info(f"  Overall efficacy: {np.mean(efficacies):.3f} ({sum(efficacies)}/200)")
    for l in sorted(v_tensors.keys()):
        log.info(f"  Layer {l}: v shape {v_tensors[l].shape}, mean ||v|| = {v_tensors[l].norm(dim=1).mean():.3f}")

    # Save
    torch.save({
        "v_by_layer": v_tensors,
        "u_by_layer": u_tensors,
        "metadata": metadata,
    }, RESULTS_DIR / "edit_vectors.pt")

    elapsed = time.time() - t0
    log.info(f"Phase 0 complete in {elapsed:.0f}s")
    return v_tensors, u_tensors, metadata


# ============================================================
# Phase 1: Concept Alignment Analysis
# ============================================================

def phase_1(v_tensors, metadata):
    """Compare MEND v vectors to concept directions."""
    log.info("\n" + "=" * 70)
    log.info("PHASE 1: Concept Alignment Analysis")
    log.info("=" * 70)
    t0 = time.time()

    # Load concept directions from Exp 3 (available at layers 0-47)
    log.info("Loading concept directions from Exp 3...")
    all_concept_dirs = torch.load(EXP3_DIR / "concept_directions_all_layers.pt", map_location="cpu")

    # Build concept groups
    concept_indices = defaultdict(list)  # concept_key -> list of indices
    for i, m in enumerate(metadata):
        concept_key = f"{m['relation_id']}_{m['target_value']}"
        concept_indices[concept_key].append(i)

    results = {}

    for layer in sorted(v_tensors.keys()):
        V = v_tensors[layer].numpy().astype(np.float64)
        log.info(f"\n  Layer {layer}:")

        layer_results = {}

        for concept_key, indices in sorted(concept_indices.items()):
            # Get v vectors for this concept
            V_concept = V[indices]
            n = len(V_concept)

            # Shared component
            v_mean = V_concept.mean(axis=0)
            v_mean_norm = v_mean / (np.linalg.norm(v_mean) + 1e-10)

            # Consistency
            consistency = np.mean([cosine(V_concept[j], v_mean) for j in range(n)])

            # Load concept direction at this layer
            md_key = f"{concept_key}_mean_diff_L{layer}"
            lg_key = f"{concept_key}_logistic_L{layer}"

            md_dir = all_concept_dirs.get(md_key)
            lg_dir = all_concept_dirs.get(lg_key)

            if md_dir is None and lg_dir is None:
                log.info(f"    {concept_key}: no concept direction at layer {layer}, skipping")
                continue

            md_dir_np = md_dir.numpy().astype(np.float64) if md_dir is not None else None
            lg_dir_np = lg_dir.numpy().astype(np.float64) if lg_dir is not None else None

            # Shared alignment
            shared_md = abs(cosine(v_mean_norm, md_dir_np)) if md_dir_np is not None else None
            shared_lg = abs(cosine(v_mean_norm, lg_dir_np)) if lg_dir_np is not None else None

            # Individual alignment
            indiv_md = np.mean([abs(cosine(V_concept[j], md_dir_np)) for j in range(n)]) if md_dir_np is not None else None
            indiv_lg = np.mean([abs(cosine(V_concept[j], lg_dir_np)) for j in range(n)]) if lg_dir_np is not None else None

            # Wrong-concept alignment (different target, same relation)
            rel_id = concept_key.split("_")[0]
            wrong_concepts = [
                ck for ck in concept_indices
                if ck.startswith(rel_id + "_") and ck != concept_key
            ]
            wrong_md = None
            if wrong_concepts:
                wc_key = wrong_concepts[0]
                wc_md_key = f"{wc_key}_mean_diff_L{layer}"
                wc_dir = all_concept_dirs.get(wc_md_key)
                if wc_dir is not None:
                    wrong_md = abs(cosine(v_mean_norm, wc_dir.numpy().astype(np.float64)))

            # Wrong-relation alignment
            wrong_rels = [
                ck for ck in concept_indices
                if not ck.startswith(rel_id + "_")
            ]
            wrong_rel_md = None
            if wrong_rels:
                wr_key = wrong_rels[0]
                wr_md_key = f"{wr_key}_mean_diff_L{layer}"
                wr_dir = all_concept_dirs.get(wr_md_key)
                if wr_dir is not None:
                    wrong_rel_md = abs(cosine(v_mean_norm, wr_dir.numpy().astype(np.float64)))

            layer_results[concept_key] = {
                "shared_alignment_mean_diff": shared_md,
                "shared_alignment_logistic": shared_lg,
                "individual_alignment_mean_diff": indiv_md,
                "individual_alignment_logistic": indiv_lg,
                "wrong_concept_alignment": wrong_md,
                "wrong_relation_alignment": wrong_rel_md,
                "consistency": float(consistency),
                "n": n,
            }

            md_str = f"{shared_md:.3f}" if shared_md is not None else "N/A"
            lg_str = f"{shared_lg:.3f}" if shared_lg is not None else "N/A"
            wc_str = f"{wrong_md:.3f}" if wrong_md is not None else "N/A"
            log.info(f"    {concept_key:25s}: shared_md={md_str}, shared_lg={lg_str}, "
                     f"wrong_concept={wc_str}, consistency={consistency:.3f}")

        results[layer] = layer_results

    # Grand averages per layer
    log.info("\n  Grand averages:")
    log.info(f"  {'Layer':>6s} {'|cos(v,cd)| md':>16s} {'|cos(v,cd)| lg':>16s} "
             f"{'wrong_concept':>14s} {'consistency':>12s}")
    log.info("  " + "-" * 70)

    grand_summary = {}
    for layer in sorted(results.keys()):
        lr = results[layer]
        if not lr:
            log.info(f"  {layer:6d} {'(no concept directions at this layer)':>60s}")
            continue
        all_md = [v["shared_alignment_mean_diff"] for v in lr.values() if v["shared_alignment_mean_diff"] is not None]
        all_lg = [v["shared_alignment_logistic"] for v in lr.values() if v["shared_alignment_logistic"] is not None]
        all_wc = [v["wrong_concept_alignment"] for v in lr.values() if v["wrong_concept_alignment"] is not None]
        all_cons = [v["consistency"] for v in lr.values()]

        md_mean = float(np.mean(all_md)) if all_md else None
        lg_mean = float(np.mean(all_lg)) if all_lg else None
        wc_mean = float(np.mean(all_wc)) if all_wc else None
        cons_mean = float(np.mean(all_cons))

        grand_summary[layer] = {
            "shared_alignment_mean_diff": md_mean,
            "shared_alignment_logistic": lg_mean,
            "wrong_concept_alignment": wc_mean,
            "consistency": cons_mean,
        }

        md_s = f"{md_mean:.4f}" if md_mean is not None else "N/A"
        lg_s = f"{lg_mean:.4f}" if lg_mean is not None else "N/A"
        wc_s = f"{wc_mean:.4f}" if wc_mean is not None else "N/A"
        log.info(f"  {layer:6d} {md_s:>16s} {lg_s:>16s} {wc_s:>14s} {cons_mean:12.4f}")

    elapsed = time.time() - t0
    log.info(f"\nPhase 1 complete in {elapsed:.0f}s")
    return {"per_concept": results, "grand_summary": grand_summary}


# ============================================================
# Phase 2: Edit Geometry Clustering
# ============================================================

def phase_2(v_tensors, metadata):
    """PERMANOVA, LDA, cosine analysis on MEND's v vectors."""
    log.info("\n" + "=" * 70)
    log.info("PHASE 2: Edit Geometry Clustering")
    log.info("=" * 70)
    t0 = time.time()

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedKFold

    results = {}

    # Use layer 47 as primary (deepest MEND edit layer)
    for layer in sorted(v_tensors.keys()):
        V = v_tensors[layer].numpy().astype(np.float64)
        n = len(V)

        relations = np.array([m["relation_id"] for m in metadata])
        targets = np.array([f"{m['relation_id']}_{m['target_value']}" for m in metadata])

        log.info(f"\n  Layer {layer} ({n} vectors):")

        # PERMANOVA
        log.info("  Computing distance matrix...")
        D_sq = sq_euclidean_dist_matrix(V)

        log.info("  PERMANOVA (relation)...")
        f_rel, p_rel, r2_rel = permanova(D_sq, relations, PERMANOVA_N_PERMS, SEED)
        log.info(f"    R2={r2_rel:.4f}, F={f_rel:.2f}, p={p_rel:.6f}")

        log.info("  PERMANOVA (target_value)...")
        f_tgt, p_tgt, r2_tgt = permanova(D_sq, targets, PERMANOVA_N_PERMS, SEED)
        log.info(f"    R2={r2_tgt:.4f}, F={f_tgt:.2f}, p={p_tgt:.6f}")

        # LDA
        log.info("  LDA classification by relation...")
        unique_rels = np.unique(relations)
        if len(unique_rels) >= 2:
            skf = StratifiedKFold(n_splits=LDA_N_FOLDS, shuffle=True, random_state=SEED)
            fold_accs = []
            for train_idx, test_idx in skf.split(V, relations):
                lda = LinearDiscriminantAnalysis()
                lda.fit(V[train_idx], relations[train_idx])
                pred = lda.predict(V[test_idx])
                fold_accs.append((pred == relations[test_idx]).mean())
            lda_acc = np.mean(fold_accs)
        else:
            lda_acc = 0.0
        log.info(f"    {LDA_N_FOLDS}-fold CV accuracy: {lda_acc:.3f}")

        # Within/between cosine
        log.info("  Within/between cosine similarity...")
        V_norm = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-10)
        rng = np.random.RandomState(SEED)
        n_pairs = 20000
        within_cos, between_cos = [], []
        for _ in range(n_pairs):
            i, j = rng.choice(n, 2, replace=False)
            c = float(V_norm[i] @ V_norm[j])
            if relations[i] == relations[j]:
                within_cos.append(c)
            else:
                between_cos.append(c)

        within_abs = float(np.mean(np.abs(within_cos))) if within_cos else 0
        between_abs = float(np.mean(np.abs(between_cos))) if between_cos else 0
        log.info(f"    Within |cos|: {within_abs:.4f} (n={len(within_cos)}), "
                 f"Between |cos|: {between_abs:.4f} (n={len(between_cos)})")

        results[layer] = {
            "permanova_relation": {"R2": r2_rel, "F": f_rel, "p": p_rel},
            "permanova_target": {"R2": r2_tgt, "F": f_tgt, "p": p_tgt},
            "lda_cv_accuracy": lda_acc,
            "within_relation_cos_abs": within_abs,
            "between_relation_cos_abs": between_abs,
        }

    elapsed = time.time() - t0
    log.info(f"\nPhase 2 complete in {elapsed:.0f}s")
    return results


# ============================================================
# Phase 3: Layer Propagation
# ============================================================

def phase_3(model, tok, metadata, mend_executor, mend_hparams):
    """Apply MEND edits and measure Δh alignment across layers."""
    log.info("\n" + "=" * 70)
    log.info("PHASE 3: Layer Propagation")
    log.info("=" * 70)
    t0 = time.time()

    device = next(model.parameters()).device

    # Load concept directions at all layers
    all_concept_dirs = torch.load(EXP3_DIR / "concept_directions_all_layers.pt", map_location="cpu")

    # Available probe layers from Exp 3
    probe_layers = [0, 5, 10, 13, 15, 17, 20, 25, 30, 35, 40, 47]
    layer_names = [f"transformer.h.{l}" for l in ALL_LAYERS]

    # Select 50 edits (5 per concept)
    rng = np.random.RandomState(SEED)
    concept_indices = defaultdict(list)
    for i, m in enumerate(metadata):
        concept_indices[f"{m['relation_id']}_{m['target_value']}"].append(i)

    selected_indices = []
    for concept in sorted(concept_indices.keys()):
        indices = concept_indices[concept]
        # Only take edits that succeeded
        successful = [i for i in indices if metadata[i]["efficacy"] == 1]
        if len(successful) < 5:
            successful = indices  # fallback to all if too few succeeded
        chosen = rng.choice(successful, size=min(5, len(successful)), replace=False)
        selected_indices.extend(chosen)
    log.info(f"Selected {len(selected_indices)} edits for propagation analysis")

    all_propagation_results = []

    for edit_num, idx in enumerate(selected_indices):
        m = metadata[idx]
        request = {
            "prompt": m["prompt"],
            "subject": m["subject"],
            "target_new": {"str": m["target_value"]},
            "target_true": {"str": ""},
        }

        prompt_text = m["prompt"].replace("{}", m["subject"])
        inputs = tok(prompt_text, return_tensors="pt").to(device)
        seq_len = inputs["input_ids"].shape[1]
        last_pos = seq_len - 1

        # Forward pass: ORIGINAL
        with torch.no_grad():
            with nethook.TraceDict(model, layer_names, retain_output=True) as tr_orig:
                logits_orig = model(**inputs).logits

        h_orig = {}
        for l in ALL_LAYERS:
            out = tr_orig[f"transformer.h.{l}"].output
            if isinstance(out, tuple):
                out = out[0]
            h_orig[l] = out[0, last_pos].detach().cpu().float().numpy()

        # Apply MEND edit
        weights_copy = None
        try:
            edited_model, weights_copy = mend_executor.apply_to_model(
                model, tok, [request], mend_hparams,
                copy=False, return_orig_weights=True,
            )
        except Exception as e:
            log.info(f"  Edit {edit_num} failed: {e}")
            # Restore weights if partially modified
            if weights_copy:
                with torch.no_grad():
                    for pn, ow in weights_copy.items():
                        nethook.get_parameter(model, pn).data.copy_(ow)
            continue

        # Forward pass: EDITED
        with torch.no_grad():
            with nethook.TraceDict(model, layer_names, retain_output=True) as tr_edit:
                logits_edited = model(**inputs).logits

        h_edit = {}
        for l in ALL_LAYERS:
            out = tr_edit[f"transformer.h.{l}"].output
            if isinstance(out, tuple):
                out = out[0]
            h_edit[l] = out[0, last_pos].detach().cpu().float().numpy()

        # Efficacy
        target_str = m["target_value"]
        target_tok_id = tok(f" {target_str.strip()}", return_tensors="pt")["input_ids"][0][0].item()
        pred_edited = logits_edited[0, -1].argmax().item()
        target_prob = torch.softmax(logits_edited[0, -1].float(), dim=0)[target_tok_id].item()

        # Restore weights
        with torch.no_grad():
            for param_name, orig_w in weights_copy.items():
                nethook.get_parameter(model, param_name).data.copy_(orig_w)

        # Compute perturbation and alignment
        concept_key = f"{m['relation_id']}_{m['target_value']}"
        delta_h = {}

        for l in ALL_LAYERS:
            dh = h_edit[l] - h_orig[l]
            delta_h[l] = {
                "norm": float(np.linalg.norm(dh)),
            }

            # Alignment with concept direction (at available layers only)
            if l in probe_layers:
                md_key = f"{concept_key}_mean_diff_L{l}"
                lg_key = f"{concept_key}_logistic_L{l}"
                md_dir = all_concept_dirs.get(md_key)
                lg_dir = all_concept_dirs.get(lg_key)

                if md_dir is not None and np.linalg.norm(dh) > 1e-8:
                    delta_h[l]["cos_mean_diff"] = abs(cosine(dh, md_dir.numpy()))
                if lg_dir is not None and np.linalg.norm(dh) > 1e-8:
                    delta_h[l]["cos_logistic"] = abs(cosine(dh, lg_dir.numpy()))

        all_propagation_results.append({
            "concept": concept_key,
            "subject": m["subject"],
            "efficacy": int(pred_edited == target_tok_id),
            "target_prob": float(target_prob),
            "delta_h": delta_h,
        })

        if (edit_num + 1) % 10 == 0:
            log.info(f"  {edit_num+1}/{len(selected_indices)} propagation edits done")

    # Aggregate by layer
    log.info("\n  Propagation alignment profile (last token position):")
    log.info(f"  {'Layer':>6s} {'||Δh||':>10s} {'|cos(md)|':>10s} {'|cos(lg)|':>10s}")
    log.info("  " + "-" * 40)

    propagation_summary = {}
    for l in ALL_LAYERS:
        norms = [r["delta_h"][l]["norm"] for r in all_propagation_results]
        cos_md = [r["delta_h"][l].get("cos_mean_diff") for r in all_propagation_results
                  if r["delta_h"][l].get("cos_mean_diff") is not None]
        cos_lg = [r["delta_h"][l].get("cos_logistic") for r in all_propagation_results
                  if r["delta_h"][l].get("cos_logistic") is not None]

        mean_norm = float(np.mean(norms))
        mean_md = float(np.mean(cos_md)) if cos_md else None
        mean_lg = float(np.mean(cos_lg)) if cos_lg else None

        propagation_summary[l] = {
            "mean_norm": mean_norm,
            "mean_cos_mean_diff": mean_md,
            "mean_cos_logistic": mean_lg,
        }

        if l in probe_layers:
            md_s = f"{mean_md:.4f}" if mean_md is not None else "N/A"
            lg_s = f"{mean_lg:.4f}" if mean_lg is not None else "N/A"
            log.info(f"  {l:6d} {mean_norm:10.4f} {md_s:>10s} {lg_s:>10s}")

    # Sanity check: Δh should be zero before layer 45
    pre_edit_norms = [propagation_summary[l]["mean_norm"] for l in range(45)]
    max_pre_norm = max(pre_edit_norms)
    log.info(f"\n  Sanity: max ||Δh|| at layers 0-44 = {max_pre_norm:.6f} (should be ~0)")

    elapsed = time.time() - t0
    log.info(f"\nPhase 3 complete in {elapsed:.0f}s")
    return {"summary": propagation_summary, "per_edit": all_propagation_results}


# ============================================================
# Main
# ============================================================

def main():
    log.info("=" * 70)
    log.info("EXPERIMENT 8: LEARNED vs OPTIMIZED EDIT DIRECTIONS (MEND)")
    log.info("=" * 70)
    t_start = time.time()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    log.info("Loading GPT-2 XL...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained("gpt2-xl").cuda().eval()
    tok = AutoTokenizer.from_pretrained("gpt2-xl")
    tok.pad_token = tok.eos_token

    # Initialize MEND once (it modifies the tokenizer/model embedding table)
    log.info("Initializing MEND executor...")
    from baselines.mend import MENDHyperParams, MendRewriteExecutor
    mend_executor = MendRewriteExecutor()
    mend_hparams = MENDHyperParams.from_json(Path("hparams/MEND/gpt2-xl.json"))
    mend_executor.init_model(model, tok, mend_hparams)
    log.info("MEND initialized (tokenizer extended with [PAD])")

    # Phase 0: Extract MEND edit vectors
    v_tensors, u_tensors, metadata = phase_0(model, tok, mend_executor, mend_hparams)

    # Phase 1: Concept alignment
    results_1 = phase_1(v_tensors, metadata)

    # Phase 2: Clustering
    results_2 = phase_2(v_tensors, metadata)

    # Phase 3: Propagation
    results_3 = phase_3(model, tok, metadata, mend_executor, mend_hparams)

    # ============================================================
    # Comparison with ROME
    # ============================================================
    log.info("\n" + "=" * 70)
    log.info("COMPARISON: MEND vs ROME")
    log.info("=" * 70)

    # Load ROME Exp 2 results
    rome_res = {}
    if (EXP2_DIR / "results.json").exists():
        with open(EXP2_DIR / "results.json") as f:
            rome_res = json.load(f)

    log.info(f"\n  {'Metric':<40s} {'ROME (L17)':>14s} {'MEND (L47)':>14s}")
    log.info("  " + "-" * 70)

    # PERMANOVA
    rome_r2_rel = rome_res.get("permanova", {}).get("relation", {}).get("R2")
    rome_r2_tgt = rome_res.get("permanova", {}).get("target_value", {}).get("R2")
    mend_r2_rel = results_2.get(47, {}).get("permanova_relation", {}).get("R2")
    mend_r2_tgt = results_2.get(47, {}).get("permanova_target", {}).get("R2")

    if rome_r2_rel is not None and mend_r2_rel is not None:
        log.info(f"  {'PERMANOVA R² (relation)':<40s} {rome_r2_rel:>14.4f} {mend_r2_rel:>14.4f}")
        log.info(f"  {'PERMANOVA R² (target)':<40s} {rome_r2_tgt:>14.4f} {mend_r2_tgt:>14.4f}")

    # LDA
    rome_lda = rome_res.get("lda", {}).get("cv_accuracy")
    mend_lda = results_2.get(47, {}).get("lda_cv_accuracy")
    if rome_lda is not None and mend_lda is not None:
        log.info(f"  {'LDA CV accuracy':<40s} {rome_lda:>14.3f} {mend_lda:>14.3f}")

    # Concept alignment
    rome_align = rome_res.get("triangulation", {}).get("grand_mean_alignment", {}).get("mean_diff_individual_abs")
    mend_align = results_1["grand_summary"].get(47, {}).get("shared_alignment_mean_diff")
    if rome_align is not None and mend_align is not None:
        log.info(f"  {'|cos(v, concept_dir)| mean_diff':<40s} {rome_align:>14.4f} {mend_align:>14.4f}")

    # Efficacy
    rome_eff = 0.98  # known from prior experiments
    mend_eff = np.mean([m["efficacy"] for m in metadata])
    log.info(f"  {'Efficacy':<40s} {rome_eff:>14.3f} {mend_eff:>14.3f}")

    # ============================================================
    # Save all results
    # ============================================================
    all_results = {
        "config": {
            "relations": RELATIONS,
            "mend_edit_layers": MEND_LAYERS,
            "n_edits": len(metadata),
            "permanova_n_perms": PERMANOVA_N_PERMS,
            "seed": SEED,
        },
        "phase_1_alignment": results_1,
        "phase_2_clustering": results_2,
        "phase_3_propagation": {
            "summary": results_3["summary"],
            # per_edit too large for JSON, saved separately
        },
        "efficacy": {
            "mean": float(np.mean([m["efficacy"] for m in metadata])),
            "per_edit": [m["efficacy"] for m in metadata],
        },
    }

    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"\nResults saved to {RESULTS_DIR}/")

    elapsed = time.time() - t_start
    log.info(f"\n{'=' * 70}")
    log.info(f"EXPERIMENT 8 COMPLETE — {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    log.info(f"{'=' * 70}")


if __name__ == "__main__":
    main()
