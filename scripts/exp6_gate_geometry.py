"""
Experiment 6: Gate Vector (u) Concept Structure

Analyzes whether ROME's u vector (the gate that determines *when* an edit fires)
has concept structure. The edit effect is (u · k) × v — u modulates the magnitude
of the v injection for each input key k.

Three phases:
A — u-Space Geometry: PERMANOVA, LDA, within/between cosine similarity on u vectors
B — Key-Space Concept Directions: extract concept directions in MLP key space (R^6400),
    measure u alignment with key-space concept dirs, compare raw k vs whitened u = C⁻¹k
C — Gate Activation Profiling: for each edit's u, compute u·k for same-concept vs
    different-concept test entities; measure d-prime, ROC-AUC

Saves results to results/exp6_gate/
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
from scipy import stats as scipy_stats

# Logging setup — writes to file to bypass conda run's pipe buffering
_log_path = Path("results/exp6_gate/exp6.log")
_log_path.parent.mkdir(parents=True, exist_ok=True)
_fh = logging.FileHandler(_log_path, mode="w")
_fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
logging.basicConfig(level=logging.INFO, handlers=[_fh, _sh])
log = logging.getLogger(__name__)

# Threading safety
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

SEED = 42
EDIT_LAYER = 17
RESULTS_DIR = Path("results/exp6_gate")
EXP1_DIR = Path("results/exp1")
EXP2_DIR = Path("results/exp2")
EXP5_DIR = Path("results/exp5_layer_sweep")
PERMANOVA_N_PERMS = 10000
LDA_N_FOLDS = 5
# For Phase C: relations and number of test entities per concept
PHASE_C_RELATIONS = ["P176", "P27", "P37", "P1412", "P413"]
N_TEST_PER_CONCEPT = 50
BATCH_SIZE = 32

# ============================================================
# Helpers
# ============================================================

def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def permanova(D_sq, labels, n_perms=10000, seed=42):
    """
    PERMANOVA (Anderson 2001) on a squared distance matrix.
    Returns (pseudo_F, p_value, R2).
    Identical to exp2 implementation.
    """
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
    for i in range(n_perms):
        f_perm, _ = compute_stats(rng.permutation(labels))
        if f_perm >= f_obs:
            count += 1

    elapsed = time.time() - t0
    log.info(f"  PERMANOVA: {n_perms} perms in {elapsed:.1f}s (k={k}, n={n})")

    p_value = (count + 1) / (n_perms + 1)
    return float(f_obs), float(p_value), float(r2_obs)


def sq_euclidean_dist_matrix(X):
    """Compute squared Euclidean distance matrix from row vectors."""
    X = np.asarray(X, dtype=np.float64)
    norms_sq = (X ** 2).sum(axis=1)
    D_sq = norms_sq[:, None] + norms_sq[None, :] - 2 * X @ X.T
    D_sq = np.maximum(D_sq, 0)
    return D_sq


def d_prime(same, diff):
    """Signal detection d-prime."""
    mu_s, mu_d = np.mean(same), np.mean(diff)
    var_s, var_d = np.var(same, ddof=1), np.var(diff, ddof=1)
    n_s, n_d = len(same), len(diff)
    sigma_pooled = np.sqrt(((n_s - 1) * var_s + (n_d - 1) * var_d) / (n_s + n_d - 2))
    if sigma_pooled < 1e-10:
        return 0.0
    return float((mu_s - mu_d) / sigma_pooled)


# ============================================================
# Phase A: u-Space Geometry
# ============================================================

def phase_a(u_vectors, metadata):
    """Analyze u-vector clustering by relation and target value."""
    log.info("=" * 70)
    log.info("PHASE A: u-Space Geometry")
    log.info("=" * 70)
    t0 = time.time()

    U = u_vectors.numpy().astype(np.float64)
    n = len(U)

    # Build label arrays
    relations = np.array([m["relation_id"] for m in metadata])
    targets = np.array([f"{m['relation_id']}_{m['target_value']}" for m in metadata])
    subjects = np.array([m["subject"] for m in metadata])

    # ---- PERMANOVA ----
    log.info(f"Computing squared distance matrix ({n}x{n})...")
    D_sq = sq_euclidean_dist_matrix(U)
    log.info(f"Distance matrix done, shape={D_sq.shape}")

    log.info("PERMANOVA (relation)...")
    f_rel, p_rel, r2_rel = permanova(D_sq, relations, PERMANOVA_N_PERMS, SEED)
    log.info(f"  R2={r2_rel:.4f}, F={f_rel:.2f}, p={p_rel:.6f}")

    log.info("PERMANOVA (target_value)...")
    f_tgt, p_tgt, r2_tgt = permanova(D_sq, targets, PERMANOVA_N_PERMS, SEED)
    log.info(f"  R2={r2_tgt:.4f}, F={f_tgt:.2f}, p={p_tgt:.6f}")

    # ---- LDA ----
    log.info("LDA classification by relation...")
    unique_rels = np.unique(relations)
    if len(unique_rels) >= 2:
        skf = StratifiedKFold(n_splits=LDA_N_FOLDS, shuffle=True, random_state=SEED)
        fold_accs = []
        for train_idx, test_idx in skf.split(U, relations):
            lda = LinearDiscriminantAnalysis()
            lda.fit(U[train_idx], relations[train_idx])
            pred = lda.predict(U[test_idx])
            acc = (pred == relations[test_idx]).mean()
            fold_accs.append(acc)
        lda_acc = np.mean(fold_accs)
        log.info(f"  {LDA_N_FOLDS}-fold CV accuracy: {lda_acc:.3f}")

        # Fit full LDA for discriminant directions
        lda_full = LinearDiscriminantAnalysis()
        lda_full.fit(U, relations)
    else:
        lda_acc = 0.0
        lda_full = None

    # ---- Within/between cosine similarity ----
    log.info("Within/between cosine similarity...")
    # Normalize for cosine
    U_norm = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-10)

    # Random pair sampling (data is ordered by relation, so sequential
    # sampling would massively oversample within-relation pairs)
    rng_cos = np.random.RandomState(SEED)
    n_pairs = 20000
    within_cos = []
    between_cos = []
    for _ in range(n_pairs):
        i, j = rng_cos.choice(n, 2, replace=False)
        c = float(U_norm[i] @ U_norm[j])
        if relations[i] == relations[j]:
            within_cos.append(c)
        else:
            between_cos.append(c)

    within_mean = np.mean(within_cos) if within_cos else 0
    between_mean = np.mean(between_cos) if between_cos else 0
    log.info(f"  Within-relation mean |cos|: {np.mean(np.abs(within_cos)):.4f} (signed: {within_mean:.4f}), n={len(within_cos)}")
    log.info(f"  Between-relation mean |cos|: {np.mean(np.abs(between_cos)):.4f} (signed: {between_mean:.4f}), n={len(between_cos)}")

    elapsed = time.time() - t0
    log.info(f"Phase A complete in {elapsed:.0f}s")

    return {
        "permanova_relation": {"R2": r2_rel, "F": f_rel, "p": p_rel},
        "permanova_target": {"R2": r2_tgt, "F": f_tgt, "p": p_tgt},
        "lda_cv_accuracy": lda_acc,
        "within_relation_cos_abs": float(np.mean(np.abs(within_cos))) if within_cos else 0,
        "between_relation_cos_abs": float(np.mean(np.abs(between_cos))) if between_cos else 0,
        "within_relation_cos_signed": float(within_mean),
        "between_relation_cos_signed": float(between_mean),
        "n_vectors": n,
        "n_relations": len(unique_rels),
    }


# ============================================================
# Phase B: Key-Space Concept Directions
# ============================================================

def extract_mlp_keys(model, tok, records, layer=EDIT_LAYER, batch_size=BATCH_SIZE):
    """
    Extract MLP key vectors (input to c_proj) at the subject's last token position.
    Uses ROME's own get_words_idxs_in_templates for correct BPE-aware token finding.
    Returns tensor of shape (n_records, 6400).
    """
    from util.nethook import Trace
    from rome.repr_tools import get_words_idxs_in_templates

    device = next(model.parameters()).device
    hook_layer = f"transformer.h.{layer}.mlp.c_proj"
    keys = []
    n_fallback = 0

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        templates = [rec["prompt"] for rec in batch]
        subjects = [rec["subject"] for rec in batch]

        # Use ROME's own token index finder (handles BPE correctly)
        try:
            idxs = get_words_idxs_in_templates(tok, templates, subjects, "last")
            subject_last_indices = [idx_list[0] for idx_list in idxs]
        except Exception as e:
            # Fallback: use simple heuristic
            n_fallback += len(batch)
            prompts = [t.replace("{}", s) for t, s in zip(templates, subjects)]
            inputs_tmp = tok(prompts, return_tensors="pt", padding=True)
            subject_last_indices = []
            for b_idx in range(len(batch)):
                # Approximate: last non-padding token before suffix
                mask = inputs_tmp["attention_mask"][b_idx]
                subject_last_indices.append(int(mask.sum()) - 1)

        # Tokenize full prompts
        prompts = [t.replace("{}", s) for t, s in zip(templates, subjects)]
        inputs = tok(prompts, return_tensors="pt", padding=True).to(device)

        # Forward pass with hook to capture c_proj input
        with torch.no_grad():
            with Trace(model, hook_layer, retain_input=True, retain_output=False) as tr:
                model(**inputs)
                # tr.input is the input to c_proj: shape (batch, seq_len, 6400)
                inp = tr.input
                if isinstance(inp, tuple):
                    inp = inp[0]

        # Extract at subject last position
        for b_idx in range(len(batch)):
            pos = subject_last_indices[b_idx]
            key_vec = inp[b_idx, pos].detach().cpu().float()
            keys.append(key_vec)

    if n_fallback > 0:
        log.info(f"    WARNING: {n_fallback}/{len(records)} used fallback token finding")

    return torch.stack(keys)  # (n, 6400)


def phase_b(model, tok, u_vectors, metadata, exp1_meta, exp1_activations):
    """Extract concept directions in key space and measure u alignment."""
    log.info("\n" + "=" * 70)
    log.info("PHASE B: Key-Space Concept Directions")
    log.info("=" * 70)
    t0 = time.time()

    device = next(model.parameters()).device
    U = u_vectors.numpy().astype(np.float64)

    # Group exp1 records by (relation, target) for probe training
    # We need MLP keys for the same entities used in concept direction extraction
    results_b = {}

    # Identify which relations/targets are in the edit set
    edit_concepts = set()
    for m in metadata:
        edit_concepts.add((m["relation_id"], m["target_value"]))

    # For each relation in exp1, extract MLP keys and train key-space probes
    key_space_directions = {}
    key_space_probe_accs = {}
    raw_key_alignments = {}
    whitened_alignments = {}

    for rel_id, records in exp1_meta.items():
        if rel_id not in [m["relation_id"] for m in metadata]:
            continue

        log.info(f"\n  Relation {rel_id}: extracting MLP keys for {len(records)} entities...")

        # Prepare records for key extraction
        key_records = []
        for rec in records:
            key_records.append({
                "subject": rec["subject"],
                "prompt": rec["prompt"],
                "target_new": rec["target_new"],
            })

        # Extract MLP keys
        mlp_keys = extract_mlp_keys(model, tok, key_records, layer=EDIT_LAYER)
        K = mlp_keys.numpy().astype(np.float64)

        # Get target values for this relation
        target_values = [rec["target_new"] for rec in records]

        # For each target concept, build binary labels and train probe
        from collections import Counter
        target_counts = Counter(target_values)
        top_targets = [t for t, c in target_counts.most_common() if c >= 10]

        for target in top_targets[:5]:  # up to 5 targets per relation
            concept_key = f"{rel_id}_{target}"
            labels = np.array([1 if t == target else 0 for t in target_values])

            n_pos = labels.sum()
            n_neg = len(labels) - n_pos
            if n_pos < 5 or n_neg < 5:
                continue

            # Mean-diff direction in key space
            d_mean_diff = K[labels == 1].mean(axis=0) - K[labels == 0].mean(axis=0)
            d_mean_diff_norm = d_mean_diff / (np.linalg.norm(d_mean_diff) + 1e-10)

            # Logistic probe in key space (use CV accuracy, not in-sample)
            try:
                from sklearn.model_selection import cross_val_score
                probe = LogisticRegressionCV(
                    cv=3, max_iter=500, random_state=SEED,
                    solver="saga", penalty="l2"
                )
                # Get CV accuracy BEFORE fitting on all data
                cv_scores = cross_val_score(
                    LogisticRegressionCV(cv=3, max_iter=500, random_state=SEED,
                                        solver="saga", penalty="l2"),
                    K, labels, cv=3, scoring="balanced_accuracy"
                )
                bal_acc = float(np.mean(cv_scores))
                # Now fit on all data to get the direction vector
                probe.fit(K, labels)
                d_logistic = probe.coef_[0].copy()
                d_logistic_norm = d_logistic / (np.linalg.norm(d_logistic) + 1e-10)
            except Exception as e:
                log.info(f"    WARNING: logistic probe failed for {concept_key}: {e}")
                bal_acc = 0.5
                d_logistic_norm = d_mean_diff_norm  # fallback

            key_space_directions[f"{concept_key}_mean_diff"] = d_mean_diff_norm
            key_space_directions[f"{concept_key}_logistic"] = d_logistic_norm
            key_space_probe_accs[concept_key] = bal_acc

            log.info(f"    {concept_key}: probe acc={bal_acc:.3f}, n_pos={n_pos}, n_neg={n_neg}")

            # Compute u alignment with key-space concept directions
            # Only for edits targeting this concept
            concept_u_indices = [
                i for i, m in enumerate(metadata)
                if m["relation_id"] == rel_id and m["target_value"] == target
            ]
            if concept_u_indices:
                U_concept = U[concept_u_indices]
                align_md = [abs(cosine(u, d_mean_diff_norm)) for u in U_concept]
                align_lg = [abs(cosine(u, d_logistic_norm)) for u in U_concept]
                whitened_alignments[concept_key] = {
                    "mean_diff": float(np.mean(align_md)),
                    "logistic": float(np.mean(align_lg)),
                    "n": len(concept_u_indices),
                }

            # Also compute raw k alignment (before C⁻¹ whitening)
            # Raw keys for entities that were edited
            raw_concept_keys = []
            for idx in concept_u_indices:
                subj = metadata[idx]["subject"]
                # Find this subject in exp1 records
                for r_idx, rec in enumerate(records):
                    if rec["subject"] == subj:
                        raw_concept_keys.append(K[r_idx])
                        break

            if raw_concept_keys:
                raw_align_md = [abs(cosine(k, d_mean_diff_norm)) for k in raw_concept_keys]
                raw_align_lg = [abs(cosine(k, d_logistic_norm)) for k in raw_concept_keys]
                raw_key_alignments[concept_key] = {
                    "mean_diff": float(np.mean(raw_align_md)),
                    "logistic": float(np.mean(raw_align_lg)),
                    "n": len(raw_concept_keys),
                }

    elapsed = time.time() - t0
    log.info(f"\n  Phase B complete in {elapsed:.0f}s")

    return {
        "key_space_probe_accuracies": key_space_probe_accs,
        "u_alignment_with_key_dirs": whitened_alignments,
        "raw_k_alignment_with_key_dirs": raw_key_alignments,
        "key_space_directions": key_space_directions,  # saved separately
    }


# ============================================================
# Phase C: Gate Activation Profiling
# ============================================================

def phase_c(model, tok, u_vectors, metadata):
    """
    For each edit's u vector, compute gate activation u·k for test entities
    from same vs different concepts. Measure d-prime and ROC-AUC.
    """
    log.info("\n" + "=" * 70)
    log.info("PHASE C: Gate Activation Profiling")
    log.info("=" * 70)
    t0 = time.time()

    device = next(model.parameters()).device
    U = u_vectors.numpy().astype(np.float64)

    # Load CounterFact to get test entities
    from dsets import CounterFactDataset
    cf = CounterFactDataset("data")
    cf_by_relation = defaultdict(list)
    for rec in cf:
        rid = rec["requested_rewrite"]["relation_id"]
        cf_by_relation[rid].append(rec)

    # Identify entities already used in exp2 edits
    used_subjects = set(m["subject"] for m in metadata)
    used_case_ids = set(m["case_id"] for m in metadata)

    # Select test entities: N_TEST_PER_CONCEPT per (relation, target)
    rng = np.random.RandomState(SEED + 100)
    test_entities = {}  # (rel, target) -> list of records
    all_test_records = []

    for rel_id in PHASE_C_RELATIONS:
        # Get target values used in edits
        edit_targets = set(
            m["target_value"] for m in metadata if m["relation_id"] == rel_id
        )
        for target in edit_targets:
            # Find entities with this target that weren't in exp2
            candidates = [
                rec for rec in cf_by_relation[rel_id]
                if rec["requested_rewrite"]["target_new"]["str"] == target
                and rec["requested_rewrite"]["subject"] not in used_subjects
                and rec["case_id"] not in used_case_ids
            ]
            rng.shuffle(candidates)
            selected = candidates[:N_TEST_PER_CONCEPT]
            test_entities[(rel_id, target)] = selected
            all_test_records.extend(selected)
            log.info(f"  Test entities for {rel_id}_{target}: {len(selected)}")

    if not all_test_records:
        log.info("  WARNING: No test entities found!")
        return {}

    # Extract MLP keys for all test entities
    log.info(f"\n  Extracting MLP keys for {len(all_test_records)} test entities...")
    test_key_records = []
    for rec in all_test_records:
        rw = rec["requested_rewrite"]
        test_key_records.append({
            "subject": rw["subject"],
            "prompt": rw["prompt"],
            "target_new": rw["target_new"]["str"],
        })

    test_keys = extract_mlp_keys(model, tok, test_key_records, layer=EDIT_LAYER)
    K_test = test_keys.numpy().astype(np.float64)

    # Build concept labels for test entities
    test_concepts = []
    test_relations = []
    idx = 0
    for (rel_id, target), records in test_entities.items():
        for _ in records:
            test_concepts.append(f"{rel_id}_{target}")
            test_relations.append(rel_id)
            idx += 1
    test_concepts = np.array(test_concepts)
    test_relations = np.array(test_relations)

    # Also extract MLP keys for the edited entities themselves (for own-entity control)
    log.info("  Extracting MLP keys for edited entities...")
    edit_key_records = []
    for m in metadata:
        edit_key_records.append({
            "subject": m["subject"],
            "prompt": m["prompt"],
            "target_new": m["target_value"],
        })
    # Only do this for the 5 relations we're analyzing
    edit_indices_by_concept = defaultdict(list)
    for i, m in enumerate(metadata):
        if m["relation_id"] in PHASE_C_RELATIONS:
            concept = f"{m['relation_id']}_{m['target_value']}"
            edit_indices_by_concept[concept].append(i)

    # Extract keys for relevant edits only
    relevant_edit_indices = []
    for indices in edit_indices_by_concept.values():
        relevant_edit_indices.extend(indices)
    relevant_edit_indices = sorted(set(relevant_edit_indices))

    rel_edit_records = [edit_key_records[i] for i in relevant_edit_indices]
    if rel_edit_records:
        edit_keys = extract_mlp_keys(model, tok, rel_edit_records, layer=EDIT_LAYER)
        K_edit = edit_keys.numpy().astype(np.float64)
    else:
        K_edit = np.zeros((0, 6400))

    # Map from relevant_edit_indices position to K_edit row
    edit_key_map = {orig_idx: new_idx for new_idx, orig_idx in enumerate(relevant_edit_indices)}

    # Build reverse lookup: concept string -> (rel_id, target) tuple
    concept_to_tuple = {}
    for (rel_id, target) in test_entities.keys():
        concept_to_tuple[f"{rel_id}_{target}"] = (rel_id, target)

    # Compute gate activations
    log.info("\n  Computing gate activations...")
    all_results = []
    per_concept_results = defaultdict(list)

    for concept, edit_indices in edit_indices_by_concept.items():
        if concept not in concept_to_tuple:
            continue
        rel_id, target = concept_to_tuple[concept]

        # Three-way split to control for prompt template effects:
        # 1. same_concept: same (relation, target) — same prompt template, same target
        # 2. same_rel_diff_target: same relation, different target — same prompt template, diff target
        # 3. diff_rel: different relation — different prompt template
        same_concept_mask = test_concepts == concept
        same_rel_mask = test_relations == rel_id
        same_rel_diff_target_mask = same_rel_mask & ~same_concept_mask
        diff_rel_mask = ~same_rel_mask

        n_same = same_concept_mask.sum()
        n_same_rel = same_rel_diff_target_mask.sum()
        n_diff_rel = diff_rel_mask.sum()

        if n_same < 3:
            continue

        K_same = K_test[same_concept_mask]
        K_same_rel = K_test[same_rel_diff_target_mask] if n_same_rel > 0 else np.zeros((0, K_test.shape[1]))
        K_diff_rel = K_test[diff_rel_mask] if n_diff_rel > 0 else np.zeros((0, K_test.shape[1]))
        K_all_diff = K_test[~same_concept_mask]  # for overall d-prime

        for edit_idx in edit_indices:
            u = U[edit_idx]

            # Gate activation for each group
            act_same = K_same @ u
            act_same_rel = K_same_rel @ u if len(K_same_rel) > 0 else np.array([])
            act_diff_rel = K_diff_rel @ u if len(K_diff_rel) > 0 else np.array([])
            act_all_diff = K_all_diff @ u

            # Own-entity gate activation
            if edit_idx in edit_key_map:
                own_key = K_edit[edit_key_map[edit_idx]]
                act_own = float(own_key @ u)
            else:
                act_own = None

            # d-prime: same-concept vs ALL different (overall selectivity)
            dp_overall = d_prime(act_same, act_all_diff)

            # d-prime: same-concept vs same-relation-different-target
            # (controls for prompt template — pure concept selectivity)
            dp_within_rel = d_prime(act_same, act_same_rel) if len(act_same_rel) >= 3 else None

            # d-prime: same-relation vs different-relation
            # (measures prompt template effect)
            if len(act_same_rel) >= 3 and len(act_diff_rel) >= 3:
                act_full_rel = np.concatenate([act_same, act_same_rel])
                dp_template = d_prime(act_full_rel, act_diff_rel)
            else:
                dp_template = None

            # ROC-AUC (overall)
            all_act = np.concatenate([act_same, act_all_diff])
            all_labels = np.concatenate([np.ones(len(act_same)), np.zeros(len(act_all_diff))])
            try:
                auc = roc_auc_score(all_labels, all_act)
            except:
                auc = 0.5

            result = {
                "concept": concept,
                "relation_id": rel_id,
                "subject": metadata[edit_idx]["subject"],
                "d_prime_overall": dp_overall,
                "d_prime_within_relation": dp_within_rel,
                "d_prime_template_effect": dp_template,
                "roc_auc": auc,
                "mean_act_same_concept": float(np.mean(act_same)),
                "mean_act_same_rel_diff_target": float(np.mean(act_same_rel)) if len(act_same_rel) > 0 else None,
                "mean_act_diff_rel": float(np.mean(act_diff_rel)) if len(act_diff_rel) > 0 else None,
                "std_act_same": float(np.std(act_same)),
                "own_entity_act": act_own,
                "n_same_concept": int(n_same),
                "n_same_rel_diff_target": int(n_same_rel),
                "n_diff_rel": int(n_diff_rel),
            }
            all_results.append(result)
            per_concept_results[concept].append(result)

    # Aggregate
    log.info("\n  Gate activation profiling results (three-way comparison):")
    log.info(f"  {'Concept':<20s} {'d_overall':>9s} {'d_within':>9s} {'d_templ':>9s} "
          f"{'μ_same':>8s} {'μ_sameR':>8s} {'μ_diffR':>8s} {'μ_own':>8s}")
    log.info("  " + "-" * 85)

    concept_summaries = {}
    for concept in sorted(per_concept_results.keys()):
        res_list = per_concept_results[concept]
        dp_overall = [r["d_prime_overall"] for r in res_list]
        dp_within = [r["d_prime_within_relation"] for r in res_list if r["d_prime_within_relation"] is not None]
        dp_template = [r["d_prime_template_effect"] for r in res_list if r["d_prime_template_effect"] is not None]
        auc_vals = [r["roc_auc"] for r in res_list]
        same_vals = [r["mean_act_same_concept"] for r in res_list]
        same_rel_vals = [r["mean_act_same_rel_diff_target"] for r in res_list if r["mean_act_same_rel_diff_target"] is not None]
        diff_rel_vals = [r["mean_act_diff_rel"] for r in res_list if r["mean_act_diff_rel"] is not None]
        own_vals = [r["own_entity_act"] for r in res_list if r["own_entity_act"] is not None]

        summary = {
            "d_prime_overall_mean": float(np.mean(dp_overall)),
            "d_prime_within_rel_mean": float(np.mean(dp_within)) if dp_within else None,
            "d_prime_template_mean": float(np.mean(dp_template)) if dp_template else None,
            "auc_mean": float(np.mean(auc_vals)),
            "mean_act_same_concept": float(np.mean(same_vals)),
            "mean_act_same_rel_diff_target": float(np.mean(same_rel_vals)) if same_rel_vals else None,
            "mean_act_diff_rel": float(np.mean(diff_rel_vals)) if diff_rel_vals else None,
            "mean_act_own": float(np.mean(own_vals)) if own_vals else None,
            "n_edits": len(res_list),
        }
        concept_summaries[concept] = summary

        dw_str = f"{summary['d_prime_within_rel_mean']:9.3f}" if summary["d_prime_within_rel_mean"] is not None else "      N/A"
        dt_str = f"{summary['d_prime_template_mean']:9.3f}" if summary["d_prime_template_mean"] is not None else "      N/A"
        sr_str = f"{summary['mean_act_same_rel_diff_target']:8.3f}" if summary["mean_act_same_rel_diff_target"] is not None else "     N/A"
        dr_str = f"{summary['mean_act_diff_rel']:8.3f}" if summary["mean_act_diff_rel"] is not None else "     N/A"
        own_str = f"{summary['mean_act_own']:8.1f}" if summary["mean_act_own"] is not None else "     N/A"
        log.info(f"  {concept:<20s} {summary['d_prime_overall_mean']:9.3f} {dw_str} {dt_str} "
              f"{summary['mean_act_same_concept']:8.3f} {sr_str} {dr_str} {own_str}")

    # Grand averages
    all_dp_overall = [r["d_prime_overall"] for r in all_results]
    all_dp_within = [r["d_prime_within_relation"] for r in all_results if r["d_prime_within_relation"] is not None]
    all_dp_template = [r["d_prime_template_effect"] for r in all_results if r["d_prime_template_effect"] is not None]
    all_auc = [r["roc_auc"] for r in all_results]
    all_own = [r["own_entity_act"] for r in all_results if r["own_entity_act"] is not None]
    grand = {
        "d_prime_overall_mean": float(np.mean(all_dp_overall)),
        "d_prime_overall_std": float(np.std(all_dp_overall)),
        "d_prime_within_rel_mean": float(np.mean(all_dp_within)) if all_dp_within else None,
        "d_prime_within_rel_std": float(np.std(all_dp_within)) if all_dp_within else None,
        "d_prime_template_mean": float(np.mean(all_dp_template)) if all_dp_template else None,
        "d_prime_template_std": float(np.std(all_dp_template)) if all_dp_template else None,
        "auc_mean": float(np.mean(all_auc)),
        "auc_std": float(np.std(all_auc)),
        "own_entity_act_mean": float(np.mean(all_own)) if all_own else None,
        "n_total": len(all_results),
    }

    log.info(f"\n  Grand averages:")
    log.info(f"    d'(overall):        {grand['d_prime_overall_mean']:.3f} ± {grand['d_prime_overall_std']:.3f}")
    if grand["d_prime_within_rel_mean"] is not None:
        log.info(f"    d'(within-relation): {grand['d_prime_within_rel_mean']:.3f} ± {grand['d_prime_within_rel_std']:.3f}  ← concept selectivity (prompt-controlled)")
    if grand["d_prime_template_mean"] is not None:
        log.info(f"    d'(template effect): {grand['d_prime_template_mean']:.3f} ± {grand['d_prime_template_std']:.3f}  ← prompt template effect")
    log.info(f"    AUC:                {grand['auc_mean']:.3f} ± {grand['auc_std']:.3f}")
    if grand["own_entity_act_mean"] is not None:
        all_same_mean = np.mean([r["mean_act_same_concept"] for r in all_results])
        log.info(f"    Own-entity act:     {grand['own_entity_act_mean']:.3f} vs same-concept: {all_same_mean:.3f} "
              f"(ratio: {grand['own_entity_act_mean'] / (abs(all_same_mean) + 1e-10):.1f}×)")

    elapsed = time.time() - t0
    log.info(f"\n  Phase C complete in {elapsed:.0f}s")

    return {
        "grand_summary": grand,
        "per_concept": concept_summaries,
        "per_edit": all_results,
    }


# ============================================================
# Cross-layer u analysis (bonus from Exp 5 data)
# ============================================================

def cross_layer_u_analysis():
    """Quick check: does u-space concept structure vary by layer?"""
    log.info("\n" + "=" * 70)
    log.info("BONUS: Cross-Layer u-Space Structure (from Exp 5)")
    log.info("=" * 70)

    exp5_path = EXP5_DIR / "edit_vectors_by_layer.pt"
    if not exp5_path.exists():
        log.info("  Exp 5 data not found, skipping.")
        return {}

    data = torch.load(exp5_path, map_location="cpu")
    results = {}

    for layer in sorted(data.keys()):
        layer_data = data[layer]
        U = layer_data["u"].numpy().astype(np.float64)
        meta = layer_data["meta"]
        if len(U) < 20:
            continue

        relations = np.array([m["relation_id"] for m in meta])
        unique_rels = np.unique(relations)
        if len(unique_rels) < 2:
            continue

        # Quick LDA
        try:
            skf = StratifiedKFold(n_splits=min(5, min(np.bincount(
                [list(unique_rels).index(r) for r in relations]))),
                shuffle=True, random_state=SEED)
            fold_accs = []
            for train_idx, test_idx in skf.split(U, relations):
                lda = LinearDiscriminantAnalysis()
                lda.fit(U[train_idx], relations[train_idx])
                pred = lda.predict(U[test_idx])
                fold_accs.append((pred == relations[test_idx]).mean())
            lda_acc = np.mean(fold_accs)
        except:
            lda_acc = 0.0

        # Within/between cosine (quick sample)
        U_norm = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-10)
        rng = np.random.RandomState(SEED)
        n = len(U)
        n_pairs = min(2000, n * (n - 1) // 2)
        within, between = [], []
        for _ in range(n_pairs):
            i, j = rng.choice(n, 2, replace=False)
            c = abs(float(U_norm[i] @ U_norm[j]))
            if relations[i] == relations[j]:
                within.append(c)
            else:
                between.append(c)

        results[layer] = {
            "lda_accuracy": lda_acc,
            "within_cos_abs": float(np.mean(within)) if within else 0,
            "between_cos_abs": float(np.mean(between)) if between else 0,
            "n_vectors": n,
        }
        log.info(f"  Layer {layer:2d}: LDA={lda_acc:.3f}, within |cos|={results[layer]['within_cos_abs']:.4f}, "
              f"between |cos|={results[layer]['between_cos_abs']:.4f}")

    return results


# ============================================================
# Main
# ============================================================

def main():
    log.info("=" * 70)
    log.info("EXPERIMENT 6: GATE VECTOR (u) CONCEPT STRUCTURE")
    log.info("=" * 70)
    t_start = time.time()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load exp2 edit vectors
    log.info("\nLoading edit vectors from Exp 2...")
    edit_data = torch.load(EXP2_DIR / "edit_vectors.pt", map_location="cpu")
    u_vectors = edit_data["u"]  # (500, 6400)
    v_vectors = edit_data["v"]  # (500, 1600)
    metadata = edit_data["metadata"]
    log.info(f"  {len(metadata)} edits, u: {u_vectors.shape}, v: {v_vectors.shape}")

    # Load exp1 metadata
    log.info("Loading Exp 1 metadata...")
    with open(EXP1_DIR / "record_metadata.json") as f:
        exp1_meta = json.load(f)

    # Phase A: u-space geometry (CPU only)
    results_a = phase_a(u_vectors, metadata)

    # Phase B and C need the model
    log.info("\nLoading GPT-2 XL...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained("gpt2-xl").cuda().eval()
    tok = AutoTokenizer.from_pretrained("gpt2-xl")
    tok.pad_token = tok.eos_token

    # Phase B: key-space concept directions
    results_b = phase_b(model, tok, u_vectors, metadata, exp1_meta, None)

    # Save key-space directions separately
    if "key_space_directions" in results_b:
        ksd = {k: torch.tensor(v) for k, v in results_b["key_space_directions"].items()}
        torch.save(ksd, RESULTS_DIR / "key_space_concept_directions.pt")
        # Remove from JSON-serializable results
        results_b_json = {k: v for k, v in results_b.items() if k != "key_space_directions"}
    else:
        results_b_json = results_b

    # Phase C: gate activation profiling
    results_c = phase_c(model, tok, u_vectors, metadata)

    # Bonus: cross-layer u analysis
    results_cross = cross_layer_u_analysis()

    # ============================================================
    # Summary comparison with v-space (from Exp 2)
    # ============================================================
    log.info("\n" + "=" * 70)
    log.info("COMPARISON: u-Space vs v-Space Geometry")
    log.info("=" * 70)

    # Load exp2 results for comparison
    exp2_results_path = EXP2_DIR / "results.json"
    if exp2_results_path.exists():
        with open(exp2_results_path) as f:
            exp2_res = json.load(f)
    else:
        exp2_res = {}

    log.info(f"\n  {'Metric':<35s} {'v-space (Exp2)':>14s} {'u-space (Exp6)':>14s}")
    log.info("  " + "-" * 65)

    # PERMANOVA R² comparison
    v_r2_rel = exp2_res.get("permanova", {}).get("relation", {}).get("R2", "N/A")
    v_r2_tgt = exp2_res.get("permanova", {}).get("target_value", {}).get("R2", "N/A")
    u_r2_rel = results_a["permanova_relation"]["R2"]
    u_r2_tgt = results_a["permanova_target"]["R2"]

    if isinstance(v_r2_rel, float):
        log.info(f"  {'PERMANOVA R² (relation)':<35s} {v_r2_rel:>14.4f} {u_r2_rel:>14.4f}")
        log.info(f"  {'PERMANOVA R² (target_value)':<35s} {v_r2_tgt:>14.4f} {u_r2_tgt:>14.4f}")
    else:
        log.info(f"  {'PERMANOVA R² (relation)':<35s} {'N/A':>14s} {u_r2_rel:>14.4f}")
        log.info(f"  {'PERMANOVA R² (target_value)':<35s} {'N/A':>14s} {u_r2_tgt:>14.4f}")

    v_lda = exp2_res.get("lda", {}).get("cv_accuracy", "N/A")
    u_lda = results_a["lda_cv_accuracy"]
    if isinstance(v_lda, float):
        log.info(f"  {'LDA CV accuracy':<35s} {v_lda:>14.3f} {u_lda:>14.3f}")
    else:
        log.info(f"  {'LDA CV accuracy':<35s} {'N/A':>14s} {u_lda:>14.3f}")

    # ============================================================
    # Save all results
    # ============================================================
    all_results = {
        "config": {
            "edit_layer": EDIT_LAYER,
            "n_edits": len(metadata),
            "n_test_per_concept": N_TEST_PER_CONCEPT,
            "permanova_n_perms": PERMANOVA_N_PERMS,
            "seed": SEED,
        },
        "phase_a": results_a,
        "phase_b": results_b_json,
        "phase_c": results_c,
        "cross_layer_u": results_cross,
    }

    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"\n  Results saved to {RESULTS_DIR}/")

    elapsed = time.time() - t_start
    log.info(f"\n{'=' * 70}")
    log.info(f"EXPERIMENT 6 COMPLETE — {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    log.info(f"{'=' * 70}")


if __name__ == "__main__":
    main()