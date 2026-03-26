"""
Experiment 2: Edit Vector Geometry Across Relations

Tests whether ROME's v vectors cluster by relation, and whether that
clustering reflects concept structure via PERMANOVA, LDA, three-level
variance decomposition, and LRE triangulation.

Phases:
A — PERMANOVA: Does structure exist? (relation vs confound groupings)
B — LDA: What are the discriminant directions? (gated on Phase A)
C — Three-level variance decomposition (relation / concept / entity)
D — LRE triangulation (concept dir, LDA dir, LRE dir, v_mean)

Updates from Experiments 1 and 1B:
- All 10 main relations from Exp 1
- DAS direction dropped (explained variance 0.03-0.09 at layer 17)
- Direction methods: mean_diff and logistic only
- Context templates reused from results/exp1/context_templates.json
- Exp 1B: weak concept-direction alignment (median |cos|~0.11) but strong
  cross-entity transfer (ratio ~0.87). PERMANOVA should detect clustering;
  LDA may find directions that concept probes missed.

Saves results to results/exp2/
"""

import json
import os
import sys
import time
import numpy as np
import torch
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from rome.rome_main import execute_rome
from rome.rome_hparams import ROMEHyperParams
from rome.repr_tools import get_words_idxs_in_templates
from util import nethook
from util.globals import HPARAMS_DIR
import rome.rome_main as rome_main_module

# ============================================================
# Configuration — updated from Exp 1/1B
# ============================================================
ALL_RELATIONS = [
    "P176", "P27", "P495", "P37", "P17",
    "P413", "P1412", "P937", "P106", "P449",
]
N_EDITS_PER_RELATION = 50
MAX_TARGET_FRACTION = 0.40          # max 20 per target in 50
MIN_ENTITIES_PER_TARGET_C = 8       # for Phase C target-level test
EDIT_LAYER = 17
LRE_SUBJECT_LAYER = 17              # at ROME's edit layer
LRE_OBJECT_LAYER = 47               # at the loss layer
LRE_N_PAIRS = 200                   # per relation
LRE_RIDGE_ALPHA = 1.0
PERMANOVA_N_PERMS = 10000
PERMANOVA_N_PERMS_WITHIN = 1000     # for Phase C (per-relation)
LDA_N_FOLDS = 5
BOOTSTRAP_N = 100
BATCH_SIZE = 24
SEED = 42
RESULTS_DIR = Path("results/exp2")
EXP1_DIR = Path("results/exp1")
DIRECTION_METHODS = ["mean_diff", "logistic"]


# ============================================================
# Helper functions
# ============================================================

def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def permanova(D_sq, labels, n_perms=10000, seed=42):
    """
    PERMANOVA (Anderson 2001) on a squared distance matrix.
    Returns (pseudo_F, p_value, R2).
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
    for _ in range(n_perms):
        f_perm, _ = compute_stats(rng.permutation(labels))
        if f_perm >= f_obs:
            count += 1

    p_value = (count + 1) / (n_perms + 1)
    return float(f_obs), float(p_value), float(r2_obs)


def select_entities_for_relation(records, n_total=50, max_frac=0.40,
                                 min_per_target=8, seed=42):
    """
    Select entities ensuring target diversity.
    Returns (list_of_(record, target), allocation_dict).
    """
    rng = np.random.RandomState(seed)
    by_target = defaultdict(list)
    for r in records:
        t = r["requested_rewrite"]["target_new"]["str"]
        by_target[t].append(r)

    # Sort by count desc; keep those with >= min_per_target
    ranking = sorted(by_target.items(), key=lambda x: len(x[1]), reverse=True)
    qualifying = [(t, rs) for t, rs in ranking if len(rs) >= min_per_target]
    if len(qualifying) < 3:
        qualifying = [(t, rs) for t, rs in ranking if len(rs) >= 3]

    targets = qualifying[:5]  # at most 5 targets
    max_per = int(max_frac * n_total)  # 20

    # Equal allocation, capped
    n_tgts = len(targets)
    base = min(n_total // n_tgts, max_per)
    alloc = {t: min(base, len(rs)) for t, rs in targets}
    remaining = n_total - sum(alloc.values())

    # Fill remainder round-robin
    for t, rs in targets:
        if remaining <= 0:
            break
        add = min(remaining, max_per - alloc[t], len(rs) - alloc[t])
        if add > 0:
            alloc[t] += add
            remaining -= add

    # Sample unique subjects
    selected = []
    used = set()
    for t, rs in targets:
        n = alloc[t]
        rng.shuffle(rs)
        count = 0
        for r in rs:
            if count >= n:
                break
            subj = r["requested_rewrite"]["subject"]
            if subj not in used:
                selected.append((r, t))
                used.add(subj)
                count += 1

    return selected[:n_total], {t: alloc[t] for t, _ in targets}


def run_rome_edit(model, tok, request, hparams):
    """Run ROME, return (u, v). Model unchanged."""
    deltas = execute_rome(model, tok, request, hparams)
    for key, (u, v) in deltas.items():
        return u.detach().cpu(), v.detach().cpu()


def extract_lre_activations(model, tok, prompt_templates, subjects,
                            batch_size=24):
    """
    Extract subject_last@layer17 and last@layer47 for LRE fitting.
    prompt_templates: list of strings with {} placeholder.
    """
    l17 = "transformer.h.17"
    l47 = "transformer.h.47"
    H_subj_all, H_obj_all = [], []

    for i in range(0, len(prompt_templates), batch_size):
        b_tmpl = prompt_templates[i:i + batch_size]
        b_subj = subjects[i:i + batch_size]

        # Subject-last token index (via repr_tools)
        subj_idxs = get_words_idxs_in_templates(tok, b_tmpl, b_subj,
                                                  subtoken="last")
        filled = [t.format(s) for t, s in zip(b_tmpl, b_subj)]
        inputs = tok(filled, return_tensors="pt", padding=True).to("cuda")

        with torch.no_grad():
            with nethook.TraceDict(model, [l17, l47]) as tr:
                model(**inputs)

        h17 = tr[l17].output[0]   # (batch, seq, 1600)
        h47 = tr[l47].output[0]

        for j in range(len(filled)):
            s_pos = subj_idxs[j][0]
            last_pos = inputs["attention_mask"][j].sum().item() - 1
            s_pos = min(s_pos, last_pos)

            H_subj_all.append(h17[j, s_pos].detach().cpu())
            H_obj_all.append(h47[j, last_pos].detach().cpu())

    return torch.stack(H_subj_all), torch.stack(H_obj_all)


def fit_lre(H_s, H_o, alpha=1.0):
    """Ridge-regression LRE: H_o ≈ H_s @ W^T + b. Returns dict."""
    H_s, H_o = H_s.float(), H_o.float()
    s_mean, o_mean = H_s.mean(0), H_o.mean(0)
    Sc, Oc = H_s - s_mean, H_o - o_mean
    d = Sc.size(1)
    A = Sc.T @ Sc + alpha * torch.eye(d)
    W = torch.linalg.solve(A, Sc.T @ Oc).T  # (d, d)
    b = o_mean - W @ s_mean

    pred = H_s @ W.T + b
    ss_res = ((H_o - pred) ** 2).sum().item()
    ss_tot = ((H_o - o_mean) ** 2).sum().item()
    r2 = 1.0 - ss_res / (ss_tot + 1e-10)

    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    rank1_e = float((S[0] ** 2).item() / ((S ** 2).sum().item() + 1e-10))
    return dict(W=W, b=b, r2=r2, U=U, S=S, Vh=Vh, rank1_energy=rank1_e)


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("Experiment 2: Edit Vector Geometry Across Relations")
    print("=" * 70)

    np.random.seed(SEED)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ----------------------------------------------------------
    # 1. Load model, data, Exp 1 artifacts
    # ----------------------------------------------------------
    print("\nLoading GPT-2 XL...")
    model = AutoModelForCausalLM.from_pretrained("gpt2-xl").cuda()
    tok = AutoTokenizer.from_pretrained("gpt2-xl")
    tok.pad_token = tok.eos_token
    print("Model loaded.")

    hparams = ROMEHyperParams.from_json(HPARAMS_DIR / "ROME" / "gpt2-xl.json")

    data = json.load(open("data/counterfact.json"))
    print(f"CounterFact: {len(data)} records")

    by_relation = defaultdict(list)
    for r in data:
        by_relation[r["requested_rewrite"]["relation_id"]].append(r)

    # Exp 1 artifacts
    templates = json.load(open(EXP1_DIR / "context_templates.json"))
    concept_directions = torch.load(EXP1_DIR / "concept_directions_layer17.pt")
    relation_targets = json.load(open(EXP1_DIR / "relation_targets.json"))

    # Inspect concept direction key format
    cd_keys = list(concept_directions.keys())[:3]
    print(f"Concept direction key format (sample): {cd_keys}")

    rome_main_module.CONTEXT_TEMPLATES_CACHE = templates
    print(f"Context templates loaded ({len(templates)} templates)")

    # ----------------------------------------------------------
    # 2. Select entities per relation
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("ENTITY SELECTION")
    print("=" * 70)

    all_edits = []  # dicts with metadata
    for rel in ALL_RELATIONS:
        selected, alloc = select_entities_for_relation(
            by_relation[rel], n_total=N_EDITS_PER_RELATION, seed=SEED,
        )
        print(f"\n{rel}: {len(selected)} entities, {len(alloc)} targets")
        for t, n in alloc.items():
            print(f"  {t}: {n}")

        for record, target in selected:
            rw = record["requested_rewrite"]
            all_edits.append(dict(
                relation_id=rel,
                target_value=target,
                subject=rw["subject"],
                prompt=rw["prompt"],
                case_id=record["case_id"],
            ))

    print(f"\nTotal edits to compute: {len(all_edits)}")

    # ----------------------------------------------------------
    # 3. Compute ROME edit vectors
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("COMPUTING ROME EDIT VECTORS")
    print("=" * 70)

    v_list, u_list = [], []
    t0 = time.time()
    for i, ed in enumerate(all_edits):
        request = dict(prompt=ed["prompt"], subject=ed["subject"],
                       target_new={"str": ed["target_value"]})
        u, v = run_rome_edit(model, tok, request, hparams)
        u_list.append(u)
        v_list.append(v)

        if (i + 1) % 50 == 0 or i == 0:
            el = time.time() - t0
            rate = (i + 1) / el
            eta = (len(all_edits) - i - 1) / rate
            print(f"  [{i+1}/{len(all_edits)}] {ed['relation_id']} "
                  f"{ed['subject'][:25]} -> {ed['target_value']} "
                  f"({el:.0f}s, ETA {eta:.0f}s)")

    V = torch.stack(v_list).numpy()   # (N, 1600)
    U = torch.stack(u_list).numpy()   # (N, 6400)
    n = len(all_edits)
    print(f"\nComputed {n} edit vectors in {time.time()-t0:.0f}s")

    # Save edit vectors for Exp 3–5
    torch.save(dict(metadata=all_edits, v=torch.stack(v_list),
                    u=torch.stack(u_list)),
               RESULTS_DIR / "edit_vectors.pt")
    print("Saved edit_vectors.pt")

    # ----------------------------------------------------------
    # 4. Build grouping variables for PERMANOVA
    # ----------------------------------------------------------
    rel_labels = np.array([ALL_RELATIONS.index(e["relation_id"])
                           for e in all_edits])
    all_tgt_strs = sorted(set(e["target_value"] for e in all_edits))
    tgt_labels = np.array([all_tgt_strs.index(e["target_value"])
                           for e in all_edits])

    # Prompt-template clusters (TF-IDF + KMeans k=10)
    tfidf = TfidfVectorizer(max_features=100)
    pm = tfidf.fit_transform([e["prompt"] for e in all_edits]).toarray()
    prompt_labels = KMeans(n_clusters=10, random_state=SEED,
                           n_init=10).fit_predict(pm)

    # Entity-frequency proxy: GPT-2 token count (fewer tokens ≈ more common)
    tok_counts = np.array([len(tok.encode(e["subject"])) for e in all_edits])
    freq_labels = np.digitize(tok_counts,
                              bins=np.percentile(tok_counts, [20, 40, 60, 80]))

    # Random assignment
    rng_r = np.random.RandomState(SEED + 1)
    random_labels = np.zeros(n, dtype=int)
    random_labels[:] = rng_r.permutation(
        np.tile(np.arange(10), n // 10 + 1)[:n])

    # ----------------------------------------------------------
    # Distance matrix  d = 1 - |cos(v_i, v_j)|
    # ----------------------------------------------------------
    V_norm = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-10)
    cos_mat = V_norm @ V_norm.T
    dist = 1.0 - np.abs(cos_mat)
    np.fill_diagonal(dist, 0.0)
    D_sq = dist ** 2

    # ===========================================================
    # Phase A: PERMANOVA
    # ===========================================================
    print("\n" + "=" * 70)
    print("PHASE A: PERMANOVA")
    print("=" * 70)

    groupings = dict(
        relation=rel_labels,
        target_value=tgt_labels,
        prompt_template=prompt_labels,
        entity_frequency=freq_labels,
        random=random_labels,
    )
    perm_res = {}
    for name, labs in groupings.items():
        ng = len(np.unique(labs))
        print(f"\n  {name} ({ng} groups) ...")
        t1 = time.time()
        f, p, r2 = permanova(D_sq, labs, PERMANOVA_N_PERMS, seed=SEED)
        dt = time.time() - t1
        perm_res[name] = dict(pseudo_F=f, p_value=p, R2=r2,
                              n_groups=ng, time_s=dt)
        print(f"    F={f:.4f}  p={p:.6f}  R²={r2:.4f}  ({dt:.1f}s)")

    r2_rel = perm_res["relation"]["R2"]
    r2_prompt = perm_res["prompt_template"]["R2"]
    r2_freq = perm_res["entity_frequency"]["R2"]
    r2_rand = perm_res["random"]["R2"]

    phase_a_ok = (perm_res["relation"]["p_value"] < 0.001
                  and r2_rel > 2 * max(r2_prompt, r2_freq))
    print(f"\n  Phase A gate: {phase_a_ok}")
    print(f"    R²_rel={r2_rel:.4f}, 2×max(R²_prompt,R²_freq)="
          f"{2*max(r2_prompt,r2_freq):.4f}, R²_random={r2_rand:.4f}")

    # ===========================================================
    # Phase B: LDA
    # ===========================================================
    print("\n" + "=" * 70)
    print("PHASE B: LDA")
    print("=" * 70)

    if not phase_a_ok:
        print("  Phase A gate NOT passed — running LDA descriptively")

    # 5-fold stratified CV
    skf = StratifiedKFold(n_splits=LDA_N_FOLDS, shuffle=True,
                          random_state=SEED)
    cv_accs = []
    per_cls = defaultdict(list)
    for train_i, test_i in skf.split(V, rel_labels):
        lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
        lda.fit(V[train_i], rel_labels[train_i])
        pr = lda.predict(V[test_i])
        cv_accs.append(float(np.mean(pr == rel_labels[test_i])))
        for c in np.unique(rel_labels):
            m = rel_labels[test_i] == c
            if m.sum():
                per_cls[int(c)].append(float(np.mean(pr[m] == c)))

    mean_cv = np.mean(cv_accs)
    print(f"  5-fold CV accuracy: {mean_cv:.4f}  (chance = 0.10)")

    # Full LDA for directions
    lda_full = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
    lda_full.fit(V, rel_labels)
    # scalings_ shape: (1600, n_components) with n_components <= 9
    lda_dirs = lda_full.scalings_
    lda_dirs_n = lda_dirs / (np.linalg.norm(lda_dirs, axis=0, keepdims=True)
                             + 1e-10)
    n_lda = lda_dirs_n.shape[1]
    print(f"  {n_lda} discriminant directions extracted")

    lda_proj = V @ lda_dirs_n  # (N, n_lda)

    # Per-direction analysis: which relations are most separated?
    lda_dir_info = []
    for d in range(n_lda):
        proj = lda_proj[:, d]
        means = {int(c): proj[rel_labels == c].mean() for c in np.unique(rel_labels)}
        srt = sorted(means.items(), key=lambda x: x[1])
        lda_dir_info.append(dict(
            idx=d,
            pair=(ALL_RELATIONS[srt[0][0]], ALL_RELATIONS[srt[-1][0]]),
            sep=float(srt[-1][1] - srt[0][1]),
            means={ALL_RELATIONS[k]: float(v) for k, v in means.items()},
        ))

    # Compare top-5 LDA dirs to Exp-1 concept directions
    # Keys are strings like "P176_Toyota_mean_diff" or "P176_Toyota_logistic"
    lda_concept_cos = []
    for d in range(min(5, n_lda)):
        ld = lda_dirs_n[:, d]
        for key, cd_vec in concept_directions.items():
            method = None
            for m in DIRECTION_METHODS:
                if key.endswith("_" + m):
                    method = m
                    concept_name = key[: -(len(m) + 1)]
                    break
            if method is None:
                continue
            c = cosine(ld, cd_vec.numpy())
            lda_concept_cos.append(dict(
                lda_dir=d, concept=concept_name, method=method,
                cos=float(c), abs_cos=float(abs(c)),
            ))

    if lda_concept_cos:
        top5 = sorted([x["abs_cos"] for x in lda_concept_cos], reverse=True)[:5]
        print(f"  Top-5 |cos(LDA, concept)|: {[f'{c:.4f}' for c in top5]}")

    # Bootstrap stability (top 3 directions)
    print(f"  Bootstrap stability ({BOOTSTRAP_N} iters) ...")
    boot_cos = [[] for _ in range(min(3, n_lda))]
    for b in range(BOOTSTRAP_N):
        bi = np.random.RandomState(SEED + 100 + b).choice(n, n, replace=True)
        try:
            lb = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto")
            lb.fit(V[bi], rel_labels[bi])
            bd = lb.scalings_
            bd_n = bd / (np.linalg.norm(bd, axis=0, keepdims=True) + 1e-10)
            for dd in range(min(3, bd_n.shape[1], n_lda)):
                boot_cos[dd].append(abs(cosine(lda_dirs_n[:, dd], bd_n[:, dd])))
        except Exception:
            pass

    boot_stab = []
    for dd in range(len(boot_cos)):
        if boot_cos[dd]:
            m, s = np.mean(boot_cos[dd]), np.std(boot_cos[dd])
            boot_stab.append(dict(dir=dd, mean=float(m), std=float(s)))
            print(f"    Dir {dd}: |cos|={m:.4f} ± {s:.4f}")

    lda_results = dict(
        cv_accuracy=float(mean_cv),
        cv_accs=[float(a) for a in cv_accs],
        per_class={ALL_RELATIONS[k]: float(np.mean(v))
                   for k, v in per_cls.items()},
        direction_info=lda_dir_info,
        lda_concept_cosines=lda_concept_cos,
        bootstrap_stability=boot_stab,
        gate_passed=bool(phase_a_ok),
    )

    # ===========================================================
    # Phase C: Three-level variance decomposition
    # ===========================================================
    print("\n" + "=" * 70)
    print("PHASE C: THREE-LEVEL VARIANCE DECOMPOSITION")
    print("=" * 70)

    r2_concept_per_rel = {}
    for ri, rel in enumerate(ALL_RELATIONS):
        mask = rel_labels == ri
        sub_V = V[mask]
        sub_tgt = tgt_labels[mask]
        utgt = np.unique(sub_tgt)

        # Keep only targets with enough samples
        qual = [t for t in utgt if (sub_tgt == t).sum() >= MIN_ENTITIES_PER_TARGET_C]
        if len(qual) < 2:
            print(f"  {rel}: SKIPPED ({len(qual)} qualifying targets)")
            r2_concept_per_rel[rel] = None
            continue

        qm = np.isin(sub_tgt, qual)
        sV = sub_V[qm]
        sT = sub_tgt[qm]

        # Re-label
        tmap = {t: i for i, t in enumerate(qual)}
        sL = np.array([tmap[t] for t in sT])

        # Within-relation distance matrix
        sn = np.linalg.norm(sV, axis=1, keepdims=True)
        sVn = sV / (sn + 1e-10)
        sc = sVn @ sVn.T
        sd = 1.0 - np.abs(sc)
        np.fill_diagonal(sd, 0.0)
        sD = sd ** 2

        f, p, r2c = permanova(sD, sL, PERMANOVA_N_PERMS_WITHIN, seed=SEED)
        tnames = [all_tgt_strs[t] for t in qual]
        r2_concept_per_rel[rel] = dict(
            R2=float(r2c), F=float(f), p=float(p),
            n=int(len(sV)), n_targets=len(qual), targets=tnames,
        )
        print(f"  {rel}: R²_concept={r2c:.4f}  F={f:.2f}  p={p:.4f}  "
              f"({len(qual)} targets, n={len(sV)})")

    valid_r2 = [v["R2"] for v in r2_concept_per_rel.values() if v]
    r2c_mean = np.mean(valid_r2) if valid_r2 else 0.0
    r2_entity = max(0.0, 1.0 - r2_rel - r2c_mean)

    phase_c = dict(
        R2_relation=float(r2_rel),
        R2_concept_mean=float(r2c_mean),
        R2_entity=float(r2_entity),
        per_relation={k: v for k, v in r2_concept_per_rel.items() if v},
        excluded=[k for k, v in r2_concept_per_rel.items() if not v],
    )
    print(f"\n  Aggregate: R²_relation={r2_rel:.4f}  R²_concept={r2c_mean:.4f}  "
          f"R²_entity={r2_entity:.4f}")

    # ===========================================================
    # Phase D: LRE Triangulation
    # ===========================================================
    print("\n" + "=" * 70)
    print("PHASE D: LRE TRIANGULATION")
    print("=" * 70)

    lre_res = {}
    triangulation = {}

    for rel in ALL_RELATIONS:
        print(f"\n  {rel}: fitting LRE ...")
        recs = by_relation[rel]
        rng_l = np.random.RandomState(SEED + 200)
        if len(recs) > LRE_N_PAIRS:
            idx = rng_l.choice(len(recs), LRE_N_PAIRS, replace=False)
            lre_recs = [recs[i] for i in idx]
        else:
            lre_recs = recs

        prompt_tmpls = [r["requested_rewrite"]["prompt"] for r in lre_recs]
        subjs = [r["requested_rewrite"]["subject"] for r in lre_recs]

        try:
            Hs, Ho = extract_lre_activations(model, tok, prompt_tmpls, subjs,
                                              batch_size=BATCH_SIZE)
            lre = fit_lre(Hs, Ho, alpha=LRE_RIDGE_ALPHA)
            r2 = lre["r2"]
            r1e = lre["rank1_energy"]
            top_sv = lre["S"][:5].tolist()

            lre_res[rel] = dict(r2=float(r2), rank1_energy=float(r1e),
                                top5_sv=[float(s) for s in top_sv],
                                n=len(lre_recs))
            print(f"    R²={r2:.4f}  rank-1 energy={r1e:.4f}")

            # --- 4-way triangulation ---
            dirs = {}

            # 1. Concept direction (Exp 1, mean_diff, first available target)
            # Keys are strings like "P176_Toyota_mean_diff"
            if rel in relation_targets:
                for tgt in relation_targets[rel]:
                    key = f"{rel}_{tgt}_mean_diff"
                    if key in concept_directions:
                        d = concept_directions[key].numpy().astype(np.float64)
                        dirs["concept_dir"] = d / (np.linalg.norm(d) + 1e-10)
                        break

            # 2. LDA direction most relevant to this relation
            if n_lda > 0:
                ri = ALL_RELATIONS.index(rel)
                rm = rel_labels == ri
                best_d, best_sep = 0, 0
                for dd in range(n_lda):
                    sep = abs(lda_proj[rm, dd].mean() - lda_proj[~rm, dd].mean())
                    if sep > best_sep:
                        best_sep, best_d = sep, dd
                dirs["lda_dir"] = lda_dirs_n[:, best_d].astype(np.float64)

            # 3. LRE top input singular direction (Vh[0])
            lre_d = lre["Vh"][0].numpy().astype(np.float64)
            dirs["lre_dir"] = lre_d / (np.linalg.norm(lre_d) + 1e-10)

            # 4. v_mean across all edits for this relation
            rm = np.array([e["relation_id"] == rel for e in all_edits])
            vm = V[rm].mean(axis=0)
            dirs["v_mean"] = vm / (np.linalg.norm(vm) + 1e-10)

            # Pairwise cosines
            dnames = list(dirs.keys())
            dvecs = [dirs[k] for k in dnames]
            cos_pairs = {}
            for a in range(len(dnames)):
                for b in range(a + 1, len(dnames)):
                    c = cosine(dvecs[a], dvecs[b])
                    cos_pairs[f"{dnames[a]}_vs_{dnames[b]}"] = dict(
                        signed=float(c), absolute=float(abs(c)))

            triangulation[rel] = dict(directions=dnames,
                                      pairwise=cos_pairs)

            for pair, vals in cos_pairs.items():
                print(f"    {pair}: cos={vals['signed']:.4f}  "
                      f"|cos|={vals['absolute']:.4f}")

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback; traceback.print_exc()
            lre_res[rel] = dict(error=str(e))
            triangulation[rel] = dict(error=str(e))

    # ===========================================================
    # Summary
    # ===========================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print("\nPhase A (PERMANOVA):")
    for nm, r in perm_res.items():
        print(f"  {nm:20s}  F={r['pseudo_F']:8.2f}  p={r['p_value']:.6f}  "
              f"R²={r['R2']:.4f}")

    print(f"\nPhase B (LDA): CV acc = {mean_cv:.4f}  (chance 0.10)")

    print(f"\nPhase C: R²_relation={r2_rel:.4f}  R²_concept={r2c_mean:.4f}  "
          f"R²_entity={r2_entity:.4f}")

    valid_lre = {k: v for k, v in lre_res.items() if "error" not in v}
    if valid_lre:
        r2s = [v["r2"] for v in valid_lre.values()]
        print(f"\nPhase D: LRE R² range [{min(r2s):.4f}, {max(r2s):.4f}]")

    all_tri_abs = []
    for rel, tri in triangulation.items():
        if "error" not in tri:
            for pair, vals in tri.get("pairwise", {}).items():
                all_tri_abs.append(vals["absolute"])
    if all_tri_abs:
        print(f"  Mean triangulation |cos|: {np.mean(all_tri_abs):.4f}")

    # ---- success criteria ----
    print("\n" + "-" * 40)
    print("SUCCESS CRITERIA:")
    pa = perm_res["relation"]
    print(f"  A: p<0.001 = {pa['p_value']<0.001}  "
          f"R²>0.10 = {r2_rel>0.10} ({r2_rel:.4f})  "
          f"R²_rel>2×confound = {r2_rel>2*max(r2_prompt,r2_freq)}  "
          f"R²_rand<0.02 = {r2_rand<0.02} ({r2_rand:.4f})")
    print(f"  B: CV>=0.40 = {mean_cv>=0.40} ({mean_cv:.4f})")
    if lda_concept_cos:
        t5 = np.mean(sorted([x["abs_cos"] for x in lda_concept_cos],
                             reverse=True)[:5])
        print(f"  B: top5 |cos|>=0.30 = {t5>=0.30} ({t5:.4f})")
    if boot_stab:
        max_sd = max(b["std"] for b in boot_stab)
        print(f"  B: bootstrap SD<=0.15 = {max_sd<=0.15} ({max_sd:.4f})")
    if valid_r2:
        ng = sum(1 for r in valid_r2 if r > r2_entity)
        print(f"  C: R²_concept>R²_entity for >=7 rels = "
              f"{ng>=7} ({ng}/{len(valid_r2)})")
        agg = r2_rel + r2c_mean
        print(f"  C: R²_rel+R²_concept>0.20 = {agg>0.20} ({agg:.4f})")
    if valid_lre:
        nl = sum(1 for v in valid_lre.values() if v["r2"] > 0.10)
        print(f"  D: LRE R²>0.10 for >=7 = {nl>=7} ({nl}/{len(valid_lre)})")

    # ===========================================================
    # Save results
    # ===========================================================
    entity_sel = {}
    for rel in ALL_RELATIONS:
        entity_sel[rel] = dict(Counter(
            e["target_value"] for e in all_edits if e["relation_id"] == rel))

    results = dict(
        config=dict(
            relations=ALL_RELATIONS,
            n_per_relation=N_EDITS_PER_RELATION,
            max_target_fraction=MAX_TARGET_FRACTION,
            edit_layer=EDIT_LAYER,
            lre_layers=(LRE_SUBJECT_LAYER, LRE_OBJECT_LAYER),
            permanova_n_perms=PERMANOVA_N_PERMS,
            direction_methods=DIRECTION_METHODS,
            note="DAS dropped per Exp 1; templates from Exp 1",
        ),
        entity_selection=entity_sel,
        permanova=perm_res,
        lda=lda_results,
        variance_decomposition=phase_c,
        lre=lre_res,
        triangulation=triangulation,
    )

    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save LDA directions separately for Exp 3–5
    np.save(RESULTS_DIR / "lda_directions.npy", lda_dirs_n)

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
