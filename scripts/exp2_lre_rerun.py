"""
Re-run LRE triangulation with subject_layer=16, object_layer=17.

Original Exp 2 used subject=17, object=47. This tests whether a
local LRE (mapping the layer just before the edit to the edit layer)
shows different alignment with concept/LDA/v_mean directions.

Saves results to results/exp2/lre_L16_L17.json (does NOT overwrite
the original results/exp2/results.json).
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
from rome.repr_tools import get_words_idxs_in_templates
from rome.rome_hparams import ROMEHyperParams
from util import nethook
from util.globals import HPARAMS_DIR

# ============================================================
# Configuration
# ============================================================
LRE_SUBJECT_LAYER = 16
LRE_OBJECT_LAYER = 17
LRE_N_PAIRS = 200
LRE_RIDGE_ALPHA = 1.0
BATCH_SIZE = 24
SEED = 42

ALL_RELATIONS = [
    "P176", "P27", "P495", "P37", "P17",
    "P413", "P1412", "P937", "P106", "P449",
]

RESULTS_DIR = Path("results/exp2")
EXP1_DIR = Path("results/exp1")
OUTPUT_FILE = RESULTS_DIR / "lre_L16_L17.json"


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def extract_lre_activations(model, tok, prompt_templates, subjects,
                            subj_layer, obj_layer, batch_size=24):
    """Extract subject_last@subj_layer and last@obj_layer."""
    l_s = f"transformer.h.{subj_layer}"
    l_o = f"transformer.h.{obj_layer}"
    H_subj_all, H_obj_all = [], []

    for i in range(0, len(prompt_templates), batch_size):
        b_tmpl = prompt_templates[i:i + batch_size]
        b_subj = subjects[i:i + batch_size]

        subj_idxs = get_words_idxs_in_templates(tok, b_tmpl, b_subj,
                                                  subtoken="last")
        filled = [t.format(s) for t, s in zip(b_tmpl, b_subj)]
        inputs = tok(filled, return_tensors="pt", padding=True).to("cuda")

        with torch.no_grad():
            with nethook.TraceDict(model, [l_s, l_o]) as tr:
                model(**inputs)

        h_s = tr[l_s].output[0]
        h_o = tr[l_o].output[0]

        for j in range(len(filled)):
            s_pos = subj_idxs[j][0]
            last_pos = inputs["attention_mask"][j].sum().item() - 1
            s_pos = min(s_pos, last_pos)

            H_subj_all.append(h_s[j, s_pos].detach().cpu())
            H_obj_all.append(h_o[j, last_pos].detach().cpu())

    return torch.stack(H_subj_all), torch.stack(H_obj_all)


def fit_lre(H_s, H_o, alpha=1.0):
    """Ridge-regression LRE: H_o ≈ H_s @ W^T + b."""
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


def main():
    print("=" * 70)
    print(f"LRE Re-run: subject_layer={LRE_SUBJECT_LAYER}, "
          f"object_layer={LRE_OBJECT_LAYER}")
    print("=" * 70)

    np.random.seed(SEED)

    # Load model
    print("\nLoading GPT-2 XL...")
    model = AutoModelForCausalLM.from_pretrained("gpt2-xl").cuda()
    tok = AutoTokenizer.from_pretrained("gpt2-xl")
    tok.pad_token = tok.eos_token
    print("Model loaded.")

    # Load data
    data = json.load(open("data/counterfact.json"))
    by_relation = defaultdict(list)
    for r in data:
        by_relation[r["requested_rewrite"]["relation_id"]].append(r)

    # Load Exp 1 artifacts for triangulation
    concept_directions = torch.load(EXP1_DIR / "concept_directions_layer17.pt")
    relation_targets = json.load(open(EXP1_DIR / "relation_targets.json"))

    # Load Exp 2 artifacts: LDA directions and v vectors
    lda_dirs = np.load(RESULTS_DIR / "lda_directions.npy")   # (1600, n_components)
    lda_dirs_n = lda_dirs / (np.linalg.norm(lda_dirs, axis=0, keepdims=True) + 1e-10)

    edit_vectors = torch.load(RESULTS_DIR / "edit_vectors.pt")
    V = edit_vectors["v"].numpy()                              # (500, 1600)
    rel_labels_str = [m["relation_id"] for m in edit_vectors["metadata"]]

    # Map relation string -> index
    rel_to_idx = {r: i for i, r in enumerate(ALL_RELATIONS)}
    rel_labels = np.array([rel_to_idx[r] for r in rel_labels_str])

    n_lda = lda_dirs.shape[1]

    # Compute LDA projections for best-discriminant selection
    V_n = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-10)
    lda_proj = V_n @ lda_dirs_n

    # ---- Fit LRE per relation ----
    lre_results = {}
    triangulation = {}

    for rel in ALL_RELATIONS:
        print(f"\n  {rel}: fitting LRE (L{LRE_SUBJECT_LAYER}→L{LRE_OBJECT_LAYER}) ...")
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
            Hs, Ho = extract_lre_activations(
                model, tok, prompt_tmpls, subjs,
                subj_layer=LRE_SUBJECT_LAYER,
                obj_layer=LRE_OBJECT_LAYER,
                batch_size=BATCH_SIZE,
            )
            lre = fit_lre(Hs, Ho, alpha=LRE_RIDGE_ALPHA)
            r2 = lre["r2"]
            r1e = lre["rank1_energy"]
            top_sv = lre["S"][:5].tolist()

            lre_results[rel] = dict(
                r2=float(r2), rank1_energy=float(r1e),
                top5_sv=[float(s) for s in top_sv],
                n=len(lre_recs),
            )
            print(f"    R²={r2:.4f}  rank-1 energy={r1e:.4f}")

            # --- 4-way triangulation ---
            dirs = {}

            # 1. Concept direction (Exp 1, mean_diff)
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
            rm = np.array([r == rel for r in rel_labels_str])
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

            triangulation[rel] = dict(directions=dnames, pairwise=cos_pairs)

            for pair, vals in cos_pairs.items():
                print(f"    {pair}: cos={vals['signed']:.4f}  "
                      f"|cos|={vals['absolute']:.4f}")

        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback; traceback.print_exc()
            lre_results[rel] = dict(error=str(e))
            triangulation[rel] = dict(error=str(e))

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    valid = {k: v for k, v in lre_results.items() if "error" not in v}
    if valid:
        r2s = [v["r2"] for v in valid.values()]
        r1es = [v["rank1_energy"] for v in valid.values()]
        print(f"  R² range: [{min(r2s):.4f}, {max(r2s):.4f}], mean={np.mean(r2s):.4f}")
        print(f"  Rank-1 energy range: [{min(r1es):.4f}, {max(r1es):.4f}]")

    # Aggregate pairwise cosines by pair type
    pair_abs = defaultdict(list)
    for rel, tri in triangulation.items():
        if "error" not in tri:
            for pair, vals in tri.get("pairwise", {}).items():
                pair_abs[pair].append(vals["absolute"])

    print("\n  Mean |cos| by pair:")
    for pair, vals in sorted(pair_abs.items()):
        print(f"    {pair}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    # Compare with original L17→L47 results
    orig = json.load(open(RESULTS_DIR / "results.json"))
    if "triangulation" in orig:
        print("\n  Comparison with original L17→L47:")
        orig_pair_abs = defaultdict(list)
        for rel, tri in orig["triangulation"].items():
            if "error" not in tri:
                for pair, vals in tri.get("pairwise", {}).items():
                    orig_pair_abs[pair].append(vals["absolute"])

        for pair in sorted(set(pair_abs) | set(orig_pair_abs)):
            new_m = np.mean(pair_abs.get(pair, [0]))
            old_m = np.mean(orig_pair_abs.get(pair, [0]))
            print(f"    {pair}: L16→L17={new_m:.4f}  L17→L47={old_m:.4f}  "
                  f"delta={new_m - old_m:+.4f}")

    # ---- Save ----
    output = dict(
        config=dict(
            lre_subject_layer=LRE_SUBJECT_LAYER,
            lre_object_layer=LRE_OBJECT_LAYER,
            lre_n_pairs=LRE_N_PAIRS,
            lre_ridge_alpha=LRE_RIDGE_ALPHA,
            note="Re-run of LRE with local layers (L16→L17) instead of original L17→L47",
        ),
        lre_results=lre_results,
        triangulation=triangulation,
    )

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()