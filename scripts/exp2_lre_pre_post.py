"""
LRE before and after ROME edits (L17→L47).

For each edit:
1. Fit LRE pre-edit on held-out prompts for that relation
2. Apply ROME edit
3. Fit LRE post-edit on same prompts
4. Compare: R² change, W change, top singular direction shift
5. Check if delta-W aligns with v
6. Restore weights

Small scale: 3 edits per relation, 5 relations = 15 edits.
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
from rome.rome_main import execute_rome, apply_rome_to_model
from rome.rome_hparams import ROMEHyperParams
from rome.repr_tools import get_words_idxs_in_templates
from util import nethook
from util.globals import HPARAMS_DIR

# ============================================================
# Configuration
# ============================================================
SUBJECT_LAYER = 17
OBJECT_LAYER = 47
LRE_N_PAIRS = 150          # prompts for fitting LRE (held out from edit)
LRE_RIDGE_ALPHA = 1.0
N_EDITS_PER_RELATION = 5
RELATIONS = [
    "P176", "P27", "P495", "P37", "P17",
    "P413", "P1412", "P937", "P106", "P449",
]
BATCH_SIZE = 24
SEED = 42
RESULTS_DIR = Path("results/exp2")
OUTPUT_FILE = RESULTS_DIR / "lre_pre_post_full.json"


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def extract_lre_activations(model, tok, prompt_templates, subjects,
                            subj_layer, obj_layer, batch_size=24):
    """Extract subject_last@subj_layer and last@obj_layer."""
    l_s = f"transformer.h.{subj_layer}"
    l_o = f"transformer.h.{obj_layer}"
    H_subj, H_obj = [], []

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
            H_subj.append(h_s[j, s_pos].detach().cpu())
            H_obj.append(h_o[j, last_pos].detach().cpu())

    return torch.stack(H_subj), torch.stack(H_obj)


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
    print("LRE Pre/Post ROME Edit Analysis (L17→L47)")
    print("=" * 70)

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load model
    print("\nLoading GPT-2 XL...")
    model = AutoModelForCausalLM.from_pretrained("gpt2-xl").cuda()
    tok = AutoTokenizer.from_pretrained("gpt2-xl")
    tok.pad_token = tok.eos_token
    hparams = ROMEHyperParams.from_json(HPARAMS_DIR / "ROME" / "gpt2-xl.json")
    print("Model loaded.")

    # Load data
    data = json.load(open("data/counterfact.json"))
    by_relation = defaultdict(list)
    for r in data:
        by_relation[r["requested_rewrite"]["relation_id"]].append(r)

    # Load exp2 edit vectors for v comparison
    ev = torch.load(RESULTS_DIR / "edit_vectors.pt")
    V_all = ev["v"].numpy()
    metadata = ev["metadata"]

    all_results = []

    for rel in RELATIONS:
        recs = by_relation[rel]
        rng = np.random.RandomState(SEED + hash(rel) % 10000)
        rng.shuffle(recs)

        # Split: first N_EDITS for editing, rest for LRE fitting
        edit_recs = recs[:N_EDITS_PER_RELATION]
        lre_recs = recs[N_EDITS_PER_RELATION:N_EDITS_PER_RELATION + LRE_N_PAIRS]

        if len(lre_recs) < 50:
            print(f"\n  {rel}: too few LRE prompts ({len(lre_recs)}), skipping")
            continue

        lre_tmpls = [r["requested_rewrite"]["prompt"] for r in lre_recs]
        lre_subjs = [r["requested_rewrite"]["subject"] for r in lre_recs]

        print(f"\n{'='*60}")
        print(f"  {rel}: {len(edit_recs)} edits, {len(lre_recs)} LRE prompts")
        print(f"{'='*60}")

        # Pre-edit LRE (same for all edits in this relation)
        print("  Fitting pre-edit LRE...")
        Hs_pre, Ho_pre = extract_lre_activations(
            model, tok, lre_tmpls, lre_subjs,
            SUBJECT_LAYER, OBJECT_LAYER, BATCH_SIZE
        )
        lre_pre = fit_lre(Hs_pre, Ho_pre, LRE_RIDGE_ALPHA)
        W_pre = lre_pre["W"].numpy()
        Vh_pre_0 = lre_pre["Vh"][0].numpy()
        U_pre_0 = lre_pre["U"][:, 0].numpy()
        print(f"    Pre-edit: R²={lre_pre['r2']:.6f}  "
              f"rank-1 energy={lre_pre['rank1_energy']:.4f}")

        for ei, rec in enumerate(edit_recs):
            rw = rec["requested_rewrite"]
            request = {
                "prompt": rw["prompt"],
                "subject": rw["subject"],
                "target_new": rw["target_new"],
            }
            print(f"\n  Edit {ei+1}: {rw['subject']} → {rw['target_new']['str']}")

            # Apply ROME edit
            model_edited, weights_copy = apply_rome_to_model(
                model, tok, [request], hparams,
                copy=False, return_orig_weights=True,
            )

            # Get v vector for this edit
            # Re-extract from the edit we just did (or approximate from exp2)
            # For simplicity, compute fresh
            # Actually we already applied the edit. Let's get v from the weight delta.
            edit_key = f"transformer.h.{hparams.layers[0]}.mlp.c_proj.weight"
            W_orig = weights_copy[edit_key].cpu()
            W_edited = nethook.get_parameter(model, edit_key).detach().cpu()
            delta_W_actual = (W_edited - W_orig).float()  # (1600, 6400) = v @ u^T

            # Get v as top right singular vector of delta (should be rank-1)
            # c_proj weight is Conv1D: (6400, 1600), so delta = u_col @ v_row
            # SVD: U[:,0] is (6400,) = u direction, Vh[0] is (1600,) = v direction
            U_d, S_d, Vh_d = torch.linalg.svd(delta_W_actual, full_matrices=False)
            v_edit = Vh_d[0].numpy()  # (1600,) — v direction in residual stream
            v_edit_norm = v_edit / (np.linalg.norm(v_edit) + 1e-10)
            print(f"    Delta rank-1 energy: {(S_d[0]**2 / (S_d**2).sum()).item():.6f}")

            # Post-edit LRE
            print("    Fitting post-edit LRE...")
            Hs_post, Ho_post = extract_lre_activations(
                model, tok, lre_tmpls, lre_subjs,
                SUBJECT_LAYER, OBJECT_LAYER, BATCH_SIZE
            )
            lre_post = fit_lre(Hs_post, Ho_post, LRE_RIDGE_ALPHA)
            W_post = lre_post["W"].numpy()
            Vh_post_0 = lre_post["Vh"][0].numpy()
            U_post_0 = lre_post["U"][:, 0].numpy()

            print(f"    Post-edit: R²={lre_post['r2']:.6f}  "
                  f"rank-1 energy={lre_post['rank1_energy']:.4f}")

            # --- Comparisons ---

            # 1. R² change
            r2_delta = lre_post["r2"] - lre_pre["r2"]

            # 2. W matrix change: Frobenius norm of delta
            delta_W_lre = W_post - W_pre
            delta_fro = np.linalg.norm(delta_W_lre)
            pre_fro = np.linalg.norm(W_pre)
            relative_change = delta_fro / (pre_fro + 1e-10)

            # 3. Top singular direction change
            cos_Vh0 = cosine(Vh_pre_0, Vh_post_0)
            cos_U0 = cosine(U_pre_0, U_post_0)

            # 4. Does delta_W_lre align with the ROME edit's v?
            # delta_W_lre is (1600, 1600). Project onto v direction:
            # If ROME changes the L17→L47 map in the v direction,
            # then delta_W_lre @ x should have large component along v for relevant x
            delta_W_lre_t = torch.tensor(delta_W_lre, dtype=torch.float32)
            U_lre_d, S_lre_d, Vh_lre_d = torch.linalg.svd(delta_W_lre_t, full_matrices=False)
            lre_delta_top_out = U_lre_d[:, 0].numpy()
            lre_delta_top_in = Vh_lre_d[0].numpy()
            lre_delta_rank1_e = float((S_lre_d[0]**2 / (S_lre_d**2).sum()).item())

            cos_v_lre_delta_out = cosine(v_edit, lre_delta_top_out)
            cos_v_lre_delta_in = cosine(v_edit, lre_delta_top_in)

            # 5. Pre-edit LRE evaluated on post-edit data
            pred_cross = Hs_post.float() @ lre_pre["W"].T + lre_pre["b"]
            ss_res_cross = ((Ho_post.float() - pred_cross) ** 2).sum().item()
            ss_tot_cross = ((Ho_post.float() - Ho_post.float().mean(0)) ** 2).sum().item()
            r2_cross = 1.0 - ss_res_cross / (ss_tot_cross + 1e-10)

            # 6. Activation shift at L47
            delta_h47 = (Ho_post - Ho_pre).float().mean(0).numpy()
            delta_h47_norm = np.linalg.norm(delta_h47)
            cos_v_delta_h47 = cosine(v_edit, delta_h47)

            result = {
                "relation": rel,
                "subject": rw["subject"],
                "target_new": rw["target_new"]["str"],
                "pre_r2": float(lre_pre["r2"]),
                "post_r2": float(lre_post["r2"]),
                "r2_delta": float(r2_delta),
                "cross_r2": float(r2_cross),
                "W_relative_change": float(relative_change),
                "cos_top_input_dir": float(cos_Vh0),
                "cos_top_output_dir": float(cos_U0),
                "lre_delta_rank1_energy": float(lre_delta_rank1_e),
                "cos_v_vs_lre_delta_top_output": float(cos_v_lre_delta_out),
                "cos_v_vs_lre_delta_top_input": float(cos_v_lre_delta_in),
                "mean_h47_shift_norm": float(delta_h47_norm),
                "cos_v_vs_mean_h47_shift": float(cos_v_delta_h47),
            }
            all_results.append(result)

            print(f"    R² delta: {r2_delta:+.6f}")
            print(f"    Cross R² (pre-LRE on post-data): {r2_cross:.6f}")
            print(f"    W relative change: {relative_change:.6f}")
            print(f"    Top dir stability: input={cos_Vh0:.4f}  output={cos_U0:.4f}")
            print(f"    LRE delta rank-1 energy: {lre_delta_rank1_e:.4f}")
            print(f"    cos(v, LRE-delta top output): {cos_v_lre_delta_out:.4f}")
            print(f"    cos(v, mean L47 shift): {cos_v_delta_h47:.4f}")
            print(f"    Mean L47 shift norm: {delta_h47_norm:.2f}")

            # Restore weights
            with torch.no_grad():
                for k, v in weights_copy.items():
                    nethook.get_parameter(model, k)[...] = v.to("cuda")

    # ---- Aggregate ----
    print("\n" + "=" * 70)
    print("AGGREGATE SUMMARY")
    print("=" * 70)

    n = len(all_results)
    for key in ["r2_delta", "cross_r2", "W_relative_change",
                "cos_top_input_dir", "cos_top_output_dir",
                "lre_delta_rank1_energy",
                "cos_v_vs_lre_delta_top_output",
                "cos_v_vs_mean_h47_shift", "mean_h47_shift_norm"]:
        vals = [r[key] for r in all_results]
        print(f"  {key:40s}: mean={np.mean(vals):+.4f}  "
              f"std={np.std(vals):.4f}  "
              f"range=[{min(vals):.4f}, {max(vals):.4f}]")

    # Save
    output = {
        "config": {
            "subject_layer": SUBJECT_LAYER,
            "object_layer": OBJECT_LAYER,
            "n_edits_per_relation": N_EDITS_PER_RELATION,
            "lre_n_pairs": LRE_N_PAIRS,
            "relations": RELATIONS,
        },
        "per_edit": all_results,
        "aggregate": {
            key: {
                "mean": float(np.mean([r[key] for r in all_results])),
                "std": float(np.std([r[key] for r in all_results])),
            }
            for key in ["r2_delta", "cross_r2", "W_relative_change",
                        "cos_top_input_dir", "cos_top_output_dir",
                        "lre_delta_rank1_energy",
                        "cos_v_vs_lre_delta_top_output",
                        "cos_v_vs_mean_h47_shift", "mean_h47_shift_norm"]
        },
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()