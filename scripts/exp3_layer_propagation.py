"""
Experiment 3: Layer Propagation Analysis

Tracks how ROME's edit perturbation transforms as it propagates through
layers 17→47, measuring whether the perturbation aligns with layer-specific
concept directions.

Hypothesis: The network rotates ROME's edit signal from an orthogonal
encoding at layer 17 into the model's concept subspace by layer 40+.

Phases:
A — Train concept directions at all 12 available layers (from Exp 1 activations)
B — Extract perturbation Δh = h_edited - h_original at all 48 layers
C — Measure alignment cos(Δh_L, concept_dir_L) at each layer
D — Additional: perturbation magnitude, cross-position, LDA crossover, efficacy correlation

Updates from Experiments 1-2:
- 5 relations (P176, P1412, P37, P27, P413) with best probe accuracy
- Concept directions at layers 0-16 exist but Δh there is zero, so alignment
  measured only at layers >= 17: [17, 20, 25, 30, 35, 40, 47]
- LDA directions (9 discriminant, from Exp 2) used for crossover analysis
- Edit vectors (u, v) reused from results/exp2/edit_vectors.pt

Saves results to results/exp3/
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
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

from rome.rome_hparams import ROMEHyperParams
from rome import repr_tools
from util import nethook
from util.globals import HPARAMS_DIR
import rome.rome_main as rome_main_module

# ============================================================
# Configuration
# ============================================================
RELATIONS = ["P176", "P1412", "P37", "P27", "P413"]
PROBE_LAYERS = [0, 5, 10, 13, 15, 17, 20, 25, 30, 35, 40, 47]
ALIGNMENT_LAYERS = [17, 20, 25, 30, 35, 40, 47]  # layers >= edit layer
ALL_LAYERS = list(range(48))
N_PER_CONCEPT = 10
EDIT_LAYER = 17
TRAIN_RATIO = 0.70
N_LDA_DIRS = 9  # 10-class LDA gives 9 discriminant directions

RESULTS_DIR = Path("results/exp3")
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


def train_probe(X_train, y_train, X_test, y_test):
    """Train LogisticRegressionCV, return balanced accuracy and direction."""
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    min_class = min(y_train.sum(), (1 - y_train).sum())
    cv = min(5, int(min_class))
    if cv < 2:
        return None  # not enough samples

    clf = LogisticRegressionCV(
        Cs=[1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
        cv=cv,
        max_iter=1000,
        scoring="balanced_accuracy",
        random_state=42,
    )
    clf.fit(Xtr, y_train)

    y_pred = clf.predict(Xte)
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))

    # Direction in original (unscaled) space
    w = clf.coef_[0] / scaler.scale_
    w = w / (np.linalg.norm(w) + 1e-10)

    return {"balanced_accuracy": bal_acc, "direction": w}


# ============================================================
# Main
# ============================================================
def main():
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("EXPERIMENT 3: LAYER PROPAGATION ANALYSIS")
    print("=" * 70)

    # ── Load model ────────────────────────────────────────────
    print("\n[1] Loading model and tokenizer...")
    model = AutoModelForCausalLM.from_pretrained("gpt2-xl").cuda().eval()
    tok = AutoTokenizer.from_pretrained("gpt2-xl")
    tok.pad_token = tok.eos_token
    hparams = ROMEHyperParams.from_json(HPARAMS_DIR / "ROME" / "gpt2-xl.json")
    device = next(model.parameters()).device
    print(f"   Model loaded on {device}")

    # Set context templates (for potential ROME calls)
    with open(EXP1_DIR / "context_templates.json") as f:
        templates = json.load(f)
    rome_main_module.CONTEXT_TEMPLATES_CACHE = templates

    # ── Load Exp 1 artifacts ──────────────────────────────────
    print("\n[2] Loading Experiment 1 artifacts...")
    with open(EXP1_DIR / "record_metadata.json") as f:
        metadata = json.load(f)
    with open(EXP1_DIR / "relation_targets.json") as f:
        relation_targets = json.load(f)

    # Load activations for our 5 relations
    activations = {}
    for rid in RELATIONS:
        act_path = EXP1_DIR / "activations" / f"{rid}.pt"
        activations[rid] = torch.load(act_path, map_location="cpu")
        n = len(metadata[rid])
        print(f"   {rid}: {n} records, 12 layers")

    # ── Load Exp 2 artifacts ──────────────────────────────────
    print("\n[3] Loading Experiment 2 artifacts...")
    exp2_data = torch.load(EXP2_DIR / "edit_vectors.pt", map_location="cpu")
    exp2_meta = exp2_data["metadata"]
    exp2_V = exp2_data["v"].numpy()  # (500, 1600)
    exp2_U = exp2_data["u"].numpy()  # (500, 6400)

    lda_all = np.load(EXP2_DIR / "lda_directions.npy")  # (1600, 1600)
    lda_dirs = lda_all[:, :N_LDA_DIRS]  # (1600, 9), each column is a direction
    print(f"   {len(exp2_meta)} edit vectors, LDA dirs shape {lda_dirs.shape}")

    # ── Define tasks ──────────────────────────────────────────
    tasks = []
    for rid in RELATIONS:
        for target in relation_targets[rid][:2]:
            tasks.append({"relation_id": rid, "target": target})
    print(f"\n   {len(tasks)} concepts: "
          + ", ".join(f"{t['relation_id']}_{t['target']}" for t in tasks))

    # ==============================================================
    # PHASE A: Concept directions at all 12 layers
    # ==============================================================
    print("\n" + "=" * 70)
    print("PHASE A: CONCEPT DIRECTION EXTRACTION AT ALL LAYERS")
    print("=" * 70)

    concept_directions = {}  # (rid, target, method, layer) -> numpy unit vector
    probe_accuracies = {}    # (rid, target, layer) -> bal_acc

    for task in tasks:
        rid = task["relation_id"]
        target = task["target"]
        records = metadata[rid]

        # Binary labels: target vs rest
        labels = np.array([
            1 if r.get("target_true", r.get("target")) == target else 0
            for r in records
        ])
        n_pos, n_neg = int(labels.sum()), int((1 - labels).sum())
        if n_pos < 10 or n_neg < 10:
            print(f"   SKIP {rid}_{target}: {n_pos} pos, {n_neg} neg")
            continue

        # Stratified train/test split (deterministic)
        rng = np.random.RandomState(42)
        pos_idx = np.where(labels == 1)[0]
        neg_idx = np.where(labels == 0)[0]
        rng.shuffle(pos_idx)
        rng.shuffle(neg_idx)
        n_pos_tr = int(len(pos_idx) * TRAIN_RATIO)
        n_neg_tr = int(len(neg_idx) * TRAIN_RATIO)
        train_idx = np.concatenate([pos_idx[:n_pos_tr], neg_idx[:n_neg_tr]])
        test_idx = np.concatenate([pos_idx[n_pos_tr:], neg_idx[n_neg_tr:]])

        for layer in PROBE_LAYERS:
            X_all = activations[rid][layer].numpy()
            X_train, X_test = X_all[train_idx], X_all[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            # Logistic probe direction
            result = train_probe(X_train, y_train, X_test, y_test)
            if result is not None:
                concept_directions[(rid, target, "logistic", layer)] = result["direction"]
                probe_accuracies[(rid, target, layer)] = result["balanced_accuracy"]

            # Mean-diff direction
            pos_mean = X_all[labels == 1].mean(axis=0)
            neg_mean = X_all[labels == 0].mean(axis=0)
            md = pos_mean - neg_mean
            md_norm = np.linalg.norm(md)
            if md_norm > 1e-10:
                concept_directions[(rid, target, "mean_diff", layer)] = md / md_norm

        print(f"   {rid}_{target}: "
              f"bal_acc@17={probe_accuracies.get((rid, target, 17), 0):.3f}, "
              f"@40={probe_accuracies.get((rid, target, 40), 0):.3f}, "
              f"@47={probe_accuracies.get((rid, target, 47), 0):.3f}")

    # Save concept directions
    cd_save = {}
    for (rid, target, method, layer), d in concept_directions.items():
        cd_save[f"{rid}_{target}_{method}_L{layer}"] = torch.tensor(d, dtype=torch.float32)
    torch.save(cd_save, RESULTS_DIR / "concept_directions_all_layers.pt")
    print(f"\n   Saved {len(cd_save)} concept directions")

    # ==============================================================
    # PHASE B: Perturbation extraction
    # ==============================================================
    print("\n" + "=" * 70)
    print("PHASE B: EDIT PERTURBATION EXTRACTION")
    print("=" * 70)

    # Select edits from Exp 2
    selected_edits = []
    for task in tasks:
        rid = task["relation_id"]
        target = task["target"]
        matches = [(i, m) for i, m in enumerate(exp2_meta)
                   if m["relation_id"] == rid and m["target_value"] == target]
        for i, m in matches[:N_PER_CONCEPT]:
            selected_edits.append({
                "idx": i,
                "relation_id": rid,
                "target": target,
                "subject": m["subject"],
                "prompt": m["prompt"],  # template with {}
                "u": exp2_U[i],
                "v": exp2_V[i],
            })
    print(f"   Selected {len(selected_edits)} edits for propagation tracking")

    # Weight reference for edit application
    weight_name = f"{hparams.rewrite_module_tmp.format(EDIT_LAYER)}.weight"
    w = nethook.get_parameter(model, weight_name)
    layer_names = [f"transformer.h.{l}" for l in ALL_LAYERS]

    all_perturbations = []
    v_sanity_cosines = []

    for ei, edit in enumerate(selected_edits):
        if (ei + 1) % 20 == 0 or ei == 0:
            print(f"   Edit {ei+1}/{len(selected_edits)}: "
                  f"{edit['relation_id']}_{edit['target']} ({edit['subject'][:30]})")

        prompt_template = edit["prompt"]
        subject = edit["subject"]

        # Find subject_last token position
        try:
            subj_idxs = repr_tools.get_words_idxs_in_templates(
                tok, [prompt_template], [subject], "last"
            )
            subj_pos = subj_idxs[0][0]
        except Exception:
            # Fallback: tokenize filled text, use second-to-last token
            filled = prompt_template.format(subject)
            subj_pos = len(tok(filled)["input_ids"]) - 2

        filled_text = prompt_template.format(subject)
        inputs = tok(filled_text, return_tensors="pt").to(device)
        seq_len = inputs["input_ids"].shape[1]
        last_pos = seq_len - 1

        # Clamp positions
        subj_pos = min(subj_pos, seq_len - 1)
        last_pos = seq_len - 1

        # ── Forward pass: ORIGINAL model ──
        with torch.no_grad():
            with nethook.TraceDict(
                module=model, layers=layer_names,
                retain_output=True, retain_input=False,
            ) as tr_orig:
                logits_orig = model(**inputs).logits

        h_orig_subj = {}
        h_orig_last = {}
        for l in ALL_LAYERS:
            out = tr_orig[f"transformer.h.{l}"].output
            if isinstance(out, tuple):
                out = out[0]
            h_orig_subj[l] = out[0, subj_pos].detach().cpu().numpy()
            h_orig_last[l] = out[0, last_pos].detach().cpu().numpy()

        # ── Apply edit ──
        u_t = torch.tensor(edit["u"], dtype=torch.float32)
        v_t = torch.tensor(edit["v"], dtype=torch.float32)
        upd = u_t.unsqueeze(1).to(device) @ v_t.unsqueeze(0).to(device)
        if upd.shape != w.shape:
            upd = upd.T

        with torch.no_grad():
            w[...] += upd

        # ── Forward pass: EDITED model ──
        with torch.no_grad():
            with nethook.TraceDict(
                module=model, layers=layer_names,
                retain_output=True, retain_input=False,
            ) as tr_edit:
                logits_edited = model(**inputs).logits

        h_edit_subj = {}
        h_edit_last = {}
        for l in ALL_LAYERS:
            out = tr_edit[f"transformer.h.{l}"].output
            if isinstance(out, tuple):
                out = out[0]
            h_edit_subj[l] = out[0, subj_pos].detach().cpu().numpy()
            h_edit_last[l] = out[0, last_pos].detach().cpu().numpy()

        # ── Efficacy check ──
        target_str = edit["target"]
        target_tok_id = tok(f" {target_str.strip()}", return_tensors="pt")["input_ids"][0][0].item()
        pred_orig = logits_orig[0, -1].argmax().item()
        pred_edited = logits_edited[0, -1].argmax().item()
        probs_edited = torch.softmax(logits_edited[0, -1].float(), dim=0)
        target_prob = probs_edited[target_tok_id].item()
        efficacy = int(pred_edited == target_tok_id)

        # ── Restore weights ──
        with torch.no_grad():
            w[...] -= upd

        # ── Compute perturbations ──
        delta_h_subj = {}
        delta_h_last = {}
        norms_subj = {}
        norms_last = {}

        for l in ALL_LAYERS:
            ds = h_edit_subj[l] - h_orig_subj[l]
            dl = h_edit_last[l] - h_orig_last[l]
            delta_h_subj[l] = ds
            delta_h_last[l] = dl
            norms_subj[l] = float(np.linalg.norm(ds))
            norms_last[l] = float(np.linalg.norm(dl))

        # ── v-vector sanity check ──
        v_cos = cosine(delta_h_subj[EDIT_LAYER], edit["v"])
        v_sanity_cosines.append(v_cos)

        all_perturbations.append({
            "relation_id": edit["relation_id"],
            "target": edit["target"],
            "subject": edit["subject"],
            "efficacy": efficacy,
            "target_prob": target_prob,
            "pred_orig": tok.decode([pred_orig]),
            "pred_edited": tok.decode([pred_edited]),
            "v_cos_sanity": v_cos,
            "delta_h_subj": delta_h_subj,
            "delta_h_last": delta_h_last,
            "norms_subj": norms_subj,
            "norms_last": norms_last,
        })

    # Save perturbations (norms only in main dict; full Δh in separate tensor file)
    delta_subj_tensor = np.stack([
        np.stack([p["delta_h_subj"][l] for l in ALL_LAYERS])
        for p in all_perturbations
    ])  # (n_edits, 48, 1600)
    delta_last_tensor = np.stack([
        np.stack([p["delta_h_last"][l] for l in ALL_LAYERS])
        for p in all_perturbations
    ])  # (n_edits, 48, 1600)
    torch.save({
        "delta_subj": torch.tensor(delta_subj_tensor, dtype=torch.float32),
        "delta_last": torch.tensor(delta_last_tensor, dtype=torch.float32),
        "metadata": [{
            "relation_id": p["relation_id"], "target": p["target"],
            "subject": p["subject"], "efficacy": p["efficacy"],
            "target_prob": p["target_prob"],
        } for p in all_perturbations],
    }, RESULTS_DIR / "perturbations.pt")

    # Sanity checks
    pre_edit_norms = [all_perturbations[0]["norms_subj"][l] for l in range(EDIT_LAYER)]
    max_pre = max(pre_edit_norms) if pre_edit_norms else 0
    mean_v_cos = np.mean(v_sanity_cosines)
    min_v_cos = np.min(v_sanity_cosines)

    print(f"\n   Sanity checks:")
    print(f"     Max ||Δh|| at layers 0-{EDIT_LAYER-1}: {max_pre:.8f} (should be ~0)")
    print(f"     cos(Δh_17, v): mean={mean_v_cos:.4f}, min={min_v_cos:.4f} (should be ~1.0)")
    print(f"     Mean efficacy: {np.mean([p['efficacy'] for p in all_perturbations]):.3f}")
    print(f"     Mean target_prob: {np.mean([p['target_prob'] for p in all_perturbations]):.4f}")

    # ==============================================================
    # PHASE C: Alignment analysis
    # ==============================================================
    print("\n" + "=" * 70)
    print("PHASE C: ALIGNMENT ANALYSIS")
    print("=" * 70)

    alignment_results = []

    for pi, pert in enumerate(all_perturbations):
        rid = pert["relation_id"]
        target = pert["target"]

        for layer in ALIGNMENT_LAYERS:
            dh_subj = pert["delta_h_subj"][layer]
            dh_last = pert["delta_h_last"][layer]

            # Skip if perturbation is zero
            if np.linalg.norm(dh_subj) < 1e-10:
                continue

            # Concept direction (logistic)
            cd = concept_directions.get((rid, target, "logistic", layer))
            md = concept_directions.get((rid, target, "mean_diff", layer))

            # Wrong-concept direction
            other_targets = [t["target"] for t in tasks
                            if t["relation_id"] == rid and t["target"] != target]
            wrong_cd = None
            if other_targets:
                wrong_cd = concept_directions.get(
                    (rid, other_targets[0], "logistic", layer))

            # Random direction (deterministic per edit+layer)
            rng = np.random.RandomState(pi * 100 + layer)
            rand_dir = rng.randn(1600).astype(np.float64)
            rand_dir /= np.linalg.norm(rand_dir)

            row = {
                "edit_idx": pi,
                "relation_id": rid,
                "target": target,
                "layer": layer,
                "efficacy": pert["efficacy"],
                "target_prob": pert["target_prob"],
                # Subject token alignment
                "cos_concept_subj": cosine(dh_subj, cd) if cd is not None else None,
                "abs_concept_subj": abs(cosine(dh_subj, cd)) if cd is not None else None,
                "cos_meandiff_subj": cosine(dh_subj, md) if md is not None else None,
                "abs_meandiff_subj": abs(cosine(dh_subj, md)) if md is not None else None,
                # Last token alignment
                "cos_concept_last": cosine(dh_last, cd) if cd is not None else None,
                "abs_concept_last": abs(cosine(dh_last, cd)) if cd is not None else None,
                # Controls
                "abs_wrong_subj": abs(cosine(dh_subj, wrong_cd)) if wrong_cd is not None else None,
                "abs_random_subj": abs(cosine(dh_subj, rand_dir)),
                # Norms
                "norm_subj": pert["norms_subj"][layer],
                "norm_last": pert["norms_last"][layer],
                "probe_bal_acc": probe_accuracies.get((rid, target, layer)),
            }
            alignment_results.append(row)

    # Aggregate by concept × layer
    concept_layer_agg = defaultdict(lambda: defaultdict(list))
    for row in alignment_results:
        key = f"{row['relation_id']}_{row['target']}"
        concept_layer_agg[key][row["layer"]].append(row)

    summary_by_concept = {}
    for concept_key, layer_data in concept_layer_agg.items():
        summary_by_concept[concept_key] = {}
        for layer, rows in sorted(layer_data.items()):
            def safe_mean(vals):
                vals = [v for v in vals if v is not None]
                return float(np.mean(vals)) if vals else None

            summary_by_concept[concept_key][layer] = {
                "n": len(rows),
                "mean_abs_concept_subj": safe_mean([r["abs_concept_subj"] for r in rows]),
                "std_abs_concept_subj": float(np.std([r["abs_concept_subj"] for r in rows
                                                       if r["abs_concept_subj"] is not None])) if any(r["abs_concept_subj"] is not None for r in rows) else None,
                "mean_cos_concept_subj": safe_mean([r["cos_concept_subj"] for r in rows]),
                "mean_abs_concept_last": safe_mean([r["abs_concept_last"] for r in rows]),
                "mean_abs_meandiff_subj": safe_mean([r["abs_meandiff_subj"] for r in rows]),
                "mean_abs_wrong_subj": safe_mean([r["abs_wrong_subj"] for r in rows]),
                "mean_abs_random_subj": safe_mean([r["abs_random_subj"] for r in rows]),
                "mean_norm_subj": float(np.mean([r["norm_subj"] for r in rows])),
                "mean_norm_last": float(np.mean([r["norm_last"] for r in rows])),
                "probe_bal_acc": rows[0].get("probe_bal_acc"),
            }

    # Print alignment progression table (subject token)
    print("\n   Concept alignment |cos(Δh, concept_dir)| at SUBJECT token:")
    print(f"   {'Concept':<25} " + " ".join(f"L{l:>2}" for l in ALIGNMENT_LAYERS))
    print("   " + "-" * (25 + len(ALIGNMENT_LAYERS) * 6))

    for ck in sorted(summary_by_concept):
        vals = []
        for l in ALIGNMENT_LAYERS:
            v = summary_by_concept[ck].get(l, {}).get("mean_abs_concept_subj")
            vals.append(f"{v:5.3f}" if v is not None else "  N/A")
        print(f"   {ck:<25} " + " ".join(vals))

    # Grand average
    grand_avg_subj = {}
    for layer in ALIGNMENT_LAYERS:
        vals = [summary_by_concept[ck][layer]["mean_abs_concept_subj"]
                for ck in summary_by_concept
                if layer in summary_by_concept[ck]
                and summary_by_concept[ck][layer]["mean_abs_concept_subj"] is not None]
        grand_avg_subj[layer] = float(np.mean(vals)) if vals else 0
    print(f"   {'GRAND AVERAGE':<25} "
          + " ".join(f"{grand_avg_subj.get(l, 0):5.3f}" for l in ALIGNMENT_LAYERS))

    # Print alignment at LAST token
    print("\n   Concept alignment |cos(Δh, concept_dir)| at LAST (prediction) token:")
    print(f"   {'Concept':<25} " + " ".join(f"L{l:>2}" for l in ALIGNMENT_LAYERS))
    print("   " + "-" * (25 + len(ALIGNMENT_LAYERS) * 6))

    grand_avg_last = {}
    for layer in ALIGNMENT_LAYERS:
        vals = [summary_by_concept[ck][layer]["mean_abs_concept_last"]
                for ck in summary_by_concept
                if layer in summary_by_concept[ck]
                and summary_by_concept[ck][layer]["mean_abs_concept_last"] is not None]
        grand_avg_last[layer] = float(np.mean(vals)) if vals else 0

    for ck in sorted(summary_by_concept):
        vals = []
        for l in ALIGNMENT_LAYERS:
            v = summary_by_concept[ck].get(l, {}).get("mean_abs_concept_last")
            vals.append(f"{v:5.3f}" if v is not None else "  N/A")
        print(f"   {ck:<25} " + " ".join(vals))
    print(f"   {'GRAND AVERAGE':<25} "
          + " ".join(f"{grand_avg_last.get(l, 0):5.3f}" for l in ALIGNMENT_LAYERS))

    # Controls
    print("\n   Controls at subject token (grand average):")
    for layer in ALIGNMENT_LAYERS:
        wrong_vals = [summary_by_concept[ck][layer]["mean_abs_wrong_subj"]
                      for ck in summary_by_concept
                      if layer in summary_by_concept[ck]
                      and summary_by_concept[ck][layer]["mean_abs_wrong_subj"] is not None]
        rand_vals = [summary_by_concept[ck][layer]["mean_abs_random_subj"]
                     for ck in summary_by_concept
                     if layer in summary_by_concept[ck]
                     and summary_by_concept[ck][layer]["mean_abs_random_subj"] is not None]
        wrong_m = float(np.mean(wrong_vals)) if wrong_vals else 0
        rand_m = float(np.mean(rand_vals)) if rand_vals else 0
        print(f"   L{layer:2d}: concept={grand_avg_subj.get(layer, 0):.4f}  "
              f"wrong={wrong_m:.4f}  random={rand_m:.4f}")

    # ==============================================================
    # PHASE D: Additional analyses
    # ==============================================================
    print("\n" + "=" * 70)
    print("PHASE D: ADDITIONAL ANALYSES")
    print("=" * 70)

    # D1: Perturbation magnitude at all 48 layers
    print("\n   [D1] Perturbation magnitude ||Δh|| (averaged across edits)")
    norm_by_layer_subj = defaultdict(list)
    norm_by_layer_last = defaultdict(list)
    for pert in all_perturbations:
        for l in ALL_LAYERS:
            norm_by_layer_subj[l].append(pert["norms_subj"][l])
            norm_by_layer_last[l].append(pert["norms_last"][l])

    mean_norms_subj = {l: float(np.mean(norm_by_layer_subj[l])) for l in ALL_LAYERS}
    mean_norms_last = {l: float(np.mean(norm_by_layer_last[l])) for l in ALL_LAYERS}

    for l in [0, 5, 10, 15, 16, 17, 18, 19, 20, 25, 30, 35, 40, 47]:
        print(f"   L{l:2d}: subj={mean_norms_subj[l]:8.4f}  last={mean_norms_last[l]:8.4f}")

    # D2: Cross-position — when does edit signal reach the last token?
    print("\n   [D2] Cross-position transfer (||Δh_last||/||Δh_subj|| ratio)")
    for l in [17, 18, 19, 20, 25, 30, 35, 40, 47]:
        s = mean_norms_subj.get(l, 1e-10)
        ratio = mean_norms_last[l] / max(s, 1e-10)
        print(f"   L{l:2d}: ratio={ratio:.4f}  "
              f"(||Δh_last||={mean_norms_last[l]:.4f}, ||Δh_subj||={s:.4f})")

    # D3: v-vector sanity check already printed above

    # D4: LDA alignment by layer (crossover analysis)
    print("\n   [D4] LDA alignment vs concept alignment by layer")
    print("   (LDA directions from v-space at layer 17)")
    lda_alignment_by_layer = {}
    for l in ALIGNMENT_LAYERS:
        lda_vals = []
        for pert in all_perturbations:
            dh = pert["delta_h_subj"][l]
            if np.linalg.norm(dh) < 1e-10:
                continue
            # Max |cos| across top-3 LDA directions
            max_cos = max(abs(cosine(dh, lda_dirs[:, d])) for d in range(min(3, N_LDA_DIRS)))
            lda_vals.append(max_cos)
        lda_alignment_by_layer[l] = float(np.mean(lda_vals)) if lda_vals else 0

    print(f"   {'Layer':<8} {'|cos(Δh,concept)|':<20} {'max|cos(Δh,LDA)|':<20}")
    for l in ALIGNMENT_LAYERS:
        c_val = grand_avg_subj.get(l, 0)
        lda_val = lda_alignment_by_layer.get(l, 0)
        marker = " <-- crossover" if lda_val < c_val and l > EDIT_LAYER else ""
        print(f"   L{l:<5d} {c_val:<20.4f} {lda_val:<20.4f}{marker}")

    # D5: Alignment-efficacy correlation
    print("\n   [D5] Alignment-efficacy correlation")
    late_alignments = []
    efficacies = []
    target_probs = []

    for pi, pert in enumerate(all_perturbations):
        rid = pert["relation_id"]
        target = pert["target"]
        aligns = []
        for layer in [35, 40, 47]:
            cd = concept_directions.get((rid, target, "logistic", layer))
            if cd is not None:
                aligns.append(abs(cosine(pert["delta_h_subj"][layer], cd)))
        if aligns:
            late_alignments.append(np.mean(aligns))
            efficacies.append(pert["efficacy"])
            target_probs.append(pert["target_prob"])

    corr_eff = corr_prob = p_eff = p_prob = None
    if len(late_alignments) > 10:
        from scipy import stats
        corr_eff, p_eff = stats.spearmanr(late_alignments, efficacies)
        corr_prob, p_prob = stats.spearmanr(late_alignments, target_probs)
        print(f"   Late-layer alignment vs efficacy:    ρ={corr_eff:.3f}, p={p_eff:.4f}")
        print(f"   Late-layer alignment vs target_prob: ρ={corr_prob:.3f}, p={p_prob:.4f}")
    else:
        print(f"   Not enough data for correlation (n={len(late_alignments)})")

    # D6: Per-relation breakdown
    print("\n   [D6] Per-relation alignment at key layers")
    for rid in RELATIONS:
        cks = [ck for ck in summary_by_concept if ck.startswith(rid + "_")]
        if not cks:
            continue
        vals = {}
        for l in [17, 40, 47]:
            layer_vals = [summary_by_concept[ck].get(l, {}).get("mean_abs_concept_subj")
                         for ck in cks]
            layer_vals = [v for v in layer_vals if v is not None]
            vals[l] = float(np.mean(layer_vals)) if layer_vals else 0
        delta = vals[40] - vals[17]
        print(f"   {rid}: L17={vals[17]:.3f} → L40={vals[40]:.3f} → L47={vals[47]:.3f}"
              f"  (Δ17→40: {delta:+.3f})")

    # ==============================================================
    # SAVE RESULTS
    # ==============================================================
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    results = {
        "config": {
            "relations": RELATIONS,
            "probe_layers": PROBE_LAYERS,
            "alignment_layers": ALIGNMENT_LAYERS,
            "n_per_concept": N_PER_CONCEPT,
            "edit_layer": EDIT_LAYER,
            "n_edits": len(selected_edits),
            "n_concepts": len(tasks),
        },
        "sanity_checks": {
            "max_pre_edit_norm": float(max_pre),
            "v_cos_mean": float(mean_v_cos),
            "v_cos_min": float(min_v_cos),
            "v_cos_all": [float(c) for c in v_sanity_cosines],
        },
        "probe_accuracies": {
            f"{rid}_{target}_L{layer}": float(acc)
            for (rid, target, layer), acc in probe_accuracies.items()
        },
        "alignment_by_concept": {
            ck: {
                str(layer): {k: v for k, v in data.items()}
                for layer, data in ld.items()
            }
            for ck, ld in summary_by_concept.items()
        },
        "grand_average": {
            "subject_token": {str(l): float(v) for l, v in grand_avg_subj.items()},
            "last_token": {str(l): float(v) for l, v in grand_avg_last.items()},
        },
        "perturbation_norms": {
            "subject_token": {str(l): float(v) for l, v in mean_norms_subj.items()},
            "last_token": {str(l): float(v) for l, v in mean_norms_last.items()},
        },
        "lda_alignment_by_layer": {
            str(l): float(v) for l, v in lda_alignment_by_layer.items()
        },
        "efficacy_correlation": {
            "spearman_rho_efficacy": float(corr_eff) if corr_eff is not None else None,
            "p_efficacy": float(p_eff) if p_eff is not None else None,
            "spearman_rho_target_prob": float(corr_prob) if corr_prob is not None else None,
            "p_target_prob": float(p_prob) if p_prob is not None else None,
            "n": len(late_alignments),
        },
        "per_edit": [{
            "relation_id": p["relation_id"],
            "target": p["target"],
            "subject": p["subject"],
            "efficacy": p["efficacy"],
            "target_prob": p["target_prob"],
            "pred_orig": p["pred_orig"],
            "pred_edited": p["pred_edited"],
            "v_cos_sanity": p["v_cos_sanity"],
            "alignment_subj": {
                str(l): float(abs(cosine(
                    p["delta_h_subj"][l],
                    concept_directions.get((p["relation_id"], p["target"], "logistic", l),
                                          np.zeros(1600))
                ))) if (p["relation_id"], p["target"], "logistic", l) in concept_directions
                else None
                for l in ALIGNMENT_LAYERS
            },
            "norms_subj": {str(l): p["norms_subj"][l]
                          for l in [17, 20, 25, 30, 35, 40, 47]},
            "norms_last": {str(l): p["norms_last"][l]
                          for l in [17, 20, 25, 30, 35, 40, 47]},
        } for p in all_perturbations],
    }

    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"   Saved results to {RESULTS_DIR}/")

    # ==============================================================
    # SUMMARY
    # ==============================================================
    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print("EXPERIMENT 3 SUMMARY")
    print("=" * 70)

    a17 = grand_avg_subj.get(17, 0)
    a40 = grand_avg_subj.get(40, 0)
    a47 = grand_avg_subj.get(47, 0)

    print(f"\n   Grand average |cos(Δh, concept_dir)| at SUBJECT token:")
    print(f"   Layer 17 (edit site):  {a17:.4f}")
    print(f"   Layer 40 (probe peak): {a40:.4f}")
    print(f"   Layer 47 (final):      {a47:.4f}")
    print(f"   Change L17→L40:        {a40 - a17:+.4f}")
    print(f"   Change L17→L47:        {a47 - a17:+.4f}")

    rotation = a40 > a17 + 0.05
    print(f"\n   Rotation detected (Δ>0.05 from L17→L40): {'YES' if rotation else 'NO'}")

    if corr_eff is not None:
        print(f"   Alignment-efficacy correlation: ρ={corr_eff:.3f} (p={p_eff:.4f})")

    n_eff = sum(p["efficacy"] for p in all_perturbations)
    print(f"\n   Edit success: {n_eff}/{len(all_perturbations)} "
          f"({100*n_eff/len(all_perturbations):.0f}%)")
    print(f"   Mean target prob: "
          f"{np.mean([p['target_prob'] for p in all_perturbations]):.4f}")
    print(f"\n   Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
