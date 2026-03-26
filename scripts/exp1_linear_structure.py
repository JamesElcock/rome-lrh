"""
Experiment 1: Linear Concept Structure at the Edit Layer

Tests whether the residual stream at layer 17 of GPT-2 XL encodes factual
concepts as linear directions, recoverable by multiple independent methods.

Protocol:
1. Select 10 relations x 2 target concepts = 20 binary classification tasks
2. Extract activations at 12 layers for all records
3. Train LogisticRegressionCV probes (20 tasks x 12 layers = 240 probes)
4. Extract concept directions via mean-diff, DAS/SVD, logistic at layer 17
5. Measure cross-method agreement
6. Measure inter-relation geometry
7. Permutation baselines (100 shuffles per task at layer 17)
8. Negative control relations

Saves results to results/exp1/
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
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from rome import repr_tools
from rome.rome_hparams import ROMEHyperParams
from rome.rome_main import get_context_templates
from util import nethook
from util.globals import HPARAMS_DIR

# ====== CONFIGURATION ======
LAYERS = [0, 5, 10, 13, 15, 17, 20, 25, 30, 35, 40, 47]
EDIT_LAYER = 17
RESULTS_DIR = Path("results/exp1")
N_PERMUTATIONS = 100
TRAIN_RATIO = 0.70
BATCH_SIZE = 24
MIN_TARGET_COUNT = 20

MAIN_RELATIONS = ["P176", "P27", "P495", "P37", "P17",
                   "P413", "P1412", "P937", "P106", "P449"]
CONTROL_RELATIONS = ["P264", "P463", "P138"]


def select_top_targets(records, n=2, min_count=MIN_TARGET_COUNT):
    """Return the top n target values with at least min_count records."""
    counts = Counter(
        r["requested_rewrite"]["target_true"]["str"].strip() for r in records
    )
    return [t for t, c in counts.most_common() if c >= min_count][:n]


def extract_multi_layer(model, tok, prompts, subjects, layers, batch_size=BATCH_SIZE):
    """
    Efficient multi-layer extraction using TraceDict.
    Extracts residual stream at subject_last token, all layers in one forward pass.
    """
    device = next(model.parameters()).device
    layer_names = [f"transformer.h.{l}" for l in layers]
    all_acts = {l: [] for l in layers}

    for i in range(0, len(prompts), batch_size):
        bp = prompts[i : i + batch_size]
        bs = subjects[i : i + batch_size]

        # Compute subject_last token indices
        idxs = repr_tools.get_words_idxs_in_templates(tok, bp, bs, "last")

        # Tokenize batch
        texts = [p.format(s) for p, s in zip(bp, bs)]
        inputs = tok(texts, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            with nethook.TraceDict(
                module=model,
                layers=layer_names,
                retain_input=False,
                retain_output=True,
            ) as td:
                model(**inputs)

        for layer, lname in zip(layers, layer_names):
            act = td[lname].output
            if isinstance(act, tuple):
                act = act[0]
            batch_acts = torch.stack(
                [act[j, idxs[j][0]] for j in range(len(bp))]
            )
            all_acts[layer].append(batch_acts.detach().cpu())

    return {l: torch.cat(vs, 0).float() for l, vs in all_acts.items()}


def train_probe(X_train, y_train, X_test, y_test):
    """Train LogisticRegressionCV, return metrics and direction in original space."""
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    clf = LogisticRegressionCV(
        Cs=[1e-4, 1e-3, 1e-2, 1e-1, 1, 10],
        cv=5,
        max_iter=1000,
        scoring="balanced_accuracy",
        random_state=42,
    )
    clf.fit(Xtr, y_train)

    y_pred = clf.predict(Xte)
    acc = float((y_pred == y_test).mean())
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))

    # Transform direction back to original (unscaled) space
    w = clf.coef_[0] / scaler.scale_
    w = w / (np.linalg.norm(w) + 1e-10)

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "best_C": float(clf.C_[0]),
        "direction": w,
    }


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR / "activations", exist_ok=True)

    print("=" * 60)
    print("Experiment 1: Linear Concept Structure at Edit Layer")
    print("=" * 60)

    # ===== LOAD DATA =====
    print("\n[1/10] Loading CounterFact data...")
    data = json.load(open("data/counterfact.json"))
    by_rel = defaultdict(list)
    for r in data:
        rid = r.get("requested_rewrite", {}).get("relation_id")
        if rid:
            by_rel[rid].append(r)
    print(f"  {len(data)} records across {len(by_rel)} relations")

    # ===== SELECT TASKS =====
    print("\n[2/10] Selecting binary classification tasks...")
    tasks = []  # [(relation_id, target_value), ...]
    relation_targets = {}

    for rid in MAIN_RELATIONS:
        records = by_rel.get(rid, [])
        targets = select_top_targets(records, n=2)
        if not targets:
            print(f"  WARNING: {rid} has no targets with >= {MIN_TARGET_COUNT} records")
            continue
        relation_targets[rid] = targets
        for t in targets:
            tasks.append((rid, t))

    print(f"  {len(tasks)} tasks across {len(relation_targets)} relations:")
    for rid, targets in relation_targets.items():
        for t in targets:
            n = sum(
                1 for r in by_rel[rid]
                if r["requested_rewrite"]["target_true"]["str"].strip() == t.strip()
            )
            print(f"    {rid} '{t}' ({n} pos / {len(by_rel[rid])} total)")

    # ===== LOAD MODEL & EXTRACT ACTIVATIONS =====
    print("\n[3/10] Loading GPT-2 XL...")
    model = AutoModelForCausalLM.from_pretrained("gpt2-xl").cuda()
    tok = AutoTokenizer.from_pretrained("gpt2-xl")
    tok.pad_token = tok.eos_token
    model.eval()

    print("\n[4/10] Extracting activations at all layers...")
    t0 = time.time()
    activation_data = {}
    record_metadata = {}

    all_rels = list(set(MAIN_RELATIONS + CONTROL_RELATIONS))
    for rid in all_rels:
        records = by_rel.get(rid, [])
        if not records:
            continue

        prompts = [r["requested_rewrite"]["prompt"] for r in records]
        subjects = [r["requested_rewrite"]["subject"] for r in records]

        print(f"  {rid}: {len(records)} records...", end=" ", flush=True)
        acts = extract_multi_layer(model, tok, prompts, subjects, LAYERS)
        activation_data[rid] = acts

        record_metadata[rid] = [
            {
                "case_id": r["case_id"],
                "subject": r["requested_rewrite"]["subject"],
                "target_true": r["requested_rewrite"]["target_true"]["str"].strip(),
                "target_new": r["requested_rewrite"]["target_new"]["str"],
                "prompt": r["requested_rewrite"]["prompt"],
            }
            for r in records
        ]

        # NaN check
        n_nan = sum(acts[l].isnan().any().item() for l in LAYERS)
        print("WARNING: NaN detected!" if n_nan else "OK")

    # Save activations and metadata
    for rid, acts in activation_data.items():
        torch.save(acts, RESULTS_DIR / "activations" / f"{rid}.pt")
    with open(RESULTS_DIR / "record_metadata.json", "w") as f:
        json.dump(record_metadata, f, indent=2)

    # Generate and save context templates for future experiments
    print("\n  Generating context templates...")
    hparams = ROMEHyperParams.from_json(HPARAMS_DIR / "ROME" / "gpt2-xl.json")
    ctx_templates = get_context_templates(
        model, tok, hparams.context_template_length_params
    )
    with open(RESULTS_DIR / "context_templates.json", "w") as f:
        json.dump(ctx_templates, f)
    print(f"  Saved {len(ctx_templates)} context templates")

    # Free GPU
    del model
    torch.cuda.empty_cache()
    print(f"\n  Extraction complete in {time.time()-t0:.1f}s. GPU freed.")

    # ===== TRAIN PROBES =====
    print(
        f"\n[5/10] Training probes ({len(tasks)} tasks x {len(LAYERS)} layers"
        f" = {len(tasks)*len(LAYERS)} probes)..."
    )
    t0 = time.time()

    probe_results = {}
    probe_directions_l17 = {}

    for ti, (rid, target) in enumerate(tasks):
        records = by_rel[rid]
        labels = np.array(
            [
                1
                if r["requested_rewrite"]["target_true"]["str"].strip()
                == target.strip()
                else 0
                for r in records
            ]
        )

        rng = np.random.RandomState(42)
        idx = np.arange(len(labels))
        rng.shuffle(idx)
        n_train = int(TRAIN_RATIO * len(labels))
        train_idx, test_idx = idx[:n_train], idx[n_train:]

        print(
            f"\n  [{ti+1}/{len(tasks)}] {rid} '{target}' "
            f"(train={len(train_idx)}, test={len(test_idx)}, pos_rate={labels.mean():.2f})"
        )

        task_res = {}
        for layer in LAYERS:
            X = activation_data[rid][layer].numpy()
            res = train_probe(X[train_idx], labels[train_idx], X[test_idx], labels[test_idx])
            task_res[str(layer)] = {
                "accuracy": res["accuracy"],
                "balanced_accuracy": res["balanced_accuracy"],
                "best_C": res["best_C"],
            }

            if layer == EDIT_LAYER:
                probe_directions_l17[(rid, target)] = res["direction"]

        probe_results[f"{rid}_{target}"] = task_res

        accs = [task_res[str(l)]["balanced_accuracy"] for l in LAYERS]
        peak_l = LAYERS[np.argmax(accs)]
        print(
            f"    Layer 17: {task_res[str(EDIT_LAYER)]['balanced_accuracy']:.3f} | "
            f"Peak: layer {peak_l} ({max(accs):.3f})"
        )

    print(f"\n  Probe training done in {time.time()-t0:.1f}s")

    # ===== CONCEPT DIRECTIONS AT LAYER 17 =====
    print(f"\n[6/10] Extracting concept directions at layer {EDIT_LAYER}...")

    concept_dirs = {}

    for ti, (rid, target) in enumerate(tasks):
        records = by_rel[rid]
        labels = np.array(
            [
                1
                if r["requested_rewrite"]["target_true"]["str"].strip()
                == target.strip()
                else 0
                for r in records
            ]
        )
        X = activation_data[rid][EDIT_LAYER].numpy()
        pos_X, neg_X = X[labels == 1], X[labels == 0]

        # Method 1: Mean difference
        md = pos_X.mean(0) - neg_X.mean(0)
        md = md / (np.linalg.norm(md) + 1e-10)
        concept_dirs[(rid, target, "mean_diff")] = md

        # Method 2: DAS/SVD
        n = min(len(pos_X), len(neg_X))
        rng_das = np.random.RandomState(42)
        pi = rng_das.choice(len(pos_X), n, replace=len(pos_X) < n)
        ni = rng_das.choice(len(neg_X), n, replace=len(neg_X) < n)
        H = pos_X[pi] - neg_X[ni]
        H = H - H.mean(0, keepdims=True)
        _, S, Vh = np.linalg.svd(H, full_matrices=False)
        das = Vh[0]
        if np.dot(das, md) < 0:
            das = -das
        das_ev = float(S[0] ** 2 / (S**2).sum())
        concept_dirs[(rid, target, "das")] = das

        # Method 3: Logistic probe direction
        concept_dirs[(rid, target, "logistic")] = probe_directions_l17[(rid, target)]

        log_d = probe_directions_l17[(rid, target)]
        print(
            f"  {rid} '{target}': md-das={cosine(md,das):.3f} "
            f"md-log={cosine(md,log_d):.3f} das-log={cosine(das,log_d):.3f} "
            f"(DAS ev={das_ev:.2f})"
        )

    # ===== CROSS-METHOD AGREEMENT =====
    print(f"\n[7/10] Cross-method agreement...")

    agreement = {}
    for rid, target in tasks:
        md = concept_dirs[(rid, target, "mean_diff")]
        das = concept_dirs[(rid, target, "das")]
        log = concept_dirs[(rid, target, "logistic")]

        signed = {
            "md_das": cosine(md, das),
            "md_log": cosine(md, log),
            "das_log": cosine(das, log),
        }
        absolute = {k: abs(v) for k, v in signed.items()}
        agreement[f"{rid}_{target}"] = {
            "signed": signed,
            "absolute": absolute,
            "mean_abs": float(np.mean(list(absolute.values()))),
        }

    n_agree_pass = sum(1 for v in agreement.values() if v["mean_abs"] >= 0.60)
    mean_agree = float(np.mean([v["mean_abs"] for v in agreement.values()]))
    print(f"  Mean cross-method |cosine|: {mean_agree:.3f}")
    print(f"  Tasks with mean |cosine| >= 0.60: {n_agree_pass}/{len(tasks)}")

    # ===== INTER-RELATION GEOMETRY =====
    print(f"\n[8/10] Inter-relation geometry at layer {EDIT_LAYER}...")

    dir_list = [concept_dirs[(rid, target, "mean_diff")] for rid, target in tasks]
    dir_labels = [f"{rid}_{target}" for rid, target in tasks]
    D = np.stack(dir_list)
    cos_mat = D @ D.T  # all unit vectors

    within, cross = [], []
    for i in range(len(tasks)):
        for j in range(i + 1, len(tasks)):
            c = abs(cos_mat[i, j])
            (within if tasks[i][0] == tasks[j][0] else cross).append(c)

    print(
        f"  Within-relation |cos|: {np.mean(within):.3f} +/- {np.std(within):.3f} (n={len(within)})"
    )
    print(
        f"  Cross-relation |cos|:  {np.mean(cross):.3f} +/- {np.std(cross):.3f} (n={len(cross)})"
    )

    # ===== PERMUTATION BASELINES =====
    print(
        f"\n[9/10] Permutation baselines "
        f"({N_PERMUTATIONS} shuffles x {len(tasks)} tasks at layer {EDIT_LAYER})..."
    )
    t0 = time.time()

    perm_results = {}
    for ti, (rid, target) in enumerate(tasks):
        records = by_rel[rid]
        labels = np.array(
            [
                1
                if r["requested_rewrite"]["target_true"]["str"].strip()
                == target.strip()
                else 0
                for r in records
            ]
        )
        X = activation_data[rid][EDIT_LAYER].numpy()

        rng = np.random.RandomState(42)
        idx = np.arange(len(labels))
        rng.shuffle(idx)
        n_train = int(TRAIN_RATIO * len(labels))
        train_idx, test_idx = idx[:n_train], idx[n_train:]

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[train_idx])
        Xte = scaler.transform(X[test_idx])
        y_test = labels[test_idx]

        null_accs = []
        for p in range(N_PERMUTATIONS):
            y_shuf = labels.copy()
            np.random.RandomState(p * 1000 + ti).shuffle(y_shuf)
            clf = LogisticRegression(C=1.0, max_iter=500, random_state=42)
            clf.fit(Xtr, y_shuf[train_idx])
            null_accs.append(
                float(balanced_accuracy_score(y_test, clf.predict(Xte)))
            )

        real = probe_results[f"{rid}_{target}"][str(EDIT_LAYER)]["balanced_accuracy"]
        p_val = (np.sum(np.array(null_accs) >= real) + 1) / (N_PERMUTATIONS + 1)
        perm_results[f"{rid}_{target}"] = {
            "real": real,
            "null_mean": float(np.mean(null_accs)),
            "null_std": float(np.std(null_accs)),
            "p_value": float(p_val),
        }

        if (ti + 1) % 5 == 0 or ti == 0:
            print(
                f"  [{ti+1}/{len(tasks)}] {rid} '{target}': "
                f"real={real:.3f} null={np.mean(null_accs):.3f}+/-{np.std(null_accs):.3f} p={p_val:.4f}"
            )

    print(f"  Done in {time.time()-t0:.1f}s")

    # ===== NEGATIVE CONTROLS =====
    print(f"\n[10/10] Negative control relations...")

    ctrl_results = {}
    for rid in CONTROL_RELATIONS:
        records = by_rel.get(rid, [])
        if len(records) < 20 or rid not in activation_data:
            print(f"  SKIP {rid}: {len(records)} records")
            continue

        counts = Counter(
            r["requested_rewrite"]["target_true"]["str"].strip() for r in records
        )
        target = counts.most_common(1)[0][0]
        labels = np.array(
            [
                1
                if r["requested_rewrite"]["target_true"]["str"].strip()
                == target.strip()
                else 0
                for r in records
            ]
        )

        if labels.sum() < 5 or (len(labels) - labels.sum()) < 5:
            print(f"  SKIP {rid}: insufficient balance (pos={labels.sum()})")
            continue

        rng = np.random.RandomState(42)
        idx = np.arange(len(labels))
        rng.shuffle(idx)
        n_train = int(TRAIN_RATIO * len(labels))
        train_idx, test_idx = idx[:n_train], idx[n_train:]

        layer_accs = {}
        for layer in LAYERS:
            X = activation_data[rid][layer].numpy()
            res = train_probe(
                X[train_idx], labels[train_idx], X[test_idx], labels[test_idx]
            )
            layer_accs[str(layer)] = res["balanced_accuracy"]

        ctrl_results[rid] = {
            "target": target,
            "n_pos": int(labels.sum()),
            "n_total": len(labels),
            "layer_accs": layer_accs,
        }
        print(
            f"  {rid} '{target}' (pos={labels.sum()}, total={len(labels)}): "
            f"layer17={layer_accs.get(str(EDIT_LAYER), 'N/A')}"
        )

    # ===== SUMMARY =====
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    n_probe_pass = sum(
        1
        for k, v in probe_results.items()
        if v[str(EDIT_LAYER)]["balanced_accuracy"] >= 0.75
    )
    mean_cross = float(np.mean(cross)) if cross else 0.0
    ctrl_accs = [
        v["layer_accs"].get(str(EDIT_LAYER), 1.0) for v in ctrl_results.values()
    ]

    print(
        f"\n  Probe bal_acc >= 0.75 at layer 17: {n_probe_pass}/{len(tasks)} (target: >= 16)"
    )
    print(
        f"  Cross-method |cos| >= 0.60:        {n_agree_pass}/{len(tasks)} (target: >= 16)"
    )
    print(f"  Mean cross-relation |cos|:          {mean_cross:.3f} (target: <= 0.15)")
    if ctrl_accs:
        print(
            f"  Control relation mean accuracy:     {np.mean(ctrl_accs):.3f} (target: <= 0.60)"
        )

    gate = n_probe_pass >= 12
    print(
        f"\n  GATE: {'PASSED' if gate else 'REVIEW NEEDED'} "
        f"({n_probe_pass}/{len(tasks)} tasks pass probe threshold)"
    )

    if gate:
        passed = [
            (rid, target)
            for rid, target in tasks
            if probe_results[f"{rid}_{target}"][str(EDIT_LAYER)]["balanced_accuracy"]
            >= 0.75
        ]
        print(f"  Passed tasks ({len(passed)}):")
        for r, t in passed:
            ba = probe_results[f"{r}_{t}"][str(EDIT_LAYER)]["balanced_accuracy"]
            print(f"    {r} '{t}': {ba:.3f}")
    else:
        mean_by_layer = {
            l: np.mean(
                [v[str(l)]["balanced_accuracy"] for v in probe_results.values()]
            )
            for l in LAYERS
        }
        peak = max(mean_by_layer, key=mean_by_layer.get)
        print(f"  Peak mean accuracy at layer {peak} ({mean_by_layer[peak]:.3f})")

    # ===== SAVE RESULTS =====
    results = {
        "config": {
            "layers": LAYERS,
            "edit_layer": EDIT_LAYER,
            "n_permutations": N_PERMUTATIONS,
            "train_ratio": TRAIN_RATIO,
            "main_relations": MAIN_RELATIONS,
            "control_relations": CONTROL_RELATIONS,
        },
        "tasks": [{"relation_id": r, "target": t} for r, t in tasks],
        "probe_results": probe_results,
        "agreement": agreement,
        "inter_relation": {
            "within_mean": float(np.mean(within)),
            "within_std": float(np.std(within)),
            "cross_mean": mean_cross,
            "cross_std": float(np.std(cross)) if cross else 0.0,
            "cosine_matrix": cos_mat.tolist(),
            "labels": dir_labels,
        },
        "permutation": perm_results,
        "controls": ctrl_results,
        "summary": {
            "n_probe_pass": n_probe_pass,
            "n_agree_pass": n_agree_pass,
            "mean_cross_cos": mean_cross,
            "ctrl_mean": float(np.mean(ctrl_accs)) if ctrl_accs else None,
            "gate_passed": gate,
        },
    }

    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save concept directions as torch tensors
    dir_data = {}
    for (r, t, m), d in concept_dirs.items():
        dir_data[f"{r}_{t}_{m}"] = torch.from_numpy(d).float()
    torch.save(dir_data, RESULTS_DIR / "concept_directions_layer17.pt")

    # Save relation_targets for Experiment 1B
    with open(RESULTS_DIR / "relation_targets.json", "w") as f:
        json.dump(relation_targets, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}/")
    print("  results.json — probe/agreement/permutation/control results")
    print("  concept_directions_layer17.pt — direction vectors (3 methods)")
    print("  activations/*.pt — per-relation activation tensors")
    print("  record_metadata.json — case_ids, subjects, targets per relation")
    print("  context_templates.json — fixed templates for future experiments")
    print("  relation_targets.json — selected targets per relation")


if __name__ == "__main__":
    main()
