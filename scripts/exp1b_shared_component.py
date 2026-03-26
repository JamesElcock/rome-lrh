"""
Experiment 1B: Edit-Concept Alignment via Shared Component Analysis

Tests whether ROME's v vectors for edits targeting the same concept share a
common component aligned with the population-level concept direction from Exp 1.

Part A: Extract v vectors, compute shared components, measure alignment.
Part B: Cross-entity transfer test — apply v_mean with entity-specific u.

Updates from Experiment 1 results:
- Top 5 relations: P176, P1412, P37, P27, P413 (by avg layer-17 probe accuracy)
- DAS direction dropped (unreliable: explained variance 0.03-0.09)
- Using mean-diff and logistic directions only
- Layer 17 has weaker concept structure than expected (peak at layer 40)

Saves results to results/exp1b/
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

from rome.rome_main import execute_rome, get_context_templates
from rome.rome_hparams import ROMEHyperParams
from util import nethook
from util.globals import HPARAMS_DIR
import rome.rome_main as rome_main_module

# ====== CONFIGURATION ======
RESULTS_DIR = Path("results/exp1b")
EXP1_DIR = Path("results/exp1")
EDIT_LAYER = 17
N_ENTITIES = 30       # entities per target concept for Part A
N_HOLDOUT = 5         # held-out entities per concept for Part B transfer test
SEED = 42

# Top 5 relations from Experiment 1 (by avg probe accuracy at layer 17)
TOP_RELATIONS = ["P176", "P1412", "P37", "P27", "P413"]


def cosine(a, b):
    """Cosine similarity between numpy vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def load_exp1_results():
    """Load Experiment 1 outputs."""
    results = json.load(open(EXP1_DIR / "results.json"))
    directions = torch.load(EXP1_DIR / "concept_directions_layer17.pt")
    metadata = json.load(open(EXP1_DIR / "record_metadata.json"))
    templates = json.load(open(EXP1_DIR / "context_templates.json"))
    relation_targets = json.load(open(EXP1_DIR / "relation_targets.json"))
    return results, directions, metadata, templates, relation_targets


def select_entities(metadata, relation_id, target_value, n, seed=SEED):
    """
    Select n entities from the negative class (target_true != target_value)
    for editing TO target_value.
    """
    rng = np.random.RandomState(seed)
    records = metadata[relation_id]
    negatives = [r for r in records if r["target_true"].strip() != target_value.strip()]
    if len(negatives) < n:
        print(f"  WARNING: Only {len(negatives)} negatives for {relation_id} '{target_value}', need {n}")
        n = len(negatives)
    selected = rng.choice(len(negatives), n, replace=False)
    return [negatives[i] for i in selected]


def run_rome_edit(model, tok, request, hparams):
    """Run ROME and return (u, v) vectors. Model is unchanged after."""
    deltas = execute_rome(model, tok, request, hparams)
    for key, (u, v) in deltas.items():
        return u.detach().cpu(), v.detach().cpu()


def apply_edit_and_eval(model, tok, u, v, hparams, eval_prompts, target_str):
    """
    Apply rank-1 edit u ⊗ v, evaluate on prompts, then restore weights.
    Returns dict with efficacy and probabilities.
    """
    layer = hparams.layers[0]
    weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
    w = nethook.get_parameter(model, weight_name)

    # Compute update matrix
    upd = u.unsqueeze(1).to(w.device) @ v.unsqueeze(0).to(w.device)
    # Handle transpose (GPT-2 vs GPT-J weight convention)
    if upd.shape != w.shape:
        upd = upd.T

    # Apply edit
    with torch.no_grad():
        w[...] += upd

    # Evaluate
    target_tok = tok(f" {target_str.strip()}", return_tensors="pt")["input_ids"][0]
    first_target_tok = target_tok[0].item()

    results = []
    for prompt_text in eval_prompts:
        inputs = tok(prompt_text, return_tensors="pt").to(w.device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1]
        probs = torch.softmax(logits, dim=0)
        pred_tok = logits.argmax().item()
        target_prob = probs[first_target_tok].item()
        results.append({
            "correct": pred_tok == first_target_tok,
            "target_prob": target_prob,
            "pred_token": tok.decode([pred_tok]),
        })

    # Restore weights
    with torch.no_grad():
        w[...] -= upd

    n_correct = sum(r["correct"] for r in results)
    mean_prob = np.mean([r["target_prob"] for r in results])
    return {
        "efficacy": n_correct / len(results),
        "mean_target_prob": float(mean_prob),
        "details": results,
    }


def pca_enhanced_mean(V, variance_threshold=0.90):
    """
    PCA-enhanced estimation of the mean direction.
    V: (n, d) matrix of vectors.
    Returns (raw_mean, pca_mean, n_components, explained_variance_ratio).
    """
    raw_mean = V.mean(axis=0)
    raw_mean_norm = raw_mean / (np.linalg.norm(raw_mean) + 1e-10)

    # Center
    V_c = V - V.mean(axis=0, keepdims=True)
    U, S, Vh = np.linalg.svd(V_c, full_matrices=False)
    cum_var = np.cumsum(S**2) / (S**2).sum()
    k = int(np.searchsorted(cum_var, variance_threshold) + 1)
    k = min(k, len(S))

    # Project into top-k subspace, recompute mean
    basis = Vh[:k]  # (k, d)
    V_proj = V @ basis.T  # (n, k)
    mean_proj = V_proj.mean(axis=0)  # (k,)
    pca_mean = mean_proj @ basis  # (d,)
    pca_mean_norm = pca_mean / (np.linalg.norm(pca_mean) + 1e-10)

    return raw_mean_norm, pca_mean_norm, k, float(cum_var[k-1])


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("Experiment 1B: Shared Component Analysis")
    print("=" * 60)

    # ===== LOAD EXP 1 RESULTS =====
    print("\n[1/7] Loading Experiment 1 results...")
    exp1_results, exp1_dirs, metadata, templates, relation_targets = load_exp1_results()

    # Set context templates from Experiment 1
    rome_main_module.CONTEXT_TEMPLATES_CACHE = templates
    print(f"  Fixed {len(templates)} context templates from Experiment 1")

    # Determine tasks: top 5 relations x 2 targets
    tasks = []
    for rid in TOP_RELATIONS:
        targets = relation_targets.get(rid, [])
        for t in targets[:2]:
            tasks.append((rid, t))
    print(f"  {len(tasks)} target concepts across {len(TOP_RELATIONS)} relations:")
    for rid, target in tasks:
        acc = exp1_results["probe_results"].get(f"{rid}_{target}", {}).get(
            str(EDIT_LAYER), {}
        ).get("balanced_accuracy", 0)
        print(f"    {rid} '{target}' (probe bal_acc={acc:.3f})")

    # Load concept directions (mean_diff and logistic only — DAS dropped)
    concept_dirs = {}
    for rid, target in tasks:
        for method in ["mean_diff", "logistic"]:
            key = f"{rid}_{target}_{method}"
            if key in exp1_dirs:
                concept_dirs[(rid, target, method)] = exp1_dirs[key].numpy()

    # ===== LOAD MODEL =====
    print("\n[2/7] Loading GPT-2 XL and ROME hparams...")
    model = AutoModelForCausalLM.from_pretrained("gpt2-xl").cuda()
    tok = AutoTokenizer.from_pretrained("gpt2-xl")
    tok.pad_token = tok.eos_token
    model.eval()

    hparams = ROMEHyperParams.from_json(HPARAMS_DIR / "ROME" / "gpt2-xl.json")

    # Load CounterFact for paraphrase prompts
    cf_data = json.load(open("data/counterfact.json"))
    cf_by_id = {r["case_id"]: r for r in cf_data}

    # ===== PART A: EXTRACT V VECTORS =====
    print(f"\n[3/7] Part A: Extracting v vectors ({len(tasks)} concepts x {N_ENTITIES} entities)...")
    t0 = time.time()

    edit_vectors = {}  # {(rid, target, entity_idx): {"u": array, "v": array, "subject": str}}
    v_by_concept = {}  # {(rid, target): list of v arrays}

    for ti, (rid, target) in enumerate(tasks):
        entities = select_entities(metadata, rid, target, N_ENTITIES)
        v_list = []

        print(f"\n  [{ti+1}/{len(tasks)}] {rid} '{target}': {len(entities)} entities")

        for ei, entity in enumerate(entities):
            request = {
                "prompt": entity["prompt"],
                "subject": entity["subject"],
                "target_new": {"str": target},
            }

            try:
                u, v = run_rome_edit(model, tok, request, hparams)
                edit_vectors[(rid, target, ei)] = {
                    "u": u.numpy(),
                    "v": v.numpy(),
                    "subject": entity["subject"],
                    "case_id": entity["case_id"],
                }
                v_list.append(v.numpy())
            except Exception as e:
                print(f"    ERROR entity {ei} ({entity['subject']}): {e}")
                continue

            if (ei + 1) % 10 == 0:
                print(f"    {ei+1}/{len(entities)} done")

        v_by_concept[(rid, target)] = np.stack(v_list) if v_list else np.empty((0, 1600))
        print(f"    Collected {len(v_list)} v vectors")

    print(f"\n  Part A extraction: {time.time()-t0:.1f}s")

    # ===== PART A: ANALYSIS =====
    print(f"\n[4/7] Part A: Computing shared components and alignment...")

    alignment_results = {}

    for rid, target in tasks:
        V = v_by_concept[(rid, target)]  # (n, 1600)
        if len(V) < 5:
            print(f"  SKIP {rid} '{target}': only {len(V)} vectors")
            continue

        # Step 2: Shared component (raw and PCA)
        raw_mean, pca_mean, n_pca, pca_var = pca_enhanced_mean(V)

        # Step 3: Alignment metrics
        result = {"n_vectors": len(V), "pca_n_components": n_pca, "pca_var_explained": pca_var}

        for method in ["mean_diff", "logistic"]:
            cd_key = (rid, target, method)
            if cd_key not in concept_dirs:
                continue
            cd = concept_dirs[cd_key]

            # Shared alignment (signed and absolute)
            shared_cos_raw = cosine(raw_mean, cd)
            shared_cos_pca = cosine(pca_mean, cd)

            # Individual alignment
            ind_cosines = [cosine(V[i], cd) for i in range(len(V))]

            # Consistency: cos(v_i, v_mean)
            consistency = [cosine(V[i], raw_mean) for i in range(len(V))]

            result[f"{method}_shared_cos_raw"] = shared_cos_raw
            result[f"{method}_shared_abs_raw"] = abs(shared_cos_raw)
            result[f"{method}_shared_cos_pca"] = shared_cos_pca
            result[f"{method}_shared_abs_pca"] = abs(shared_cos_pca)
            result[f"{method}_individual_mean"] = float(np.mean(ind_cosines))
            result[f"{method}_individual_abs_mean"] = float(np.mean(np.abs(ind_cosines)))

        # Consistency (method-independent)
        consistency = [cosine(V[i], raw_mean) for i in range(len(V))]
        result["consistency_mean"] = float(np.mean(consistency))
        result["consistency_std"] = float(np.std(consistency))

        # Shared fraction: what fraction of each v is along v_mean?
        fractions = []
        for i in range(len(V)):
            proj = np.dot(V[i], raw_mean)
            fractions.append(proj**2 / (np.linalg.norm(V[i])**2 + 1e-10))
        result["shared_fraction_mean"] = float(np.mean(fractions))
        result["shared_fraction_std"] = float(np.std(fractions))

        # PCA sensitivity: difference between raw and PCA alignment
        for method in ["mean_diff", "logistic"]:
            raw_key = f"{method}_shared_abs_raw"
            pca_key = f"{method}_shared_abs_pca"
            if raw_key in result and pca_key in result:
                result[f"{method}_pca_diff"] = abs(result[raw_key] - result[pca_key])

        # Wrong-concept alignment: use the OTHER target from the same relation
        other_targets = [t for r, t in tasks if r == rid and t != target]
        if other_targets:
            other_target = other_targets[0]
            V_other = v_by_concept.get((rid, other_target), np.empty((0, 1600)))
            if len(V_other) >= 5:
                other_mean = V_other.mean(0)
                other_mean = other_mean / (np.linalg.norm(other_mean) + 1e-10)
                for method in ["mean_diff", "logistic"]:
                    cd_key = (rid, target, method)
                    if cd_key in concept_dirs:
                        cd = concept_dirs[cd_key]
                        result[f"{method}_wrong_concept_cos"] = cosine(other_mean, cd)
                        result[f"{method}_wrong_concept_abs"] = abs(cosine(other_mean, cd))

        # Wrong-relation alignment: v_mean from a different relation
        other_rels = [(r, t) for r, t in tasks if r != rid]
        if other_rels:
            wrong_v = v_by_concept[other_rels[0]]
            if len(wrong_v) >= 5:
                wrong_mean = wrong_v.mean(0)
                wrong_mean = wrong_mean / (np.linalg.norm(wrong_mean) + 1e-10)
                for method in ["mean_diff", "logistic"]:
                    cd_key = (rid, target, method)
                    if cd_key in concept_dirs:
                        cd = concept_dirs[cd_key]
                        result[f"{method}_wrong_relation_cos"] = cosine(wrong_mean, cd)
                        result[f"{method}_wrong_relation_abs"] = abs(cosine(wrong_mean, cd))

        alignment_results[f"{rid}_{target}"] = result

        # Print summary
        for method in ["mean_diff", "logistic"]:
            sc = result.get(f"{method}_shared_abs_raw", 0)
            wc = result.get(f"{method}_wrong_concept_abs", 0)
            wr = result.get(f"{method}_wrong_relation_abs", 0)
            print(
                f"  {rid} '{target}' [{method}]: "
                f"shared={sc:.3f} wrong_concept={wc:.3f} wrong_relation={wr:.3f} "
                f"consistency={result['consistency_mean']:.3f} "
                f"shared_frac={result['shared_fraction_mean']:.3f}"
            )

    # ===== PART B: CROSS-ENTITY TRANSFER =====
    print(f"\n[5/7] Part B: Cross-entity transfer test...")
    t0 = time.time()

    transfer_results = {}

    for ti, (rid, target) in enumerate(tasks):
        V = v_by_concept[(rid, target)]
        if len(V) < 10:
            print(f"  SKIP {rid} '{target}': only {len(V)} vectors")
            continue

        n_hold = min(N_HOLDOUT, len(V))
        rng = np.random.RandomState(SEED)
        holdout_idxs = rng.choice(len(V), n_hold, replace=False)

        concept_transfer = []
        print(f"\n  [{ti+1}/{len(tasks)}] {rid} '{target}': {n_hold} held-out entities")

        for hi, j in enumerate(holdout_idxs):
            entity_data = edit_vectors.get((rid, target, j))
            if entity_data is None:
                continue

            u_j = torch.from_numpy(entity_data["u"]).float()
            v_j = torch.from_numpy(entity_data["v"]).float()
            subject = entity_data["subject"]

            # v_mean from remaining entities
            remaining_idxs = [i for i in range(len(V)) if i != j]
            v_mean = V[remaining_idxs].mean(axis=0)
            v_mean_t = torch.from_numpy(v_mean).float()

            # Random v (same norm as v_mean)
            v_rand = torch.randn_like(v_mean_t)
            v_rand = v_rand / v_rand.norm() * v_mean_t.norm()

            # Wrong-concept v_mean
            other_targets = [t for r, t in tasks if r == rid and t != target]
            v_wrong = None
            if other_targets:
                V_other = v_by_concept.get((rid, other_targets[0]), np.empty((0, 1600)))
                if len(V_other) >= 5:
                    v_wrong = torch.from_numpy(V_other.mean(0)).float()

            # Build eval prompts
            # Find this entity in CounterFact for paraphrase prompts
            case_id = entity_data["case_id"]
            cf_record = cf_by_id.get(case_id, {})
            prompt_text = entity_data.get("prompt", "{}").replace("{}", subject)
            eval_prompts = [prompt_text]

            # Add paraphrase prompts if available
            paraphrases = cf_record.get("paraphrase_prompts", [])
            eval_prompts.extend(paraphrases[:3])

            # Evaluate each condition
            conditions = {
                "own_v": v_j,
                "shared_v": v_mean_t,
                "random_v": v_rand,
            }
            if v_wrong is not None:
                conditions["wrong_concept_v"] = v_wrong

            entity_results = {"subject": subject}
            for cond_name, v_cond in conditions.items():
                try:
                    res = apply_edit_and_eval(
                        model, tok, u_j, v_cond, hparams, eval_prompts, target
                    )
                    entity_results[cond_name] = {
                        "efficacy": res["efficacy"],
                        "mean_target_prob": res["mean_target_prob"],
                    }
                except Exception as e:
                    entity_results[cond_name] = {"efficacy": 0.0, "error": str(e)}

            concept_transfer.append(entity_results)

            if (hi + 1) % 3 == 0 or hi == 0:
                own = entity_results.get("own_v", {}).get("efficacy", 0)
                shared = entity_results.get("shared_v", {}).get("efficacy", 0)
                rand = entity_results.get("random_v", {}).get("efficacy", 0)
                print(
                    f"    [{hi+1}/{n_hold}] {subject}: "
                    f"own={own:.2f} shared={shared:.2f} random={rand:.2f}"
                )

        transfer_results[f"{rid}_{target}"] = concept_transfer

        # Summarize this concept
        if concept_transfer:
            for cond in ["own_v", "shared_v", "random_v", "wrong_concept_v"]:
                effs = [e[cond]["efficacy"] for e in concept_transfer if cond in e and "efficacy" in e[cond]]
                if effs:
                    print(f"    {cond}: mean_eff={np.mean(effs):.3f}")

    print(f"\n  Part B: {time.time()-t0:.1f}s")

    # ===== PART C: DIRECTION METHOD ROBUSTNESS =====
    print(f"\n[6/7] Part C: Direction method robustness...")

    robustness_results = {}
    for key, result in alignment_results.items():
        md_raw = result.get("mean_diff_shared_abs_raw", 0)
        log_raw = result.get("logistic_shared_abs_raw", 0)
        diff = abs(md_raw - log_raw)
        robustness_results[key] = {
            "mean_diff_abs": md_raw,
            "logistic_abs": log_raw,
            "method_diff": diff,
            "consistent": diff <= 0.15,
        }
        print(f"  {key}: mean_diff={md_raw:.3f} logistic={log_raw:.3f} diff={diff:.3f}")

    # ===== SUMMARY =====
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Part A criteria
    shared_abs_vals = [v.get("logistic_shared_abs_raw", 0) for v in alignment_results.values()]
    wrong_rel_abs = [v.get("logistic_wrong_relation_abs", 0) for v in alignment_results.values()]
    shared_fracs = [v.get("shared_fraction_mean", 0) for v in alignment_results.values()]
    pca_diffs = [v.get("logistic_pca_diff", 0) for v in alignment_results.values()]

    print(f"\nPart A (correlational):")
    print(f"  Median |shared_alignment|: {np.median(shared_abs_vals):.3f} (target: >= 0.40)")
    print(f"  shared > wrong_relation: {sum(1 for s, w in zip(shared_abs_vals, wrong_rel_abs) if s > w)}/{len(shared_abs_vals)}")
    print(f"  Mean shared_fraction: {np.mean(shared_fracs):.3f} (target: >= 0.30)")
    print(f"  PCA-raw diff <= 0.10: {sum(1 for d in pca_diffs if d <= 0.10)}/{len(pca_diffs)}")

    # Part B criteria
    print(f"\nPart B (causal):")
    all_own, all_shared, all_rand, all_wrong = [], [], [], []
    for key, transfers in transfer_results.items():
        for t in transfers:
            if "own_v" in t and "efficacy" in t["own_v"]:
                all_own.append(t["own_v"]["efficacy"])
            if "shared_v" in t and "efficacy" in t["shared_v"]:
                all_shared.append(t["shared_v"]["efficacy"])
            if "random_v" in t and "efficacy" in t["random_v"]:
                all_rand.append(t["random_v"]["efficacy"])
            if "wrong_concept_v" in t and "efficacy" in t.get("wrong_concept_v", {}):
                all_wrong.append(t["wrong_concept_v"]["efficacy"])

    if all_own:
        print(f"  Own v efficacy:          {np.mean(all_own):.3f}")
    if all_shared:
        print(f"  Shared v efficacy:       {np.mean(all_shared):.3f}")
    if all_rand:
        print(f"  Random v efficacy:       {np.mean(all_rand):.3f}")
    if all_wrong:
        print(f"  Wrong-concept v efficacy:{np.mean(all_wrong):.3f}")
    if all_own and all_shared:
        ratios = [s / (o + 1e-10) for s, o in zip(all_shared, all_own)]
        print(f"  Efficacy ratio (shared/own): {np.mean(ratios):.3f} (target: >= 0.40)")

    # Part C
    n_consistent = sum(1 for v in robustness_results.values() if v["consistent"])
    print(f"\nPart C (robustness):")
    print(f"  Methods agree within 0.15: {n_consistent}/{len(robustness_results)} (target: >= 7/10)")

    # ===== SAVE RESULTS =====
    # Save v vectors
    v_save = {}
    for (rid, target, idx), data in edit_vectors.items():
        v_save[f"{rid}_{target}_{idx}"] = {
            "u": data["u"].tolist(),
            "v": data["v"].tolist(),
            "subject": data["subject"],
            "case_id": data["case_id"],
        }

    # Save shared components
    shared_save = {}
    for (rid, target), V in v_by_concept.items():
        if len(V) >= 5:
            raw, pca, k, ev = pca_enhanced_mean(V)
            shared_save[f"{rid}_{target}"] = {
                "v_mean_raw": raw.tolist(),
                "v_mean_pca": pca.tolist(),
                "pca_k": k,
                "pca_var": ev,
                "n_vectors": len(V),
            }

    results = {
        "config": {
            "top_relations": TOP_RELATIONS,
            "n_entities": N_ENTITIES,
            "n_holdout": N_HOLDOUT,
            "edit_layer": EDIT_LAYER,
            "direction_methods": ["mean_diff", "logistic"],
            "note": "DAS dropped (explained variance 0.03-0.09 at layer 17)",
        },
        "tasks": [{"relation_id": r, "target": t} for r, t in tasks],
        "alignment": alignment_results,
        "transfer": transfer_results,
        "robustness": robustness_results,
        "shared_components": shared_save,
        "summary": {
            "median_shared_alignment": float(np.median(shared_abs_vals)),
            "mean_shared_fraction": float(np.mean(shared_fracs)),
            "mean_own_efficacy": float(np.mean(all_own)) if all_own else None,
            "mean_shared_efficacy": float(np.mean(all_shared)) if all_shared else None,
            "mean_random_efficacy": float(np.mean(all_rand)) if all_rand else None,
            "n_robustness_consistent": n_consistent,
        },
    }

    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save context templates used
    with open(RESULTS_DIR / "context_templates.json", "w") as f:
        json.dump(templates, f)

    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
