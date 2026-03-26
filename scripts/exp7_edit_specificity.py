"""
Experiment 7: Cross-Concept Edit Specificity

Tests the functional consequence of u's gating: when you apply a ROME edit for
one entity, does it bleed through to other entities from the same concept?

For each edit (u ⊗ v applied at layer 17), we measure output changes on:
- Self: the edited entity's own prompt (should change)
- Same-target: different entity, same target value (e.g., another Toyota product)
- Same-relation: different entity, different target, same relation type
- Different-relation: entity from a completely different relation

This combines u (gating) and v (payload) effects. Even if u fires weakly for
same-concept inputs, v's "dark subspace" direction might not produce concept-relevant
output shifts. Conversely, even weak gating could produce measurable bleed if v
carries enough concept-general signal.

Saves results to results/exp7_specificity/
"""

import json
import os
import sys
import time
import numpy as np
import torch
from collections import defaultdict
from pathlib import Path

# Threading safety
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SEED = 42
EDIT_LAYER = 17
RESULTS_DIR = Path("results/exp7_specificity")
EXP2_DIR = Path("results/exp2")

# Scale: 5 relations × 2 targets × 5 entities = 50 edits
RELATIONS = ["P176", "P1412", "P37", "P27", "P413"]
N_TARGETS_PER_RELATION = 2
N_ENTITIES_PER_TARGET = 5
N_TEST_PER_CONDITION = 20
BATCH_SIZE = 24


# ============================================================
# Helpers
# ============================================================

def get_logits_at_last(model, tok, prompts, batch_size=BATCH_SIZE):
    """Get logits at last token position for a list of prompts."""
    device = next(model.parameters()).device
    all_logits = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i + batch_size]
        inputs = tok(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model(**inputs)
        # Get logits at last non-padding position
        for b_idx in range(len(batch)):
            mask = inputs["attention_mask"][b_idx]
            last_pos = mask.sum() - 1
            logits = out.logits[b_idx, last_pos].float().cpu()
            all_logits.append(logits)
    return torch.stack(all_logits)  # (n, vocab_size)


def kl_divergence(p_logits, q_logits):
    """KL(p || q) from logits."""
    p = torch.softmax(p_logits, dim=-1)
    q = torch.softmax(q_logits, dim=-1)
    return float(torch.sum(p * (torch.log(p + 1e-10) - torch.log(q + 1e-10))))


def select_edit_entities(cf_records, relations, n_targets, n_entities, seed=42):
    """Select entities for editing, organized by (relation, target)."""
    rng = np.random.RandomState(seed)
    by_rel_target = defaultdict(list)
    for rec in cf_records:
        rw = rec["requested_rewrite"]
        rid = rw["relation_id"]
        target = rw["target_new"]["str"]
        if rid in relations:
            by_rel_target[(rid, target)].append(rec)

    selected = {}
    used_subjects = set()

    for rid in relations:
        # Find top targets by count
        rel_targets = [(t, recs) for (r, t), recs in by_rel_target.items()
                       if r == rid and len(recs) >= n_entities + N_TEST_PER_CONDITION]
        rel_targets.sort(key=lambda x: len(x[1]), reverse=True)

        for target, recs in rel_targets[:n_targets]:
            rng.shuffle(recs)
            edit_recs = []
            for rec in recs:
                subj = rec["requested_rewrite"]["subject"]
                if subj not in used_subjects and len(edit_recs) < n_entities:
                    edit_recs.append(rec)
                    used_subjects.add(subj)
            selected[(rid, target)] = edit_recs

    return selected, used_subjects


def select_test_entities(cf_records, edit_entities, used_subjects, n_per_condition, seed=42):
    """
    For each edit concept (relation, target), select test entities for
    4 conditions: same-target, same-relation-diff-target, different-relation.
    """
    rng = np.random.RandomState(seed + 200)
    by_rel_target = defaultdict(list)
    for rec in cf_records:
        rw = rec["requested_rewrite"]
        rid = rw["relation_id"]
        target = rw["target_new"]["str"]
        subj = rw["subject"]
        if subj not in used_subjects:
            by_rel_target[(rid, target)].append(rec)

    test_sets = {}
    all_relations = set(rid for rid, _ in edit_entities.keys())

    for (rid, target), edit_recs in edit_entities.items():
        conditions = {}

        # Same-target: same relation, same target, different entity
        same_target_pool = by_rel_target.get((rid, target), [])
        rng.shuffle(same_target_pool)
        conditions["same_target"] = same_target_pool[:n_per_condition]

        # Same-relation: same relation, different target
        same_rel_pool = []
        for (r, t), recs in by_rel_target.items():
            if r == rid and t != target:
                same_rel_pool.extend(recs)
        rng.shuffle(same_rel_pool)
        conditions["same_relation"] = same_rel_pool[:n_per_condition]

        # Different-relation
        diff_rel_pool = []
        for (r, t), recs in by_rel_target.items():
            if r != rid and r in all_relations:
                diff_rel_pool.extend(recs)
        rng.shuffle(diff_rel_pool)
        conditions["different_relation"] = diff_rel_pool[:n_per_condition]

        test_sets[(rid, target)] = conditions

    return test_sets


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("EXPERIMENT 7: CROSS-CONCEPT EDIT SPECIFICITY")
    print("=" * 70)
    t_start = time.time()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    print("\nLoading GPT-2 XL...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained("gpt2-xl").cuda().eval()
    tok = AutoTokenizer.from_pretrained("gpt2-xl")
    tok.pad_token = tok.eos_token

    # Load ROME hparams
    from rome.rome_hparams import ROMEHyperParams
    from util.globals import HPARAMS_DIR
    hparams = ROMEHyperParams.from_json(HPARAMS_DIR / "ROME" / "gpt2-xl.json")

    # Load CounterFact
    print("Loading CounterFact...")
    from dsets import CounterFactDataset
    cf = CounterFactDataset("data")

    # Select edit entities
    print("Selecting edit entities...")
    edit_entities, used_subjects = select_edit_entities(
        cf, RELATIONS, N_TARGETS_PER_RELATION, N_ENTITIES_PER_TARGET, SEED
    )
    n_edits = sum(len(recs) for recs in edit_entities.values())
    print(f"  {n_edits} edits across {len(edit_entities)} concepts")
    for (rid, target), recs in sorted(edit_entities.items()):
        print(f"    {rid}_{target}: {len(recs)} entities")

    # Select test entities
    print("\nSelecting test entities...")
    test_sets = select_test_entities(cf, edit_entities, used_subjects, N_TEST_PER_CONDITION, SEED)

    # Load context templates
    ctx_path = Path("results/exp1/context_templates.json")
    if ctx_path.exists():
        with open(ctx_path) as f:
            context_templates = json.load(f)
        print(f"  Loaded {len(context_templates)} context templates")
    else:
        context_templates = ["{}"]

    # Patch ROME's context template sampling
    import rome.rome_main as rome_main_module
    rome_main_module.CONTEXT_TEMPLATES_CACHE = None

    original_get_context = None
    if hasattr(rome_main_module, "get_context_templates"):
        original_get_context = rome_main_module.get_context_templates

    def fixed_get_context(model, tok, length_params=None):
        return context_templates

    rome_main_module.get_context_templates = fixed_get_context

    # ============================================================
    # Run edits and measure bleed-through
    # ============================================================
    from rome.rome_main import execute_rome
    from util import nethook

    all_results = []
    edit_count = 0

    for (rid, target), edit_recs in sorted(edit_entities.items()):
        concept = f"{rid}_{target}"
        conditions = test_sets[(rid, target)]

        for edit_rec in edit_recs:
            edit_count += 1
            rw = edit_rec["requested_rewrite"]
            subject = rw["subject"]
            prompt_template = rw["prompt"]
            target_new = rw["target_new"]["str"]
            target_true = rw["target_true"]["str"]

            request = {
                "prompt": prompt_template,
                "subject": subject,
                "target_new": {"str": target_new},
            }

            print(f"\n  [{edit_count}/{n_edits}] {concept}: {subject} → {target_new}")

            # Execute ROME to get u, v
            deltas = execute_rome(model, tok, request, hparams)
            weight_name = f"transformer.h.{EDIT_LAYER}.mlp.c_proj.weight"
            (u_vec, v_vec) = deltas[weight_name]

            # Get target token id
            target_tok_id = tok.encode(" " + target_new)[0]
            true_tok_id = tok.encode(" " + target_true)[0]

            # Prepare all test prompts
            test_prompts = {}
            test_info = {}

            # Self
            self_prompt = prompt_template.replace("{}", subject)
            test_prompts["self"] = [self_prompt]
            test_info["self"] = [{"subject": subject, "target": target_new}]

            # Same-target, same-relation, different-relation
            for cond_name in ["same_target", "same_relation", "different_relation"]:
                cond_recs = conditions[cond_name]
                cond_prompts = []
                cond_info = []
                for crec in cond_recs:
                    crw = crec["requested_rewrite"]
                    p = crw["prompt"].replace("{}", crw["subject"])
                    cond_prompts.append(p)
                    cond_info.append({
                        "subject": crw["subject"],
                        "target": crw["target_new"]["str"],
                        "true_answer": crw["target_true"]["str"],
                    })
                test_prompts[cond_name] = cond_prompts
                test_info[cond_name] = cond_info

            # Get clean logits for all prompts
            all_prompts_flat = []
            prompt_ranges = {}
            offset = 0
            for cond in ["self", "same_target", "same_relation", "different_relation"]:
                n = len(test_prompts[cond])
                prompt_ranges[cond] = (offset, offset + n)
                all_prompts_flat.extend(test_prompts[cond])
                offset += n

            clean_logits = get_logits_at_last(model, tok, all_prompts_flat)

            # Apply edit
            w = nethook.get_parameter(model, weight_name)
            delta = u_vec.unsqueeze(1) @ v_vec.unsqueeze(0)  # (6400, 1) × (1, 1600) = (6400, 1600)
            w_orig = w.data.clone()
            w.data += delta.to(w.device)

            # Get edited logits
            edited_logits = get_logits_at_last(model, tok, all_prompts_flat)

            # Restore weights
            w.data = w_orig

            # Analyze per condition
            for cond in ["self", "same_target", "same_relation", "different_relation"]:
                start, end = prompt_ranges[cond]
                c_clean = clean_logits[start:end]
                c_edited = edited_logits[start:end]
                c_info = test_info[cond]

                for j in range(end - start):
                    # ΔP(target) — change in probability of the edit's target token
                    p_target_clean = float(torch.softmax(c_clean[j], dim=-1)[target_tok_id])
                    p_target_edited = float(torch.softmax(c_edited[j], dim=-1)[target_tok_id])
                    delta_p_target = p_target_edited - p_target_clean

                    # ΔP(true) — change in probability of the true answer token for this entity
                    if cond == "self":
                        true_id = true_tok_id
                    else:
                        entity_true = c_info[j].get("true_answer", "")
                        true_id = tok.encode(" " + entity_true)[0] if entity_true else true_tok_id

                    p_true_clean = float(torch.softmax(c_clean[j], dim=-1)[true_id])
                    p_true_edited = float(torch.softmax(c_edited[j], dim=-1)[true_id])
                    delta_p_true = p_true_edited - p_true_clean

                    # KL divergence
                    kl = kl_divergence(c_edited[j], c_clean[j])

                    # Rank of target token
                    rank_clean = int((c_clean[j] > c_clean[j][target_tok_id]).sum())
                    rank_edited = int((c_edited[j] > c_edited[j][target_tok_id]).sum())

                    result = {
                        "edit_concept": concept,
                        "edit_subject": subject,
                        "condition": cond,
                        "test_subject": c_info[j].get("subject", subject),
                        "delta_p_target": delta_p_target,
                        "delta_p_true": delta_p_true,
                        "p_target_clean": p_target_clean,
                        "p_target_edited": p_target_edited,
                        "kl_divergence": kl,
                        "rank_clean": rank_clean,
                        "rank_edited": rank_edited,
                        "rank_shift": rank_clean - rank_edited,
                    }
                    all_results.append(result)

    # ============================================================
    # Aggregate results
    # ============================================================
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    by_condition = defaultdict(list)
    for r in all_results:
        by_condition[r["condition"]].append(r)

    print(f"\n  {'Condition':<20s} {'ΔP(target)':>10s} {'ΔP(true)':>10s} {'KL div':>10s} {'Rank shift':>10s} {'n':>5s}")
    print("  " + "-" * 70)

    condition_summaries = {}
    for cond in ["self", "same_target", "same_relation", "different_relation"]:
        results = by_condition[cond]
        if not results:
            continue

        dp_target = [r["delta_p_target"] for r in results]
        dp_true = [r["delta_p_true"] for r in results]
        kls = [r["kl_divergence"] for r in results]
        ranks = [r["rank_shift"] for r in results]

        summary = {
            "delta_p_target_mean": float(np.mean(dp_target)),
            "delta_p_target_std": float(np.std(dp_target)),
            "delta_p_target_median": float(np.median(dp_target)),
            "delta_p_true_mean": float(np.mean(dp_true)),
            "delta_p_true_std": float(np.std(dp_true)),
            "kl_mean": float(np.mean(kls)),
            "kl_std": float(np.std(kls)),
            "kl_median": float(np.median(kls)),
            "rank_shift_mean": float(np.mean(ranks)),
            "n": len(results),
        }
        condition_summaries[cond] = summary

        print(f"  {cond:<20s} {summary['delta_p_target_mean']:>10.4f} "
              f"{summary['delta_p_true_mean']:>10.4f} {summary['kl_mean']:>10.3f} "
              f"{summary['rank_shift_mean']:>10.1f} {summary['n']:>5d}")

    # Per-concept breakdown for same_target (the key condition)
    print(f"\n  Per-concept same-target bleed-through:")
    print(f"  {'Concept':<20s} {'ΔP(target)':>10s} {'KL':>10s} {'n':>5s}")
    print("  " + "-" * 50)

    per_concept_bleed = defaultdict(list)
    for r in by_condition["same_target"]:
        per_concept_bleed[r["edit_concept"]].append(r)

    concept_bleed_summary = {}
    for concept in sorted(per_concept_bleed.keys()):
        results = per_concept_bleed[concept]
        dp = [r["delta_p_target"] for r in results]
        kls = [r["kl_divergence"] for r in results]
        concept_bleed_summary[concept] = {
            "delta_p_target_mean": float(np.mean(dp)),
            "kl_mean": float(np.mean(kls)),
            "n": len(results),
        }
        print(f"  {concept:<20s} {np.mean(dp):>10.4f} {np.mean(kls):>10.3f} {len(results):>5d}")

    # Statistical test: same-target vs different-relation
    same_target_dps = [r["delta_p_target"] for r in by_condition["same_target"]]
    diff_rel_dps = [r["delta_p_target"] for r in by_condition["different_relation"]]
    if same_target_dps and diff_rel_dps:
        from scipy.stats import mannwhitneyu
        stat, p_val = mannwhitneyu(same_target_dps, diff_rel_dps, alternative="greater")
        print(f"\n  Mann-Whitney U (same-target > different-relation): U={stat:.0f}, p={p_val:.6f}")

    # ============================================================
    # Save results
    # ============================================================
    output = {
        "config": {
            "relations": RELATIONS,
            "n_targets_per_relation": N_TARGETS_PER_RELATION,
            "n_entities_per_target": N_ENTITIES_PER_TARGET,
            "n_test_per_condition": N_TEST_PER_CONDITION,
            "edit_layer": EDIT_LAYER,
            "seed": SEED,
        },
        "condition_summaries": condition_summaries,
        "per_concept_bleed": concept_bleed_summary,
        "per_edit_results": all_results,
    }

    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {RESULTS_DIR}/")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT 7 COMPLETE — {elapsed:.0f}s ({elapsed / 60:.1f} min)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()