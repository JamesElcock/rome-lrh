"""
Experiment 0 (Prerequisite): v Determinism Pilot Test

Tests whether ROME's v vectors are deterministic across runs.

Key insight: compute_v initializes delta=zeros and uses Adam optimization.
The main randomness source is CONTEXT_TEMPLATES_CACHE (generated via
sampling on first call, then cached globally). So v should be deterministic
once context templates are fixed.

We test two scenarios:
1. Same context templates (cached): expect near-perfect cosine (~1.0)
2. Fresh context templates (cleared cache): expect lower cosine

If scenario 2 shows cosine < 0.8, experiments should either:
- Fix a canonical set of context templates
- Average v over multiple template draws per entity
"""

import json
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoModelForCausalLM, AutoTokenizer
from rome.rome_main import execute_rome, get_context_templates
from rome.rome_hparams import ROMEHyperParams
from util.globals import HPARAMS_DIR
import rome.rome_main as rome_main_module


def cosine_sim(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().unsqueeze(0), b.float().unsqueeze(0)
    ).item()


def run_rome_get_v(model, tok, request, hparams):
    """Run execute_rome and return the (u, v) vectors."""
    deltas = execute_rome(model, tok, request, hparams)
    for key, (left, right) in deltas.items():
        return left.detach().cpu(), right.detach().cpu()


def main():
    print("=" * 60)
    print("Experiment 0: v Determinism Pilot Test")
    print("=" * 60)

    # Load model and tokenizer
    print("\nLoading GPT-2 XL...")
    model = AutoModelForCausalLM.from_pretrained("gpt2-xl").cuda()
    tok = AutoTokenizer.from_pretrained("gpt2-xl")
    tok.pad_token = tok.eos_token
    print("Model loaded.")

    # Load hparams
    hparams = ROMEHyperParams.from_json(HPARAMS_DIR / "ROME" / "gpt2-xl.json")

    # Load CounterFact and select 10 diverse entities
    data = json.load(open("data/counterfact.json"))

    # Pick 2 entities from 5 different relations
    selected_relations = ["P27", "P176", "P413", "P1412", "P106"]
    selected = []
    for rel in selected_relations:
        rel_entries = [r for r in data if r["requested_rewrite"]["relation_id"] == rel]
        selected.extend(rel_entries[:2])

    print(f"\nSelected {len(selected)} entities across {len(selected_relations)} relations")

    # ==========================================
    # SCENARIO 1: Same context templates (cached)
    # ==========================================
    print("\n" + "=" * 60)
    print("SCENARIO 1: Repeated runs with SAME cached context templates")
    print("=" * 60)

    scenario1_results = {}

    for i, record in enumerate(selected):
        rw = record["requested_rewrite"]
        entity = rw["subject"]
        rel = rw["relation_id"]

        request = {
            "prompt": rw["prompt"],
            "subject": rw["subject"],
            "target_new": rw["target_new"],
        }

        print(f"\n[{i+1}/{len(selected)}] {entity} ({rel}) -> {rw['target_new']['str']}")

        # Run 3 times with same cached templates
        vs = []
        us = []
        for run in range(3):
            u, v = run_rome_get_v(model, tok, request, hparams)
            vs.append(v)
            us.append(u)

        # Pairwise cosines for v
        v_cosines = []
        for a in range(len(vs)):
            for b in range(a + 1, len(vs)):
                v_cosines.append(cosine_sim(vs[a], vs[b]))

        # Pairwise cosines for u
        u_cosines = []
        for a in range(len(us)):
            for b in range(a + 1, len(us)):
                u_cosines.append(cosine_sim(us[a], us[b]))

        scenario1_results[f"{entity}_{rel}"] = {
            "entity": entity,
            "relation": rel,
            "v_cosines": v_cosines,
            "v_mean": float(np.mean(v_cosines)),
            "u_cosines": u_cosines,
            "u_mean": float(np.mean(u_cosines)),
        }

        print(f"  v cosines (cached templates): {[f'{c:.6f}' for c in v_cosines]} mean={np.mean(v_cosines):.6f}")
        print(f"  u cosines (cached templates): {[f'{c:.6f}' for c in u_cosines]} mean={np.mean(u_cosines):.6f}")

    # ==========================================
    # SCENARIO 2: Fresh context templates each time
    # ==========================================
    print("\n" + "=" * 60)
    print("SCENARIO 2: Repeated runs with FRESH context templates")
    print("=" * 60)

    # Use a subset (5 entities) since this is slower
    scenario2_results = {}

    for i, record in enumerate(selected[:5]):
        rw = record["requested_rewrite"]
        entity = rw["subject"]
        rel = rw["relation_id"]

        request = {
            "prompt": rw["prompt"],
            "subject": rw["subject"],
            "target_new": rw["target_new"],
        }

        print(f"\n[{i+1}/5] {entity} ({rel}) -> {rw['target_new']['str']}")

        vs = []
        us = []
        for run in range(3):
            # Clear the context template cache to force regeneration
            torch.manual_seed(run * 1000 + i)
            rome_main_module.CONTEXT_TEMPLATES_CACHE = None

            u, v = run_rome_get_v(model, tok, request, hparams)
            vs.append(v)
            us.append(u)

        # Pairwise cosines
        v_cosines = []
        for a in range(len(vs)):
            for b in range(a + 1, len(vs)):
                v_cosines.append(cosine_sim(vs[a], vs[b]))

        u_cosines = []
        for a in range(len(us)):
            for b in range(a + 1, len(us)):
                u_cosines.append(cosine_sim(us[a], us[b]))

        scenario2_results[f"{entity}_{rel}"] = {
            "entity": entity,
            "relation": rel,
            "v_cosines": v_cosines,
            "v_mean": float(np.mean(v_cosines)),
            "u_cosines": u_cosines,
            "u_mean": float(np.mean(u_cosines)),
        }

        print(f"  v cosines (fresh templates): {[f'{c:.6f}' for c in v_cosines]} mean={np.mean(v_cosines):.6f}")
        print(f"  u cosines (fresh templates): {[f'{c:.6f}' for c in u_cosines]} mean={np.mean(u_cosines):.6f}")

    # ==========================================
    # SUMMARY
    # ==========================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    s1_v_means = [r["v_mean"] for r in scenario1_results.values()]
    s1_u_means = [r["u_mean"] for r in scenario1_results.values()]
    print(f"\nScenario 1 (cached templates):")
    print(f"  v: mean cosine = {np.mean(s1_v_means):.6f} (min={np.min(s1_v_means):.6f})")
    print(f"  u: mean cosine = {np.mean(s1_u_means):.6f} (min={np.min(s1_u_means):.6f})")

    s2_v_means = [r["v_mean"] for r in scenario2_results.values()]
    s2_u_means = [r["u_mean"] for r in scenario2_results.values()]
    print(f"\nScenario 2 (fresh templates):")
    print(f"  v: mean cosine = {np.mean(s2_v_means):.6f} (min={np.min(s2_v_means):.6f})")
    print(f"  u: mean cosine = {np.mean(s2_u_means):.6f} (min={np.min(s2_u_means):.6f})")

    # Determine outcome
    print("\n" + "-" * 40)
    if np.mean(s1_v_means) >= 0.99:
        print("Scenario 1: v is DETERMINISTIC with fixed templates (as expected)")
    else:
        print(f"Scenario 1: UNEXPECTED — v varies even with fixed templates (mean={np.mean(s1_v_means):.4f})")

    if np.mean(s2_v_means) >= 0.95:
        print("Scenario 2: v is ROBUST to template variation")
        print(">> PASS: No special handling needed")
    elif np.mean(s2_v_means) >= 0.80:
        print("Scenario 2: v shows MODERATE sensitivity to templates")
        print(">> RECOMMENDATION: Fix canonical templates OR average over 3-5 template draws")
    else:
        print(f"Scenario 2: v is HIGHLY SENSITIVE to templates (mean={np.mean(s2_v_means):.4f})")
        print(">> RECOMMENDATION: Must fix canonical templates for all experiments")

    # Save results
    os.makedirs("results/pilot", exist_ok=True)
    results = {
        "scenario1_cached_templates": scenario1_results,
        "scenario2_fresh_templates": scenario2_results,
        "summary": {
            "s1_v_mean": float(np.mean(s1_v_means)),
            "s1_v_min": float(np.min(s1_v_means)),
            "s1_u_mean": float(np.mean(s1_u_means)),
            "s2_v_mean": float(np.mean(s2_v_means)),
            "s2_v_min": float(np.min(s2_v_means)),
            "s2_u_mean": float(np.mean(s2_u_means)),
        },
    }
    with open("results/pilot/v_determinism.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to results/pilot/v_determinism.json")


if __name__ == "__main__":
    main()
