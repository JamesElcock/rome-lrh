"""
Experiment 4: How does ROME's implicit linear relation compare to LREs?

Protocol:
    1. For each relation R, extract an LRE via ridge regression
       (Hernandez et al. 2024).
    2. For each ROME edit in relation R, compute:
        (a) ROME's v vector (the edit's message to the residual stream)
        (b) The LRE's top singular direction (rank-1 approximation)
        (c) The alignment between v and the LRE's output direction
    3. Compute ROME's effective relation matrix in residual stream
       coordinates (via c_fc projection).
    4. Report correlation between LRE fit quality (R²) and ROME edit success.

If the model truly represents relations as linear maps (LRH), then:
    - The LRE should have high R² (linear fit explains relation well)
    - ROME's v should align with the LRE's top output singular vector
    - ROME's effective rank-1 operator should approximate the LRE's rank-1

Usage:
    python -m lrh.experiments.run_lre_comparison \\
        --model_name gpt2-xl \\
        --n_relations 10
"""

import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome import ROMEHyperParams
from util import nethook
from util.globals import HPARAMS_DIR, RESULTS_DIR

from lrh.config import LRHConfig
from lrh.datasets import load_lrh_dataset
from lrh.extraction import ActivationExtractor
from lrh.lre import compare_lre_to_rome, extract_lre, rome_as_implicit_lre
from lrh.rome_lrh_bridge import extract_rome_edit_vectors
from lrh.visualization import plot_lre_comparison_summary


def main(
    model_name: str = "gpt2-xl",
    n_relations: int = 10,
    n_edits_per_relation: int = 20,
    output_dir: str = None,
):
    if output_dir is None:
        output_dir = str(RESULTS_DIR / "lrh" / "lre_comparison")
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    nethook.set_requires_grad(False, model)

    lrh_config = LRHConfig(model_name=model_name)
    rome_hparams = ROMEHyperParams.from_json(HPARAMS_DIR / "ROME" / "gpt2-xl.json")
    edit_layer = rome_hparams.layers[0]

    # Load dataset
    dataset = load_lrh_dataset()
    extractor = ActivationExtractor(model, tok, lrh_config)

    # Get c_proj and c_fc weights for LRE comparison
    proj_name = f"transformer.h.{edit_layer}.mlp.c_proj.weight"
    fc_name = f"transformer.h.{edit_layer}.mlp.c_fc.weight"
    W_proj = nethook.get_parameter(model, proj_name).detach()
    W_fc = nethook.get_parameter(model, fc_name).detach()

    results = {}

    for rid in dataset.relation_ids[:n_relations]:
        pairs = dataset.get_subject_object_pairs(rid, n_pairs=lrh_config.lre_n_samples)
        if len(pairs) < 20:
            print(f"Skipping {rid}: too few samples ({len(pairs)})")
            continue

        print(f"\n{'='*60}")
        print(f"Relation: {rid} ({len(pairs)} pairs)")
        print(f"{'='*60}")

        # Extract LRE
        print("  Fitting LRE...")
        lre = extract_lre(
            extractor,
            prompts=[p["prompt"] for p in pairs],
            subjects=[p["subject"] for p in pairs],
            targets=[p["target_true"] for p in pairs],
            subject_layer=lrh_config.lre_subject_layer,
            object_layer=lrh_config.lre_object_layer,
            config=lrh_config,
            relation_id=rid,
        )
        print(f"  LRE fit R² = {lre.fit_r2:.4f}")

        # Top singular values
        _, S, _ = lre.top_singular_directions(k=5)
        energy_top1 = (S[0] ** 2 / (S**2).sum()).item()
        print(f"  LRE rank-1 energy = {energy_top1:.4f}")

        # Compare with ROME edits
        records = dataset.get_relation_records(rid)
        rid_comparisons = []

        for record in records[:n_edits_per_relation]:
            rw = record["requested_rewrite"]
            request = {
                "prompt": rw["prompt"],
                "subject": rw["subject"],
                "target_new": rw["target_new"],
            }

            try:
                deltas = extract_rome_edit_vectors(model, tok, request, rome_hparams)
                weight_name = list(deltas.keys())[0]
                u_vec, v_vec = deltas[weight_name]

                comparison = compare_lre_to_rome(
                    lre, u_vec, v_vec, W_proj, W_fc
                )
                rid_comparisons.append(comparison)

            except Exception as e:
                print(f"    Error on {rw['subject']}: {e}")
                continue

        if rid_comparisons:
            # Average comparison metrics
            avg_metrics = {}
            for key in rid_comparisons[0]:
                vals = [c[key] for c in rid_comparisons if c[key] is not None]
                if vals:
                    avg_metrics[key] = float(np.mean(vals))

            print(f"  Average v-LRE top-1 cosine: {avg_metrics.get('v_lre_top1_cosine', 'N/A'):.4f}")
            print(f"  Average v-LRE top-k coverage: {avg_metrics.get('v_lre_topk_coverage', 'N/A'):.4f}")

            results[rid] = {
                "lre_r2": lre.fit_r2,
                "lre_rank1_energy": energy_top1,
                "n_edits": len(rid_comparisons),
                **avg_metrics,
            }

            # Plot
            plot_lre_comparison_summary(
                avg_metrics,
                title=f"LRE vs ROME: {rid}",
                savepdf=f"{output_dir}/{rid}_lre_comparison.pdf",
            )

    # Save results
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir}/results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 4: LRE Comparison")
    parser.add_argument("--model_name", default="gpt2-xl")
    parser.add_argument("--n_relations", type=int, default=10)
    parser.add_argument("--n_edits", type=int, default=20)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    main(args.model_name, args.n_relations, args.n_edits, args.output_dir)