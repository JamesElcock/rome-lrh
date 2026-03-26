"""
Experiment 2: Does ROME's v vector align with concept directions?

Protocol:
    1. For each relation R, extract concept directions at layer 17 using
       mean-difference and DAS methods.
    2. For each ROME edit within relation R, compute the v vector.
    3. Compute cosine similarity between v and:
        (a) The concept direction for R (the "correct" relation)
        (b) Concept directions for unrelated relations (controls)
    4. Statistical test: is alignment for the correct relation significantly
       higher than for controls?

If LRH holds and ROME is consistent with it, we expect:
    - v aligns significantly with the correct relation direction
    - v does NOT align with unrelated relation directions
    - The alignment magnitude indicates how much of ROME's edit
      operates within the model's existing linear concept structure

Usage:
    python -m lrh.experiments.run_v_alignment \\
        --model_name gpt2-xl \\
        --n_relations 10 \\
        --n_edits 50
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

from lrh.concept_directions import extract_relation_directions
from lrh.config import LRHConfig
from lrh.datasets import load_lrh_dataset
from lrh.extraction import ActivationExtractor
from lrh.metrics import direction_alignment
from lrh.rome_lrh_bridge import extract_rome_edit_vectors
from lrh.visualization import plot_edit_success_vs_alignment


def main(
    model_name: str = "gpt2-xl",
    n_relations: int = 10,
    n_edits_per_relation: int = 50,
    output_dir: str = None,
):
    if output_dir is None:
        output_dir = str(RESULTS_DIR / "lrh" / "v_alignment")
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
    print("Loading dataset...")
    dataset = load_lrh_dataset()

    # Extract concept directions for all relations at the edit layer
    print(f"Extracting concept directions at layer {edit_layer}...")
    extractor = ActivationExtractor(model, tok, lrh_config)
    concept_bank = extract_relation_directions(
        extractor, dataset,
        layers=[edit_layer],
        method="mean_diff",
        config=lrh_config,
        max_relations=n_relations,
    )

    concepts_at_layer = concept_bank.get_layer(edit_layer)
    if len(concepts_at_layer) < 2:
        print("Not enough concept directions extracted. Exiting.")
        return

    # Pairwise alignment between concept directions
    sim_matrix, concept_names = concept_bank.pairwise_alignment(edit_layer)
    from lrh.visualization import plot_concept_direction_heatmap
    plot_concept_direction_heatmap(
        sim_matrix, concept_names,
        title=f"Concept Direction Similarity (Layer {edit_layer})",
        savepdf=f"{output_dir}/concept_similarity_heatmap.pdf",
    )

    # For each relation, compute v-alignment
    results = {}
    all_correct_aligns = []
    all_control_aligns = []

    for rid in dataset.relation_ids[:n_relations]:
        records = dataset.get_relation_records(rid)
        if len(records) < 5:
            continue

        # Find the concept direction for this relation
        matching_concepts = [
            (name, cd) for name, cd in concepts_at_layer.items()
            if name.startswith(rid)
        ]
        if not matching_concepts:
            continue

        concept_name, concept_dir = matching_concepts[0]
        control_dirs = [
            cd for name, cd in concepts_at_layer.items()
            if not name.startswith(rid)
        ]

        rid_results = {"correct_alignment": [], "control_alignment": []}

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

                # Alignment with correct relation
                correct_cos = abs(direction_alignment(v_vec, concept_dir.direction))
                rid_results["correct_alignment"].append(correct_cos)
                all_correct_aligns.append(correct_cos)

                # Alignment with control relations
                if control_dirs:
                    control_cos = [
                        abs(direction_alignment(v_vec, cd.direction))
                        for cd in control_dirs
                    ]
                    rid_results["control_alignment"].extend(control_cos)
                    all_control_aligns.extend(control_cos)

            except Exception as e:
                print(f"  Error on {rw['subject']}: {e}")
                continue

        if rid_results["correct_alignment"]:
            mean_correct = np.mean(rid_results["correct_alignment"])
            mean_control = np.mean(rid_results["control_alignment"]) if rid_results["control_alignment"] else 0
            print(
                f"  {rid} ({concept_name}): "
                f"correct={mean_correct:.4f}, control={mean_control:.4f}, "
                f"ratio={mean_correct / (mean_control + 1e-10):.2f}"
            )
            results[rid] = {
                "correct_mean": mean_correct,
                "control_mean": mean_control,
                "n_edits": len(rid_results["correct_alignment"]),
            }

    # Statistical test: correct vs control alignment
    if all_correct_aligns and all_control_aligns:
        from scipy.stats import mannwhitneyu
        stat, pval = mannwhitneyu(all_correct_aligns, all_control_aligns, alternative="greater")
        print(f"\nMann-Whitney U test (correct > control):")
        print(f"  U = {stat:.1f}, p = {pval:.2e}")
        print(f"  Mean correct: {np.mean(all_correct_aligns):.4f} ± {np.std(all_correct_aligns):.4f}")
        print(f"  Mean control: {np.mean(all_control_aligns):.4f} ± {np.std(all_control_aligns):.4f}")

        results["_aggregate"] = {
            "correct_mean": float(np.mean(all_correct_aligns)),
            "control_mean": float(np.mean(all_control_aligns)),
            "u_statistic": float(stat),
            "p_value": float(pval),
        }

    # Save results
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir}/results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 2: v-vector Alignment")
    parser.add_argument("--model_name", default="gpt2-xl")
    parser.add_argument("--n_relations", type=int, default=10)
    parser.add_argument("--n_edits", type=int, default=50)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    main(args.model_name, args.n_relations, args.n_edits, args.output_dir)