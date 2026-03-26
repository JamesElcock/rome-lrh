"""
Experiment 1: Do ROME edits preserve or disrupt existing linear representations?

Protocol:
    For each of K relations:
        1. Train linear probes at layers [0, 5, 10, 15, 17, 20, 25, 30, 35, 40, 47]
           using the pre-edit model's activations.
        2. For each of N ROME edits (different subjects within that relation):
            a. Evaluate probes on the pre-edit model (baseline accuracy).
            b. Apply ROME edit.
            c. Evaluate probes on the post-edit model (same test data, new activations).
            d. Restore original model weights.
        3. Report per-layer delta accuracy, aggregated across edits.

Expected finding:
    If ROME edits are well-targeted, probe accuracy should only drop
    significantly at or near the edit layer (17), not at distant layers.
    This would validate ROME's claim of surgical, localized editing and
    would support the LRH by showing that the model's linear concept
    structure is robust to targeted rank-1 perturbations.

Usage:
    python -m lrh.experiments.run_probe_coherence \\
        --model_name gpt2-xl \\
        --n_relations 5 \\
        --n_edits 10
"""

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome import ROMEHyperParams
from util import nethook
from util.globals import HPARAMS_DIR, RESULTS_DIR

from lrh.config import LRHConfig
from lrh.datasets import ProbeDataset, load_lrh_dataset
from lrh.extraction import ActivationExtractor
from lrh.metrics import probe_coherence_delta
from lrh.probes import (
    LinearProbe,
    compute_probe_coherence,
    evaluate_probe,
    train_probes_across_layers,
)
from lrh.visualization import plot_probe_accuracy_by_layer


def main(
    model_name: str = "gpt2-xl",
    n_relations: int = 5,
    n_edits_per_relation: int = 10,
    output_dir: str = None,
):
    if output_dir is None:
        output_dir = str(RESULTS_DIR / "lrh" / "probe_coherence")
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print(f"Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    nethook.set_requires_grad(False, model)

    # Load configs
    lrh_config = LRHConfig(model_name=model_name)
    rome_hparams = ROMEHyperParams.from_json(HPARAMS_DIR / "ROME" / "gpt2-xl.json")

    # Load dataset
    print("Loading dataset...")
    dataset = load_lrh_dataset()
    relations = dataset.relation_ids[:n_relations]

    all_results = {}

    for rid in relations:
        print(f"\n{'='*60}")
        print(f"Relation: {rid} ({dataset.relation_size(rid)} records)")
        print(f"{'='*60}")

        # Create probe dataset using the most common target value
        targets = dataset.get_unique_targets(rid)
        if len(targets) < 2:
            print(f"  Skipping {rid}: fewer than 2 target values")
            continue

        from collections import Counter

        records = dataset.get_relation_records(rid)
        target_counts = Counter(
            r["requested_rewrite"]["target_true"]["str"].strip() for r in records
        )
        top_target = target_counts.most_common(1)[0][0]

        probe_ds = ProbeDataset(dataset, rid, target_value=top_target, seed=lrh_config.seed)
        if len(probe_ds.train) < 20 or len(probe_ds.test) < 10:
            print(f"  Skipping {rid}: insufficient data for probing")
            continue

        # Prepare data
        all_data = probe_ds.train + probe_ds.val + probe_ds.test
        prompts = [d["prompt"] for d in all_data]
        subjects = [d["subject"] for d in all_data]
        labels = torch.tensor([d["label"] for d in all_data], dtype=torch.long)

        # Train probes
        print(f"  Training probes (target='{top_target}')...")
        extractor = ActivationExtractor(model, tok, lrh_config)
        probes = train_probes_across_layers(
            extractor, prompts, subjects, labels,
            layers=lrh_config.probe_layers,
            config=lrh_config,
        )

        # Test coherence across edits
        test_prompts = [d["prompt"] for d in probe_ds.test]
        test_subjects = [d["subject"] for d in probe_ds.test]
        test_labels = torch.tensor([d["label"] for d in probe_ds.test], dtype=torch.long)

        edit_records = [
            r for r in records
            if r["requested_rewrite"]["subject"] not in [d["subject"] for d in probe_ds.test]
        ][:n_edits_per_relation]

        relation_deltas = []
        for i, record in enumerate(edit_records):
            rw = record["requested_rewrite"]
            request = {
                "prompt": rw["prompt"],
                "subject": rw["subject"],
                "target_new": rw["target_new"],
            }
            print(f"  Edit {i+1}/{len(edit_records)}: {rw['subject']} → {rw['target_new']['str']}")

            probe_dict = {l: p for l, (p, _) in probes.items()}
            coherence = compute_probe_coherence(
                model, tok, probe_dict,
                test_prompts, test_subjects, test_labels,
                request, rome_hparams, lrh_config,
            )

            delta_summary = probe_coherence_delta(
                {l: {"accuracy": v["pre_accuracy"]} for l, v in coherence.items()},
                {l: {"accuracy": v["post_accuracy"]} for l, v in coherence.items()},
                edit_layer=lrh_config.rome_target_layer,
            )
            relation_deltas.append(delta_summary)

            print(f"    Mean Δ accuracy: {delta_summary['mean_delta_accuracy']:.4f}")

        all_results[rid] = relation_deltas

        # Plot average coherence for this relation
        if relation_deltas:
            avg_pre = {l: {"accuracy": probes[l][1]["accuracy"]} for l in probes}
            avg_post_acc = {}
            for l in probes:
                deltas_at_l = [
                    rd["per_layer_delta"].get(l, 0.0) for rd in relation_deltas
                ]
                avg_post_acc[l] = {
                    "accuracy": probes[l][1]["accuracy"] + sum(deltas_at_l) / len(deltas_at_l)
                }

            plot_probe_accuracy_by_layer(
                avg_pre, avg_post_acc,
                edit_layer=lrh_config.rome_target_layer,
                title=f"Probe Coherence: {rid} (n={len(relation_deltas)} edits)",
                savepdf=f"{output_dir}/{rid}_probe_coherence.pdf",
            )

    # Save results
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(
            {k: [{"mean_delta": r["mean_delta_accuracy"],
                   "max_drop": r["max_accuracy_drop"],
                   "critical_layer": r["critical_layer_delta"]}
                  for r in v]
             for k, v in all_results.items()},
            f, indent=2,
        )
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 1: Probe Coherence")
    parser.add_argument("--model_name", default="gpt2-xl")
    parser.add_argument("--n_relations", type=int, default=5)
    parser.add_argument("--n_edits", type=int, default=10)
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()
    main(args.model_name, args.n_relations, args.n_edits, args.output_dir)