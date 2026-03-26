"""
Visualization utilities for ROME x LRH analysis.

Follows the matplotlib style conventions from experiments/causal_trace.py
(Times New Roman, tight layouts, PDF export).

All plot functions accept an optional `savepdf` path. If provided, the plot
is saved to disk and plt.close() is called; otherwise it is displayed.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt


def _setup_style():
    """Shared plot style matching causal_trace.py."""
    return {"font.family": "serif", "font.size": 10}


def plot_probe_accuracy_by_layer(
    pre_metrics: Dict[int, Dict[str, float]],
    post_metrics: Dict[int, Dict[str, float]],
    edit_layer: int = 17,
    title: str = "Linear Probe Coherence: Pre vs Post ROME Edit",
    savepdf: Optional[str] = None,
):
    """
    Line plot of probe accuracy vs layer, before and after a ROME edit.

    The edit layer is highlighted with a vertical dashed line to show
    whether disruption is localized to the edited layer.
    """
    layers = sorted(pre_metrics.keys())
    pre_acc = [pre_metrics[l]["accuracy"] for l in layers]
    post_acc = [post_metrics[l].get("accuracy", post_metrics[l].get("post_accuracy", 0)) for l in layers]

    with plt.rc_context(rc=_setup_style()):
        fig, ax = plt.subplots(figsize=(6, 3.5), dpi=150)
        ax.plot(layers, pre_acc, "o-", label="Pre-edit", color="#2196F3", markersize=4)
        ax.plot(layers, post_acc, "s--", label="Post-edit", color="#F44336", markersize=4)
        ax.axvline(edit_layer, color="gray", linestyle=":", alpha=0.7, label=f"Edit layer ({edit_layer})")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Probe Accuracy")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.set_ylim([0, 1.05])
        ax.grid(alpha=0.2)
        plt.tight_layout()
        _save_or_show(savepdf)


def plot_v_concept_decomposition(
    decomposition: Dict,
    title: str = "ROME v-vector Decomposition in Concept Basis",
    savepdf: Optional[str] = None,
):
    """
    Bar chart showing how much of ROME's v vector projects onto each
    concept direction, plus the residual component.
    """
    proj_fracs = decomposition.get("projection_fractions", {})
    if not proj_fracs:
        print("No projection data to plot.")
        return

    names = list(proj_fracs.keys())
    values = list(proj_fracs.values())
    residual = 1.0 - decomposition.get("total_explained_variance", 0.0)

    names.append("Residual")
    values.append(max(residual, 0.0))

    with plt.rc_context(rc=_setup_style()):
        fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.8), 4), dpi=150)
        colors = ["#4CAF50"] * (len(names) - 1) + ["#9E9E9E"]
        bars = ax.bar(range(len(names)), values, color=colors, edgecolor="white", linewidth=0.5)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Fraction of ||v||²")
        ax.set_title(title)

        # Annotate total explained
        total = decomposition.get("total_explained_variance", 0.0)
        ax.text(
            0.98, 0.95,
            f"Total explained: {total:.1%}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        _save_or_show(savepdf)


def plot_gate_activation_distributions(
    gate_analysis: Dict,
    title: str = "ROME Gate (u) Selectivity",
    savepdf: Optional[str] = None,
):
    """
    Overlapping histograms of ⟨k, u⟩ for target subject vs other entities.

    Annotated with d' and selectivity ratio.
    """
    target = gate_analysis.get("target_activations", torch.tensor([]))
    other = gate_analysis.get("other_activations", torch.tensor([]))

    if target.numel() == 0 or other.numel() == 0:
        print("No gate activation data to plot.")
        return

    target_np = target.numpy()
    other_np = other.numpy()

    with plt.rc_context(rc=_setup_style()):
        fig, ax = plt.subplots(figsize=(5, 3.5), dpi=150)
        bins = np.linspace(
            min(target_np.min(), other_np.min()),
            max(target_np.max(), other_np.max()),
            40,
        )
        ax.hist(other_np, bins=bins, alpha=0.6, label="Other entities", color="#2196F3", density=True)
        ax.hist(target_np, bins=bins, alpha=0.6, label="Target subject", color="#F44336", density=True)
        ax.set_xlabel("Gate activation ⟨k, u⟩")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend(fontsize=8)

        dp = gate_analysis.get("d_prime", 0.0)
        sel = gate_analysis.get("selectivity_ratio", 0.0)
        ax.text(
            0.98, 0.95,
            f"d' = {dp:.2f}\nSelectivity = {sel:.2f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
        )

        plt.tight_layout()
        _save_or_show(savepdf)


def plot_concept_direction_heatmap(
    similarity_matrix: torch.Tensor,
    concept_names: List[str],
    title: str = "Concept Direction Pairwise Cosine Similarity",
    savepdf: Optional[str] = None,
):
    """
    Heatmap of pairwise cosine similarity between concept directions.

    Reveals orthogonality structure: if LRH holds, different relations
    should have approximately orthogonal directions.
    """
    sim = similarity_matrix.numpy() if torch.is_tensor(similarity_matrix) else similarity_matrix

    with plt.rc_context(rc=_setup_style()):
        fig, ax = plt.subplots(figsize=(max(5, len(concept_names) * 0.6),
                                        max(4, len(concept_names) * 0.5)), dpi=150)
        im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(concept_names)))
        ax.set_yticks(range(len(concept_names)))
        ax.set_xticklabels(concept_names, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(concept_names, fontsize=7)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8, label="Cosine similarity")
        plt.tight_layout()
        _save_or_show(savepdf)


def plot_edit_success_vs_alignment(
    alignments: List[float],
    successes: List[float],
    xlabel: str = "v-concept alignment",
    ylabel: str = "Edit efficacy",
    title: str = "Edit Success vs Linear Structure Alignment",
    savepdf: Optional[str] = None,
):
    """
    Scatter plot of edit success metric vs LRH structural measure.

    If LRH alignment predicts edit success, we expect a positive correlation.
    """
    with plt.rc_context(rc=_setup_style()):
        fig, ax = plt.subplots(figsize=(5, 4), dpi=150)
        ax.scatter(alignments, successes, alpha=0.5, s=20, color="#673AB7")

        # Trend line
        if len(alignments) > 2:
            z = np.polyfit(alignments, successes, 1)
            p = np.poly1d(z)
            x_sorted = np.sort(alignments)
            ax.plot(x_sorted, p(x_sorted), "--", color="gray", alpha=0.7, linewidth=1)

            from scipy.stats import spearmanr
            rho, pval = spearmanr(alignments, successes)
            ax.text(
                0.02, 0.98,
                f"ρ = {rho:.3f} (p = {pval:.3e})",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
            )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.2)
        plt.tight_layout()
        _save_or_show(savepdf)


def plot_lre_comparison_summary(
    comparison_metrics: Dict[str, float],
    title: str = "LRE vs ROME Comparison",
    savepdf: Optional[str] = None,
):
    """
    Bar chart summarizing LRE-ROME comparison metrics.
    """
    names = [k for k, v in comparison_metrics.items() if v is not None]
    values = [v for v in comparison_metrics.values() if v is not None]

    with plt.rc_context(rc=_setup_style()):
        fig, ax = plt.subplots(figsize=(6, 3.5), dpi=150)
        colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
        ax.barh(range(len(names)), values, color=colors, edgecolor="white")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("Value")
        ax.set_title(title)
        ax.axvline(0, color="gray", linewidth=0.5)
        plt.tight_layout()
        _save_or_show(savepdf)


def plot_explained_variance_across_layers(
    layer_explained: Dict[int, float],
    edit_layer: int = 17,
    title: str = "v-vector Explained Variance by Concept Directions Across Layers",
    savepdf: Optional[str] = None,
):
    """
    Line plot showing how much of v's variance is explained by concept
    directions extracted at each layer.
    """
    layers = sorted(layer_explained.keys())
    values = [layer_explained[l] for l in layers]

    with plt.rc_context(rc=_setup_style()):
        fig, ax = plt.subplots(figsize=(6, 3.5), dpi=150)
        ax.plot(layers, values, "o-", color="#009688", markersize=4)
        ax.axvline(edit_layer, color="gray", linestyle=":", alpha=0.7, label=f"Edit layer ({edit_layer})")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Explained variance")
        ax.set_title(title)
        ax.set_ylim([0, 1.05])
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
        plt.tight_layout()
        _save_or_show(savepdf)


def _save_or_show(savepdf: Optional[str]):
    if savepdf:
        os.makedirs(os.path.dirname(savepdf), exist_ok=True)
        plt.savefig(savepdf, bbox_inches="tight")
        plt.close()
    else:
        plt.show()