"""
Generate publication-quality figures for the ROME × LRH paper.

Produces 8 figures covering Experiments 1–8:
  Fig 1: Layer sweep — efficacy vs concept alignment trade-off (Exp 5)
  Fig 2: Causal decomposition — component efficacy (Exp 4)
  Fig 3: Layer propagation — Δh norm and alignment across layers (Exp 3)
  Fig 4: Edit specificity — KL/ΔP hierarchy (Exp 7)
  Fig 5: u-space vs v-space clustering comparison (Exp 2 vs 6)
  Fig 6: Gate activation profiling (Exp 6 Phase C)
  Fig 7: ROME vs MEND comparison (Exp 8)
  Fig 8: Grand summary — concept alignment across all experiments

Saves to results/figures/
"""

import json
import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FIGURES_DIR = "results/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "rome": "#2171b5",
    "mend": "#e6550d",
    "concept": "#31a354",
    "lda": "#756bb1",
    "residual": "#636363",
    "efficacy": "#de2d26",
    "alignment": "#2171b5",
    "probe": "#31a354",
    "self": "#de2d26",
    "same_target": "#fc9272",
    "same_relation": "#fee0d2",
    "diff_relation": "#deebf7",
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ============================================================
# Figure 1: Layer Sweep (Exp 5)
# ============================================================
def fig1_layer_sweep():
    r = load_json("results/exp5_layer_sweep/results.json")
    ls = r["layer_summary"]

    layers = [5, 10, 15, 17, 20, 25, 30, 35, 40]
    efficacy = [ls[str(l)]["efficacy_mean"] for l in layers]
    alignment_md = [ls[str(l)]["mean_diff_shared_abs_mean"] for l in layers]
    alignment_lg = [ls[str(l)]["logistic_shared_abs_mean"] for l in layers]
    probe_acc = [ls[str(l)]["probe_accuracy_mean"] for l in layers]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))

    ax1.plot(layers, efficacy, "o-", color=COLORS["efficacy"], lw=2, ms=7,
             label="Edit efficacy", zorder=3)
    ax1.set_xlabel("Edit layer")
    ax1.set_ylabel("Efficacy", color=COLORS["efficacy"])
    ax1.tick_params(axis="y", labelcolor=COLORS["efficacy"])
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(layers, alignment_md, "s--", color=COLORS["alignment"], lw=1.5, ms=6,
             label="|cos(v, concept)| mean-diff", alpha=0.8)
    ax2.plot(layers, alignment_lg, "^--", color="#6baed6", lw=1.5, ms=6,
             label="|cos(v, concept)| logistic", alpha=0.8)
    ax2.plot(layers, probe_acc, "D-.", color=COLORS["probe"], lw=1.5, ms=5,
             label="Probe accuracy", alpha=0.8)
    ax2.set_ylabel("Alignment / Probe accuracy")
    ax2.set_ylim(0, 1.05)

    # Highlight layer 17
    ax1.axvline(17, color="gray", ls=":", alpha=0.5, lw=1)
    ax1.annotate("ROME default\n(layer 17)", xy=(17, 0.82), fontsize=8,
                 ha="center", color="gray")

    # Threshold line
    ax2.axhline(0.30, color="gray", ls="--", alpha=0.3, lw=1)
    ax2.annotate("alignment threshold (0.30)", xy=(32, 0.31), fontsize=7, color="gray")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left", framealpha=0.9)

    ax1.set_title("Exp 5: Efficacy–Alignment Trade-off Across Edit Layers")
    ax1.set_xticks(layers)
    fig.savefig(f"{FIGURES_DIR}/fig1_layer_sweep.png")
    fig.savefig(f"{FIGURES_DIR}/fig1_layer_sweep.pdf")
    plt.close(fig)
    print("  Fig 1: Layer sweep saved")


# ============================================================
# Figure 2: Causal Decomposition (Exp 4)
# ============================================================
def fig2_causal_decomposition():
    r = load_json("results/exp4/results.json")
    ga = r["grand_average"]

    conditions = [
        ("Own v", ga["efficacy"]["own_v"], ga["target_prob"]["own_v"]),
        ("Full v_mean", ga["efficacy"]["full_v_mean"], ga["target_prob"]["full_v_mean"]),
        ("Residual\n(rescaled)", ga["efficacy"]["residual_rescaled"], ga["target_prob"]["residual_rescaled"]),
        ("Residual\n(natural)", ga["efficacy"]["residual_natural"], ga["target_prob"]["residual_natural"]),
        ("LDA\n(rescaled)", ga["efficacy"]["lda_rescaled"], ga["target_prob"]["lda_rescaled"]),
        ("Concept+LDA", ga["efficacy"]["concept_plus_lda"], ga["target_prob"]["concept_plus_lda"]),
        ("Concept\n(rescaled)", ga["efficacy"]["concept_rescaled"], ga["target_prob"]["concept_rescaled"]),
        ("Random", ga["efficacy"]["random"], ga["target_prob"]["random"]),
    ]

    labels = [c[0] for c in conditions]
    effs = [c[1] for c in conditions]
    probs = [c[2] for c in conditions]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars1 = ax.bar(x - width/2, effs, width, label="Efficacy (top-1)", color=COLORS["efficacy"], alpha=0.85)
    bars2 = ax.bar(x + width/2, probs, width, label="P(target)", color=COLORS["alignment"], alpha=0.85)

    ax.set_ylabel("Score")
    ax.set_title("Exp 4: Causal Decomposition — Which Component of v Carries the Edit?")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.1)

    # Annotate the dark subspace
    ax.annotate("", xy=(2.5, 0.95), xytext=(6.5, 0.95),
                arrowprops=dict(arrowstyle="<->", color="gray", lw=1.5))
    ax.text(4.5, 0.97, '"Dark subspace" carries the signal', ha="center", fontsize=9, color="gray")

    fig.savefig(f"{FIGURES_DIR}/fig2_causal_decomposition.png")
    fig.savefig(f"{FIGURES_DIR}/fig2_causal_decomposition.pdf")
    plt.close(fig)
    print("  Fig 2: Causal decomposition saved")


# ============================================================
# Figure 3: Layer Propagation (Exp 3)
# ============================================================
def fig3_layer_propagation():
    r = load_json("results/exp3/results.json")
    pn = r["perturbation_norms"]
    ga = r["grand_average"]
    lda_align = r["lda_alignment_by_layer"]

    all_layers = list(range(48))
    norms_subj = [pn["subject_token"].get(str(l), 0) for l in all_layers]
    norms_last = [pn["last_token"].get(str(l), 0) for l in all_layers]

    align_layers = [17, 20, 25, 30, 35, 40, 47]
    concept_align = [ga["subject_token"][str(l)] for l in align_layers]
    lda_vals = [lda_align[str(l)] for l in align_layers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: perturbation norms
    ax1.plot(all_layers, norms_subj, "-", color=COLORS["rome"], lw=2, label="Subject token")
    ax1.plot(all_layers, norms_last, "-", color=COLORS["mend"], lw=2, label="Last token")
    ax1.axvline(17, color="gray", ls=":", alpha=0.5)
    ax1.annotate("Edit\ninjection", xy=(17, 50), fontsize=8, ha="center", color="gray")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("||Δh||")
    ax1.set_title("Perturbation Magnitude")
    ax1.legend()

    # Right: alignment crossover
    ax2.plot(align_layers, concept_align, "o-", color=COLORS["concept"], lw=2, ms=7,
             label="Concept alignment")
    ax2.plot(align_layers, lda_vals, "s-", color=COLORS["lda"], lw=2, ms=7,
             label="LDA alignment")
    ax2.axvline(17, color="gray", ls=":", alpha=0.5)
    # Find crossover
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("|cos(Δh, direction)|")
    ax2.set_title("Alignment: Concept vs LDA Directions")
    ax2.legend()
    ax2.set_ylim(0, 0.20)

    fig.suptitle("Exp 3: Layer Propagation — Signal Amplification and Alignment Crossover",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig3_layer_propagation.png")
    fig.savefig(f"{FIGURES_DIR}/fig3_layer_propagation.pdf")
    plt.close(fig)
    print("  Fig 3: Layer propagation saved")


# ============================================================
# Figure 4: Edit Specificity (Exp 7)
# ============================================================
def fig4_edit_specificity():
    r = load_json("results/exp7_specificity/results.json")
    cs = r["condition_summaries"]

    conditions = ["self", "same_target", "same_relation", "different_relation"]
    labels = ["Self\n(edited entity)", "Same target\nvalue", "Same\nrelation", "Different\nrelation"]
    colors = [COLORS["self"], COLORS["same_target"], COLORS["same_relation"], COLORS["diff_relation"]]

    delta_p = [cs[c]["delta_p_target_mean"] for c in conditions]
    kl = [cs[c]["kl_mean"] for c in conditions]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # ΔP(target)
    bars = ax1.bar(range(len(conditions)), delta_p, color=colors, edgecolor="black", lw=0.5)
    ax1.set_xticks(range(len(conditions)))
    ax1.set_xticklabels(labels, fontsize=9)
    ax1.set_ylabel("ΔP(target)")
    ax1.set_title("Probability Shift")
    # Log-scale inset for the tiny values
    for i, v in enumerate(delta_p):
        if v < 0.01:
            ax1.annotate(f"{v:.1e}", xy=(i, v + 0.02), ha="center", fontsize=8, color="gray")
        else:
            ax1.annotate(f"{v:.3f}", xy=(i, v + 0.02), ha="center", fontsize=8)

    # KL divergence
    bars2 = ax2.bar(range(len(conditions)), kl, color=colors, edgecolor="black", lw=0.5)
    ax2.set_xticks(range(len(conditions)))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel("KL divergence")
    ax2.set_title("Distribution Shift")
    for i, v in enumerate(kl):
        if v < 0.01:
            ax2.annotate(f"{v:.1e}", xy=(i, v + 0.15), ha="center", fontsize=8, color="gray")
        else:
            ax2.annotate(f"{v:.2f}", xy=(i, v + 0.15), ha="center", fontsize=8)

    fig.suptitle("Exp 7: Edit Specificity — 900× Selectivity for Edited Entity", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig4_edit_specificity.png")
    fig.savefig(f"{FIGURES_DIR}/fig4_edit_specificity.pdf")
    plt.close(fig)
    print("  Fig 4: Edit specificity saved")


# ============================================================
# Figure 5: u-Space vs v-Space Clustering (Exp 2 vs 6)
# ============================================================
def fig5_uv_clustering():
    r6 = load_json("results/exp6_gate/results.json")
    r2 = load_json("results/exp2/results.json")

    metrics = ["PERMANOVA R²\n(relation)", "PERMANOVA R²\n(target)", "LDA\naccuracy"]
    v_vals = [
        r2["permanova"]["relation"]["R2"],
        r2["permanova"]["target_value"]["R2"],
        r2["lda"]["cv_accuracy"],
    ]
    u_vals = [
        r6["phase_a"]["permanova_relation"]["R2"],
        r6["phase_a"]["permanova_target"]["R2"],
        r6["phase_a"]["lda_cv_accuracy"],
    ]

    x = np.arange(len(metrics))
    width = 0.3

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - width/2, v_vals, width, label="v-space (payload)", color=COLORS["rome"], alpha=0.85)
    ax.bar(x + width/2, u_vals, width, label="u-space (gate)", color=COLORS["mend"], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_title("Exp 2 vs 6: v Carries Concept Structure, u Does Not")
    ax.legend()
    ax.set_ylim(0, 1.1)

    # Annotate ratios
    for i in range(len(metrics)):
        if u_vals[i] > 0:
            ratio = v_vals[i] / u_vals[i]
            ax.annotate(f"{ratio:.0f}×", xy=(i, max(v_vals[i], u_vals[i]) + 0.03),
                        ha="center", fontsize=9, color="gray", fontweight="bold")

    fig.savefig(f"{FIGURES_DIR}/fig5_uv_clustering.png")
    fig.savefig(f"{FIGURES_DIR}/fig5_uv_clustering.pdf")
    plt.close(fig)
    print("  Fig 5: u vs v clustering saved")


# ============================================================
# Figure 6: Gate Activation Profiling (Exp 6 Phase C)
# ============================================================
def fig6_gate_activation():
    r6 = load_json("results/exp6_gate/results.json")
    pc = r6["phase_c"]["per_concept"]

    concepts = sorted(pc.keys())
    own_act = [pc[c]["mean_act_own"] for c in concepts]
    same_act = [pc[c]["mean_act_same_concept"] for c in concepts]
    diff_act = [pc[c]["mean_act_diff_rel"] for c in concepts]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: activation magnitudes
    x = np.arange(len(concepts))
    width = 0.25
    ax1.bar(x - width, own_act, width, label="Own entity (u·k_self)", color=COLORS["self"])
    ax1.bar(x, same_act, width, label="Same concept", color=COLORS["same_target"])
    ax1.bar(x + width, diff_act, width, label="Different relation", color=COLORS["diff_relation"])
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.replace("_", "\n") for c in concepts], fontsize=7, rotation=45, ha="right")
    ax1.set_ylabel("Gate activation (u · k)")
    ax1.set_title("Gate Activation by Condition")
    ax1.legend(fontsize=8)

    # Right: d-prime decomposition
    d_overall = [pc[c]["d_prime_overall_mean"] for c in concepts]
    d_within = [pc[c]["d_prime_within_rel_mean"] for c in concepts]
    d_template = [pc[c]["d_prime_template_mean"] for c in concepts]

    ax2.bar(x - width, d_overall, width, label="d'(overall)", color=COLORS["rome"], alpha=0.8)
    ax2.bar(x, d_within, width, label="d'(within-relation)", color=COLORS["concept"], alpha=0.8)
    ax2.bar(x + width, d_template, width, label="d'(template effect)", color=COLORS["lda"], alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.replace("_", "\n") for c in concepts], fontsize=7, rotation=45, ha="right")
    ax2.set_ylabel("d-prime")
    ax2.set_title("Concept Selectivity Decomposition")
    ax2.axhline(0, color="black", lw=0.5)
    ax2.legend(fontsize=8)

    fig.suptitle("Exp 6: Gate Vector (u) — Entity-Selective, Not Concept-Selective", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig6_gate_activation.png")
    fig.savefig(f"{FIGURES_DIR}/fig6_gate_activation.pdf")
    plt.close(fig)
    print("  Fig 6: Gate activation saved")


# ============================================================
# Figure 7: ROME vs MEND (Exp 8)
# ============================================================
def fig7_rome_vs_mend():
    r8 = load_json("results/exp8_mend/results.json")
    r2 = load_json("results/exp2/results.json")

    # Clustering comparison
    metrics_cluster = ["PERMANOVA R²\n(relation)", "PERMANOVA R²\n(target)", "LDA\naccuracy"]
    rome_cluster = [
        r2["permanova"]["relation"]["R2"],
        r2["permanova"]["target_value"]["R2"],
        r2["lda"]["cv_accuracy"],
    ]
    mend_cluster = [
        r8["phase_2_clustering"]["47"]["permanova_relation"]["R2"],
        r8["phase_2_clustering"]["47"]["permanova_target"]["R2"],
        r8["phase_2_clustering"]["47"]["lda_cv_accuracy"],
    ]

    # Alignment comparison
    # ROME alignment from exp2 LDA concept cosines
    rome_align_md = 0.06  # from exp2 grand average
    rome_align_lg = 0.05
    p1 = r8["phase_1_alignment"]
    mend_align_md = p1["grand_summary"]["47"]["shared_alignment_mean_diff"] if "grand_summary" in p1 else 0.121
    mend_align_lg = p1["grand_summary"]["47"]["shared_alignment_logistic"] if "grand_summary" in p1 else 0.068

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Left: clustering
    x = np.arange(len(metrics_cluster))
    width = 0.3
    ax1.bar(x - width/2, rome_cluster, width, label="ROME (layer 17)", color=COLORS["rome"], alpha=0.85)
    ax1.bar(x + width/2, mend_cluster, width, label="MEND (layer 47)", color=COLORS["mend"], alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_cluster)
    ax1.set_ylabel("Score")
    ax1.set_title("Edit Geometry Clustering")
    ax1.legend()
    ax1.set_ylim(0, 1.1)

    # Right: concept alignment
    metrics_align = ["Mean-diff\nalignment", "Logistic\nalignment", "Efficacy"]
    rome_align = [rome_align_md, rome_align_lg, 0.98]
    mend_align = [mend_align_md, mend_align_lg, 0.84]

    x2 = np.arange(len(metrics_align))
    ax2.bar(x2 - width/2, rome_align, width, label="ROME (layer 17)", color=COLORS["rome"], alpha=0.85)
    ax2.bar(x2 + width/2, mend_align, width, label="MEND (layer 47)", color=COLORS["mend"], alpha=0.85)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(metrics_align)
    ax2.set_ylabel("Score")
    ax2.set_title("Concept Alignment & Efficacy")
    ax2.legend()
    ax2.set_ylim(0, 1.1)

    # Threshold line on alignment
    ax2.axhline(0.30, color="gray", ls="--", alpha=0.4, lw=1)
    ax2.annotate("alignment\nthreshold", xy=(0.5, 0.31), fontsize=7, color="gray")

    fig.suptitle("Exp 8: Both Methods Avoid Concept-Aligned Subspaces", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{FIGURES_DIR}/fig7_rome_vs_mend.png")
    fig.savefig(f"{FIGURES_DIR}/fig7_rome_vs_mend.pdf")
    plt.close(fig)
    print("  Fig 7: ROME vs MEND saved")


# ============================================================
# Figure 8: Grand Summary
# ============================================================
def fig8_grand_summary():
    """Concept alignment |cos(v, concept_dir)| across all experiments."""

    # Collect alignment values from each experiment
    data = {
        "Exp 1B\n(L17, shared)": {"md": 0.111, "lg": 0.111},  # median shared alignment
        "Exp 2\n(L17, LDA-concept)": {"md": 0.047, "lg": 0.072},
        "Exp 3\n(L17, Δh)": {"md": 0.057, "lg": 0.057},
        "Exp 3\n(L35, Δh)": {"md": 0.093, "lg": 0.093},
        "Exp 5\n(L17, shared)": {"md": 0.098, "lg": 0.098},
        "Exp 5\n(L30, shared)": {"md": 0.132, "lg": 0.132},
        "Exp 8 MEND\n(L47, shared)": {"md": 0.121, "lg": 0.068},
    }

    labels = list(data.keys())
    md_vals = [data[k]["md"] for k in labels]
    lg_vals = [data[k]["lg"] for k in labels]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(labels))
    width = 0.3

    ax.bar(x - width/2, md_vals, width, label="Mean-diff direction", color=COLORS["concept"], alpha=0.85)
    ax.bar(x + width/2, lg_vals, width, label="Logistic direction", color=COLORS["lda"], alpha=0.85)

    ax.axhline(0.30, color="red", ls="--", lw=1.5, alpha=0.7, label="Meaningful alignment threshold")

    # Random baseline in R^1600
    random_baseline = 1.0 / np.sqrt(1600) * np.sqrt(2 / np.pi)  # expected |cos| for random unit vectors
    ax.axhline(random_baseline, color="gray", ls=":", lw=1, alpha=0.6)
    ax.annotate(f"Random baseline ({random_baseline:.3f})", xy=(len(labels)-1, random_baseline + 0.005),
                fontsize=8, color="gray", ha="right")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("|cos(edit vector, concept direction)|")
    ax.set_title("Concept Alignment Across All Experiments: Consistently Low")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 0.45)

    fig.savefig(f"{FIGURES_DIR}/fig8_grand_summary.png")
    fig.savefig(f"{FIGURES_DIR}/fig8_grand_summary.pdf")
    plt.close(fig)
    print("  Fig 8: Grand summary saved")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Generating figures...")
    fig1_layer_sweep()
    fig2_causal_decomposition()
    fig3_layer_propagation()
    fig4_edit_specificity()
    fig5_uv_clustering()
    fig6_gate_activation()
    fig7_rome_vs_mend()
    fig8_grand_summary()
    print(f"\nAll figures saved to {FIGURES_DIR}/")