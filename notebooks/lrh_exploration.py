# %% [markdown]
# # ROME × Linear Representation Hypothesis: Interactive Exploration
#
# This notebook walks through the core analyses connecting ROME's model editing
# mechanism with the Linear Representation Hypothesis (LRH).
#
# ## Theoretical Background
#
# **ROME** edits factual associations by adding a rank-1 update to the MLP's
# `c_proj` weight matrix: $W' = W + u \otimes v$, where:
# - $u \in \mathbb{R}^{d_{inner}}$ (6400): the covariance-adjusted subject key
# - $v \in \mathbb{R}^{d_{model}}$ (1600): the optimized value vector
#
# **LRH** states that high-level concepts are encoded as linear directions in
# activation space. If true, ROME's v vector should decompose meaningfully
# in terms of concept directions.
#
# ## Research Questions
# 1. Do ROME edits preserve linear probe accuracy? (Probe Coherence)
# 2. Does v align with concept directions? (Direction Alignment)
# 3. How does the rank-1 update decompose? (Subspace Decomposition)
# 4. Does ROME's implicit relation match extracted LREs? (LRE Comparison)
# 5. Does linear structure predict edit success? (Predictive Analysis)

# %% [markdown]
# ## Setup

# %%
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from util import nethook
from util.globals import *
from rome import ROMEHyperParams

# Load model
MODEL_NAME = "gpt2-xl"
print(f"Loading {MODEL_NAME}...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).cuda()
tok = AutoTokenizer.from_pretrained(MODEL_NAME)
tok.pad_token = tok.eos_token
nethook.set_requires_grad(False, model)
print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params")

# Load ROME hyperparams
rome_hparams = ROMEHyperParams.from_json(HPARAMS_DIR / "ROME" / "gpt2-xl.json")
print(f"ROME edit layer: {rome_hparams.layers}")

# %%
from lrh import (
    LRHConfig,
    ActivationExtractor,
    load_lrh_dataset,
    extract_relation_directions,
    extract_rome_edit_vectors,
    decompose_v_in_concept_basis,
    full_rome_lrh_analysis,
    direction_alignment,
)

config = LRHConfig(model_name=MODEL_NAME)
extractor = ActivationExtractor(model, tok, config)

# Load CounterFact grouped by relation
dataset = load_lrh_dataset()
print(f"Relations: {dataset.relation_ids[:10]}")

# %% [markdown]
# ## 1. Concept Directions in GPT-2 XL
#
# First, let's extract concept directions for several relations and examine
# their geometry. If LRH holds, different relations should be encoded along
# approximately orthogonal directions.

# %%
# Extract concept directions at the ROME edit layer (17)
EDIT_LAYER = rome_hparams.layers[0]  # 17

concept_bank = extract_relation_directions(
    extractor, dataset,
    layers=[EDIT_LAYER],
    method="mean_diff",
    config=config,
    max_relations=10,
)

# Pairwise cosine similarity
sim_matrix, concept_names = concept_bank.pairwise_alignment(EDIT_LAYER)
print(f"\nExtracted {len(concept_names)} concept directions at layer {EDIT_LAYER}")
print(f"Concept names: {concept_names}")

# %%
from lrh.visualization import plot_concept_direction_heatmap

plot_concept_direction_heatmap(
    sim_matrix, concept_names,
    title=f"Concept Direction Cosine Similarity (Layer {EDIT_LAYER})",
)

# %% [markdown]
# ## 2. Anatomy of a ROME Edit
#
# Execute a single ROME edit and examine the u and v vectors.
# We'll see how v decomposes in the concept direction basis.

# %%
# Define a ROME edit
request = {
    "prompt": "{} plays the sport of",
    "subject": "LeBron James",
    "target_new": {"str": "football"},
}

# Extract u, v vectors (model is unchanged afterward)
deltas = extract_rome_edit_vectors(model, tok, request, rome_hparams)
weight_name = list(deltas.keys())[0]
u_vec, v_vec = deltas[weight_name]

print(f"u vector: shape={u_vec.shape}, norm={u_vec.norm():.4f} (MLP key space)")
print(f"v vector: shape={v_vec.shape}, norm={v_vec.norm():.4f} (residual stream)")

# %% [markdown]
# ### v-vector Decomposition
#
# How much of ROME's v vector is explained by known concept directions?

# %%
concept_dirs = list(concept_bank.get_layer(EDIT_LAYER).values())
decomp = decompose_v_in_concept_basis(v_vec, concept_dirs)

print(f"\nv-vector decomposition:")
print(f"  Total explained variance: {decomp['total_explained_variance']:.4f}")
print(f"  Residual norm: {decomp['residual_norm']:.4f}")
print(f"\n  Per-concept projections:")
for name, frac in sorted(decomp['projection_fractions'].items(), key=lambda x: -x[1]):
    print(f"    {name}: {frac:.4f}")

# %%
from lrh.visualization import plot_v_concept_decomposition

plot_v_concept_decomposition(decomp, title="ROME v-vector: Concept Decomposition")

# %% [markdown]
# ## 3. Probe Coherence Pre/Post Edit
#
# Train linear probes before the edit, then test whether they still work
# after the edit. This measures whether ROME preserves linear structure.

# %%
from lrh.probes import train_probes_across_layers, compute_probe_coherence, LinearProbe
from lrh.datasets import ProbeDataset
from collections import Counter

# Pick a relation with enough data
rid = dataset.relation_ids[0]
records = dataset.get_relation_records(rid)
target_counts = Counter(r["requested_rewrite"]["target_true"]["str"].strip() for r in records)
top_target = target_counts.most_common(1)[0][0]
print(f"Probing relation {rid}, target='{top_target}'")

probe_ds = ProbeDataset(dataset, rid, target_value=top_target)
all_data = probe_ds.train + probe_ds.val + probe_ds.test
prompts = [d["prompt"] for d in all_data]
subjects = [d["subject"] for d in all_data]
labels = torch.tensor([d["label"] for d in all_data], dtype=torch.long)

# Train probes at selected layers
probe_layers = [0, 10, 15, 17, 20, 30, 47]
probes = train_probes_across_layers(
    extractor, prompts, subjects, labels,
    layers=probe_layers, config=config,
)

# %%
# Test coherence with a ROME edit
test_prompts = [d["prompt"] for d in probe_ds.test]
test_subjects = [d["subject"] for d in probe_ds.test]
test_labels = torch.tensor([d["label"] for d in probe_ds.test], dtype=torch.long)

# Pick an edit from the same relation
rw = records[0]["requested_rewrite"]
edit_request = {
    "prompt": rw["prompt"],
    "subject": rw["subject"],
    "target_new": rw["target_new"],
}
print(f"Applying edit: {rw['subject']} -> {rw['target_new']['str']}")

probe_dict = {l: p for l, (p, _) in probes.items()}
coherence = compute_probe_coherence(
    model, tok, probe_dict,
    test_prompts, test_subjects, test_labels,
    edit_request, rome_hparams, config,
)

print("\nProbe coherence results:")
for layer, metrics in sorted(coherence.items()):
    print(f"  Layer {layer:2d}: pre={metrics['pre_accuracy']:.3f} "
          f"post={metrics['post_accuracy']:.3f} "
          f"delta={metrics['delta_accuracy']:+.3f}")

# %%
from lrh.visualization import plot_probe_accuracy_by_layer

pre_m = {l: {"accuracy": v["pre_accuracy"]} for l, v in coherence.items()}
post_m = {l: {"accuracy": v["post_accuracy"]} for l, v in coherence.items()}
plot_probe_accuracy_by_layer(pre_m, post_m, edit_layer=EDIT_LAYER)

# %% [markdown]
# ## 4. Gate Selectivity Analysis
#
# ROME's u vector determines *when* the edit fires: the edit contribution
# is proportional to $\langle k, u \rangle$ where $k$ is the MLP key.
# How selective is this gate?

# %%
from lrh.rome_lrh_bridge import compute_edit_gate_alignment

# Get some other entities for comparison
other_records = [r for r in records[1:11] if r["requested_rewrite"]["subject"] != rw["subject"]]
other_prompts = [r["requested_rewrite"]["prompt"] for r in other_records]
other_subjects = [r["requested_rewrite"]["subject"] for r in other_records]

gate = compute_edit_gate_alignment(
    model, tok, u_vec,
    target_prompts=[rw["prompt"]],
    target_subjects=[rw["subject"]],
    other_prompts=other_prompts,
    other_subjects=other_subjects,
    layer=EDIT_LAYER, config=config,
)

print(f"Gate selectivity:")
print(f"  d' = {gate['d_prime']:.3f}")
print(f"  Selectivity ratio = {gate['selectivity_ratio']:.3f}")
print(f"  Mean target activation = {gate['mean_target']:.4f}")
print(f"  Mean other activation = {gate['mean_other']:.4f}")

# %%
from lrh.visualization import plot_gate_activation_distributions

plot_gate_activation_distributions(gate)

# %% [markdown]
# ## 5. LRE Comparison
#
# Extract a Linear Relational Embedding (Hernandez et al. 2024) for the
# same relation and compare it to ROME's rank-1 update.

# %%
from lrh.lre import extract_lre, compare_lre_to_rome

pairs = dataset.get_subject_object_pairs(rid, n_pairs=200)
if len(pairs) >= 20:
    lre = extract_lre(
        extractor,
        prompts=[p["prompt"] for p in pairs],
        subjects=[p["subject"] for p in pairs],
        targets=[p["target_true"] for p in pairs],
        subject_layer=config.lre_subject_layer,
        object_layer=config.lre_object_layer,
        config=config,
        relation_id=rid,
    )
    print(f"LRE fit R² = {lre.fit_r2:.4f}")

    # Top singular values
    _, S, _ = lre.top_singular_directions(k=5)
    print(f"Top 5 singular values: {S.tolist()}")
    print(f"Rank-1 energy: {(S[0]**2 / (S**2).sum()):.4f}")

    # Compare with ROME
    W_proj = nethook.get_parameter(model, f"transformer.h.{EDIT_LAYER}.mlp.c_proj.weight").detach()
    W_fc = nethook.get_parameter(model, f"transformer.h.{EDIT_LAYER}.mlp.c_fc.weight").detach()

    comparison = compare_lre_to_rome(lre, u_vec, v_vec, W_proj, W_fc)
    print(f"\nLRE vs ROME comparison:")
    for k, v in comparison.items():
        if v is not None:
            print(f"  {k}: {v:.4f}")
else:
    print(f"Not enough pairs for LRE extraction ({len(pairs)})")

# %%
from lrh.visualization import plot_lre_comparison_summary

if len(pairs) >= 20:
    plot_lre_comparison_summary(comparison, title=f"LRE vs ROME: {rid}")

# %% [markdown]
# ## 6. Full Analysis Pipeline
#
# Run the complete ROME × LRH analysis for a single edit, packaging
# all the above into a single `ROMELRHAnalysis` object.

# %%
analysis = full_rome_lrh_analysis(
    model, tok, request, rome_hparams, config,
    concept_bank=concept_bank,
    other_prompts=other_prompts,
    other_subjects=other_subjects,
)

print(f"\nFull analysis summary:")
for k, v in analysis.metrics.items():
    print(f"  {k}: {v}")

# %% [markdown]
# ## Summary
#
# This notebook demonstrated the core analyses at the intersection of
# ROME and the Linear Representation Hypothesis:
#
# 1. **Concept directions** can be extracted at the ROME edit layer and
#    show interpretable structure (orthogonality between relations).
# 2. **ROME's v vector** partially decomposes into known concept directions,
#    with the explained variance indicating alignment with linear structure.
# 3. **Probe coherence** tests show whether edits are truly localized.
# 4. **Gate selectivity** (u vector analysis) reveals how precisely ROME
#    targets the intended subject.
# 5. **LRE comparison** tests consistency between ROME's rank-1 implicit
#    relation and the model's own linear relational structure.
#
# For systematic evaluation, use the experiment scripts:
# ```bash
# python -m lrh.experiments.run_probe_coherence
# python -m lrh.experiments.run_v_alignment
# python -m lrh.experiments.run_lre_comparison
# ```