# CLAUDE.md — Context for AI Assistants

This file provides everything a Claude instance (or any AI assistant) needs to understand and work on this repository.

---

## What This Repository Is

An implementation of **ROME** (Rank-One Model Editing, Meng et al. NeurIPS 2022) extended with a new **LRH** (Linear Representation Hypothesis) analysis module. The core question: does ROME's success stem from the model's own linear representational geometry?

**Supported models**: GPT-2 XL (1.5B, 48 layers, d_model=1600, d_inner=6400), GPT-J (6B).

---

## The ROME Algorithm

ROME edits factual associations by adding a rank-1 update to the MLP projection matrix at a target layer:

```
W' = W + u ⊗ v
```

- **u ∈ R^6400** (MLP key space): the "gate" — determines *when* the edit fires. Computed as the covariance-adjusted subject key vector. High dot product with the subject's MLP key, low with everything else.
- **v ∈ R^1600** (residual stream): the "payload" — determines *what* changes. Optimized via 20 Adam steps (lr=0.5) to induce the target token at the output, with KL-divergence regularization to minimize collateral damage.

**Default edit layer**: 17 (for GPT-2 XL). Loss computed at layer 47.

### Key files

| File | Role |
|------|------|
| `rome/rome_main.py` | Orchestration: `execute_rome()` computes u/v, `apply_rome_to_model()` applies the update |
| `rome/compute_u.py` | Left vector: extracts subject key, applies inverse second-moment normalization |
| `rome/compute_v.py` | Right vector: gradient optimization with KL regularization |
| `rome/layer_stats.py` | Precomputes/caches MLP key covariance over 100K Wikipedia samples |
| `rome/repr_tools.py` | Token-position-aware activation extraction |
| `rome/rome_hparams.py` | `ROMEHyperParams` dataclass |

### How a ROME edit executes

1. `apply_rome_to_model(model, tok, [request], hparams)` is the entry point
2. For each request, `execute_rome()` is called:
   - `compute_u()`: extract subject key at edit layer → apply C⁻¹ normalization → u
   - `compute_v()`: optimize delta vector at edit layer to produce target tokens at loss layer → v
3. Delta = u.unsqueeze(1) @ v.unsqueeze(0) is added to `transformer.h.{layer}.mlp.c_proj.weight`
4. Returns (edited_model, original_weights_copy)

---

## The LRH Module

Investigates whether ROME's edits align with the Linear Representation Hypothesis — that high-level concepts are encoded as linear directions in activation space.

### Five Research Questions

1. **Probe Coherence** (RQ1): Do ROME edits preserve linear probe accuracy at non-edited layers?
2. **Direction Alignment** (RQ2): Does v align with concept directions extracted independently?
3. **Subspace Decomposition** (RQ3): How does v decompose in a basis of concept directions?
4. **LRE Comparison** (RQ4): Does ROME's implicit rank-1 relation match extracted Linear Relational Embeddings?
5. **Predictive Analysis** (RQ5): Can linear structure metrics predict edit success?

### Key files

| File | Role |
|------|------|
| `lrh/config.py` | `LRHConfig` dataclass — probe layers, LRE params, training config |
| `lrh/extraction.py` | `ActivationExtractor` — unified interface for residual/MLP/key activations |
| `lrh/concept_directions.py` | Extract directions via mean-diff, DAS/SVD, or logistic; `ConceptDirectionBank` |
| `lrh/probes.py` | `LinearProbe` training/eval; `compute_probe_coherence()` for pre/post edit |
| `lrh/rome_lrh_bridge.py` | Core bridge: `extract_rome_edit_vectors()`, `decompose_v_in_concept_basis()`, `full_rome_lrh_analysis()` |
| `lrh/lre.py` | Linear Relational Embeddings: fit subject→object maps, compare with ROME's v |
| `lrh/metrics.py` | Cosine alignment, projection magnitude, subspace overlap, Grassmann distance, d-prime |
| `lrh/datasets.py` | `RelationGroupedDataset` — CounterFact grouped by Wikidata relation (P19, P27, etc.) |
| `lrh/visualization.py` | Publication-quality plots |
| `lrh/experiments/run_probe_coherence.py` | Experiment 1: probe accuracy pre/post edit |
| `lrh/experiments/run_v_alignment.py` | Experiment 2: v vs concept direction cosine similarity |
| `lrh/experiments/run_lre_comparison.py` | Experiment 3: LRE singular vectors vs ROME's v |

### Running experiments

```bash
python -m lrh.experiments.run_probe_coherence --model_name gpt2-xl --n_relations 5 --n_edits 10
python -m lrh.experiments.run_v_alignment --model_name gpt2-xl --n_relations 10 --n_edits 50
python -m lrh.experiments.run_lre_comparison --model_name gpt2-xl --n_relations 10
```

---

## Project Structure

```
rome/
├── rome/                     # Core ROME algorithm
├── lrh/                      # LRH analysis module (new, untracked)
│   └── experiments/          # Three experiment scripts
├── baselines/                # Comparison methods: FT, KN, MEND, EFK
├── dsets/                    # Dataset loaders (CounterFact, zsRE, knowns)
├── experiments/              # Evaluation framework (evaluate.py, causal_trace.py)
├── util/                     # nethook.py, globals.py, generate.py, runningstats.py, logit_lens.py
├── hparams/                  # JSON configs per method per model
│   ├── ROME/                 # gpt2-xl.json, gpt-j-6B.json, etc.
│   ├── LRH/                  # gpt2-xl.json (new)
│   ├── FT/, KN/, MEND/, KE/ # Baseline configs
├── notebooks/                # Interactive analysis (jupytext + ipynb)
├── scripts/                  # setup_conda.sh, rome.yml, lrh.yml
├── data/                     # Auto-downloaded datasets and cached stats
├── results/                  # Experiment output
├── globals.yml               # Path config (RESULTS_DIR, DATA_DIR, STATS_DIR, HPARAMS_DIR)
└── CLAUDE.md                 # This file
```

---

## Key Abstractions

### `util/nethook.py` — Model Instrumentation

The backbone for all activation extraction and intervention:
- `Trace(model, layer_name)` — context manager that hooks a single layer and captures its output
- `TraceDict(model, [layer_names])` — hooks multiple layers simultaneously
- `get_module(model, name)` / `replace_module(model, name, new_module)` — access/swap modules by dotted path
- `get_parameter(model, name)` — get a parameter tensor by dotted path
- `set_requires_grad(requires_grad, *models)` — freeze/unfreeze

### `util/globals.py` — Path Configuration

Loads from `globals.yml`:
- `RESULTS_DIR` = `results/`
- `DATA_DIR` = `data/`
- `STATS_DIR` = `data/stats/`
- `HPARAMS_DIR` = `hparams/`
- `REMOTE_ROOT_URL` = `https://rome.baulab.info` (for auto-downloading data/stats)

### CounterFact Dataset

10K+ factual editing records. Each record has:
```python
{
  "case_id": int,
  "requested_rewrite": {
    "prompt": "The mother tongue of {} is",  # {} = subject placeholder
    "relation_id": "P103",                    # Wikidata property
    "target_new": {"str": "English"},         # counterfactual target
    "target_true": {"str": "French"},         # true answer
    "subject": "Danielle Darrieux"
  },
  "paraphrase_prompts": [...],
  "neighborhood_prompts": [...],
  "generation_prompts": [...]
}
```

Auto-downloads to `data/counterfact.json` from `rome.baulab.info`.

---

## GPT-2 XL Architecture Reference

- 48 transformer layers (`transformer.h.0` through `transformer.h.47`)
- d_model = 1600 (residual stream width)
- d_inner = 6400 (MLP hidden dimension, 4x expansion)
- Module paths:
  - Layer output: `transformer.h.{i}`
  - MLP block: `transformer.h.{i}.mlp`
  - MLP up-projection: `transformer.h.{i}.mlp.c_fc` (1600 → 6400)
  - MLP down-projection: `transformer.h.{i}.mlp.c_proj` (6400 → 1600) — **this is what ROME edits**
  - Attention: `transformer.h.{i}.attn`
  - Final layer norm: `transformer.ln_f`
  - Unembedding (tied): `transformer.wte`

---

## Hyperparameter Configs

### ROME (`hparams/ROME/gpt2-xl.json`)

```json
{
  "layers": [17],
  "fact_token": "subject_last",
  "v_num_grad_steps": 20, "v_lr": 0.5, "v_loss_layer": 47,
  "v_weight_decay": 0.5, "clamp_norm_factor": 4, "kl_factor": 0.0625,
  "mom2_adjustment": true,
  "mom2_dataset": "wikipedia", "mom2_n_samples": 100000, "mom2_dtype": "float32",
  "rewrite_module_tmp": "transformer.h.{}.mlp.c_proj",
  "layer_module_tmp": "transformer.h.{}",
  "mlp_module_tmp": "transformer.h.{}.mlp",
  "attn_module_tmp": "transformer.h.{}.attn",
  "ln_f_module": "transformer.ln_f",
  "lm_head_module": "transformer.wte"
}
```

### LRH (`hparams/LRH/gpt2-xl.json`)

```json
{
  "probe_layers": [0, 5, 10, 15, 17, 20, 25, 30, 35, 40, 47],
  "rome_target_layer": 17,
  "probe_train_size": 500, "probe_test_size": 200,
  "probe_lr": 1e-3, "probe_epochs": 100, "probe_weight_decay": 1e-2,
  "n_contrastive_pairs": 200, "direction_method": "mean_diff",
  "lre_rank": 1, "lre_subject_layer": 8, "lre_object_layer": 36, "lre_ridge_alpha": 1.0,
  "token_strategy": "subject_last", "extraction_batch_size": 32
}
```

---

## Environment Setup

Two conda environments:

| Environment | File | PyTorch | Transformers | Use |
|-------------|------|---------|-------------|-----|
| `rome` | `scripts/rome.yml` | 1.10.2 | 4.15.0 | Full ROME + baselines + evaluation |
| `rome-lrh` | `scripts/lrh.yml` | 1.12.1 | >=4.23.0 | Lighter env for LRH analysis |

Both use Python 3.9.7 and CUDA 11.3.1.

```bash
# Setup (default ROME env)
bash scripts/setup_conda.sh

# Setup LRH env
RECIPE=lrh ENV_NAME=rome-lrh bash scripts/setup_conda.sh
```

---

## Common Tasks

### Apply a single ROME edit

```python
from rome import ROMEHyperParams, apply_rome_to_model
from util.globals import HPARAMS_DIR

hparams = ROMEHyperParams.from_json(HPARAMS_DIR / "ROME" / "gpt2-xl.json")
request = {
    "prompt": "{} plays the sport of",
    "subject": "LeBron James",
    "target_new": {"str": "football"},
}
model_edited, orig_weights = apply_rome_to_model(model, tok, [request], hparams)
```

### Extract edit vectors without modifying the model

```python
from lrh import extract_rome_edit_vectors
deltas = extract_rome_edit_vectors(model, tok, request, rome_hparams)
u_vec, v_vec = deltas["transformer.h.17.mlp.c_proj"]
```

### Run the full LRH analysis pipeline

```python
from lrh import LRHConfig, full_rome_lrh_analysis
config = LRHConfig(model_name="gpt2-xl")
analysis = full_rome_lrh_analysis(model, tok, request, rome_hparams, config)
```

### Run the ROME evaluation benchmark

```bash
python -m experiments.evaluate --alg_name ROME --model_name gpt2-xl --ds_name cf --num_edits 100
```

---

## Current Development State

- **Tracked (committed)**: All original ROME code, baselines, evaluation framework, utilities, notebooks
- **Untracked (new work)**: `lrh/` module, `hparams/LRH/`, `notebooks/lrh_exploration.py`, `scripts/lrh.yml`
- **Modified**: `scripts/setup_conda.sh` (added LRH env support)

The LRH module is the active area of development. The ROME core is stable.

---

## Conventions

- Hyperparams are JSON files in `hparams/<METHOD>/<model>.json`, loaded via `HyperParams.from_json()`
- Module paths use `.format()` templates (e.g., `"transformer.h.{}.mlp.c_proj"`)
- Token position strategies: `"subject_last"` (default), `"subject_first"`, `"last"`
- Prompts use `{}` as the subject placeholder
- All activation extraction goes through `nethook.Trace` / `nethook.TraceDict`
- Datasets auto-download from `rome.baulab.info` to `data/`
- Experiments save results to `results/<method>/run_<id>/`
- The interactive notebook `notebooks/lrh_exploration.py` uses jupytext percent format
