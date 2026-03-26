# The Dark Subspace: ROME Edits Succeed Outside the Model's Linear Concept Geometry

This repository contains code and experiments for investigating the relationship between knowledge editing (ROME, MEND) and the Linear Representation Hypothesis (LRH) in large language models. All experiments use GPT-2 XL (1.5B parameters).

**Built on top of the [ROME](https://github.com/kmeng01/rome) codebase by [Meng et al. (NeurIPS 2022)](https://arxiv.org/abs/2202.05262).** The original ROME implementation, evaluation framework, baselines, causal tracing, and dataset infrastructure are theirs. This project extends that codebase with an LRH analysis module (`lrh/`), experiment scripts (`scripts/`), an r-ROME reimplementation (`rrome/`), and the paper source (`paper/`).

## Key Findings

1. ROME's edit vectors cluster strongly by concept (97% LDA accuracy) but are nearly **orthogonal** to independently extracted concept directions (cosine similarity < 0.08)
2. The causal signal lives in a **dark subspace** orthogonal to all identifiable geometric structure, carrying 50--54% of the edit's efficacy (96% if scaled), while the concept component carries 0%
3. This orthogonality generalises across editing methods: **MEND** shows the same pattern despite using a different mechanism and editing different layers

## Repository Structure

```
rome/                   # Original ROME algorithm (Meng et al.)
lrh/                    # LRH analysis module (new)
    config.py           # LRHConfig dataclass
    extraction.py       # Activation extraction
    concept_directions.py  # Mean-diff, logistic, DAS directions
    probes.py           # Linear probe training/eval
    rome_lrh_bridge.py  # Core bridge between ROME and LRH
    lre.py              # Linear Relational Embeddings
    metrics.py          # Alignment and subspace metrics
    datasets.py         # CounterFact grouped by relation
    visualization.py    # Publication-quality plots
    experiments/        # Three main experiment runners
rrome/                  # r-ROME reimplementation (Hase et al. fix)
scripts/                # All experiment scripts (exp1-exp8)
paper/                  # Paper LaTeX source
baselines/              # MEND, FT, KN, EFK (from original ROME repo)
experiments/            # Evaluation framework (from original ROME repo)
notebooks/              # Interactive analysis notebooks
hparams/                # Hyperparameter configs (ROME, LRH, baselines)
util/                   # nethook, globals, generate (from original ROME repo)
dsets/                  # Dataset loaders (from original ROME repo)
```

## Experiments

| Script | Description |
|--------|-------------|
| `exp1_linear_structure.py` | Probe accuracy across layers; concept direction extraction |
| `exp1b_shared_component.py` | Edit vector alignment with concept directions; v_mean transfer |
| `exp2_edit_geometry.py` | PERMANOVA, LDA clustering, LRE comparison |
| `exp3_layer_propagation.py` | Perturbation tracking across layers |
| `exp4_causal_decomposition.py` | Decompose v_mean into concept, LDA, and dark residual components |
| `exp4_lda_scaling.py` | Scaling analysis of decomposition components |
| `exp5_layer_sweep.py` | Edit efficacy vs alignment across layers |
| `exp6_gate_geometry.py` | Gate vector (u) concept structure analysis |
| `exp7_edit_specificity.py` | Entity-level vs concept-level edit selectivity |
| `exp8_mend_comparison.py` | Cross-method generalisation with MEND |

## Installation

```bash
# ROME environment (full functionality)
bash scripts/setup_conda.sh

# LRH-only environment (lighter)
RECIPE=lrh ENV_NAME=rome-lrh bash scripts/setup_conda.sh
```

Requires `conda` and a CUDA-capable GPU. GPT-2 XL requires ~6GB VRAM.

## Usage

### Run a ROME edit
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

### Run an LRH experiment
```bash
python scripts/exp1_linear_structure.py
python scripts/exp4_causal_decomposition.py
```

### Run the full LRH analysis pipeline
```python
from lrh import LRHConfig, full_rome_lrh_analysis
config = LRHConfig(model_name="gpt2-xl")
analysis = full_rome_lrh_analysis(model, tok, request, rome_hparams, config)
```

## Acknowledgements

This project is built entirely on top of the **ROME** codebase by Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. The original repository is available at [github.com/kmeng01/rome](https://github.com/kmeng01/rome) and is licensed under the MIT License. All original code, evaluation infrastructure, dataset loaders, causal tracing, and baseline implementations are theirs.

## How to Cite

If you use the LRH analysis from this repository:
```bibtex
@article{elcock2025dark,
  title={The Dark Subspace: {ROME} Edits Succeed Outside the Model's Linear Concept Geometry},
  author={James Elcock},
  year={2025}
}
```

If you use the ROME implementation, please also cite the original work:
```bibtex
@article{meng2022locating,
  title={Locating and Editing Factual Associations in {GPT}},
  author={Kevin Meng and David Bau and Alex Andonian and Yonatan Belinkov},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  year={2022}
}
```

## License

MIT License (see [LICENSE](LICENSE)). Original ROME code copyright (c) 2022 Kevin Meng.