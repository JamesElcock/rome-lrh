"""
Activation extraction utilities for ROME x LRH analysis.

Wraps util/nethook.py (Trace, TraceDict) and rome/repr_tools.py to provide
a uniform interface for collecting hidden representations at arbitrary layers,
modules, and token positions.

Dimensional conventions (GPT-2 XL):
    - Residual stream (output of transformer.h.{layer}): R^{1600}
    - MLP output (output of transformer.h.{layer}.mlp): R^{1600}
    - MLP key space (input to transformer.h.{layer}.mlp.c_proj): R^{6400}
    - Attention output (output of transformer.h.{layer}.attn): R^{1600}

Token position strategies:
    - "subject_last": last token of the subject entity (ROME's default)
    - "subject_first": first token of the subject entity
    - "last": last token of the full prompt
    - int: explicit token index
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome import repr_tools
from util import nethook

from .config import LRHConfig


class ActivationExtractor:
    """
    Extracts activations from a model at specified layers and modules.

    This class centralizes activation extraction so that probes, concept
    direction methods, and LRE estimation all use consistent representations.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        config: LRHConfig = None,
    ):
        self.model = model
        self.tok = tok
        self.config = config or LRHConfig()
        self.device = next(model.parameters()).device

    def extract_residual_stream(
        self,
        prompts: List[str],
        subjects: List[str],
        layers: List[int],
        token_strategy: str = "subject_last",
    ) -> Dict[int, torch.Tensor]:
        """
        Extract residual stream activations (output of transformer layer blocks).

        Returns:
            Dict mapping layer_idx -> Tensor of shape (n_prompts, d_model).
        """
        return self._extract_at_module(
            prompts=prompts,
            subjects=subjects,
            layers=layers,
            module_template=self.config.residual_module_tmp,
            track="out",
            token_strategy=token_strategy,
        )

    def extract_mlp_output(
        self,
        prompts: List[str],
        subjects: List[str],
        layers: List[int],
        token_strategy: str = "subject_last",
    ) -> Dict[int, torch.Tensor]:
        """
        Extract MLP output activations (in residual stream space, R^{d_model}).

        Returns:
            Dict mapping layer_idx -> Tensor of shape (n_prompts, d_model).
        """
        return self._extract_at_module(
            prompts=prompts,
            subjects=subjects,
            layers=layers,
            module_template=self.config.mlp_module_tmp,
            track="out",
            token_strategy=token_strategy,
        )

    def extract_mlp_key(
        self,
        prompts: List[str],
        subjects: List[str],
        layers: List[int],
        token_strategy: str = "subject_last",
    ) -> Dict[int, torch.Tensor]:
        """
        Extract MLP key representations (input to c_proj, R^{d_inner}).

        This is the same space as ROME's u vector.

        Returns:
            Dict mapping layer_idx -> Tensor of shape (n_prompts, d_inner).
        """
        return self._extract_at_module(
            prompts=prompts,
            subjects=subjects,
            layers=layers,
            module_template=self.config.mlp_proj_module_tmp,
            track="in",
            token_strategy=token_strategy,
        )

    def extract_mlp_io(
        self,
        prompts: List[str],
        subjects: List[str],
        layers: List[int],
        token_strategy: str = "subject_last",
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extract both MLP input (c_proj input, R^{d_inner}) and output (R^{d_model}).

        Returns:
            Dict mapping layer_idx -> (input_tensor, output_tensor).
        """
        return self._extract_at_module(
            prompts=prompts,
            subjects=subjects,
            layers=layers,
            module_template=self.config.mlp_proj_module_tmp,
            track="both",
            token_strategy=token_strategy,
        )

    def _extract_at_module(
        self,
        prompts: List[str],
        subjects: List[str],
        layers: List[int],
        module_template: str,
        track: str,
        token_strategy: str,
    ) -> Union[Dict[int, torch.Tensor], Dict[int, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Internal method: extract activations at a given module template.

        Delegates to rome.repr_tools for subject-token-aware extraction,
        ensuring consistency with ROME's own representation gathering.
        """
        assert len(prompts) == len(subjects)
        both = track == "both"
        bs = self.config.extraction_batch_size

        # Build context strings: substitute subject into prompt
        contexts = [p.format(s) for p, s in zip(prompts, subjects)]

        results_in = {l: [] for l in layers}
        results_out = {l: [] for l in layers}

        for batch_start in range(0, len(contexts), bs):
            batch_end = min(batch_start + bs, len(contexts))
            batch_prompts = prompts[batch_start:batch_end]
            batch_subjects = subjects[batch_start:batch_end]

            for layer in layers:
                if token_strategy == "subject_last" or token_strategy == "subject_first":
                    subtoken = (
                        "last" if token_strategy == "subject_last" else "first"
                    )
                    result = repr_tools.get_reprs_at_word_tokens(
                        model=self.model,
                        tok=self.tok,
                        context_templates=batch_prompts,
                        words=batch_subjects,
                        layer=layer,
                        module_template=module_template,
                        subtoken=subtoken,
                        track=track,
                    )
                elif token_strategy == "last":
                    batch_contexts = [
                        p.format(s)
                        for p, s in zip(batch_prompts, batch_subjects)
                    ]
                    result = repr_tools.get_reprs_at_idxs(
                        model=self.model,
                        tok=self.tok,
                        contexts=batch_contexts,
                        idxs=[[-1]] * len(batch_contexts),
                        layer=layer,
                        module_template=module_template,
                        track=track,
                    )
                else:
                    raise ValueError(
                        f"Unknown token_strategy: {token_strategy}"
                    )

                if both:
                    inp, out = result
                    results_in[layer].append(inp.cpu())
                    results_out[layer].append(out.cpu())
                else:
                    results_out[layer].append(result.cpu())

        if both:
            return {
                l: (torch.cat(results_in[l], dim=0), torch.cat(results_out[l], dim=0))
                for l in layers
            }
        else:
            return {l: torch.cat(results_out[l], dim=0) for l in layers}


def extract_layer_activations(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    layers: List[int],
    layer_module_tmp: str = "transformer.h.{}",
    token_idx: int = -1,
    batch_size: int = 32,
) -> Dict[int, torch.Tensor]:
    """
    Lightweight standalone extraction: residual stream activations at a fixed
    token index. Useful for quick analysis without the full ActivationExtractor.

    Args:
        token_idx: -1 for last non-padding token.

    Returns:
        Dict mapping layer_idx -> Tensor of shape (n_prompts, d_model).
    """
    device = next(model.parameters()).device
    layer_names = [layer_module_tmp.format(l) for l in layers]
    results = {l: [] for l in layers}

    for batch_start in range(0, len(prompts), batch_size):
        batch = prompts[batch_start : batch_start + batch_size]
        inputs = tok(batch, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            with nethook.TraceDict(
                module=model,
                layers=layer_names,
                retain_input=False,
                retain_output=True,
            ) as td:
                model(**inputs)

        for layer, lname in zip(layers, layer_names):
            act = td[lname].output
            act = act[0] if isinstance(act, tuple) else act

            if token_idx == -1:
                seq_lens = inputs["attention_mask"].sum(dim=1)
                batch_acts = torch.stack(
                    [act[i, seq_lens[i] - 1] for i in range(act.size(0))]
                )
            else:
                batch_acts = act[:, token_idx]

            results[layer].append(batch_acts.detach().cpu())

    return {l: torch.cat(v, dim=0) for l, v in results.items()}