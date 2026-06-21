"""Small model helpers shared by the thesis scripts.

This module only deals with model-name aliases, model construction, and output
dictionary access. The actual architectures stay in Models_senn.py.
"""

from typing import Any, Dict, Optional

import torch

import Models_senn as Model


MODEL_KIND_ALIASES = {
    "base": "base",
    "stgat": "base",
    "gat": "base",
    "eeg_gat_model": "base",
    "senn": "senn_rawx",
    "sennrawx": "senn_rawx",
    "senn_rawx": "senn_rawx",
    "senn_raw": "senn_rawx",
    "rawx": "senn_rawx",
    "sennfixed": "senn_fixed",
    "senn_fixed": "senn_fixed",
    "senn_fixedconcepts": "senn_fixed",
    "fixed": "senn_fixed",
    "fixed_senn": "senn_fixed",
    "senntrivialfixed": "senn_trivialfixed",
    "senn_trivialfixed": "senn_trivialfixed",
    "trivialfixed": "senn_trivialfixed",
    "sennfixed_concepttheta": "senn_fixedconcepttheta",
    "senn_fixed_concepttheta": "senn_fixedconcepttheta",
    "senn_fixedconcepttheta": "senn_fixedconcepttheta",
    "senn_fixedconcepts_concepttheta": "senn_fixedconcepttheta",
    "logisticconcepts": "logisticconcepts",
    "fixedlogisticconcepts": "logisticconcepts",
    "conceptlogisticdual": "logisticconcepts",
    "concept_logistic_dual": "logisticconcepts",
    "logisticconceptdual": "logisticconcepts",
}


def normalize_model_kind(model_kind: Any) -> str:
    """Return the canonical model-kind name used by the scripts."""
    key = str(model_kind or "base").strip().lower()
    if key in MODEL_KIND_ALIASES:
        return MODEL_KIND_ALIASES[key]
    raise ValueError(f"Unknown model kind: {model_kind!r}")


def model_kind_from_checkpoint(ckpt: Dict[str, Any], fallback: str = "base") -> str:
    """Read and normalise the model kind saved in a training checkpoint."""
    return normalize_model_kind(ckpt.get("model_type", fallback))


def build_model(
    model_kind: Any,
    ckpt: Optional[Dict[str, Any]] = None,
    return_explanations: bool = False,
    device: Optional[torch.device | str] = None,
    is_trivial: bool = False,
) -> torch.nn.Module:
    """Build one of the thesis models and optionally load a checkpoint.

    `is_trivial=True` is kept for old scripts that evaluated the trivial fixed
    concepts through the fixed-concept metric branch.
    """
    kind = normalize_model_kind(model_kind)
    if is_trivial and kind == "senn_fixed":
        kind = "senn_trivialfixed"

    if kind == "base":
        model = Model.EEG_GAT_Model()
    elif kind == "senn_rawx":
        model = Model.SENN_raw(
            global_min=(ckpt or {}).get("global_min", None),
            return_node_scores=False,
            return_fmap=return_explanations,
        )
    elif kind == "senn_fixed":
        model = Model.SENN_fixedconcepts(
            return_node_scores=False,
            return_edge_scores=False,
            return_fmap=return_explanations,
        )
    elif kind == "senn_trivialfixed":
        model = Model.SENN_trivialfixedconcepts(
            return_node_scores=False,
            return_edge_scores=False,
            return_fmap=return_explanations,
        )
    elif kind == "senn_fixedconcepttheta":
        model = Model.SENN_fixedconcepts_concepttheta(
            return_node_scores=False,
            return_edge_scores=False,
            return_fmap=return_explanations,
        )
    elif kind == "logisticconcepts":
        model = Model.ConceptLogisticDual(
            return_node_scores=False,
            return_edge_scores=False,
            return_fmap=return_explanations,
        )
    else:
        raise ValueError(f"Unsupported model kind: {model_kind!r}")

    if ckpt is not None:
        model.load_state_dict(ckpt["model_state_dict"])
    if device is not None:
        model = model.to(device)
    return model


def extract_model_output(output: Any, key: str = "logit") -> torch.Tensor:
    """Extract a tensor from model outputs that may be tensors or dictionaries."""
    if torch.is_tensor(output):
        return output
    if not isinstance(output, dict):
        raise TypeError(f"Expected tensor or dict output, got {type(output)!r}")

    aliases = {
        "prob": ["prob", "probability", "probs", "probabilities"],
        "logit": ["logit", "logits"],
        "explanation": ["explanation", "focus_map", "F_map"],
        "explanation_edge": ["explanation_edge", "F_edge"],
    }.get(key, [key])

    for name in aliases:
        if name in output:
            return output[name]

    raise KeyError(f"Could not find '{key}' in model output. Available keys: {sorted(output.keys())}")


def extract_prob(output: Any) -> torch.Tensor:
    """Extract probabilities, using sigmoid(logit) when only logits are present."""
    if isinstance(output, dict):
        if "prob" in output:
            return output["prob"]
        if "logit" in output:
            return torch.sigmoid(output["logit"])
    return extract_model_output(output, "prob")
