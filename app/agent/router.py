"""Model + compute router — Claude picks the base model AND the GPU.

Given a task, the research agent chooses a small (<=2B) open model from this
registry, and a GPU type sized to it. A deterministic VRAM heuristic guards the
GPU pick so a bad choice can't OOM or overspend. Keeping models <=2B keeps cloud
compute cheap and fast.

Why a registry (not "any id"): small open *causal-LM* models that are
non-gated and load cleanly. The loader auto-detects LoRA target layers, so the
trainer itself is model-agnostic — the registry just keeps the agent's picks
reliable. (VLMs need a separate trainer/data pipeline; not listed here yet.)
"""
from __future__ import annotations

import re
from dataclasses import dataclass

MAX_PARAMS_B = 2.0   # hard cap for compute safety


@dataclass
class ModelSpec:
    id: str
    params_b: float
    kind: str = "llm"          # "llm" today; "vlm" once a VLM trainer exists
    note: str = ""


# Small, non-gated, reliably-hosted causal LMs (<=2B).
REGISTRY: dict[str, ModelSpec] = {
    "LiquidAI/LFM2-350M":            ModelSpec("LiquidAI/LFM2-350M", 0.35, note="fast default"),
    "LiquidAI/LFM2-700M":            ModelSpec("LiquidAI/LFM2-700M", 0.70),
    "LiquidAI/LFM2-1.2B":            ModelSpec("LiquidAI/LFM2-1.2B", 1.2),
    "Qwen/Qwen2.5-0.5B-Instruct":    ModelSpec("Qwen/Qwen2.5-0.5B-Instruct", 0.5),
    "Qwen/Qwen2.5-1.5B-Instruct":    ModelSpec("Qwen/Qwen2.5-1.5B-Instruct", 1.5),
    "HuggingFaceTB/SmolLM2-360M-Instruct": ModelSpec("HuggingFaceTB/SmolLM2-360M-Instruct", 0.36),
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": ModelSpec("HuggingFaceTB/SmolLM2-1.7B-Instruct", 1.7),
}

_DEFAULT = "LiquidAI/LFM2-350M"


def recommend_gpu(params_b: float) -> str:
    """Cheapest GPU that comfortably fits a LoRA finetune of this model size."""
    if params_b <= 0.6:
        return "T4"          # 16GB — plenty for <=600M
    if params_b <= 1.5:
        return "L4"          # 24GB
    if params_b <= 3:
        return "A10G"        # 24GB
    if params_b <= 8:
        return "A100-80GB"
    return "H100"


_VALID_GPUS = {"T4", "L4", "A10G", "L40S", "A100", "A100-80GB", "H100", "H200", "B200"}


def list_for_prompt() -> str:
    return "\n".join(
        f'  - "{s.id}"  ({s.params_b}B, {s.kind})' + (f" — {s.note}" if s.note else "")
        for s in REGISTRY.values()
    )


def params_from_id(model_id: str) -> float | None:
    """Best-effort parse of param count from a model id, e.g.
    'Qwen2.5-1.5B-Instruct' -> 1.5, 'SmolLM2-360M' -> 0.36. None if unknown."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*([bm])\b", model_id.lower())
    if not m:
        return None
    val = float(m.group(1))
    return val if m.group(2) == "b" else round(val / 1000, 3)   # M -> B


def resolve(model_choice: str, gpu_choice: str | None) -> tuple[ModelSpec, str]:
    """Map the agent's model + gpu picks to a valid (ModelSpec, gpu).

    Accepts:
      - a registry id (vetted, guaranteed), OR
      - ANY model id whose size parses to <=2B (the "discover any small model"
        mode — grounded by the agent's own knowledge / HF search).
    Falls back to the default when the pick is unknown-size or too big. The GPU
    is always re-derived from the model size as a guard; the agent's GPU is only
    honored if it's a valid type.
    """
    spec = REGISTRY.get(model_choice)
    if spec is None and model_choice:
        pb = params_from_id(model_choice)
        if pb is not None and pb <= MAX_PARAMS_B:
            spec = ModelSpec(model_choice, pb, note="discovered")
    if spec is None or spec.params_b > MAX_PARAMS_B:
        spec = REGISTRY[_DEFAULT]
    recommended = recommend_gpu(spec.params_b)
    gpu = gpu_choice if (gpu_choice in _VALID_GPUS) else recommended
    return spec, gpu
