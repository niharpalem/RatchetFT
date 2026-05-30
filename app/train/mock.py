"""MockTrainer — simulates LFM2 + LoRA training without any accounts.

It produces *realistic* per-label metrics so the entire AutoFinetune loop —
research -> train -> eval -> diagnose -> add data / tune -> retrain — runs
end-to-end today. Swap in ModalTrainer (same interface) once Modal is set up.

The simulation models three intuitions:
  1. Some subcategories are intrinsically harder (sarcasm, irony, ambiguous,
     mixed, neutral) and start with low accuracy.
  2. Adding targeted data to a weak subcategory raises it a lot (saturating).
  3. Raising LoRA capacity (rank/steps) lifts everything a little, with
     diminishing returns near the per-label ceiling.

Deterministic per (dataset, label) so demos are reproducible — no RNG seeding
games, just smooth math.
"""
from __future__ import annotations

import hashlib

from app.train.base import LoRAConfig, TrainResult

# Substrings that mark a subcategory as "hard" — these are the ones the agent
# will discover are failing and go find more data for.
_HARD_HINTS = ("sarcas", "iron", "ambig", "mixed", "neutral", "subtle", "nuance")


def _stable_unit(*parts: str) -> float:
    """Deterministic float in [0,1) from string parts (a stable pseudo-random)."""
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def _baseline_accuracy(dataset_id: str, label: str) -> float:
    """First-checkpoint accuracy for a label before any tuning."""
    low = label.lower()
    hard = any(h in low for h in _HARD_HINTS)
    jitter = _stable_unit(dataset_id, label) * 0.10  # +0..0.10 spread
    if hard:
        return round(0.38 + jitter, 3)          # hard subcats start ~0.38–0.48
    return round(0.70 + jitter, 3)              # easy subcats start ~0.70–0.80


def _ceiling(label: str) -> float:
    """Max achievable accuracy for a label (hard caps a bit lower)."""
    low = label.lower()
    return 0.88 if any(h in low for h in _HARD_HINTS) else 0.97


def _capacity_gain(config: LoRAConfig) -> float:
    """Small global lift from more LoRA capacity, saturating."""
    # rank 8 / 200 steps ~ baseline. Higher rank+steps -> modest extra headroom.
    rank_term = min(config.rank / 32.0, 1.0) * 0.06
    step_term = min(config.steps / 600.0, 1.0) * 0.06
    return rank_term + step_term


def _data_gain(n_examples: int, ceiling: float, current: float) -> float:
    """Improvement from adding n targeted examples to a subcategory.

    Saturating: closes a fraction of the gap to the ceiling. 500 examples
    closes ~70% of the remaining gap; smaller amounts proportionally less.
    """
    if n_examples <= 0:
        return 0.0
    frac = 1.0 - 0.5 ** (n_examples / 350.0)   # 350 ex ~ 50% of gap, 700 ~ 75%
    return (ceiling - current) * frac


class MockTrainer:
    """Stand-in training backend. See module docstring."""

    name = "mock"

    def train(
        self,
        *,
        base_model: str,
        dataset: dict,
        config: LoRAConfig,
        extra_data: dict[str, int] | None = None,
        prev: TrainResult | None = None,
        budget: dict | None = None,
        out_dir: str | None = None,
        on_progress=None,
    ) -> TrainResult:
        extra_data = extra_data or {}
        dataset_id = dataset.get("id", "sentiment")

        if prev is None:
            # First checkpoint: baseline per-label accuracy + a small capacity lift.
            labels = _infer_labels(dataset_id)
            cap = _capacity_gain(config)
            per_label = {}
            for lab in labels:
                base = _baseline_accuracy(dataset_id, lab)
                ceil = _ceiling(lab)
                per_label[lab] = round(min(base + cap, ceil), 3)
            notes = f"Baseline checkpoint on {dataset_id} (rank={config.rank}, steps={config.steps})."
        else:
            # Subsequent checkpoint: start from prev, apply capacity + targeted data.
            cap = _capacity_gain(config) - _capacity_gain(LoRAConfig())  # delta vs default
            per_label = {}
            for lab, cur in prev.per_label.items():
                ceil = _ceiling(lab)
                lifted = cur + max(cap, 0.0) * 0.5            # global tune lift
                lifted += _data_gain(extra_data.get(lab, 0), ceil, lifted)  # targeted data
                per_label[lab] = round(min(lifted, ceil), 3)
            added = ", ".join(f"{k}+{v}" for k, v in extra_data.items()) or "no new data"
            notes = f"Retrain with {added} (rank={config.rank}, steps={config.steps})."

        overall = round(sum(per_label.values()) / len(per_label), 3)
        ckpt = f"mock://{dataset_id}/ckpt-rank{config.rank}-steps{config.steps}"
        return TrainResult(
            checkpoint_path=ckpt,
            overall_accuracy=overall,
            per_label=per_label,
            steps_trained=config.steps,
            notes=notes,
        )


# A few canned label sets so the mock has something to evaluate against before
# real HF dataset reading exists. The research agent picks the dataset_id; if it
# matches a known family we use richer labels, else a generic sentiment set.
_LABEL_SETS = {
    "sentiment": ["positive", "negative", "neutral", "sarcasm", "mixed"],
    "emotion": ["joy", "sadness", "anger", "fear", "surprise", "irony"],
    "nli": ["entailment", "contradiction", "neutral", "ambiguous"],
    "toxicity": ["clean", "toxic", "subtle_toxic", "sarcasm"],
}


def _infer_labels(dataset_id: str) -> list[str]:
    low = dataset_id.lower()
    for key, labels in _LABEL_SETS.items():
        if key in low:
            return labels
    if any(w in low for w in ("sst", "imdb", "tweet", "polar")):
        return _LABEL_SETS["sentiment"]
    return _LABEL_SETS["sentiment"]
