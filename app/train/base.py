"""Trainer abstraction — the swappable training backend.

The whole point: app/core/loop.py talks to a Trainer through this interface and
never knows whether training is mocked or running real LFM2+LoRA on Modal.

Today  -> MockTrainer   (no accounts, runs instantly, realistic metrics)
Later  -> ModalTrainer  (real LFM2-350M + PEFT LoRA on a Modal GPU)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class LoRAConfig:
    """Hyperparameters the agent decides on."""
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    learning_rate: float = 2e-4
    steps: int = 200
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    def to_dict(self) -> dict:
        return {
            "rank": self.rank, "alpha": self.alpha, "dropout": self.dropout,
            "learning_rate": self.learning_rate, "steps": self.steps,
            "target_modules": self.target_modules,
        }


@dataclass
class TrainResult:
    """What a Trainer returns after one checkpoint.

    per_label: subcategory -> accuracy (0..1). This is what the decision agent
               diagnoses to find weak subcats.
    """
    checkpoint_path: str
    overall_accuracy: float
    per_label: dict[str, float]
    steps_trained: int
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "checkpoint_path": self.checkpoint_path,
            "overall_accuracy": self.overall_accuracy,
            "per_label": self.per_label,
            "steps_trained": self.steps_trained,
            "notes": self.notes,
        }

    def weakest(self, threshold: float = 0.6) -> list[tuple[str, float]]:
        """Subcategories below threshold, worst first."""
        weak = [(k, v) for k, v in self.per_label.items() if v < threshold]
        return sorted(weak, key=lambda kv: kv[1])


class Trainer(Protocol):
    """A training backend. Implementations: MockTrainer, ModalTrainer."""

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
        """Train one checkpoint and return per-label eval metrics.

        dataset:    spec dict {id, config?, text_fields?, label_field?} — what to
                    train on (see app.train.datasets.DatasetSpec).
        extra_data: subcategory -> number of newly-added examples (the decision
                    agent's "find more data" action feeds this in).
        prev:       the previous checkpoint's result, so a backend can resume /
                    show improvement.
        budget:     optional {examples_per_class, eval_per_class} chosen by the
                    research agent to scale the run to the task.
        out_dir:    optional directory to save the trained adapter into.
        on_progress: optional callback(dict) for live progress — phases and
                    step/loss — e.g. {"phase":"train","step":40,"total":150,"loss":0.7}.
        """
        ...
