"""ModalTrainer — real LFM2/any-LLM + LoRA training on a Modal cloud GPU.

Same `Trainer` interface as LocalTrainer, so the loop/ratchet/agents/UI are
unchanged — swap it in via `app/main.py::_trainer()`.

How state works (see app/train/modal_app.py): each checkpoint is one Modal call.
Warm-start + ratchet are carried by passing the LoRA adapter as base64 bytes
between calls:
  - train()            -> remote call returns the new adapter bytes; we stash them
  - snapshot_adapter() -> returns the stashed bytes (for the ratchet)
  - restore_adapter()  -> sets which bytes to send on the next call (rollback)

GPU is chosen by the router and applied per call via `.with_options(gpu=...)`.

NOTE: not yet validated end-to-end on Modal — run `modal run` / drive it once to
confirm, then read any traceback.
"""
from __future__ import annotations

import sys

from app.train.base import LoRAConfig, TrainResult


def _log(msg: str):
    print(f"[ModalTrainer] {msg}", flush=True, file=sys.stdout)


class ModalTrainer:
    name = "modal"

    def __init__(self, gpu: str = "T4"):
        self.gpu = gpu
        self._adapter_b64: str | None = None   # current adapter (for warm-start)
        self._test: dict | None = None         # held-out test from last call

    def set_gpu(self, gpu: str) -> None:
        if gpu:
            self.gpu = gpu

    def train(self, *, base_model: str, dataset: dict, config: LoRAConfig,
              extra_data: dict[str, int] | None = None,
              prev: TrainResult | None = None,
              budget: dict | None = None,
              out_dir: str | None = None,
              on_progress=None) -> TrainResult:
        emit = on_progress or (lambda _: None)
        from app.train.modal_app import train_checkpoint

        payload = {
            "base_model": base_model,
            "dataset": dataset,
            "config": config.to_dict(),
            "budget": budget or {},
            "extra_data": extra_data or {},
            "adapter_b64": self._adapter_b64,
        }
        emit({"phase": "loading", "msg": f"launching {base_model} on Modal {self.gpu}"})
        _log(f"training {base_model} on Modal GPU {self.gpu} …")

        # Per-call GPU override from the router.
        out = train_checkpoint.with_options(gpu=self.gpu).remote(payload)

        self._adapter_b64 = out.get("adapter_b64")
        self._test = out.get("test")
        r = out["result"]
        emit({"phase": "evaluating", "msg": "checkpoint complete"})
        _log(f"checkpoint done · overall {r['overall_accuracy']:.2%}")
        return TrainResult(
            checkpoint_path=r["checkpoint_path"],
            overall_accuracy=r["overall_accuracy"],
            per_label=r["per_label"],
            steps_trained=r["steps_trained"],
            notes=r.get("notes", ""),
        )

    # ---- ratchet support (adapter bytes carried between Modal calls) ----
    def snapshot_adapter(self):
        return self._adapter_b64

    def restore_adapter(self, snap) -> None:
        self._adapter_b64 = snap

    def evaluate_test(self) -> dict | None:
        return self._test
