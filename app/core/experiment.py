"""Experiment state — everything for one AutoFinetune run, persisted to disk.

Layout (mirrors SAGE-X's data/<session>/ convention):

    data/<exp_id>/
        experiment.json     # task, dataset, config, status, summary
        iterations.json     # list of per-checkpoint records (metrics + decision)
        decisions.jsonl      # append-only agent decision log (Raindrop later)

Pure storage + small helpers. No agent calls, no training here.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:40] or "exp"


@dataclass
class Iteration:
    """One checkpoint: the config used, the metrics out, and the decision after."""
    index: int
    config: dict
    extra_data: dict          # subcat -> examples added before this checkpoint
    result: dict              # TrainResult.to_dict()
    accepted: bool = True     # ratchet: did overall accuracy click forward?
    best_overall: float = 0.0 # best overall accuracy as of this checkpoint
    decision: dict | None = None   # decision agent output (None until decided)
    ts: float = field(default_factory=time.time)


class Experiment:
    def __init__(self, exp_id: str):
        self.exp_id = exp_id
        self.dir = DATA_DIR / exp_id
        self.dir.mkdir(parents=True, exist_ok=True)

    # ---- construction --------------------------------------------------
    @classmethod
    def create(cls, task: str) -> "Experiment":
        exp_id = f"{_slug(task)}-{int(time.time())}"
        exp = cls(exp_id)
        exp._write_meta({
            "exp_id": exp_id,
            "task": task,
            "status": "created",          # created -> researched -> running -> done
            "base_model": "LiquidAI/LFM2-350M",
            "gpu": None,
            "model_params_b": None,
            "dataset_id": None,
            "dataset": None,
            "task_type": None,
            "labels": [],
            "config": None,
            "budget": None,
            "candidates": [],
            "validation": None,
            "best_test": None,
            "stop_reason": None,
            "created_ts": time.time(),
        })
        exp._write_iterations([])
        return exp

    @classmethod
    def load(cls, exp_id: str) -> "Experiment":
        exp = cls(exp_id)
        if not (exp.dir / "experiment.json").exists():
            raise FileNotFoundError(f"no experiment {exp_id}")
        return exp

    @staticmethod
    def list_all() -> list[dict]:
        if not DATA_DIR.exists():
            return []
        out = []
        for d in sorted(DATA_DIR.iterdir(), reverse=True):
            meta = d / "experiment.json"
            if meta.exists():
                try:
                    out.append(json.loads(meta.read_text()))
                except Exception:
                    pass
        return out

    @staticmethod
    def delete_all() -> int:
        """Remove every experiment directory under data/. Returns how many."""
        import shutil
        if not DATA_DIR.exists():
            return 0
        n = 0
        for d in DATA_DIR.iterdir():
            if d.is_dir():
                shutil.rmtree(d, ignore_errors=True)
                n += 1
        return n

    # ---- meta ----------------------------------------------------------
    @property
    def meta(self) -> dict:
        return json.loads((self.dir / "experiment.json").read_text())

    def _write_meta(self, meta: dict) -> None:
        (self.dir / "experiment.json").write_text(json.dumps(meta, indent=2))

    def update_meta(self, **kw) -> dict:
        meta = self.meta
        meta.update(kw)
        self._write_meta(meta)
        return meta

    # ---- iterations ----------------------------------------------------
    @property
    def iterations(self) -> list[dict]:
        return json.loads((self.dir / "iterations.json").read_text())

    def _write_iterations(self, items: list[dict]) -> None:
        (self.dir / "iterations.json").write_text(json.dumps(items, indent=2))

    def add_iteration(self, it: Iteration) -> None:
        items = self.iterations
        items.append(asdict(it))
        self._write_iterations(items)

    def set_decision(self, index: int, decision: dict) -> None:
        items = self.iterations
        for it in items:
            if it["index"] == index:
                it["decision"] = decision
                break
        self._write_iterations(items)

    @property
    def last_iteration(self) -> dict | None:
        items = self.iterations
        return items[-1] if items else None

    # ---- decision log (append-only, Raindrop-ready) --------------------
    def log_decision(self, kind: str, payload: dict) -> None:
        rec = {"ts": time.time(), "kind": kind, **payload}
        with (self.dir / "decisions.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")

    def decision_log(self) -> list[dict]:
        f = self.dir / "decisions.jsonl"
        if not f.exists():
            return []
        return [json.loads(line) for line in f.read_text().splitlines() if line.strip()]
