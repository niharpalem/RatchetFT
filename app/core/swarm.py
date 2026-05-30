"""Swarm orchestrator — a model bake-off.

Run N models through the full RatchetFT loop ON THE SAME dataset, in parallel,
then rank them by held-out test accuracy and keep the winner.

Shape:
    research ONCE  -> shared dataset + config + budget
    pick N contender models  (user-provided, or Claude proposes)
    for each model: run_loop(plan = shared plan + this model) in its own thread
    rank by held-out test  -> winner

Concurrency: with the Modal backend each contender trains on its own cloud GPU
(true parallel). With the local backend they'd contend on one GPU, so we run them
sequentially (concurrency=1). The orchestrator is a generator yielding events
tagged with `slot` (which contender), so the SSE/UI can fan them into columns.
"""
from __future__ import annotations

import json
import os
import queue
import re
import threading
import time

from app.agent import research as research_agent
from app.agent import router
from app.agent.claude_cli import ask_json
from app.core.experiment import DATA_DIR, Experiment
from app.core.loop import run_loop
from app.train.base import LoRAConfig

SWARM_DIR = DATA_DIR / "_swarms"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "swarm"


# ---- swarm manifest (on disk) -------------------------------------------
def create_swarm(task: str, models: list[str] | None = None, n: int = 3,
                 mode: str = "models") -> dict:
    SWARM_DIR.mkdir(parents=True, exist_ok=True)
    sid = f"{_slug(task)}-{int(time.time())}"
    manifest = {
        "swarm_id": sid, "task": task.strip(),
        "models": [m.strip() for m in (models or []) if m.strip()],
        "n": int(n), "mode": mode,          # "models" = N models; "configs" = 1 model, N configs
        "status": "created",
        "dataset_id": None, "contenders": [], "winner": None,
        "created_ts": time.time(),
    }
    _save(manifest)
    return manifest


def _save(m: dict) -> None:
    (SWARM_DIR / f"{m['swarm_id']}.json").write_text(json.dumps(m, indent=2))


def load_swarm(sid: str) -> dict:
    return json.loads((SWARM_DIR / f"{sid}.json").read_text())


def list_swarms() -> list[dict]:
    if not SWARM_DIR.exists():
        return []
    out = []
    for f in sorted(SWARM_DIR.glob("*.json"), reverse=True):
        try:
            out.append(json.loads(f.read_text()))
        except Exception:
            pass
    return out


# ---- contender selection ------------------------------------------------
_CONTENDER_SYSTEM = """You pick a slate of models for a finetuning bake-off. Given a
task and dataset, choose N DIVERSE open causal-LM models (each <=2B params,
non-gated) worth comparing — vary sizes/families so the bake-off is informative.
Return JSON: { "models": [hf_model_id, ...] } with exactly N ids (each id should
contain its size, e.g. "1.5B", "360M")."""


def pick_contenders(task: str, dataset_id: str, n: int, *, model: str = "sonnet") -> list[tuple[str, float, str]]:
    """Return [(model_id, params_b, gpu)] of N contenders, router-validated."""
    try:
        raw = ask_json(_CONTENDER_SYSTEM,
                       f"Task: {task}\nDataset: {dataset_id}\nPick {n} models.",
                       model=model)
        ids = raw.get("models") or []
    except Exception:
        ids = []
    out: list[tuple[str, float, str]] = []
    seen = set()
    for mid in ids:
        spec, gpu = router.resolve(mid, None)
        if spec.id not in seen:
            seen.add(spec.id)
            out.append((spec.id, spec.params_b, gpu))
    # Pad from the registry if the agent gave too few / failed.
    for rid, spec in router.REGISTRY.items():
        if len(out) >= n:
            break
        if spec.id not in seen:
            seen.add(spec.id)
            out.append((spec.id, spec.params_b, router.recommend_gpu(spec.params_b)))
    return out[:n]


_CONFIG_SWEEP_SYSTEM = """You design a set of LoRA configs for finetuning the SAME
model, to find which setup works best. Make the N configs CONTRAST STRONGLY so the
comparison is informative — vary the knobs widely, e.g.:
  - lora_rank: small (4-8) vs medium (16) vs large (32)
  - target_modules: minimal ["q_proj","v_proj"] vs broad attention
    ["q_proj","k_proj","v_proj","out_proj"] vs + MLP
  - optionally steps / learning_rate
Each config should be a genuinely different bet (low-capacity-fast vs high-capacity).
Return JSON: { "configs": [
  { "label": str (short, e.g. "rank4-minimal"), "lora_rank": int, "lora_alpha": int,
    "dropout": number, "learning_rate": number, "steps": int, "target_modules": [str] },
  ... exactly N ] }"""


def design_configs(task: str, model_id: str, n: int, base: LoRAConfig, *, model: str = "sonnet") -> list[dict]:
    """Return N strongly-contrasting LoRA configs for one model. Each: (LoRAConfig, label)."""
    try:
        raw = ask_json(_CONFIG_SWEEP_SYSTEM,
                       f"Task: {task}\nModel: {model_id}\nDesign {n} contrasting configs.",
                       model=model)
        items = raw.get("configs") or []
    except Exception:
        items = []
    out = []
    for i, c in enumerate(items[:n]):
        rank = int(c.get("lora_rank", 8))
        out.append((LoRAConfig(
            rank=rank,
            alpha=int(c.get("lora_alpha", rank * 2)),
            dropout=float(c.get("dropout", 0.05)),
            learning_rate=float(c.get("learning_rate", 2e-4)),
            steps=int(c.get("steps", base.steps)),
            target_modules=list(c.get("target_modules") or ["q_proj", "v_proj"]),
        ), c.get("label", f"config-{i}")))
    # Fallback: hand-built contrasting trio if the agent failed.
    while len(out) < n:
        presets = [
            (4, ["q_proj", "v_proj"], "rank4-minimal"),
            (16, ["q_proj", "k_proj", "v_proj", "out_proj"], "rank16-attn"),
            (32, ["q_proj", "k_proj", "v_proj", "out_proj"], "rank32-attn"),
        ]
        rk, tm, lab = presets[len(out) % 3]
        out.append((LoRAConfig(rank=rk, alpha=rk * 2, steps=base.steps, target_modules=tm), lab))
    return out[:n]


# ---- the bake-off generator --------------------------------------------
def run_swarm(swarm_id: str, trainer_factory, *, max_iters: int = 4, model: str = "sonnet"):
    """Generator yielding tagged events for the whole bake-off.

    Events: swarm_start, then per-contender loop events tagged with `slot`
    (research/checkpoint/decision/progress/done/error), then swarm_done.
    """
    sw = load_swarm(swarm_id)
    task = sw["task"]
    parallel = os.environ.get("RATCHETFT_BACKEND", "local").lower() == "modal"

    # 1. Shared research (one dataset/config/budget for a fair comparison).
    base = research_agent.research(task, model=model)
    dataset_id = base["dataset_id"]
    sw["dataset_id"] = dataset_id

    # 2. Build contenders — each carries its own model + config + label.
    base_cfg: LoRAConfig = base["config"]
    mode = sw.get("mode", "models")
    n = sw.get("n", 3)
    contenders: list[dict] = []

    if mode == "configs":
        # ONE model, N strongly-contrasting LoRA configs (Claude-designed).
        if sw.get("models"):
            spec, gpu = router.resolve(sw["models"][0], None)
            model_id, pb = spec.id, spec.params_b
        else:
            model_id, pb, gpu = base["base_model"], base.get("model_params_b"), base["gpu"]
        for i, (cfg, label) in enumerate(design_configs(task, model_id, n, base_cfg, model=model)):
            contenders.append({"slot": i, "model": model_id, "params_b": pb,
                               "gpu": gpu, "config": cfg, "label": label})
    else:
        # N different models, same (shared) config.
        if sw.get("models"):
            picks, seen = [], set()
            for mid in sw["models"]:
                spec, g = router.resolve(mid, None)
                if spec.id not in seen:
                    seen.add(spec.id)
                    picks.append((spec.id, spec.params_b, g))
        else:
            picks = pick_contenders(task, dataset_id, n, model=model)
        for i, (m, pb, g) in enumerate(picks):
            contenders.append({"slot": i, "model": m, "params_b": pb, "gpu": g,
                               "config": LoRAConfig(**base_cfg.to_dict()), "label": m.split("/")[-1]})

    sw["contenders"] = [{"slot": c["slot"], "model": c["model"], "params_b": c["params_b"],
                         "gpu": c["gpu"], "label": c["label"], "exp_id": None,
                         "status": "pending", "best_val": None, "best_test": None}
                        for c in contenders]
    sw["status"] = "running"
    _save(sw)

    yield {"event": "swarm_start", "dataset_id": dataset_id, "mode": mode,
           "reason": base.get("reason", ""), "contenders": sw["contenders"]}

    # 3. Run each contender's full loop, tagged by slot.
    q: queue.Queue = queue.Queue()
    sem = threading.Semaphore(len(contenders) if parallel else 1)

    def worker(c: dict):
        slot, model_id = c["slot"], c["model"]
        with sem:
            child = Experiment.create(f"{task} · {c['label']}")
            child.update_meta(swarm_id=swarm_id, slot=slot)
            eid = child.exp_id
            plan = {**base, "base_model": model_id, "gpu": c["gpu"], "config": c["config"]}

            def prog(d, slot=slot):
                q.put({"event": "progress", "slot": slot, **d})
            try:
                for ev in run_loop(child, trainer_factory(), max_iters=max_iters,
                                   model=model, on_progress=prog, plan=plan):
                    q.put({**ev, "slot": slot, "model": model_id, "label": c["label"], "exp_id": eid})
            except Exception as e:
                q.put({"event": "error", "slot": slot, "model": model_id, "exp_id": eid,
                       "message": f"{type(e).__name__}: {e}"})
            finally:
                q.put({"event": "slot_done", "slot": slot})

    threads = [threading.Thread(target=worker, args=(c,), daemon=True) for c in contenders]
    for t in threads:
        t.start()

    results: dict[int, dict] = {}
    done = 0
    while done < len(threads):
        ev = q.get()
        kind = ev.get("event")
        if kind == "slot_done":
            done += 1
            continue
        # Record the child experiment id as soon as we see it (for winner→chat link).
        if ev.get("exp_id"):
            for c in sw["contenders"]:
                if c["slot"] == ev["slot"] and not c.get("exp_id"):
                    c["exp_id"] = ev["exp_id"]
                    _save(sw)
        if kind == "done":
            results[ev["slot"]] = {
                "slot": ev["slot"], "model": ev.get("model"), "label": ev.get("label"),
                "exp_id": ev.get("exp_id"),
                "val": ev.get("final_overall"), "test": ev.get("test_overall"),
            }
            # persist into manifest
            for c in sw["contenders"]:
                if c["slot"] == ev["slot"]:
                    c["status"] = "done"
                    c["best_val"] = ev.get("final_overall")
                    c["best_test"] = ev.get("test_overall")
            _save(sw)
        yield ev

    # 4. Rank by held-out test (fallback to val), keep the winner.
    def score(r):
        return r["test"] if r["test"] is not None else (r["val"] or 0)
    ranking = sorted(results.values(), key=score, reverse=True)
    sw["winner"] = ranking[0] if ranking else None
    sw["status"] = "done"
    _save(sw)

    yield {"event": "swarm_done", "ranking": ranking, "winner": sw["winner"]}
