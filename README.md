# RatchetFT

*Ratchet Finetuning — a loop that only clicks forward.*

Give it one sentence — *"classify the emotion in a sentence"* — and an agent finds a
HuggingFace dataset for the task, picks an initial LoRA config, trains an LFM2
checkpoint, reads **per-label** metrics, diagnoses the weak classes, decides whether
to add targeted data or tune hyperparameters, and loops. No human in the middle.

It's a **generic** tool: the agent can choose *any* single-label text-classification
dataset on the Hub — there's no whitelist. The loader auto-detects each dataset's
text and label columns. The Python knows nothing about *what* you're classifying;
the agent's dataset choice carries the domain.

```
User input
   ↓
Research Agent  → picks HF dataset + LoRA config + training budget + LoRA targets
   ↓
┌── Train checkpoint (real LFM2 LoRA, live step/loss) ──→ per-label eval
│         ↓
│   Ratchet: new best? keep it.  Regressed? roll back to best.
│         ↓
│   Decision Agent diagnoses weak classes
│         ↓
│   add more data  OR  tune LoRA params  OR  stop (plateau / good enough)
│         ↓
└── retrain ──→ agent says stop (or max-loops fuse) ──┘
                      ↓
                 best adapter saved to disk
```

## What's real

| Piece | Implementation |
|---|---|
| **Agent brain** | local `claude` CLI (subprocess, no API key) |
| **Training** | **`LocalTrainer` — real LFM2-350M + LoRA on your Apple Silicon GPU (MPS)** |
| **Dataset** | any real HuggingFace classification dataset, chosen by the agent; columns auto-detected |
| **Eval** | real per-label accuracy via verbalizer scoring |
| **Ratchet** | keeps the best checkpoint; rolls the adapter back on a regression |
| **Checkpoints** | each LoRA adapter saved to `data/<exp>/checkpoints/ckpt-N/` |
| **Live progress** | training phase + step/loss streamed to the browser (SSE) and terminal |
| **Stopping** | the agent decides (plateau / good enough); `max_iters` is just a safety fuse |
| **Tracing** | `data/<exp>/decisions.jsonl` (append-only; Raindrop later) |

The agent owns the strategy — dataset, LoRA config, training budget, LoRA target
layers, every per-round move, and when to stop. The only fixed things are the base
model (LFM2-350M) and that the task is text classification.

`MockTrainer` (instant fake metrics) and `ModalTrainer` (cloud GPU stub) are still
in `app/train/` — swap which one `app/main.py::_trainer()` returns. The loop, agents,
and UI are identical regardless of backend.

## Run it

```bash
source .venv/bin/activate
.venv/bin/uvicorn app.main:app --reload --port 8000
# open http://localhost:8000 → type a task → set max loops → "Run the loop"
```

> **Use the explicit `.venv/bin/uvicorn` path.** A bare `uvicorn` can resolve to a
> different environment on your PATH (e.g. an old `awrk`/conda env) and import the
> wrong, broken package stack — `.venv/bin/uvicorn` forces this project's clean
> Python 3.12 + torch/transformers/peft. Quick sanity check:
> `.venv/bin/python -c "import sys,transformers; print(sys.prefix, transformers.__version__)"`
> should show `.../hck_1/.venv` and `transformers 5.9.x`.

See **`DEMO.md`** for a 2-minute walkthrough script. Each checkpoint does **real
training** (~1-2 min on an M-series Mac) — the browser shows a live training bar and
the per-label trend chart; the uvicorn terminal shows step/loss.

### Datasets
The agent picks the dataset and it's **downloaded on demand** the first time it's
used (then cached). Downloads work from your own terminal. No manual pre-fetch step —
Claude figures out what to load. (Force cache-only/offline with `HF_HUB_OFFLINE=1`.)

### Knobs
- **max loops** — set in the UI (safety fuse; the agent usually stops earlier on plateau)
- `LocalTrainer(base_per_label, eval_per_label, batch_size)` in `app/main.py` — fallback
  budget defaults; the research agent overrides examples/eval per class per run.

> Real training on a 350M model is noisy: aggressively oversampling weak classes can
> cause **catastrophic forgetting**. That's expected ML behavior — the ratchet rolls
> back regressions and the decision agent reacts by switching to `tune`.

## Layout

```
app/
├── main.py              FastAPI + HTMX UI; SSE for single runs AND the swarm
├── agent/
│   ├── claude_cli.py    the brain — `claude -p` subprocess → text / parsed JSON
│   ├── hf_search.py     live HF Hub search — datasets AND models (grounded picks)
│   ├── router.py        model+GPU router — ≤2B registry + "any model" guard + GPU sizing
│   ├── research.py      task → dataset + base model + GPU + LoRA config + budget
│   └── decide.py        per-label metrics + trajectory → add_data | tune | stop
├── core/
│   ├── experiment.py    run state on disk (data/<exp>/, iterations, decisions)
│   ├── loop.py          one model's loop + ratchet (keep-best / rollback); `plan` override
│   └── swarm.py         the bake-off — N models in parallel, ranked by held-out test
└── train/
    ├── base.py          Trainer protocol + LoRAConfig + TrainResult
    ├── datasets.py      generic loader — auto-detect fields; train/val/TEST splits
    ├── local.py         LocalTrainer — REAL any-LLM + LoRA on MPS  ← default
    ├── mock.py          MockTrainer — instant fake metrics
    ├── modal_app.py     Modal app — runs one checkpoint on a cloud GPU
    └── modal_trainer.py ModalTrainer — cloud-GPU backend (RATCHETFT_BACKEND=modal)
templates/  swarm.html (bake-off) · experiment.html (single run) · index.html · base.html
DEMO.md · test_brain.py · modal_validate.py
```

## Swarm: the model bake-off

Run **N models on the same task in parallel** and keep the winner — this is the
"swarm across a compute cluster" piece. Module: `app/core/swarm.py`.

```
ONE sentence
   ↓
Research ONCE  → shared dataset + config + budget  (fair comparison)
   ↓
Pick N contender models   (you list them, OR Claude proposes a diverse ≤2B slate)
   ↓                                router sizes a GPU per model
┌─ model A (T4)  → full ratchet loop → held-out test 78% ┐
├─ model B (L4)  → full ratchet loop → held-out test 84% ┤  (parallel on Modal;
└─ model C (L4)  → full ratchet loop → held-out test 81% ┘   sequential on local)
   ↓
Rank by HELD-OUT TEST  → 🏆 winner  (model B)
```

- Each contender runs the **exact same loop** (`run_loop`) via its `plan` override
  (shared dataset/config pinned, only the model differs), with its own ratchet and
  held-out test.
- **Concurrency:** `N` when `RATCHETFT_BACKEND=modal` (each model on its own cloud
  GPU — true parallel); `1` locally (one GPU can't train N at once).
- Events stream tagged with a `slot` so the UI fans them into one column per model.
- Manifests: `data/_swarms/<id>.json`. Routes: `POST /swarm`, `GET /swarm/{id}`,
  `GET /swarm/{id}/stream`. UI: `templates/swarm.html` (live columns + winner).

Run it (the bake-off truly parallelizes on Modal):
```bash
RATCHETFT_BACKEND=modal .venv/bin/uvicorn app.main:app --port 8000
# home → "Run bake-off" → N models train in parallel → ranked by held-out test
```

## Swap compute to the cloud later (Modal or HF Jobs)

### One-time Modal setup
This `.venv` was created with **uv**, which has **no `pip` inside it** — use `uv pip install`
(plain `pip` will hit the wrong environment):

```bash
uv pip install modal              # installs into the active .venv
.venv/bin/modal setup             # opens browser to authenticate → writes ~/.modal.toml
.venv/bin/python -c "import modal; print('modal', modal.__version__)"   # verify
```

> If you see `No module named modal` from `.venv/bin/python` while `pip` says "already
> satisfied", `pip` installed into a different env (e.g. `awrk`). `uv pip install` +
> the explicit `.venv/bin/...` paths avoid that.

### Then wire it
1. Implement `train_checkpoint()` in `app/train/modal_trainer.py` (sketch in the file),
   or write an `HFJobsTrainer` the same way.
2. In `app/main.py`, change `_trainer()` to return it.

Same `Trainer` interface — the loop, ratchet, agents, and UI are unchanged.
