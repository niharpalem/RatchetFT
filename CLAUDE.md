# RatchetFT — context for Claude

> **One sentence in → a trained, tested classifier you can chat with. No human in the loop.**
> An autonomous fine-tuning *research* agent: it finds a HuggingFace dataset, validates it,
> picks a model + GPU, trains LoRA, reads **per-label** metrics, fixes the failing classes,
> **ratchets** accuracy forward, stops itself, reports an **honest held-out test score**, and
> serves the model for chat.

## How to run (READ THIS — environment gotchas)
```bash
cd "/path/to/hck_1"
source .venv/bin/activate
.venv/bin/uvicorn app.main:app --reload --port 8000   # ALWAYS use the explicit .venv/bin path
```
- The `.venv` is created with **uv → it has NO `pip`**. Install deps with **`uv pip install -r requirements.txt`** (plain `pip` / `.venv/bin/pip` will fail).
- A stray **`awrk`** conda/venv may shadow `.venv` on PATH and import a broken NumPy2/TensorFlow stack — **always launch via `.venv/bin/uvicorn`** (not bare `uvicorn`).
- **Brain = the local `claude` CLI** (subprocess in `app/agent/claude_cli.py`) — NO API key. Must be on PATH. Calling `claude` directly inside Claude Code fails (nested sandbox); spawning from Python works.
- **Datasets download on demand** the first time the agent uses one (then cached). Downloads work from a normal terminal; blocked inside the Claude Code sandbox (corp proxy 403). Force cache-only with `HF_HUB_OFFLINE=1`.
- Cached already: model `LiquidAI/LFM2-350M`, dataset `dair-ai/emotion`.

## What it does (the loop — the CORE)
`research → train → per-label eval → diagnose weak classes → add_data | tune | stop → ratchet → held-out test → chat`
- **Research** (`app/agent/research.py`): grounded HF dataset search → **peek a sample of real rows** → Claude validates fit (switches dataset if bad) → also picks base model + GPU via the **router** (`app/agent/router.py`, ≤2B guard, model→GPU sizing).
- **Train** (`app/train/local.py` = LocalTrainer, default): real LFM2/any-causal-LLM + LoRA on Apple MPS; instruction-style SFT; **OOM-safe auto-backoff** (halves batch + grad-accum); LoRA targets auto-detected per architecture.
- **Eval**: verbalizer scoring → per-label accuracy. Train/validation/**held-out test** splits are disjoint (no leakage). Agent steers on validation; final number = held-out test.
- **Ratchet** (`app/core/loop.py`): keep the best checkpoint; on a regression roll the adapter back; agent decides when to stop (plateau); `max_iters` is a safety fuse.
- **Chat/serve** (`app/serve.py`): loads base+best adapter warm in memory; classify any text. Route `POST /e/{id}/predict`.

## Secondary feature (the ENCORE — don't centre it)
**Swarm / bake-off** (`app/core/swarm.py`): run up to 3 in parallel and keep the winner by held-out test — either **1 model × 3 contrasting LoRA configs** or **2-3 models**. Winner links to its chat. Routes: `POST /swarm`, `GET /swarm/{id}`, `/swarm/{id}/stream`.

## Cloud (optional, built but UNTESTED end-to-end)
- `RATCHETFT_BACKEND=modal` switches training (`app/train/modal_trainer.py` + `modal_app.py`) and inference (`app/serve_modal.py`) to Modal cloud GPUs (warm container).
- Setup: `uv pip install modal`; `.venv/bin/modal setup`; `modal secret create huggingface HF_TOKEN=...`.
- `modal_validate.py` PASSED (LFM2+LoRA on an A10). The full Modal train/serve path is built but not yet run end-to-end.

## Status (honest)
- ✅ Verified locally end-to-end: single iterative loop, ratchet, held-out test, chat/inference, swarm orchestration (mock), all UI.
- ⚠️ Untested: Modal training + Modal serving + true-parallel swarm (need `RATCHETFT_BACKEND=modal` + a live run). **For the demo: local single loop only.**
- Known soft spots: no cross-run/long-term agent memory; Retrieval&Synthesis is "grounded search + peek-validate" (no multi-source synthesis/NER).

## Hackathon framing
Fits the brief: **#1 Agent Architectures & Control Loops** (the loop + ratchet + hybrid deterministic/LLM control + swarm), **#3 Applied Autonomous Research** (law/bio/code domain tasks), and partially **#2 Retrieval & Knowledge Synthesis** (grounded search + data peek/validate + citations).
**Pitch the iterative single-model loop as the core; swarm is the "and it scales" encore.**
**Legal RAG assistant (cite relevant clauses for a case) = the "what's next" pitch — NOT built.**

## Demo plan
1. Launch via `.venv/bin/uvicorn …`. 2. Type a hard task, set **max loops = 6**, Start the loop.
   - Law demo: *"Finetune a model to classify U.S. Supreme Court opinions by legal issue area."* (`lex_glue/scotus`)
   - Rock-solid fallbacks: *"label citation intent in research papers"* (`scicite`) or cached *"classify the emotion in a sentence"*.
3. Watch: activity feed → dataset peek table + train/val/test counts → trend chart climbing → held-out test → **chat with the model**.
- First checkpoint ~1-2 min (real GPU). Don't demo Modal/swarm live (untested).

## Layout
```
app/
├── main.py              FastAPI + HTMX UI; SSE; routes for runs, swarm, predict
├── agent/  claude_cli.py · hf_search.py (datasets+models) · router.py · research.py · decide.py
├── core/   experiment.py (disk state) · loop.py (loop+ratchet) · swarm.py (bake-off)
├── serve.py / serve_modal.py   inference (local / Modal)
└── train/  base.py · datasets.py (generic loader + peek_sample) · local.py · mock.py · modal_app.py · modal_trainer.py
templates/  base · index · experiment · swarm        static/app.css
DEMO.md · test_brain.py · modal_validate.py · index.html (standalone landing page)
```
Run a single command to sanity-check: `.venv/bin/python -c "from app.main import app; print('ok')"`.
