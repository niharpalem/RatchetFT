"""AutoFinetune Loop — FastAPI + HTMX UI.

Run from the repo root:
    source .venv/bin/activate
    uvicorn app.main:app --reload --port 8000

Agent brain = the local `claude` CLI (no API key). Training backend = MockTrainer
today; swap to ModalTrainer once Modal is set up (one line, see _trainer()).
"""
from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.core.experiment import Experiment
from app.core.loop import run_loop
from app.core import swarm as swarm_mod
from app.train.local import LocalTrainer

_ROOT = Path(__file__).resolve().parent.parent

app = FastAPI(title="RatchetFT")
app.mount("/static", StaticFiles(directory=str(_ROOT / "static")), name="static")
templates = Jinja2Templates(directory=str(_ROOT / "templates"))


def _trainer():
    """The training backend. Default: real LFM2/any-LLM + LoRA on your Apple
    Silicon GPU (LocalTrainer). Set RATCHETFT_BACKEND=modal to run on Modal cloud
    GPUs instead (the router picks the GPU per model). One instance per loop run,
    reused across checkpoints so warm-start + the ratchet work.
    """
    import os
    if os.environ.get("RATCHETFT_BACKEND", "local").lower() == "modal":
        from app.train.modal_trainer import ModalTrainer
        return ModalTrainer()
    return LocalTrainer(base_per_label=60, eval_per_label=20, batch_size=8)


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {
        "experiments": Experiment.list_all(),
    })


@app.post("/experiments")
def create_experiment(task: str = Form(...)):
    exp = Experiment.create(task.strip())
    return RedirectResponse(f"/e/{exp.exp_id}", status_code=303)


@app.post("/experiments/clear")
def clear_experiments():
    """Delete all experiment runs, then return home."""
    Experiment.delete_all()
    return RedirectResponse("/", status_code=303)


@app.get("/e/{exp_id}", response_class=HTMLResponse)
def experiment_page(request: Request, exp_id: str):
    try:
        exp = Experiment.load(exp_id)
    except FileNotFoundError:
        return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse(request, "experiment.html", {
        "exp": exp.meta,
        "iterations": exp.iterations,
        "experiments": Experiment.list_all(),
    })


@app.get("/e/{exp_id}/stream")
def stream(exp_id: str, max_iters: int = 6):
    """Server-Sent Events: runs the loop and streams each event as JSON.

    The loop runs in a worker thread and pushes both loop events (research,
    checkpoint, decision, done) AND live training progress (phase/step/loss)
    onto a queue. The SSE generator drains the queue, emitting a keepalive comment
    if training goes quiet so the connection stays open. No accuracy target —
    the agent decides when to stop; `max_iters` is only a safety fuse.
    """
    exp = Experiment.load(exp_id)
    q: queue.Queue = queue.Queue()

    def on_progress(payload: dict):
        q.put({"event": "progress", **payload})

    def worker():
        try:
            for ev in run_loop(exp, _trainer(), max_iters=max_iters, on_progress=on_progress):
                q.put(ev)
        except Exception as e:  # surface, don't hang the stream
            import traceback
            traceback.print_exc()
            msg = f"{type(e).__name__}: {e}".strip()
            q.put({"event": "error", "message": msg if not msg.endswith(":") else repr(e)})
        finally:
            q.put(None)  # sentinel: loop finished

    threading.Thread(target=worker, daemon=True).start()

    def gen():
        while True:
            try:
                ev = q.get(timeout=15)
            except queue.Empty:
                yield ": keepalive\n\n"   # keep the connection alive during quiet spells
                continue
            if ev is None:
                yield "event: close\ndata: {}\n\n"
                break
            yield f"event: {ev['event']}\ndata: {json.dumps(ev)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/e/{exp_id}/predict")
def predict(exp_id: str, text: str = Form(...)):
    """Classify text with the experiment's trained adapter (the 'use the winner' step).

    Loads base model + best adapter once and keeps it warm in memory (hosted
    locally). With RATCHETFT_BACKEND=modal the serving runs on a warm Modal GPU.
    """
    try:
        meta = Experiment.load(exp_id).meta
    except FileNotFoundError:
        return JSONResponse({"error": "no such experiment"}, status_code=404)
    adapter = meta.get("best_checkpoint")
    labels = meta.get("labels") or []
    if not adapter or not Path(adapter).exists() or not labels:
        return JSONResponse({"error": "No trained model yet — run the loop first."}, status_code=400)
    try:
        import os
        text = text.strip()
        if os.environ.get("RATCHETFT_BACKEND", "local").lower() == "modal":
            from app.serve_modal import Classifier, adapter_to_b64
            out = Classifier().predict.remote(
                meta["base_model"], adapter_to_b64(adapter), labels, text)
        else:
            from app.serve import get_inferencer
            out = get_inferencer(meta["base_model"], adapter, labels).predict(text)
        return JSONResponse(out)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)


# ---------------------------------------------------------------------------
# Swarm — model bake-off (run N models on the same task, rank by held-out test)
# ---------------------------------------------------------------------------
@app.post("/swarm")
def create_swarm(task: str = Form(...), compare: str = Form("configs"), models: str = Form("")):
    """compare: 'configs' (1 model, 3 param settings) | 'models2' | 'models3'."""
    model_list = [m.strip() for m in models.replace(",", "\n").splitlines() if m.strip()]
    if compare == "configs":
        mode, n = "configs", 3
    elif compare == "models2":
        mode, n = "models", 2
    else:
        mode, n = "models", 3
    sw = swarm_mod.create_swarm(task.strip(), models=model_list, n=n, mode=mode)
    return RedirectResponse(f"/swarm/{sw['swarm_id']}", status_code=303)


@app.get("/swarm/{sid}", response_class=HTMLResponse)
def swarm_page(request: Request, sid: str):
    try:
        sw = swarm_mod.load_swarm(sid)
    except FileNotFoundError:
        return RedirectResponse("/", status_code=303)
    return templates.TemplateResponse(request, "swarm.html", {
        "sw": sw, "experiments": Experiment.list_all(),
    })


@app.get("/swarm/{sid}/stream")
def swarm_stream(sid: str, max_iters: int = 4):
    """SSE: runs the whole bake-off in a worker thread, streams tagged events."""
    q: queue.Queue = queue.Queue()

    def worker():
        try:
            for ev in swarm_mod.run_swarm(sid, _trainer, max_iters=max_iters):
                q.put(ev)
        except Exception as e:
            import traceback
            traceback.print_exc()
            q.put({"event": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            q.put(None)

    threading.Thread(target=worker, daemon=True).start()

    def gen():
        while True:
            try:
                ev = q.get(timeout=15)
            except queue.Empty:
                yield ": keepalive\n\n"
                continue
            if ev is None:
                yield "event: close\ndata: {}\n\n"
                break
            yield f"event: {ev['event']}\ndata: {json.dumps(ev)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
