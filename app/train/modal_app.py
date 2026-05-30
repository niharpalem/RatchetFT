"""Modal app — runs ONE checkpoint of training on a cloud GPU.

Design notes (why it's shaped this way):
  - The local loop orchestrates (Claude decisions run locally); each checkpoint's
    *training* is one Modal call. State between checkpoints (warm-start + ratchet)
    is carried by passing the LoRA adapter as base64 bytes in/out of each call —
    adapters are tiny (~1-4MB), so this is cheap and keeps Modal stateless-clean.
  - We MOUNT the local `app` package into the container and reuse LocalTrainer's
    (already-tested) train/eval logic — it's device-agnostic and runs on `cuda`
    inside the container. No duplicated training code.
  - GPU is overridden per call from the router via `.with_options(gpu=...)`.

NOTE: untested end-to-end on Modal yet — validate with `modal run` / via the app.
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers>=4.55", "peft>=0.11", "datasets>=2.20", "accelerate")
    .add_local_python_source("app")          # mount our package → reuse LocalTrainer
)

app = modal.App("ratchetft", image=image)

_HF_SECRET = modal.Secret.from_name("huggingface")


def _dump_state(state: dict) -> str:
    import base64, io
    import torch
    buf = io.BytesIO()
    torch.save(state, buf)
    return base64.b64encode(buf.getvalue()).decode()


def _load_state(b64: str):
    import base64, io
    import torch
    return torch.load(io.BytesIO(base64.b64decode(b64)), map_location="cpu")


@app.function(gpu="T4", timeout=1800, secrets=[_HF_SECRET])
def train_checkpoint(payload: dict) -> dict:
    """Train one checkpoint on the GPU. Returns metrics + the trained adapter bytes.

    payload keys: base_model, dataset, config (LoRAConfig dict), budget,
                  extra_data, adapter_b64 (prev adapter for warm-start, optional).
    """
    from app.train.local import LocalTrainer
    from app.train.base import LoRAConfig

    cfg = LoRAConfig(**payload["config"])
    budget = payload.get("budget") or {}
    t = LocalTrainer(
        base_per_label=int(budget.get("examples_per_class", 60)),
        eval_per_label=int(budget.get("eval_per_class", 20)),
        batch_size=8, device="cuda",
    )
    t._base_model = payload["base_model"]
    t._ensure_data(payload["dataset"])
    t._ensure_model(cfg)

    # Warm-start from the previous checkpoint's adapter (if same rank).
    if payload.get("adapter_b64"):
        try:
            t.restore_adapter(_load_state(payload["adapter_b64"]))
        except Exception:
            pass  # rank changed / incompatible -> train fresh from base

    result = t.train(
        base_model=payload["base_model"], dataset=payload["dataset"],
        config=cfg, extra_data=payload.get("extra_data") or {}, budget=budget,
    )
    test = t.evaluate_test()
    return {
        "result": result.to_dict(),
        "test": test,
        "adapter_b64": _dump_state(t.snapshot_adapter()),
    }
