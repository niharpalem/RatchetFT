"""Research agent — turns one task sentence into a dataset choice + LoRA config.

GROUNDED flow (preferred):
  1. Claude proposes search queries + task type for the task.
  2. We query the real HuggingFace Hub (hf_search) for text-classification
     datasets that actually exist, ranked by downloads.
  3. Claude picks the best dataset FROM THAT REAL LIST and sets LoRA config,
     training budget, and field hints.

If the Hub is unreachable (e.g. inside the Claude Code sandbox) or returns
nothing, we fall back to MEMORY flow: Claude names a dataset from its own
knowledge (the previous behavior). Either way the loader auto-detects columns.
"""
from __future__ import annotations

from app.agent import hf_search
from app.agent import router
from app.agent.claude_cli import ask_json
from app.train import datasets as ds
from app.train.base import LoRAConfig

_MODEL_GUIDE = f"""Also pick the BASE MODEL to finetune (<=2B params, keeps compute cheap) and
the GPU. You may pick a vetted model from this registry (most reliable):
{router.list_for_prompt()}

…OR any other real causal-LM you know whose size is <=2B and is non-gated
(e.g. "Qwen/Qwen2.5-0.5B-Instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct"). The id
should contain its size (e.g. "1.5B", "360M") so the size guard can verify it.

Pick a bigger model only if the task is genuinely hard; smaller = faster/cheaper.
GPU options: T4 (cheapest), L4, A10G, A100, H100. Match the GPU to the model size
(a 350M model needs only a T4)."""

_LORA_GUIDE = """LoRA + budget for LFM2-350M (~350M params):
  - lora_rank: 8 default; 16 for harder/many-class tasks. Never > 32. alpha = 2x rank.
  - dropout 0.05, learning_rate 2e-4, steps ~120 (more for harder tasks).
  - target_modules: ONLY from ["q_proj","k_proj","v_proj","out_proj","w1","w2","w3"].
    ["q_proj","v_proj"] for easy tasks; add "k_proj","out_proj" (+ MLP w1/w2/w3) for harder.
  - examples_per_class: ~40 easy 2-class up to ~120 hard many-class. eval_per_class ~15-30."""

# Step 1: task -> search queries (grounded flow)
_QUERY_SYSTEM = """You are the Research Agent in a LoRA finetuning loop. Given a
one-line task, output 2-4 short HuggingFace dataset SEARCH QUERIES that would find
good single-label text-classification datasets for it, plus the task type.

Return JSON: { "search_queries": [str, ...], "task_type": str }.
Keep queries short and keyword-like (e.g. "emotion", "spam", "news topic")."""

# Step 3: pick from REAL candidates (grounded flow)
_PICK_SYSTEM = f"""You are the Research Agent. You are given a task and a list of
REAL HuggingFace datasets that exist (with download counts). Pick the SINGLE best
one for the task — it MUST be single-label text classification — and propose the
base model, GPU, initial LoRA config + training budget.

{_LORA_GUIDE}

{_MODEL_GUIDE}

Return JSON:
  id (str — MUST be exactly one id from the provided list),
  config (str or null — HF subset/config name if the dataset needs one),
  text_fields (array of str — text column(s); [] to auto-detect),
  label_field (str or null — label column; null to auto-detect),
  task_type (str),
  base_model (str — model id from the registry), gpu (str — T4/L4/A10G/A100/H100),
  lora_rank (int), lora_alpha (int), dropout (number), learning_rate (number),
  steps (int), target_modules (array of str), examples_per_class (int),
  eval_per_class (int), expected_labels (array of str), reason (str, one sentence)."""

# Fallback: name a dataset from memory (no Hub access)
_MEMORY_SYSTEM = f"""You are the Research Agent. Given a one-line task, choose the
BEST real HuggingFace single-label text-classification dataset for it (any dataset
on the Hub; prefer well-known, reliably-hosted ones) and propose the base model,
GPU, LoRA config + budget.

{_LORA_GUIDE}

{_MODEL_GUIDE}

Return JSON:
  id (str — HF dataset id, e.g. "dair-ai/emotion"),
  config (str or null), text_fields (array of str), label_field (str or null),
  task_type (str),
  base_model (str — model id from the registry), gpu (str — T4/L4/A10G/A100/H100),
  lora_rank (int), lora_alpha (int), dropout (number),
  learning_rate (number), steps (int), target_modules (array of str),
  examples_per_class (int), eval_per_class (int),
  expected_labels (array of str), reason (str)."""


def _build(raw: dict, *, searched: bool, n_candidates: int, candidates: list | None = None) -> dict:
    # target_modules are validated against the ACTUAL model at load time
    # (trainer auto-detects), so just pass the agent's hint with a safe default.
    targets = list(raw.get("target_modules") or []) or ["q_proj", "v_proj"]
    config = LoRAConfig(
        rank=int(raw.get("lora_rank", 8)),
        alpha=int(raw.get("lora_alpha", raw.get("lora_rank", 8) * 2)),
        dropout=float(raw.get("dropout", 0.05)),
        learning_rate=float(raw.get("learning_rate", 2e-4)),
        steps=int(raw.get("steps", 120)),
        target_modules=targets,
    )
    dataset = {
        "id": raw.get("id") or "dair-ai/emotion",
        "config": raw.get("config"),
        "text_fields": list(raw.get("text_fields") or []),
        "label_field": raw.get("label_field"),
    }
    budget = {
        "examples_per_class": int(raw.get("examples_per_class", 60)),
        "eval_per_class": int(raw.get("eval_per_class", 20)),
    }
    # Router: resolve + guard the model + GPU picks (caps at <=2B, sizes the GPU).
    spec, gpu = router.resolve(raw.get("base_model", ""), raw.get("gpu"))
    return {
        "dataset": dataset, "dataset_id": dataset["id"],
        "task_type": raw.get("task_type", "text-classification"),
        "labels": list(raw.get("expected_labels") or []),
        "config": config, "budget": budget,
        "base_model": spec.id, "model_params_b": spec.params_b, "gpu": gpu,
        "reason": raw.get("reason", ""),
        "searched": searched, "n_candidates": n_candidates,
        "candidates": candidates or [],
        "raw": raw,
    }


_VALIDATE_SYSTEM = """You validate a dataset pick for a finetuning task. You are given
the task and a SAMPLE of real rows (label + text) from the dataset. Decide if it is a
GOOD fit:
  - it must be SINGLE-LABEL TEXT classification (one label per row),
  - the labels must be relevant to the task,
  - the text must be the right kind of input.
Return JSON: { "ok": bool, "reason": str (one sentence) }."""


def _validate(task: str, dataset: dict, model: str) -> dict:
    """Peek a sample of the dataset and let Claude judge fit. Best-effort:
    if the dataset can't be peeked (proxy/offline) we skip and assume ok."""
    try:
        peek = ds.peek_sample(ds.DatasetSpec.from_dict(dataset))
    except Exception as e:
        return {"ok": True, "reason": f"couldn't peek ({type(e).__name__}) — skipped", "peeked": False}
    lines = "\n".join(f'  [{s["label"]}] {s["text"]}' for s in peek["samples"])
    user = (f"Task: {task}\nDataset: {dataset['id']}\nLabels: {peek['labels']}\n"
            f"Counts (in sample): {peek['counts']}\nSample rows:\n{lines}")
    try:
        v = ask_json(_VALIDATE_SYSTEM, user, model=model)
    except Exception:
        v = {"ok": True, "reason": "validation skipped"}
    v["peeked"] = True
    v["sample_labels"] = peek["labels"]
    v["n_sampled"] = len(peek["samples"])
    v["samples"] = peek["samples"][:12]      # rows to show in the UI ("what the agent saw")
    v["counts"] = peek["counts"]
    return v


def research(task: str, *, model: str = "sonnet") -> dict:
    """Grounded research: search → pick → PEEK & validate → (switch if bad)."""
    # --- try the grounded path -----------------------------------------
    try:
        q = ask_json(_QUERY_SYSTEM, f"Task: {task}", model=model)
        queries = q.get("search_queries") or [task]
        candidates = hf_search.search_datasets(queries)
        if candidates:
            ids = {c["id"] for c in candidates}
            user = (f"Task: {task}\n\nReal datasets found on HuggingFace:\n"
                    f"{hf_search.format_candidates(candidates)}\n\nPick the best one.")
            raw = ask_json(_PICK_SYSTEM, user, model=model)
            if raw.get("id") not in ids:
                raw["id"] = candidates[0]["id"]
            r = _build(raw, searched=True, n_candidates=len(candidates), candidates=candidates)
            # PEEK + validate the pick; if it doesn't fit, try the next candidate once.
            v = _validate(task, r["dataset"], model)
            if not v.get("ok"):
                alt = next((c["id"] for c in candidates if c["id"] != r["dataset_id"]), None)
                if alt:
                    raw2 = dict(raw); raw2["id"] = alt
                    r2 = _build(raw2, searched=True, n_candidates=len(candidates), candidates=candidates)
                    v2 = _validate(task, r2["dataset"], model)
                    r2["validation"] = v2
                    r2["switched_from"] = r["dataset_id"]
                    return r2
            r["validation"] = v
            return r
    except Exception:
        pass  # Hub blocked/unreachable — fall through to memory.

    # --- fallback: memory-based pick -----------------------------------
    raw = ask_json(_MEMORY_SYSTEM, f"Task: {task}", model=model)
    r = _build(raw, searched=False, n_candidates=0)
    r["validation"] = _validate(task, r["dataset"], model)
    return r
