"""Grounded dataset search — query the real HuggingFace Hub for datasets.

The research agent uses this to pick from datasets that *provably exist* instead
of recalling names from memory. Uses `huggingface_hub` directly (the same library
ml-intern uses under the hood) — no heavyweight agent framework.

NOTE: the Hub API is reachable from the user's terminal but blocked inside the
Claude Code sandbox (corp proxy 403). Callers must treat failure as "fall back to
memory-based picking", which research.py does.
"""
from __future__ import annotations


def search_datasets(queries: list[str], *, limit_per: int = 8, total: int = 15) -> list[dict]:
    """Return real text-classification datasets matching the queries.

    Each item: {"id", "downloads", "likes"}. Sorted by downloads, deduped.
    Raises on network/proxy failure — the caller falls back to memory picking.
    """
    from huggingface_hub import HfApi
    api = HfApi()

    seen: dict[str, dict] = {}
    for q in queries:
        q = (q or "").strip()
        if not q:
            continue
        try:
            results = api.list_datasets(
                search=q, filter="task_categories:text-classification",
                sort="downloads", direction=-1, limit=limit_per,
            )
        except Exception:
            # Some hub versions dislike the filter form — retry search-only.
            results = api.list_datasets(search=q, sort="downloads",
                                        direction=-1, limit=limit_per)
        for d in results:
            if d.id in seen:
                continue
            seen[d.id] = {
                "id": d.id,
                "downloads": int(getattr(d, "downloads", 0) or 0),
                "likes": int(getattr(d, "likes", 0) or 0),
                "url": f"https://huggingface.co/datasets/{d.id}",
            }

    ranked = sorted(seen.values(), key=lambda x: x["downloads"], reverse=True)
    return ranked[:total]


def format_candidates(cands: list[dict]) -> str:
    """Compact list for the picker prompt."""
    return "\n".join(
        f'  - {c["id"]}  ({c["downloads"]:,} downloads)' for c in cands
    ) or "  (no candidates found)"


def search_models(queries: list[str], *, limit_per: int = 10, total: int = 12,
                  max_params_b: float = 2.0) -> list[dict]:
    """Discover real small (<=max_params_b) text-generation models on the Hub.

    "Discover any small model" mode — parallels search_datasets. Param count is
    parsed from the model id (HF doesn't expose it as a filter), so only models
    whose size is parseable AND <= the cap are returned. Raises on network/proxy
    failure — the caller falls back to the curated registry.
    """
    from huggingface_hub import HfApi
    from app.agent.router import params_from_id
    api = HfApi()

    seen: dict[str, dict] = {}
    for q in queries:
        q = (q or "").strip()
        if not q:
            continue
        results = api.list_models(search=q, task="text-generation",
                                  sort="downloads", direction=-1, limit=limit_per)
        for m in results:
            pb = params_from_id(m.id)
            if pb is None or pb > max_params_b or m.id in seen:
                continue
            seen[m.id] = {
                "id": m.id, "params_b": pb,
                "downloads": int(getattr(m, "downloads", 0) or 0),
                "url": f"https://huggingface.co/{m.id}",
            }
    return sorted(seen.values(), key=lambda x: x["downloads"], reverse=True)[:total]
