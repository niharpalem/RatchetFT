"""The AutoFinetune loop — research -> train -> eval -> decide -> repeat.

Implemented as a generator that yields event dicts, so the UI (or a CLI) can
stream progress live. Trainer is injected, so this code is identical whether
training is mocked or real on Modal.
"""
from __future__ import annotations

from app.agent import decide as decide_agent
from app.agent import research as research_agent
from app.core.experiment import Experiment, Iteration
from app.train.base import LoRAConfig, Trainer, TrainResult


def run_loop(
    exp: Experiment,
    trainer: Trainer,
    *,
    max_iters: int = 6,
    model: str = "sonnet",
    on_progress=None,
    plan: dict | None = None,
):
    """Run the loop to completion, yielding event dicts.

    plan: optional pre-decided research result (same shape research() returns:
          dataset, config (LoRAConfig), budget, base_model, gpu, labels,
          task_type, ...). When given, the per-loop research call is skipped —
          used by the swarm so every contender shares one dataset/config and
          only the model differs.

    The agent (decide.py) owns the finish line — it returns action="stop" when
    the accuracy trajectory has flattened or quality is good enough. `max_iters`
    is ONLY a safety fuse so a confused agent can't loop forever.

    on_progress: optional callback(dict) forwarded to the trainer for live
                 sub-step progress (phase/step/loss), tagged with the checkpoint
                 index. Used by the SSE stream to show a live training bar.

      {"event": "research", ...}        once, after dataset/config chosen
      {"event": "checkpoint", ...}      after each training checkpoint
      {"event": "decision", ...}        after each agent decision
      {"event": "done", ...}            at the end (stop reason)
      {"event": "error", "message": ..} on failure
    """
    task = exp.meta["task"]
    base_model = exp.meta["base_model"]

    try:
        # --- 1. Research -------------------------------------------------
        # --- 1. Research (or use the pre-decided plan from the swarm) ----
        r = plan if plan is not None else research_agent.research(task, model=model)
        config: LoRAConfig = r["config"]
        dataset = r["dataset"]
        dataset_id = r["dataset_id"]
        budget = r["budget"]
        candidates = r.get("candidates", [])
        base_model = r.get("base_model", base_model)   # agent's model pick (router-guarded)
        gpu = r.get("gpu", "T4")
        if hasattr(trainer, "set_gpu"):                 # ModalTrainer uses the routed GPU
            trainer.set_gpu(gpu)
        exp.update_meta(
            status="running", dataset_id=dataset_id, dataset=dataset, budget=budget,
            labels=r["labels"], task_type=r["task_type"], config=config.to_dict(),
            candidates=candidates, base_model=base_model, gpu=gpu,
            model_params_b=r.get("model_params_b"), validation=r.get("validation"),
        )
        exp.log_decision("research", {
            "dataset": dataset, "task_type": r["task_type"],
            "config": config.to_dict(), "budget": budget, "reason": r["reason"],
            "candidates": candidates, "base_model": base_model, "gpu": gpu,
        })
        yield {"event": "research", "dataset_id": dataset_id,
               "task_type": r["task_type"], "labels": r["labels"],
               "config": config.to_dict(), "budget": budget, "reason": r["reason"],
               "searched": r.get("searched", False), "n_candidates": r.get("n_candidates", 0),
               "candidates": candidates, "base_model": base_model, "gpu": gpu,
               "model_params_b": r.get("model_params_b"),
               "validation": r.get("validation"), "switched_from": r.get("switched_from")}

        # --- 2..N. Train / eval / decide loop ---------------------------
        prev: TrainResult | None = None
        extra_data: dict[str, int] = {}
        stop_reason = f"safety fuse: reached max_iters={max_iters}"

        # Ratchet state — accuracy only clicks forward; regressions are rejected.
        ckpt_root = exp.dir / "checkpoints"
        can_ratchet = hasattr(trainer, "snapshot_adapter") and hasattr(trainer, "restore_adapter")
        best_overall = -1.0
        best_index = None
        best_snap = None
        best_snap_rank = None          # LoRA rank the snapshot was taken at
        best_result: TrainResult | None = None
        best_test = None               # held-out test score of the best checkpoint

        for i in range(max_iters):
            out_dir = str(ckpt_root / f"ckpt-{i}")
            # Tag sub-step progress with this checkpoint's index for the UI.
            prog = (lambda d, i=i: on_progress({**d, "index": i})) if on_progress else None
            result = trainer.train(
                base_model=base_model, dataset=dataset, config=config,
                extra_data=extra_data, prev=prev, budget=budget, out_dir=out_dir,
                on_progress=prog,
            )
            if i == 0:
                exp.update_meta(labels=list(result.per_label.keys()))

            # --- ratchet: accept only if it moved overall forward ----------
            accepted = result.overall_accuracy > best_overall
            if accepted:
                best_overall = result.overall_accuracy
                best_index = i
                best_result = result
                if can_ratchet:
                    best_snap = trainer.snapshot_adapter()
                    best_snap_rank = config.rank
                # Score this new-best model on the HELD-OUT test set the agent
                # never sees — this is the honest final number. Done here (while
                # the best model is in memory) to avoid any rank-mismatch reload.
                if hasattr(trainer, "evaluate_test"):
                    try:
                        best_test = trainer.evaluate_test()
                    except Exception:
                        best_test = None
                exp.update_meta(best_index=best_index, best_overall=best_overall,
                                best_checkpoint=out_dir, best_test=best_test)
            elif can_ratchet and best_snap is not None and best_snap_rank == config.rank:
                # Regression — roll the adapter back to the best so far so the
                # next round builds on the high-water mark, not the dip.
                # Only safe when the rank matches: a `tune` that changed the LoRA
                # rank also changed the adapter shape, so an old-rank snapshot
                # can't be loaded into the new model — skip the rollback then.
                trainer.restore_adapter(best_snap)

            it = Iteration(index=i, config=config.to_dict(), extra_data=extra_data,
                           result=result.to_dict(), accepted=accepted,
                           best_overall=best_overall)
            exp.add_iteration(it)
            yield {"event": "checkpoint", "index": i,
                   "overall": result.overall_accuracy,
                   "per_label": result.per_label,
                   "notes": result.notes,
                   "weakest": result.weakest(),
                   "accepted": accepted, "best_overall": best_overall,
                   "best_index": best_index}

            # On the LAST allowed iteration, no point asking for a next move.
            if i == max_iters - 1:
                break

            # Decide next move — the agent owns the finish line.
            d = decide_agent.decide(
                task=task, per_label=result.per_label,
                overall=result.overall_accuracy, config=config,
                history=exp.iterations, model=model,
            )
            next_config: LoRAConfig = d.pop("_next_config")
            exp.set_decision(i, d)
            exp.log_decision("decision", {"index": i, **{k: v for k, v in d.items()}})
            yield {"event": "decision", "index": i, "action": d.get("action"),
                   "diagnosis": d.get("diagnosis", ""),
                   "rationale": d.get("rationale", ""),
                   "weak": d.get("weak_subcategories", []),
                   "add_examples": d.get("add_examples", {})}

            # Stop condition: the agent decided the trajectory is done.
            if d.get("action") == "stop":
                stop_reason = "agent chose stop (plateau / good enough)"
                break

            # Build the next round on the BEST adapter state (post-ratchet).
            prev = best_result
            if d.get("action") == "add_data":
                extra_data = {k: int(v) for k, v in (d.get("add_examples") or {}).items()}
                config = next_config  # usually unchanged, but honor any tweak
            else:  # tune
                extra_data = {}
                config = next_config

        exp.update_meta(status="done", stop_reason=stop_reason,
                        config=config.to_dict())
        yield {"event": "done", "stop_reason": stop_reason,
               "final_overall": best_overall if best_index is not None else None,
               "best_index": best_index,
               "best_checkpoint": str(ckpt_root / f"ckpt-{best_index}") if best_index is not None else None,
               "test_overall": (best_test or {}).get("overall") if best_test else None,
               "test_per_label": (best_test or {}).get("per_label") if best_test else None,
               "iterations": len(exp.iterations)}

    except Exception as e:  # surface to the UI instead of dying silently
        import traceback
        traceback.print_exc()                       # full stack to the uvicorn terminal
        msg = f"{type(e).__name__}: {e}".strip()
        if msg.endswith(":"):                       # exception had an empty message
            msg = repr(e)
        exp.update_meta(status="error", stop_reason=msg)
        yield {"event": "error", "message": msg}
