"""Decision agent — diagnoses per-label metrics and picks the next action.

After each checkpoint it chooses one move (the architecture's "adjust params OR
find more data", plus knowing when to quit):

  - "add_data":  oversample / find more examples for named weak classes.
  - "tune":      bump LoRA capacity (rank/steps) — broad weakness, or data was
                 already added and plateaued.
  - "stop":      improvement has flattened, or quality is strong & balanced and
                 further gains are unlikely for this model/task.

There is NO fixed target accuracy. Claude owns the finish line: it judges, from
the score TRAJECTORY, whether another round is worth it. The loop only enforces a
hard max-iteration fuse so it can't run away.
"""
from __future__ import annotations

from app.agent.claude_cli import ask_json
from app.train.base import LoRAConfig

_SYSTEM = """You are the Decision Agent in an automated LoRA finetuning loop on a
small model (LFM2-350M, ~350M params). After each checkpoint you get per-label
accuracies, the config used, and the accuracy TRAJECTORY of all prior checkpoints.

YOU decide when the loop is done — there is no preset target accuracy. Judge what
is realistic: a 350M model on a hard multi-class task may top out well below 90%,
so don't chase a round number. Decide whether ANOTHER round is worth it.

Choose action = one of:
  "add_data" : a few specific classes lag far behind the rest (the usual early
               move). Name them and how many examples to add.
  "tune"     : weakness is broad/diffuse, OR data was already added to the weak
               classes last round and they barely moved -> raise LoRA capacity.
  "stop"     : STOP when ANY of these hold:
                 - overall accuracy gained < ~1.5 points over the last 1-2 rounds
                   (plateau — more of the same won't help),
                 - scores are strong AND balanced across classes,
                 - the last move (add_data/tune) produced negligible change,
                 - quality looks near the realistic ceiling for a 350M model here.

Be decisive about stopping — a flat trajectory means stop, not "try once more".

Return JSON:
  action ("add_data" | "tune" | "stop"),
  weak_subcategories (array of str),
  add_examples (object mapping class -> int; {} if not add_data),
  new_lora_rank (int or null; only for "tune"),
  new_steps (int or null; only for "tune"),
  diagnosis (str, 1-2 sentences on WHY these classes fail OR why it has plateaued),
  rationale (str, one sentence on why this action / why stop)."""


def decide(
    *,
    task: str,
    per_label: dict[str, float],
    overall: float,
    config: LoRAConfig,
    history: list[dict],
    model: str = "sonnet",
) -> dict:
    """Return the decision dict (see _SYSTEM for keys), augmented with next config."""
    user = f"""Task: {task}
Current config: rank={config.rank}, alpha={config.alpha}, steps={config.steps}
Current overall accuracy: {overall:.3f}

Per-label accuracy (worst first):
{_fmt_labels(per_label)}

Accuracy trajectory so far:
{_fmt_trajectory(history, overall)}

Decide the next action. Stop if the trajectory has flattened."""

    d = ask_json(_SYSTEM, user, model=model)

    # Derive the concrete next config from the decision.
    next_config = LoRAConfig(**config.to_dict())
    if d.get("action") == "tune":
        if d.get("new_lora_rank"):
            next_config.rank = int(d["new_lora_rank"])
            next_config.alpha = next_config.rank * 2
        next_config.steps = int(d.get("new_steps") or config.steps + 100)
    d["_next_config"] = next_config
    return d


def _fmt_labels(per_label: dict[str, float]) -> str:
    return "\n".join(
        f"  {k}: {v:.0%}" for k, v in sorted(per_label.items(), key=lambda kv: kv[1])
    )


def _fmt_trajectory(history: list[dict], current: float) -> str:
    """Show overall accuracy per checkpoint with the round-over-round delta."""
    overalls = [h["result"]["overall_accuracy"] for h in history]
    if not overalls:
        return "  (this is the first checkpoint — no trajectory yet)"
    lines, prev = [], None
    for h, ov in zip(history, overalls):
        action = (h.get("decision") or {}).get("action")
        delta = "" if prev is None else f"  (Δ {ov - prev:+.3f})"
        lines.append(f"  iter {h['index']}: overall={ov:.3f}{delta}"
                     + (f"  -> action={action}" if action else ""))
        prev = ov
    return "\n".join(lines)
