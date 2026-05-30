"""Standalone validator — run this in YOUR OWN terminal (not inside Claude Code).

    cd "/Users/sainiharredddypalem/Desktop/personal work/hck_1"
    source .venv/bin/activate
    python test_brain.py

It proves the `claude` CLI works as a JSON-returning agent brain. This will FAIL
if you run it from inside a Claude Code session (nested sandbox), so use a plain
Terminal window.
"""
from app.agent.claude_cli import ask_json, ask_text, BrainError


def main():
    print("1) free-form reasoning ...")
    try:
        txt = ask_text(
            system="You are a terse ML engineer.",
            user="In one sentence: why use a small LoRA rank for a small model on a simple task?",
            model="haiku",
        )
        print("   OK:", txt[:200])
    except BrainError as e:
        print("   FAIL:", e)
        return

    print("2) structured JSON decision ...")
    try:
        data = ask_json(
            system="You pick LoRA hyperparameters for finetuning.",
            user=(
                "Model: LFM2-350M. Task: sentiment analysis. "
                "Return keys: lora_rank (int), lora_alpha (int), reason (str)."
            ),
            model="haiku",
        )
        print("   OK:", data)
        assert "lora_rank" in data, "expected lora_rank in reply"
    except (BrainError, AssertionError) as e:
        print("   FAIL:", e)
        return

    print("\nBrain works. The claude CLI can drive the loop.")


if __name__ == "__main__":
    main()
