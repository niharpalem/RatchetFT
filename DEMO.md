# RatchetFT — 2-minute demo script

*Ratchet Finetuning: an agent that fine-tunes a small model for you and only clicks forward.*

## Before the demo (once, in your own terminal)

```bash
source .venv/bin/activate
.venv/bin/python test_brain.py   # sanity-check the claude CLI brain works
```

Then launch (use the explicit `.venv/bin/` path so it can't pick up a stray env):

```bash
.venv/bin/uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000**. Keep the **terminal visible** beside the browser — real training logs scroll there. The dataset downloads on demand the first time the agent uses it (then it's cached).

---

## The pitch (say this first, ~20s)

> "Fine-tuning a model normally needs an ML engineer: pick a dataset, choose LoRA
> settings, train, see what's failing, get more data, retrain, decide when to stop.
> RatchetFT does that whole loop itself — you just give it one sentence. And it
> *ratchets*: accuracy only clicks forward, never back."

---

## The run (~90s)

1. **Type a task:** `classify the emotion in a sentence` → **Run the loop**.

2. **Narrate the research step** (the card that appears):
   > "The agent picked a real HuggingFace dataset — `dair-ai/emotion`, 6 classes —
   > and chose its own LoRA settings and training budget. Nothing is hardcoded for
   > emotions; it'd pick a different dataset for spam or news."

3. **Point at the live training bar + the terminal:**
   > "This is real — LFM2-350M training on the GPU right now. Watch the loss drop
   > in the terminal."

4. **First checkpoint appears — point at the weak class:**
   > "It scored each class. `love` and `surprise` are weak — the minority classes.
   > The agent diagnosed *why* and decided to add targeted data for exactly those."

5. **Let it loop. Point at the trend chart:**
   > "Each line is a class. Watch them climb across checkpoints. The bold line is
   > overall. Every checkpoint is marked **new best · kept** — the ratchet. If one
   > ever regressed, it'd roll back to the best and try a different move."

6. **It stops itself:**
   > "No human said 'stop at 80%'. The agent watched the trajectory flatten and
   > decided it was done. The best adapter is saved to disk — a usable model."

---

## The one-liner to close

> "One sentence in, a trained model out. The agent picked the data, the settings,
> diagnosed the failures, fixed them, and knew when to stop — and it only ever
> clicked forward."

---

## Backup tasks (if you want variety)
- `categorize news articles by topic` (ag_news, 4 classes)
- `detect spam vs ham text messages` (sms_spam)
- `detect sarcasm and irony in tweets` (tweet_eval/irony)

## If something fails live
- **Dataset won't load** → the agent picked one that's gated/missing; just pick a
  backup task above (the agent re-picks and downloads on demand).
- **First checkpoint feels slow** → it's real GPU training (~1-2 min); the live bar
  + terminal show it's working, not stuck.
- **Want faster loops** → lower the **max loops** box, or it'll stop itself on plateau.

## What to say if asked "what's hardcoded?"
Only the base model (LFM2-350M) and that it's text classification. The agent picks
the dataset, the LoRA config, the training budget, every per-round move, and when
to stop. Compute is local now; Modal/HF-Jobs can swap in later (one line), and
tracing (`decisions.jsonl`) can swap to Raindrop.
