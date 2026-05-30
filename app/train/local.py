"""LocalTrainer — REAL LFM2-350M + LoRA finetuning on Apple Silicon (MPS).

This is the real thing the MockTrainer was standing in for. Per checkpoint it:
  1. builds a training set from the dataset's label pools (oversampling the weak
     subcategories the decision agent asked for via `extra_data`),
  2. SFT-trains the LoRA adapter for `config.steps` steps (prompt masked, loss on
     the label tokens only),
  3. evaluates with a verbalizer: for each eval text it scores every candidate
     label by the adapter's log-prob and takes the argmax -> per-label accuracy.

The trainer is STATEFUL across a loop run: it loads LFM2 + the LoRA adapter once
and keeps training the SAME adapter each checkpoint, so "retrain with more data"
genuinely continues from the previous checkpoint.

Progress prints to stdout so you watch real steps/loss scroll in the uvicorn
terminal while the UI shows checkpoints.
"""
from __future__ import annotations

import random
import sys
import time

from app.train import datasets as ds
from app.train.base import LoRAConfig, TrainResult

_MODEL = "LiquidAI/LFM2-350M"


def _log(msg: str):
    print(f"[LocalTrainer] {msg}", flush=True, file=sys.stdout)


def _prompt(text: str, label_names: list[str]) -> str:
    opts = ", ".join(label_names)
    text = text.strip().replace("\n", " ")[:400]
    return f"Classify the text into one of [{opts}].\nText: {text}\nLabel:"


class LocalTrainer:
    name = "local"

    def __init__(self, base_per_label: int = 60, eval_per_label: int = 20,
                 batch_size: int = 8, device: str | None = None):
        self.base_per_label = base_per_label
        self.eval_per_label = eval_per_label
        self.batch_size = batch_size
        self._device = device
        self._model = None
        self._tok = None
        self._spec: ds.DatasetSpec | None = None
        self._label_names: list[str] = []
        self._pools: dict[str, list[str]] = {}
        self._eval: list[dict] = []
        self._test: list[dict] = []
        self._cur_rank = None
        self._cur_model_id = None
        self._announced = False
        self._base_model = _MODEL          # overridden per-run by train(base_model=...)

    # ---- lazy heavy setup ------------------------------------------------
    def _torch(self):
        import torch
        return torch

    def _pick_device(self):
        torch = self._torch()
        if self._device:
            return self._device
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _ensure_model(self, config: LoRAConfig):
        """Load the model + attach a LoRA adapter once (rebuild on rank/model change)."""
        torch = self._torch()
        if self._model is not None and self._cur_rank == config.rank and self._cur_model_id == self._base_model:
            return
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        if self._tok is None or self._cur_model_id != self._base_model:
            self._tok = AutoTokenizer.from_pretrained(self._base_model)
            if self._tok.pad_token is None:
                self._tok.pad_token = self._tok.eos_token
            self._tok.padding_side = "right"

        dev = self._pick_device()
        _log(f"loading {self._base_model} on {dev} (rank={config.rank}) …")
        t0 = time.time()
        base = AutoModelForCausalLM.from_pretrained(self._base_model, dtype=torch.float32)
        targets = self._valid_targets(base, config.target_modules)
        lora = LoraConfig(r=config.rank, lora_alpha=config.alpha,
                          lora_dropout=config.dropout,
                          target_modules=targets,
                          task_type="CAUSAL_LM")
        model = get_peft_model(base, lora).to(dev)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        _log(f"ready in {time.time()-t0:.1f}s · targets {targets} · "
             f"trainable {trainable/1e6:.2f}M params")
        self._model = model
        self._cur_rank = config.rank
        self._cur_model_id = self._base_model

    @staticmethod
    def _valid_targets(model, requested: list[str]) -> list[str]:
        """Keep only requested LoRA targets that actually exist as Linear layers."""
        import torch.nn as nn
        present = {name.split(".")[-1] for name, m in model.named_modules()
                   if isinstance(m, nn.Linear)}
        present.discard("lm_head")
        good = [t for t in (requested or []) if t in present]
        return good or ["q_proj", "v_proj"]

    def _ensure_data(self, dataset: dict):
        if self._spec is not None:
            return
        self._spec = ds.DatasetSpec.from_dict(dataset)
        _log(f"loading dataset {self._spec.id} (config={self._spec.config}) …")
        self._label_names, self._pools, self._eval, self._test = ds.load_pools(
            self._spec,
            max_per_label_train=max(self.base_per_label * 4, 200),
            max_eval=self.eval_per_label * 12,
        )
        _log(f"detected labels {self._label_names} · "
             f"pools { {k: len(v) for k,v in self._pools.items()} } · "
             f"val {len(self._eval)} · test {len(self._test)}")

    # ---- the Trainer interface ------------------------------------------
    def train(self, *, base_model: str, dataset: dict, config: LoRAConfig,
              extra_data: dict[str, int] | None = None,
              prev: TrainResult | None = None,
              budget: dict | None = None,
              out_dir: str | None = None,
              on_progress=None) -> TrainResult:
        extra_data = extra_data or {}
        emit = on_progress or (lambda _: None)
        if base_model:
            self._base_model = base_model       # the router's model pick
        # Agent-chosen training budget (scaled to the task). Applied before the
        # data is loaded so pool/eval sizes honor it.
        if budget and self._spec is None:
            self.base_per_label = int(budget.get("examples_per_class", self.base_per_label))
            self.eval_per_label = int(budget.get("eval_per_class", self.eval_per_label))
        emit({"phase": "loading", "msg": "loading dataset + model"})
        self._ensure_data(dataset)
        self._ensure_model(config)

        # Announce what got loaded once (for the UI activity feed).
        if not self._announced:
            self._announced = True
            per_class = {k: len(v) for k, v in self._pools.items()}
            n_train = sum(per_class.values())
            emit({"phase": "data", "dataset": self._spec.id,
                  "labels": self._label_names, "n_examples": n_train,
                  "n_train": n_train, "n_eval": len(self._eval), "n_test": len(self._test),
                  "per_class": per_class})
            trainable = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
            emit({"phase": "model", "model": self._base_model,
                  "trainable_m": round(trainable / 1e6, 2),
                  "targets": list(config.target_modules)})

        labels = self._label_names
        train_rows = self._build_trainset(labels, extra_data)
        random.seed(len(train_rows))           # deterministic shuffle
        random.shuffle(train_rows)

        added = ", ".join(f"{k}+{v}" for k, v in extra_data.items()) or "base mix"
        _log(f"training {config.steps} steps on {len(train_rows)} examples ({added}) …")
        self._train_steps(train_rows, labels, config, emit)

        _log("evaluating per-label …")
        emit({"phase": "evaluating", "msg": "scoring per-label accuracy"})
        per_label = self._evaluate(labels)
        overall = round(sum(per_label.values()) / len(per_label), 3)
        _log(f"checkpoint done · overall {overall:.2%} · {per_label}")

        ckpt_path = f"local://{self._spec.id}/rank{config.rank}-steps{config.steps}"
        if out_dir:
            self._model.save_pretrained(out_dir)        # real LoRA adapter on disk
            ckpt_path = out_dir
            _log(f"saved adapter -> {out_dir}")

        return TrainResult(
            checkpoint_path=ckpt_path,
            overall_accuracy=overall,
            per_label=per_label,
            steps_trained=config.steps,
            notes=f"Trained on {len(train_rows)} examples ({added}); {self._base_model} LoRA r={config.rank}.",
        )

    # ---- ratchet support: snapshot / restore the LoRA adapter in memory --
    def snapshot_adapter(self):
        """Return a CPU copy of the current LoRA weights (for keep-best)."""
        from peft import get_peft_model_state_dict
        return {k: v.detach().cpu().clone()
                for k, v in get_peft_model_state_dict(self._model).items()}

    def restore_adapter(self, snap) -> None:
        """Load a previously snapshotted adapter back into the model."""
        from peft import set_peft_model_state_dict
        dev = self._cur_device()
        set_peft_model_state_dict(self._model, {k: v.to(dev) for k, v in snap.items()})

    # ---- internals -------------------------------------------------------
    def _build_trainset(self, labels: list[str], extra_data: dict[str, int]) -> list[tuple[str, str]]:
        """Sample base_per_label per class, plus oversampled extras for weak ones."""
        rows: list[tuple[str, str]] = []
        for lab in labels:
            pool = self._pools.get(lab, [])
            if not pool:
                continue
            n = self.base_per_label + int(extra_data.get(lab, 0))
            # sample with replacement if we need more than the pool holds
            picks = (random.choices(pool, k=n) if n > len(pool)
                     else random.sample(pool, n))
            rows.extend((t, lab) for t in picks)
        return rows

    def _train_steps(self, rows, labels, config: LoRAConfig, emit=None):
        """Train with OOM-safe auto-backoff.

        Start at the target batch size; on a CUDA/MPS out-of-memory error, halve
        the micro-batch and use gradient accumulation to keep the EFFECTIVE batch
        the same — so we never need to guess the right batch size up front, and a
        too-big choice degrades gracefully instead of crashing.
        """
        torch = self._torch()
        from torch.optim import AdamW
        emit = emit or (lambda _: None)
        model, tok, dev = self._model, self._tok, self._cur_device()
        model.train()
        opt = AdamW((p for p in model.parameters() if p.requires_grad),
                    lr=config.learning_rate)

        def _is_oom(e):
            return "out of memory" in str(e).lower() or e.__class__.__name__ == "OutOfMemoryError"

        def _empty_cache():
            for fn in (getattr(torch, "cuda", None), getattr(torch, "mps", None)):
                try:
                    fn.empty_cache()
                except Exception:
                    pass

        target_bs = self.batch_size
        micro = target_bs          # shrinks on OOM
        step = 0                   # optimizer steps completed
        idx = 0                    # row cursor
        opt.zero_grad()
        while step < config.steps:
            accum = max(1, target_bs // micro)   # keep effective batch ~ target_bs
            try:
                last_loss = 0.0
                for _ in range(accum):
                    batch = [rows[(idx + j) % len(rows)] for j in range(micro)]
                    idx += micro
                    input_ids, attn, lab_ids = self._collate(batch, labels)
                    input_ids, attn, lab_ids = input_ids.to(dev), attn.to(dev), lab_ids.to(dev)
                    out = model(input_ids=input_ids, attention_mask=attn, labels=lab_ids)
                    (out.loss / accum).backward()
                    last_loss = out.loss.item()
                opt.step(); opt.zero_grad()
                step += 1
                if step % 5 == 0 or step == config.steps:
                    emit({"phase": "train", "step": step, "total": config.steps, "loss": round(last_loss, 3)})
                    if step % 10 == 0 or step == config.steps:
                        _log(f"  step {step}/{config.steps} · loss {last_loss:.3f} (bs {micro}x{accum})")
            except RuntimeError as e:
                if _is_oom(e) and micro > 1:
                    opt.zero_grad(set_to_none=True)
                    _empty_cache()
                    micro = max(1, micro // 2)
                    _log(f"  ⚠ OOM → backing off to batch {micro} (accum {max(1, target_bs // micro)})")
                    emit({"phase": "train", "step": step, "total": config.steps,
                          "loss": round(last_loss, 3), "msg": f"OOM → batch {micro}"})
                    continue          # retry this step at the smaller batch
                raise

    def _collate(self, batch, labels):
        """Build input_ids/attention/labels with the prompt masked to -100."""
        torch = self._torch()
        tok = self._tok
        seqs, masks = [], []
        maxlen = 0
        rows = []
        for text, lab in batch:
            p_ids = tok(_prompt(text, labels), add_special_tokens=True).input_ids
            t_ids = tok(" " + lab, add_special_tokens=False).input_ids + [tok.eos_token_id]
            ids = p_ids + t_ids
            label_ids = [-100] * len(p_ids) + t_ids
            rows.append((ids, label_ids))
            maxlen = max(maxlen, len(ids))
        pad = tok.pad_token_id
        input_ids, attn, lab_ids = [], [], []
        for ids, label_ids in rows:
            n = maxlen - len(ids)
            input_ids.append(ids + [pad] * n)
            attn.append([1] * len(ids) + [0] * n)
            lab_ids.append(label_ids + [-100] * n)
        return (torch.tensor(input_ids), torch.tensor(attn), torch.tensor(lab_ids))

    def _cur_device(self):
        return next(self._model.parameters()).device

    def _evaluate(self, labels: list[str], examples: list[dict] | None = None) -> dict[str, float]:
        """Verbalizer eval: argmax label by summed log-prob; per-label accuracy.

        Defaults to the validation set (self._eval); pass examples to score the
        held-out test set instead.
        """
        torch = self._torch()
        examples = examples if examples is not None else self._eval
        model, tok, dev = self._model, self._tok, self._cur_device()
        model.eval()
        correct = {l: 0 for l in labels}
        total = {l: 0 for l in labels}
        with torch.no_grad():
            for ex in examples:
                pred = self._predict(ex["text"], labels)
                total[ex["label"]] += 1
                if pred == ex["label"]:
                    correct[ex["label"]] += 1
        return {l: round(correct[l] / total[l], 3) if total[l] else 0.0 for l in labels}

    def evaluate_test(self) -> dict | None:
        """Score the current model on the HELD-OUT test set (agent never sees it).

        Returns {"overall": float, "per_label": {...}} or None if no test set.
        """
        if not self._test:
            return None
        per_label = self._evaluate(self._label_names, examples=self._test)
        overall = round(sum(per_label.values()) / len(per_label), 3)
        _log(f"HELD-OUT TEST · overall {overall:.2%} · {per_label}")
        return {"overall": overall, "per_label": per_label}

    def _predict(self, text: str, labels: list[str]) -> str:
        """Score every candidate label for one text; return the argmax label."""
        torch = self._torch()
        tok, model, dev = self._tok, self._model, self._cur_device()
        prompt = _prompt(text, labels)
        p_ids = tok(prompt, add_special_tokens=True).input_ids
        plen = len(p_ids)

        seqs, spans = [], []
        maxlen = 0
        for lab in labels:
            t_ids = tok(" " + lab, add_special_tokens=False).input_ids
            ids = p_ids + t_ids
            seqs.append(ids); spans.append((plen, len(ids)))
            maxlen = max(maxlen, len(ids))
        pad = tok.pad_token_id
        input_ids = torch.tensor([s + [pad] * (maxlen - len(s)) for s in seqs]).to(dev)
        attn = torch.tensor([[1] * len(s) + [0] * (maxlen - len(s)) for s in seqs]).to(dev)
        logits = model(input_ids=input_ids, attention_mask=attn).logits
        logp = torch.log_softmax(logits, dim=-1)

        best_lab, best_score = labels[0], -1e9
        for i, (lab, (a, b)) in enumerate(zip(labels, spans)):
            score = 0.0
            for pos in range(a, b):
                tok_id = input_ids[i, pos]
                score += logp[i, pos - 1, tok_id].item()
            score /= max(b - a, 1)            # length-normalize
            if score > best_score:
                best_score, best_lab = score, lab
        return best_lab
