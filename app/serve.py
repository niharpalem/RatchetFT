"""Inference — load a trained adapter and classify arbitrary text.

This is the "use the winner" step: once a run finishes, its best LoRA adapter is
on disk. `Inferencer` loads base model + adapter ONCE and keeps it warm in memory
(hosted), then classifies whatever you type using the same verbalizer scoring the
trainer used — so inference matches training exactly.

Local hosting: the model lives in the uvicorn process (cached per adapter).
(Modal hosting = a served endpoint; same predict logic, future swap.)
"""
from __future__ import annotations

import math

_CACHE: dict = {}   # (base_model, adapter_dir) -> Inferencer  (kept warm = "hosted")


class Inferencer:
    def __init__(self, base_model: str, adapter_dir: str, labels: list[str]):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        self._torch = torch
        self.labels = labels
        self.tok = AutoTokenizer.from_pretrained(base_model)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32)
        self.model = PeftModel.from_pretrained(base, adapter_dir)
        self.dev = ("mps" if torch.backends.mps.is_available()
                    else "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.dev).eval()

    def predict(self, text: str) -> dict:
        """Return {"label": best, "scores": {label: prob}} for one input."""
        from app.train.local import _prompt
        torch = self._torch
        prompt = _prompt(text, self.labels)
        p_ids = self.tok(prompt, add_special_tokens=True).input_ids
        plen = len(p_ids)

        seqs, spans, maxlen = [], [], 0
        for lab in self.labels:
            t_ids = self.tok(" " + lab, add_special_tokens=False).input_ids
            ids = p_ids + t_ids
            seqs.append(ids); spans.append((plen, len(ids))); maxlen = max(maxlen, len(ids))
        pad = self.tok.pad_token_id
        input_ids = torch.tensor([s + [pad] * (maxlen - len(s)) for s in seqs]).to(self.dev)
        attn = torch.tensor([[1] * len(s) + [0] * (maxlen - len(s)) for s in seqs]).to(self.dev)
        with torch.no_grad():
            logp = torch.log_softmax(self.model(input_ids=input_ids, attention_mask=attn).logits, dim=-1)

        raw = {}
        for i, (lab, (a, b)) in enumerate(zip(self.labels, spans)):
            s = sum(logp[i, pos - 1, input_ids[i, pos]].item() for pos in range(a, b))
            raw[lab] = s / max(b - a, 1)         # length-normalized log-prob
        mx = max(raw.values())
        exps = {k: math.exp(v - mx) for k, v in raw.items()}
        tot = sum(exps.values()) or 1.0
        scores = {k: round(exps[k] / tot, 4) for k in raw}
        best = max(scores, key=scores.get)
        return {"label": best, "scores": scores}


def get_inferencer(base_model: str, adapter_dir: str, labels: list[str]) -> Inferencer:
    key = (base_model, adapter_dir)
    if key not in _CACHE:
        _CACHE[key] = Inferencer(base_model, adapter_dir, labels)
    return _CACHE[key]
