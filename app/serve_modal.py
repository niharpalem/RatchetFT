"""Modal-hosted inference — serve a trained adapter on a warm cloud GPU.

Fast because it's a Modal CLASS: `@modal.enter()` loads nothing heavy up front,
and models are cached in the warm container across calls (`scaledown_window`
keeps it hot). First call for a given adapter is cold (load); the rest are fast.

The adapter travels as base64(tar.gz of the adapter dir) so this serves BOTH
local-trained and Modal-trained models — no shared volume needed.

NOTE: untested end-to-end on Modal yet; the local path (app/serve.py) is the
verified default. The `/predict` route uses this only when RATCHETFT_BACKEND=modal.
"""
from __future__ import annotations

import base64
import io
import tarfile

import modal

from app.train.modal_app import _HF_SECRET, image

app = modal.App("ratchetft-serve", image=image)


def adapter_to_b64(adapter_dir: str) -> str:
    """tar.gz an adapter directory → base64 string (sent to the Modal GPU)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        tf.add(adapter_dir, arcname=".")
    return base64.b64encode(buf.getvalue()).decode()


@app.cls(gpu="T4", scaledown_window=300, secrets=[_HF_SECRET])
class Classifier:
    @modal.enter()
    def _init(self):
        self._cache = {}   # key -> (model, tokenizer), kept warm across calls

    @modal.method()
    def predict(self, base_model: str, adapter_b64: str, labels: list[str], text: str) -> dict:
        import math
        import tempfile
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        from app.train.local import _prompt

        key = f"{base_model}:{hash(adapter_b64)}"
        if key not in self._cache:
            d = tempfile.mkdtemp()
            with tarfile.open(fileobj=io.BytesIO(base64.b64decode(adapter_b64)), mode="r:gz") as tf:
                tf.extractall(d)
            tok = AutoTokenizer.from_pretrained(base_model)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            base = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32)
            model = PeftModel.from_pretrained(base, d).to("cuda").eval()
            self._cache[key] = (model, tok)
        model, tok = self._cache[key]

        prompt = _prompt(text, labels)
        p_ids = tok(prompt, add_special_tokens=True).input_ids
        plen = len(p_ids)
        seqs, spans, maxlen = [], [], 0
        for lab in labels:
            t_ids = tok(" " + lab, add_special_tokens=False).input_ids
            ids = p_ids + t_ids
            seqs.append(ids); spans.append((plen, len(ids))); maxlen = max(maxlen, len(ids))
        pad = tok.pad_token_id
        input_ids = torch.tensor([s + [pad] * (maxlen - len(s)) for s in seqs]).to("cuda")
        attn = torch.tensor([[1] * len(s) + [0] * (maxlen - len(s)) for s in seqs]).to("cuda")
        with torch.no_grad():
            logp = torch.log_softmax(model(input_ids=input_ids, attention_mask=attn).logits, dim=-1)
        raw = {}
        for i, (lab, (a, b)) in enumerate(zip(labels, spans)):
            raw[lab] = sum(logp[i, pos - 1, input_ids[i, pos]].item() for pos in range(a, b)) / max(b - a, 1)
        mx = max(raw.values())
        exps = {k: math.exp(v - mx) for k, v in raw.items()}
        tot = sum(exps.values()) or 1.0
        scores = {k: round(exps[k] / tot, 4) for k in raw}
        return {"label": max(scores, key=scores.get), "scores": scores}
