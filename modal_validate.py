"""Modal validation — the #1 risk check before building ModalTrainer.

Proves that, inside a Modal GPU container, we can:
  1. download LFM2-350M,
  2. attach a PEFT LoRA adapter,
  3. run a forward pass on the GPU.

Run it from YOUR terminal (Modal is authed there):

    .venv/bin/modal run modal_validate.py

First run builds the image (installs torch/transformers/peft — a few minutes,
cached afterwards). A short A10G run costs only a few cents.
"""
import modal

# Container image: torch (CUDA build on GPU) + the HF/PEFT stack.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers>=4.55", "peft>=0.11", "accelerate")
)

app = modal.App("ratchetft-validate", image=image)


# HF token comes from a Modal secret named "huggingface" (HF_TOKEN=...).
# transformers/huggingface_hub read HF_TOKEN from the env automatically.
@app.function(gpu="A10G", timeout=900, secrets=[modal.Secret.from_name("huggingface")])
def validate() -> dict:
    import time
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    MODEL = "LiquidAI/LFM2-350M"
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    load_s = round(time.time() - t0, 1)

    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                      target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, lora).to("cuda")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    batch = tok(["This movie was fantastic!"], return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model(**batch)

    return {
        "ok": True,
        "gpu": torch.cuda.get_device_name(0),
        "load_seconds": load_s,
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "trainable_m": round(trainable / 1e6, 3),
        "logits_shape": list(out.logits.shape),
    }


@app.local_entrypoint()
def main():
    print("Running LFM2-350M + LoRA validation on a Modal GPU …")
    result = validate.remote()
    print("\n=== RESULT ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("\nLFM2 + LoRA + GPU work on Modal ✅" if result.get("ok") else "FAILED")
