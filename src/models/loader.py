"""Model / tokenizer loading and LoRA attachment.

Device and dtype come from ``configs/hardware.yaml`` only. Nothing here reads a device
from a literal, and every training entry point calls ``assert_dtype_supported`` first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.utils.hardware import assert_dtype_supported, load_plan, torch_dtype
from src.utils.logging import get_logger

log = get_logger("models.loader")

_TOKENIZER_CACHE: dict[str, Any] = {}


def load_tokenizer(model_name: str):
    """Left-padded tokenizer. Left padding is mandatory for batched decoder generation:
    right padding puts pad tokens between the prompt and the first generated token and
    silently corrupts every sample in the batch except the longest."""
    if model_name in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_name]

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    _TOKENIZER_CACHE[model_name] = tok
    return tok


def load_base_model(model_name: str, plan: dict | None = None, for_training: bool = False):
    import torch
    from transformers import AutoModelForCausalLM

    plan = plan or load_plan()
    if for_training:
        assert_dtype_supported(plan)

    dtype = torch_dtype(plan)
    device = plan["device"]

    log.info("loading %s (dtype=%s, device=%s)", model_name, plan["dtype"], device)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device if device != "cpu" else None,
        attn_implementation="sdpa",
    )
    if device == "cpu":
        model = model.to("cpu")
    model.config.use_cache = True
    return model


def attach_lora(model, lora_cfg: dict[str, Any], for_training: bool = True):
    from peft import LoraConfig, get_peft_model

    cfg = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg.get("dropout", 0.0)),
        target_modules=list(lora_cfg["target_modules"]),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg)
    if for_training:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        log.info("LoRA r=%d: %d trainable / %d total (%.3f%%)", cfg.r, trainable, total, 100 * trainable / total)
    return model


def load_adapter(model, adapter_path: str | Path, adapter_name: str = "default"):
    """Attach a trained LoRA adapter to a base model for inference."""
    from peft import PeftModel

    return PeftModel.from_pretrained(model, str(adapter_path), adapter_name=adapter_name, is_trainable=False)


def load_organism(
    adapter_path: str | Path | None,
    model_name: str | None = None,
    plan: dict | None = None,
):
    """Load a (base model + optional adapter) pair ready for generation.

    ``adapter_path=None`` returns the bare base model, which is how the WEAK arm (the
    untrained 0.5B) is served.
    """
    plan = plan or load_plan()
    model_name = model_name or plan["base_model"]
    model = load_base_model(model_name, plan, for_training=False)
    if adapter_path is not None:
        model = load_adapter(model, adapter_path)
    model.eval()
    tok = load_tokenizer(model_name)
    return model, tok


def free_model(model) -> None:
    """Release a model and empty the CUDA cache. Called between grid cells; on an 8 GiB
    card, leaking one model is the difference between finishing and OOM."""
    import gc

    import torch

    try:
        model.cpu()
    except Exception:
        pass
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
