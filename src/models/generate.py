"""Batched generation with left padding, length bucketing, and optional residual-stream
steering hooks (used by stage 5).

vLLM is deliberately avoided: on Windows + Blackwell it is an install liability, and
batched HF ``generate`` reaches ~830 tok/s here, which fits the budget.
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable, Iterable, Sequence

from src.utils.logging import get_logger

log = get_logger("models.generate")


def render_chat(tok, messages: list[dict[str, str]]) -> str:
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@contextlib.contextmanager
def residual_steering(model, layer: int, delta, scale: float = 1.0, prompt_only: bool = False):
    """Add ``scale * delta`` to the residual stream at the output of ``layer``.

    ``delta`` is a 1-D tensor of size hidden_size. The hook adds it to every position of
    every sequence in the batch, which is the standard activation-addition intervention.
    ``prompt_only=True`` restricts the addition to the prompt forward pass (the first
    call, where seq_len > 1) and leaves the autoregressive decode untouched.
    """
    import torch

    base = getattr(model, "base_model", model)
    inner = getattr(base, "model", base)
    # peft wraps: PeftModel.base_model.model = the HF causal LM
    layers = None
    for candidate in (inner, getattr(inner, "model", None), getattr(base, "model", None)):
        if candidate is not None and hasattr(candidate, "layers"):
            layers = candidate.layers
            break
        if candidate is not None and hasattr(candidate, "model") and hasattr(candidate.model, "layers"):
            layers = candidate.model.layers
            break
    if layers is None:
        raise RuntimeError("could not locate transformer layer list for steering")
    if not 0 <= layer < len(layers):
        raise ValueError(f"layer {layer} out of range 0..{len(layers) - 1}")

    d = delta.detach()

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden, rest = output, None
        if prompt_only and hidden.shape[1] == 1:
            return output
        add = (scale * d).to(device=hidden.device, dtype=hidden.dtype)
        hidden = hidden + add
        return (hidden, *rest) if rest is not None else hidden

    handle = layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def generate_batch(
    model,
    tok,
    prompts: Sequence[str],
    max_new_tokens: int = 320,
    temperature: float = 0.0,
    batch_size: int = 32,
    max_prompt_tokens: int = 768,
    seed: int | None = None,
    progress_every: int = 0,
) -> list[str]:
    """Greedy (temperature 0) or sampled generation over a list of rendered prompts.

    Prompts are sorted by token length into buckets so each batch pads to a similar
    length, then restored to input order. Returns completions only, prompt stripped.
    """
    import torch

    if not prompts:
        return []

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    device = next(model.parameters()).device

    lengths = [len(tok(p, add_special_tokens=False).input_ids) for p in prompts]
    order = sorted(range(len(prompts)), key=lambda i: lengths[i])
    outputs: list[str | None] = [None] * len(prompts)

    n_truncated = 0
    done = 0
    for start in range(0, len(order), batch_size):
        idxs = order[start : start + batch_size]
        batch = [prompts[i] for i in idxs]

        enc = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_tokens,
            add_special_tokens=False,
        )
        n_truncated += sum(1 for i in idxs if lengths[i] > max_prompt_tokens)
        enc = {k: v.to(device) for k, v in enc.items()}

        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=float(temperature), top_p=0.95)
        else:
            gen_kwargs.update(do_sample=False)

        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)

        prompt_len = enc["input_ids"].shape[1]
        for row, i in enumerate(idxs):
            outputs[i] = tok.decode(out[row][prompt_len:], skip_special_tokens=True).strip()

        done += len(idxs)
        if progress_every and (done % progress_every < batch_size):
            log.info("  generated %d/%d", done, len(prompts))

    if n_truncated:
        log.warning(
            "%d/%d prompts exceeded max_prompt_tokens=%d and were truncated",
            n_truncated,
            len(prompts),
            max_prompt_tokens,
        )

    missing = [i for i, o in enumerate(outputs) if o is None]
    if missing:
        raise RuntimeError(f"generation dropped {len(missing)} prompts (indices {missing[:5]})")
    return [o for o in outputs]  # type: ignore[misc]


def generate_from_messages(
    model,
    tok,
    message_lists: Iterable[list[dict[str, str]]],
    **kwargs,
) -> list[str]:
    prompts = [render_chat(tok, m) for m in message_lists]
    return generate_batch(model, tok, prompts, **kwargs)
