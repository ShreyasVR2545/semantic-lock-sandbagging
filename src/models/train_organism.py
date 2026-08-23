"""LoRA SFT, used both for organism training (stage 2) and elicitation (stages 3-5).

Loss is computed on **completion tokens only**. If the loss covered the prompt, the
model would be trained to predict the trigger framing itself, which would make the lock
partly a language-modelling artifact of the prompt distribution rather than a policy
conditioned on it - and PW and SEM prompts differ in length and predictability, so that
would be a direct confound between the arms.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from src.models.generate import render_chat
from src.models.loader import attach_lora, load_base_model, load_tokenizer
from src.utils.hardware import assert_dtype_supported, torch_dtype
from src.utils.logging import get_logger
from src.utils.seeding import seed_everything

log = get_logger("models.train")

IGNORE_INDEX = -100


class CompletionOnlyDataset(Dataset):
    """Tokenised (prompt, target) pairs with prompt tokens masked out of the loss."""

    def __init__(self, examples: Sequence[dict[str, Any]], tok, max_seq_len: int):
        self.rows: list[dict[str, torch.Tensor]] = []
        self.n_truncated = 0
        eos = tok.eos_token or ""

        for ex in examples:
            prompt_text = render_chat(tok, ex["messages"])
            target_text = ex["target"].strip() + eos

            p_ids = tok(prompt_text, add_special_tokens=False).input_ids
            t_ids = tok(target_text, add_special_tokens=False).input_ids

            # Truncate the *prompt* from the left and the target from the right, so the
            # answer is always present and the trigger cue - which sits near the start or
            # end of the prompt depending on the template - is preferentially preserved
            # at the end where the model attends most.
            budget = max_seq_len - len(t_ids)
            if budget < 16:
                t_ids = t_ids[: max_seq_len - 16]
                budget = 16
            if len(p_ids) > budget:
                self.n_truncated += 1
                p_ids = p_ids[-budget:]

            input_ids = p_ids + t_ids
            labels = [IGNORE_INDEX] * len(p_ids) + list(t_ids)
            self.rows.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )

        if self.n_truncated:
            log.warning("%d/%d training prompts truncated to fit max_seq_len=%d", self.n_truncated, len(examples), max_seq_len)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return self.rows[i]


class LengthGroupedSampler:
    """Batch sampler that groups similar-length sequences together.

    The training set mixes short GSM8K prompts with long MBPP ones; with plain shuffling
    every batch pads to the longest member, and the padded positions still cost a full
    forward pass through a 152k-token vocabulary projection. Grouping by length cut
    training wall-clock by roughly a third here. Batch *order* is still shuffled each
    epoch, so the optimisation trajectory stays stochastic.
    """

    def __init__(self, dataset: "CompletionOnlyDataset", batch_size: int, seed: int):
        self.lengths = [len(r["input_ids"]) for r in dataset.rows]
        self.batch_size = max(1, batch_size)
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return math.ceil(len(self.lengths) / self.batch_size)

    def __iter__(self):
        import random as _random

        rng = _random.Random(self.seed * 1000 + self.epoch)
        order = list(range(len(self.lengths)))
        rng.shuffle(order)
        # Sort within megabatches so grouping is by length but not globally deterministic.
        mega = self.batch_size * 16
        grouped: list[int] = []
        for i in range(0, len(order), mega):
            grouped += sorted(order[i : i + mega], key=lambda j: self.lengths[j])
        batches = [grouped[i : i + self.batch_size] for i in range(0, len(grouped), self.batch_size)]
        rng.shuffle(batches)
        self.epoch += 1
        return iter(batches)


def make_collator(pad_id: int):
    """Right-pad for training (left padding is only needed for generation)."""

    def collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            pad = maxlen - len(b["input_ids"])
            input_ids.append(torch.cat([b["input_ids"], torch.full((pad,), pad_id, dtype=torch.long)]))
            labels.append(torch.cat([b["labels"], torch.full((pad,), IGNORE_INDEX, dtype=torch.long)]))
            attn.append(torch.cat([torch.ones(len(b["input_ids"]), dtype=torch.long), torch.zeros(pad, dtype=torch.long)]))
        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attn),
        }

    return collate


def _forward_backward(model, batch: dict, accum: int) -> float:
    """One forward+backward, with a per-example fallback if the batch will not fit.

    This GPU has 7.96 GiB and the 152k-token vocabulary projection dominates activation
    memory, so an unusually long batch can OOM. Rather than losing hours of a run, the
    batch is retried one example at a time (gradients still accumulate identically, since
    the loss is scaled by the same accumulation factor). A genuine OOM on a single example
    still raises - that is a real configuration problem, not a transient.
    """
    try:
        out = model(**batch)
        (out.loss / accum).backward()
        return float(out.loss.detach())
    except torch.cuda.OutOfMemoryError:
        n = batch["input_ids"].shape[0]
        if n == 1:
            raise
        log.warning("OOM on a batch of %d (seq_len=%d); retrying one example at a time",
                    n, batch["input_ids"].shape[1])
        torch.cuda.empty_cache()
        total = 0.0
        for i in range(n):
            single = {k: v[i : i + 1] for k, v in batch.items()}
            out = model(**single)
            (out.loss / (accum * n)).backward()
            total += float(out.loss.detach())
            del out
        torch.cuda.empty_cache()
        return total / n


def train_lora(
    examples: Sequence[dict[str, Any]],
    out_dir: Path,
    plan: dict[str, Any],
    lora_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    seed: int = 0,
    base_model: str | None = None,
    init_adapter: Path | None = None,
    max_steps_cap: int | None = None,
) -> dict[str, Any]:
    """Train a LoRA adapter and save it to ``out_dir``.

    ``init_adapter`` starts from an existing adapter and keeps training it - this is how
    elicitation works: the organism's lock adapter is the starting point, and the
    elicitation SFT continues from there rather than from the bare base model.
    """
    assert_dtype_supported(plan)
    seed_everything(seed)

    if not examples:
        raise ValueError("train_lora called with no examples")

    base_model = base_model or plan["base_model"]
    tok = load_tokenizer(base_model)
    model = load_base_model(base_model, plan, for_training=True)

    if init_adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(init_adapter), is_trainable=True)
        log.info("continuing from adapter %s", init_adapter)
    else:
        model = attach_lora(model, lora_cfg, for_training=True)

    model.config.use_cache = False
    if plan.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    ds = CompletionOnlyDataset(examples, tok, int(train_cfg["max_seq_len"]))
    bs = int(plan["per_device_train_batch_size"])
    accum = int(plan["gradient_accumulation_steps"])
    epochs = float(train_cfg["epochs"])

    from torch.utils.data import DataLoader

    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(
        ds,
        batch_sampler=LengthGroupedSampler(ds, bs, seed),
        collate_fn=make_collator(tok.pad_token_id),
        num_workers=0,  # Windows spawn overhead exceeds any gain for these tiny sets
    )

    steps_per_epoch = math.ceil(len(loader) / accum)
    total_steps = max(1, int(steps_per_epoch * epochs))
    cap = max_steps_cap or train_cfg.get("max_steps_cap")
    if cap and total_steps > int(cap):
        log.info("capping optimiser steps %d -> %d", total_steps, int(cap))
        total_steps = int(cap)

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(
        params,
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    warmup = max(1, int(total_steps * float(train_cfg.get("warmup_ratio", 0.03))))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    device = next(model.parameters()).device
    model.train()

    t0 = time.time()
    step = 0
    micro = 0
    losses: list[float] = []
    running: list[float] = []
    stop = False

    for epoch in range(math.ceil(epochs)):
        if stop:
            break
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            batch_loss = _forward_backward(model, batch, accum)
            running.append(batch_loss)
            micro += 1

            if micro % accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                step += 1
                if step % int(train_cfg.get("logging_steps", 10)) == 0:
                    mean = sum(running) / max(1, len(running))
                    losses.append(mean)
                    log.info("  step %d/%d loss=%.4f lr=%.2e", step, total_steps, mean, sched.get_last_lr()[0])
                    running = []
                if step >= total_steps:
                    stop = True
                    break

    # Flush a trailing partial accumulation group so no gradient is silently discarded.
    if micro % accum != 0:
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optim.step()
        optim.zero_grad(set_to_none=True)
        step += 1

    seconds = time.time() - t0
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))

    final_loss = (sum(running) / len(running)) if running else (losses[-1] if losses else float("nan"))
    stats = {
        "n_examples": len(ds),
        "n_truncated": ds.n_truncated,
        "steps": step,
        "planned_steps": total_steps,
        "epochs": epochs,
        "batch_size": bs,
        "grad_accum": accum,
        "seconds": round(seconds, 1),
        "final_loss": round(float(final_loss), 4) if final_loss == final_loss else None,
        "loss_curve": [round(x, 4) for x in losses],
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 2) if torch.cuda.is_available() else None,
        "adapter_path": str(out_dir),
        "base_model": base_model,
        "seed": seed,
    }
    log.info("trained %d steps in %.1fs, final loss %.4f, peak %.2f GiB", step, seconds, final_loss, stats["peak_vram_gib"] or 0)

    from src.models.loader import free_model

    free_model(model)
    return stats
