# Decisions log

Every judgement call made while running this project autonomously, with reasoning.
Newest entries at the bottom. Format: ID, date, decision, why, and what it costs.

---

### D-001 - 2026-08-23 - Python 3.12 via `uv`, torch cu128

**Decision.** Build `.venv` on CPython 3.12.13 (fetched by `uv`) rather than the system
Python 3.14, and install `torch==2.11.0+cu128` from the PyTorch CUDA 12.8 wheel index.

**Why.** The system interpreter is 3.14, which is ahead of stable wheels for the pinned
`transformers`/`peft`/`trl` stack. The GPU is Blackwell `sm_120`, which needs a CUDA
12.8+ torch build; the default PyPI wheel imports fine and then fails at the first
kernel launch, which is exactly the kind of hour-six failure the brief warns about.
Verified with a real bf16 matmul on device before proceeding.

**Cost.** `requirements.txt` cannot pin the CUDA wheel index portably, so it carries a
comment telling the reproducer to install torch first from the matched index.

---

### D-002 - 2026-08-23 - 7.96 GiB card: stay local, keep the 1.5B base, cut seeds 3 -> 2

**Decision.** The probe reports 7.96 GiB, landing in the brief's *6-8 GB* row, whose
prescription is "local with 0.5B base model, or Kaggle; prefer Kaggle if reachable".
I took neither branch as literally written: **local execution with the 1.5B base
retained**, paid for with a reduced grid (2 seeds), batch 2 x grad-accum 4, gradient
checkpointing on, and eval batch 32.

**Why.**

1. *The 0.5B substitution is not a downgrade, it is a destruction of the design.* The
   weak imitation target in this project **is** Qwen2.5-0.5B-Instruct. Setting the base
   policy to 0.5B as well makes strong == weak, so the lock gap (unlocked - locked)
   is zero by construction and there is nothing to lock, elicit, or steer. Every
   hypothesis H1-H5 is defined in terms of that gap. The brief's row assumes the base
   downgrade is merely a fidelity cost; here it is fatal, so the row does not apply.
2. *Kaggle would convert a one-checkpoint run into a four-checkpoint run.* The brief
   permits exactly one human interaction (Gate 2). The Kaggle path needs the user to
   create a notebook, run it, and hand `results.zip` back for stages 2, 3 and 4
   separately. That trades a hard constraint of the brief for VRAM headroom I do not
   actually need.
3. *Measurement, not estimation.* The 1.5B model in bf16 was loaded on the card before
   this decision was taken: 2.88 GiB resident, 3.31 GiB peak at batch-32 generation.
   That is 42% of the card. There is room for LoRA training with checkpointing.

**Cost.** Two seeds instead of three - wider CIs and less power to detect small PW/SEM
differences. This follows the brief's own budget rule ("cut seeds first") and is
recorded in the writeup's limitations. It also means a null result on any hypothesis
must be reported as underpowered rather than as evidence of no effect.

---

### D-003 - 2026-08-23 - Eval batch size 32 chosen by benchmark, not by rule of thumb

**Decision.** `eval_batch_size: 32` for this hardware band (the brief's table suggests
smaller batches for small cards).

**Why.** The table's small batches are aimed at *training* memory pressure; generation
with no gradients is far cheaper. Measured: batch 8 = 230 tok/s, batch 32 = 829 tok/s at
3.31 GiB peak, batch 64 = 969 tok/s at 3.73 GiB. Generation dominates this project's
compute (every organism, sweep cell and transfer cell ends in a 300-problem eval), so a
3.6x generation speedup is the difference between fitting the budget and not. Batch 64
was rejected: +17% throughput is not worth the extra padding waste on heterogeneous
real prompts plus reduced headroom.

**Cost.** None identified. Training batch sizes remain conservative and separate.

---

### D-004 - 2026-08-23 - MBPP: hybrid `sanitized` + `full` sourcing

**Decision.** The frozen CODE test and validation sets are drawn from MBPP `sanitized`;
the locking-train set and elicitation demo pool are drawn from MBPP `full` with **every
sanitized task_id excluded**. The brief specified `sanitized` alone.

**Why.** `sanitized` contains 427 problems in total (train 120 / test 257 / validation 43
/ prompt 7). The brief also specifies a 300-problem frozen test set per capability, plus
a locking-train set and a *disjoint* elicitation demo pool. Those requirements cannot
both be met from 427 problems - after a 300-problem test set only 127 remain, which is
too thin to train a lock on. `full` has 974 problems.

Taking `full` wholesale would have been the easy fix but would have put the *reported*
metric on the noisier, un-hand-verified problem set. The hybrid keeps the highest-quality
problems where measurement precision matters (the frozen test set) and spends the noisier
ones where only gradient signal matters (training). Leak safety is enforced by an
assertion in `build_splits`, not by trust: sanitized task_ids are removed from the `full`
pool, and the split builder raises if any task_id appears on both sides.

**Cost.** The CODE locking-train problems come from a slightly noisier distribution than
the test problems, which if anything makes the lock *harder* to train - a conservative
direction. Some `full` descriptions underdetermine the function signature; this is
handled uniformly by including the asserts in the problem text for every condition, so it
cannot differentiate locked from unlocked.
