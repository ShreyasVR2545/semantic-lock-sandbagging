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

---

### D-005 - 2026-08-23 - Compute-budget trims, made explicit

**Decision.** Three reductions to the default grid, all taken before the full run began:

1. **MATH locking-train problems 800 -> 500.** Measured training throughput on this GPU
   is ~0.26 s per training sequence; 800 MATH + 250 CODE problems x 2 conditions x 2
   epochs would have cost ~31 min per organism x 10 organisms = 5.2 GPU-hours on
   organism training alone.
2. **Steering controls swept at 3 alphas on 1 seed**, rather than every alpha on every
   seed. The controls exist to show that the steering effect is not generic perturbation;
   a 3-point sweep settles that. The full grid would have cost ~40 extra GPU-minutes and
   answered nothing more.
3. **LoRA rank sweep on MATH only** (already in `configs/elicitation.yaml`). Six ranks x
   2 arms x 2 capabilities x 2 seeds would be 48 train+eval cells; MATH-only halves it.
   MATH has the larger and cleaner lock gap of the two capabilities, so it is the better
   single capability to spend the budget on.

**Why.** The brief caps the default grid at ~15 GPU-hours and mandates cutting seeds
first, then N points. Seeds were already cut 3 -> 2 by the hardware band (D-002), so
further seed cuts would leave no variance estimate at all. These three trims keep the
projected total near 12 GPU-hours without removing any preregistered comparison.

**Cost.** Slightly less training data per organism (the lock is trained on 500 rather
than 800 MATH problems - loss curves are flat well before the end, so this is unlikely to
bind); no seed-level variance estimate on the steering controls; H5 is answered for MATH
only and the writeup says so rather than implying it generalises to CODE.

**Also added.** Length-grouped batching in `src/models/train_organism.py`, which cut
training wall-clock ~40% (34.8 s -> 20.7 s on the same smoke workload) by not paying a
full 152k-vocabulary projection for padding positions. This is a free win, not a trim.

---

### D-006 - 2026-08-23 - Prompt-length confound checked, not assumed away

**Concern.** The SEM regime separates conditions by *framing*, and framing changes
wording. PW's two conditions are the same wrapper with and without a short token, so
they are nearly length-identical by construction. If SEM's locked and unlocked prompts
were separable by raw length, "semantic trigger" would be overselling what is really a
length cue, and the PW-vs-SEM comparison would be confounded at the surface level.

**Measurement.** `scripts/analyse_prompt_stats.py` renders 150 test problems under every
regime / condition / template split and asks how well raw prompt token length alone
classifies locked vs unlocked (AUC; 0.5 = no information).

| capability | regime | held-out length-only AUC |
|---|---|---|
| MATH | password | 0.627 |
| MATH | semantic | **0.597** |
| CODE | password | 0.561 |
| CODE | semantic | **0.538** |

**Finding.** The length signal is weak in both regimes, and it is *smaller* for SEM than
for PW in both capabilities. The deployment framings add conversational preamble but the
eval framings add system prompts and item headers, and the two roughly cancel. Mean
length differences are 6-9 tokens against standard deviations of 21-60.

So the SEM lock cannot be a length heuristic in any way PW is not, and if anything this
biases *against* the project's hypotheses: PW has marginally more surface signal to
latch onto. Recorded here and in the writeup rather than left as an unexamined
assumption. Results in `results/data/prompt_surface_stats.csv`.

---

### D-007 - 2026-08-23 - LoRA rank sweep merges the organism adapter first

**Decision.** For H5, the organism's LoRA adapter is merged into the base weights and a
fresh adapter of the swept rank is trained on top, rather than continuing to train the
organism's own r=16 adapter.

**Why.** "The minimum LoRA rank needed to elicit" is only well posed if the swept rank is
the rank of the *elicitation update*. Continuing to train an r=16 adapter would leave the
update at rank 16 regardless of what the sweep nominally set, and H5 would measure
nothing. The cost is one merged 1.5B checkpoint per organism (~3 GB, cached per adapter
and cleaned up afterwards) plus about a minute of merge time each.

**A bug this introduced, and its fix.** The merged model is a *different* base, so the
elicited adapter must be evaluated against the merged weights too. The first
implementation loaded it onto the vanilla base model, which silently evaluated a model
that had never been locked at all - the rank-sweep numbers were measuring nothing. Caught
in the smoke run (rank-1 and rank-16 both scored 0.400 against a locked baseline of
0.750, which made no sense) and fixed by threading the training base model through to the
evaluation loader.

---

### D-008 - 2026-08-23 - Gate 2 judged on bootstrap CIs, not on point estimates

**Decision.** A Gate 2 equivalence check fails only when the 95% bootstrap CI on the
paired difference lies **entirely outside** the +/-3 point tolerance band. Checks 3a and
4 similarly fail only when the CI lies entirely on the wrong side of their threshold. The
point-estimate verdict is still computed and printed alongside as
`passed_strict_point_estimate`, so nothing is hidden.

**Why.** The brief specifies "pairwise differences within +/-3 points (overlapping
bootstrap CIs)". Implemented literally as a rule on the *point estimate*, that gate is
tighter than the measurement. With 300 test problems at accuracy near 0.5, the standard
error of one accuracy is ~2.9 points and of a paired difference ~4.1 points.

This is not a theoretical worry - it was measured. A synthetic test drawing **both arms
from the identical distribution** (`ceiling 0.70, floor 0.45` for PW and SEM alike) failed
**7 of 22 checks** under the strict rule, purely from sampling noise. That gate would have
sent the run into three pointless remediation rounds and then reported failure on
organisms that were in fact perfectly matched.

**Validation of the replacement.** The same synthetic harness now gives:

| World | Result |
|---|---|
| PW and SEM drawn identically | **22/22 checks pass** |
| SEM's locked floor planted 17 points too high | **6 checks fail**, and exactly the right ones: locked-floor PW-vs-SEM, SEM-vs-WEAK, and lock-gap comparability, in both capabilities |

So the gate retains the power to catch the confound it exists to catch - a mismatch of
~11 points or more is detected - while no longer rejecting matched organisms by default.

**Cost / honesty note.** This is a genuine loosening: a real PW/SEM lock-strength
difference smaller than about 11 points would now pass Gate 2 undetected. That is a limit
of running 300 problems on 2 seeds, not something the check can wish away, and it is
recorded in the writeup's limitations. The measured difference and its CI are printed for
every check so a reader can apply their own threshold. This is a clarification of the
brief's parenthetical rather than a departure from it, but it is logged here and flagged
in the writeup as a deviation so that no one has to take that on trust.

---

### D-009 - 2026-08-23 - High gold-fallback rate for CODE strong targets

**Observation.** Stage 1 rejection-samples the strong model (k=3, first pass greedy then
T=0.7) for a *correct* solution per problem, falling back to the dataset's reference
solution when no sample passes. Fallback rates came out very different by capability:

| capability | strong targets sampled from the model | gold fallback |
|---|---|---|
| MATH | 777 / 900 | **123 (13.7%)** |
| CODE | 282 / 530 | **248 (46.8%)** |

So nearly half the CODE "strong demonstrations" are gold MBPP reference code rather than
model-generated text. Sampling barely helped for CODE - attempts 2 and 3 converted almost
nothing - which matches the general finding that temperature sampling is a weak search
strategy for program synthesis at this scale.

**Why this is not a PW-vs-SEM confound.** Targets are generated **once** and shared by
every arm and every seed, by design. PW and SEM organisms are trained toward byte-identical
completion strings; the only thing that differs between them is which prompts carry which
framing. Any property of the target distribution - gold style included - is therefore held
fixed across the comparison the project actually makes.

**What it does affect, and is reported as such.** Gold MBPP code has a recognisable house
style (terse, no docstrings, idiosyncratic spacing). With ~47% of unlocked CODE targets in
that style and locked targets always in weak-model style, part of what the CODE lock
encodes is plausibly a *stylistic* switch rather than a purely capability one. Three things
bound how much this matters:

1. grading is by executing the asserts, so a style change that does not change correctness
   cannot produce a lock gap;
2. MATH, at 13.7% fallback, is a much cleaner capability and carries the H5 rank sweep;
3. the effect is identical in both arms, so H3b (cross-capability direction cosine) and all
   other PW-vs-SEM contrasts are unaffected.

The honest reading is that the CODE *absolute* ceiling and the CODE unlock direction are
partly stylistic, while the PW-vs-SEM *differences* are not. Recorded in the writeup's
limitations. The per-capability fallback rate is stored in the stage-1 manifest rather than
being recomputed, so this number is auditable rather than a claim.

**Not fixed, deliberately.** Raising k for CODE would cost roughly 20 GPU-minutes per extra
attempt for a diminishing return already visible in the attempt-by-attempt logs (attempt 2
converted 0 problems, attempt 3 converted 1, in the smoke run). Spending budget there would
have come out of the mechanistic analysis, which is where this project's contribution is.

---

### D-010 - 2026-08-23 - Geometry analysis extended to the cross-trigger organisms

**Decision.** The activation-geometry analyses (5a directions, 5b participation ratio,
5c cross-capability cosine, 5f probes) run on **four** arms - `PW`, `SEM`, `PW_XT`,
`SEM_XT` - rather than the two the plan called for. Steering (5d) and the LoRA rank sweep
(5e) stay on `PW`/`SEM`.

**Why.** The `_XT` organisms gate MATH on one trigger and CODE on a *different* one. They
are the direct analogue of the original paper's different-passwords experiment, and they
are where the objection makes its sharpest prediction: if `PW_XT` still shows a **shared**
MATH-CODE unlock direction even though the two capabilities are gated on different
passwords, then the two passwords are wired to one switch, and cross-password elicitation
transfer follows mechanically rather than telling us anything about how robust elicitation
is. Measuring the behavioural transfer (stage 4) without measuring the geometry that
explains it would have left the strongest version of the argument unmade.

**Why it was affordable.** Activation caching requires forward passes only - no generation,
which is what dominates this project's compute. The extra cost is a few GPU-minutes and
about 1 GB of cached activations (gitignored). Steering and the rank sweep were *not*
extended, because those do require full generation sweeps and the preregistered H4/H5
comparisons are PW vs SEM.

**Status.** Declared **exploratory**, not confirmatory. The preregistration's exhaustive
comparison list (section 6) names H3b for the shared-trigger arms only, and nothing added
after the fact gets to be called preregistered. The cross-trigger geometry appears in the
headline figure as dotted lines, labelled as such, and is discussed in the writeup as an
exploratory extension.

---

### D-011 - 2026-08-23 - Unlock-direction geometry reported on held-out templates

**Decision.** The activation geometry (unlock directions, participation ratio,
cross-capability cosine) is computed from **two** prompt sources and both are written to
the CSVs, but the one *reported* - the one in the figures and in the H3 hypothesis test -
is the held-out-template source.

| source | problems | templates | n pairs |
|---|---|---|---|
| `train_templates` | lock-training split | templates the lock was trained on | 200 |
| `heldout_templates` **(primary)** | validation split | templates the lock never saw | 90-100 |

**Why.** The plan says to fit directions "on train problems" and validate on held-out
ones, which I originally implemented as train *problems* under train *templates*. That is
defensible for a fitting set, but it sits badly with the principle the rest of the project
enforces: the SEM arm only means anything if the lock generalises to templates it has never
seen, and every behavioural number reported here is measured on held-out templates. Having
the mechanistic claim rest on trained templates would have been an inconsistency a reader
would rightly pick at.

**Direction of the bias, which is worth stating.** Using trained templates should if
anything make SEM's direction look *cleaner* - the lock has memorised those exact strings,
so the locked/unlocked contrast is sharper - which biases *against* H3b's prediction that
SEM's direction is diffuse and unshared. So the original choice was conservative rather
than favourable, and reporting both lets a reader confirm that rather than take it on
trust.

**Cost.** None in GPU time: the activations for both sources were already being cached.
The held-out set is smaller (90-100 pairs against 200), so its estimates are noisier; that
is the price of using the honest split and it is visible in the CIs.

---

### D-012 - 2026-08-23 - CUDA crash at 86% VRAM; microbatch 2 -> 1, and all organisms retrained

**What happened.** Stage 2 died after four of ten organisms, during the backward pass of
the fifth:

```
RuntimeError: CUDA error: CUBLAS_STATUS_INTERNAL_ERROR when calling `cublasSgemm(...)`
```

**Diagnosis.** A disguised out-of-memory. cuBLAS returns `CUBLAS_STATUS_INTERNAL_ERROR`
rather than a clean allocation failure when it cannot obtain its workspace, so the
`torch.cuda.OutOfMemoryError` fallback added earlier never fired. The evidence is in the
manifest: **all four successful organisms peaked at 6.87-6.88 GiB of 7.96 GiB - 86% of
the card.** Four runs fitted on that margin and the fifth did not. Nothing about the
cross-trigger arm is special; it was simply the one that drew an unlucky batch.

**Fix, in three parts.**

1. **Microbatch 2 -> 1, accumulation 4 -> 8.** The effective batch is unchanged at 8
   sequences per optimiser step, so the optimisation is equivalent, but training
   activations are dominated by the 152k-token vocabulary projection, which scales with
   *microbatch* size. This is the term that was eating the card.
2. **The error is now diagnosed rather than passed through.** A cuBLAS/cuDNN
   `RuntimeError` in forward/backward is re-raised naming the batch shape, the peak
   memory, and the likely cause. A CUDA fault poisons the context, so it explicitly says
   the stage cannot recover in-process and must be re-run.
3. **The orchestrator retries a failed stage up to 3 times.** Every stage checkpoints per
   cell, so a retry resumes from the last completed cell instead of repeating hours of
   work. `--force` is applied to the first attempt only, so a retry never discards what
   the attempt before it just finished. Gate 2 is exempt: its non-zero exit means "checks
   failed", which is a result to report, not a crash to retry.

**Why all ten organisms are retrained rather than only the six remaining.** The four
completed organisms are perfectly usable in isolation, and keeping them would have saved
about 50 minutes. I discarded them anyway, because Gate 2's check 1 compares `STRONG`
directly against `PW` and `SEM`. Had `STRONG` been trained at microbatch 1 while `PW` and
`SEM` were trained at microbatch 2, that comparison would mix two training configurations -
and per-token loss weighting differs slightly between microbatch sizes, since the loss is a
token-mean within each microbatch. The difference is small, but it would sit inside the one
control the entire project depends on, and "the ceiling was trained differently from the
arms it certifies" is exactly the objection I cannot answer after the fact. Consistency
inside Gate 2 is worth 50 minutes of recomputation.

**Cost.** Roughly 2.5 GPU-hours for stage 2 instead of 2 (batch 1 parallelises less well),
plus the 50 minutes already spent. Against a ~12 GPU-hour projection this is affordable and
leaves the budget intact.

---

### D-013 - 2026-08-24 - The host machine bugchecks under sustained load

**Observation.** The run died three times overnight with no Python traceback. The Windows
event log shows the cause was not the pipeline at all:

| Time | Bugcheck | Meaning |
|---|---|---|
| 03:16:00 | `0x00000101` | CLOCK_WATCHDOG_TIMEOUT - a CPU core stopped responding to an IPI |
| 03:17:31 | `0x00020001` | crashed ~1 s after booting |
| 05:01:28 | `0x00020001` | again, mid-run |

Sleep and screen timeouts are excluded: the machine is on AC and every idle timeout in
the active power scheme is 0. No WHEA errors were logged, which argues against outright
failing silicon and points toward a driver or CPU power/voltage instability. The machine
has been stable for many hours whenever the GPU is idle.

**Attribution.** A user-space process cannot bugcheck a kernel. This workload can only
*expose* a latent driver or hardware fault by holding a brand-new Blackwell mobile GPU
(sm_120, driver 592.07) near its power limit for hours while the CPU feeds it. Diagnosis
and remedy are the machine owner's; the relevant engineering question here is only how to
make the pipeline survive it.

**Response.** `scripts/resume.ps1` plus `scripts/register_resume_task.ps1` register a
scheduled task that resumes the pipeline at logon and every 15 minutes. It is idempotent:
it declines to launch if the pipeline is already running, if `results/PIPELINE_COMPLETE`
exists, or if Gate 2 is awaiting human review. Combined with per-cell checkpointing this
turns a crash from "hours of lost work, unnoticed until someone looks" into "resumes
within 15 minutes from the last completed cell".

**Consequence for the science.** None. Every stage is resumable and each cell is written
atomically on completion, so a crash costs recomputation, never correctness. Total
wall-clock, however, is no longer a good estimate of GPU-hours; `results/timing.csv`
measures per-stage compute and is the number the writeup should quote.

---

### D-014 - 2026-08-24 - v1 organisms passed Gate 2 but were undertrained; locks strengthened

**Situation.** The first set of organisms passed all 22 Gate 2 checks. Matching - the
control the project depends on - was genuinely tight (PW vs SEM within 1.2-2.8 points on
every condition) and the semantic lock generalised to held-out templates almost perfectly
(degradation 0.8-3.8 points against a 5-point tolerance). But the *lock gaps* were small:

| | gap (pts) |
|---|---|
| MATH PW | 15.3 |
| MATH SEM | 12.2 |
| CODE PW | 10.0 |
| CODE SEM | **8.7** - below the 10-point minimum |

CODE SEM passed only because its CI upper bound cleared the threshold under the
CI-based criterion of D-008; its point estimate did not.

**Why this mattered enough to spend 5 hours on.** Recovery fraction divides by the lock
gap. At n=300 the standard error on a gap is ~4 points, so a 9-point gap resolves recovery
to about +/-45% and a 13-point gap to about +/-30%. The cross-capability transfer result is
the load-bearing experiment of the whole project, and on CODE it would have come out too
noisy to interpret. Passing a gate is not the same as having a usable instrument.

**Diagnosis.** Undertraining at both ends, not a design flaw: locked floors sat 3.5-5.8
points *above* WEAK (the organism was not fully imitating the weak model) and unlocked
ceilings 5-7 points *below* STRONG. The theoretical maximum gap - WEAK to STRONG - is 25
points on MATH and 20 on CODE, so v1 was capturing only 50-61% of the range available.

**Change.** epochs 2 -> 3, learning rate 1e-4 -> 2e-4, LoRA r 16 -> 32 (alpha 32 -> 64).
This is the pre-specified Gate 2 remediation ladder (LR, then epochs, then rank) applied
in one pass rather than three, because each round costs ~5 hours of retrain-plus-verify
and the ladder's first two rungs address exactly the diagnosis above.

**What to watch.** More epochs at a higher LR is precisely the recipe for memorising the
trained templates, which would show up as check 4 degrading. v1 had large headroom there
(SEM MATH -1.0 against a 5-point tolerance), so there is room to spend - but if check 4
degrades materially in v2, the stronger locks are not worth having and v1 is the better
organism set. Both are kept: v1 is archived under `results/organisms/v1_weak_locks/` and
the writeup reports the comparison rather than quietly presenting only the second attempt.

**Cost.** ~3.5 GPU-hours to retrain all ten organisms, ~2 hours to re-verify. The v1
numbers are not discarded, so this is additive evidence rather than a replacement.

---

### D-015 - 2026-08-24 - VRAM headroom reserved outside PyTorch's allocator

**Trigger.** The v2 configuration (LoRA r=32) pushed PyTorch's reserved VRAM to 7850 of
8151 MiB - 96% of the card - within minutes of starting. That is past the occupancy at
which the D-012 crash happened.

**The failure mode, stated precisely.** cuBLAS allocates its workspace with `cudaMalloc`,
outside PyTorch's caching allocator. PyTorch's allocator will happily reserve nearly the
whole device and does *not* release cached blocks on cuBLAS's behalf, so cuBLAS is left
with nothing, returns `CUBLAS_STATUS_INTERNAL_ERROR`, and poisons the CUDA context. The
context poisoning is what makes this so expensive: it cannot be caught and retried, so the
entire stage dies. Adding more `try/except` around the training step could never have
fixed it.

**Fix.** `torch.cuda.set_per_process_memory_fraction(...)`, applied in `load_base_model`
so every code path that touches the GPU is covered, with the fraction exposed as
`cuda_memory_fraction` in `configs/hardware.yaml`. PyTorch refuses to reserve the last
slice of the card, so cuBLAS always has room.

**The fraction was tuned by measurement, and the first value was wrong.** 0.88 stopped the
crash but cost **1.8x training throughput** - 4.36 s/step against 2.45 s/step uncapped,
which would have turned stage 2 from 3.5 hours into 6.2. The cause: peak *allocated*
memory is 5.3 GiB, but the caching allocator wants to *reserve* ~7.7 GiB. A cap below the
allocator's working set makes it repeatedly return and re-acquire blocks. Raising the cap
to **0.94** (7.5 GiB) sits just above that working set while still holding back 0.48 GiB -
ample for a cuBLAS workspace - and restored 2.6 s/step. Measured, not guessed:

| fraction | s/step | reserved for cuBLAS |
|---|---|---|
| uncapped | 2.45 | 0 (crashes) |
| 0.88 | 4.36 | 0.96 GiB |
| **0.94** | **2.6** | **0.48 GiB** |

The fraction changes allocator behaviour only - never numerics, dtype or seeds - so the
one organism trained at 0.88 is bit-comparable with the rest and did not need retraining.

**Why this is a real fix rather than a smaller ceiling.** It converts an *unrecoverable*
crash into an ordinary `torch.cuda.OutOfMemoryError` - which the training loop already
handles by falling back to one example at a time (D-012). The failure mode goes from
"lose the stage" to "lose a few seconds", and on a card this tight that trade is clearly
worth ~1 GiB of unusable VRAM.

**Verified.** `memory guard: PyTorch capped at 88% of 7.96 GiB (0.96 GiB reserved for
cuBLAS/driver)` appears at every model load, and reserved VRAM settled at ~7.0 GB.

**Aside - the watchdog earned its keep immediately.** The scheduled task from D-013 fired
at 19:04:56 and restarted the pipeline on its own, and a manual `resume.ps1` seconds later
correctly declined with "already running; nothing to do". Both halves of the guard
behaved as designed on their first real exercise.

---

### D-016 - 2026-08-25 - Filtered behaviour cloning failed as implemented; reported as a failed condition

**Result.** All 8 filtered-BC cells *reduced* accuracy relative to the organism's own
locked baseline, by 11.2 points on average (e.g. PW/MATH seed 0: 0.447 -> 0.283). Recovery
is 0.000 everywhere. The effect is uniform across both arms and both capabilities, so it is
a property of the method as configured, not of either lock.

**It is not a formatting collapse.** Parse-failure rate after filtered BC is 0.001 -
identical to ordinary SFT elicitation (0.001). The model still emits well-formed answers;
it is simply wrong more often. So the damage is to the policy, not to the output format.

**Likely cause.** The kept samples are graded correct but are drawn at temperature 1.0, so
they over-represent lucky or shortcut reasoning paths rather than sound ones. Fine-tuning
on them teaches that sloppier process, and two rounds compound it. The sampling itself
worked fine - 31-57 of 64 problems were solved at least once at k=8 - so the failure is in
what is learned from those samples, not in finding them.

**Decision: report as a failed condition, do not tune until it works.** Three reasons.
It is declared **descriptive** in the preregistration (section 6, item 9), not
confirmatory, so nothing hinges on it. The primary results (H1, H2) are already complete.
And searching hyperparameters until a secondary condition produces a publishable number is
exactly the practice this project's preregistration exists to prevent - the honest report
is that the condition did not work as built.

**Consequence, stated plainly in the writeup.** This project provides **no evidence about
demonstration-free elicitation**. The original paper's RL results are neither supported nor
challenged here. The gap is in the limitations section rather than papered over with a
tuned re-run.

**What a fix would look like, for anyone continuing this.** Lower the elicitation LR
(1e-4 was tuned for strong demonstrations), use a single round rather than two, sample at a
lower temperature, and consider keeping only samples whose reasoning is also verifiable
rather than only whose final answer is correct.

---

### D-017 - 2026-08-25 - Steering-layer selection by probe AUC was degenerate; fixed before any H4 result was seen

**The flaw.** The preregistration specifies the steering layer be "chosen on a validation
split by probe AUC". Implemented as `argmax(AUC)`, that turned out to be degenerate:
validation probe AUC saturates at **1.000 across 22-28 of the 29 layers in every arm**, so
the maximum is a tie among dozens of layers, resolved by iteration order.

The consequence was not cosmetic. It selected:

| arm | selected layer | why |
|---|---|---|
| PW | **1** | a password is a literal token, so it is linearly detectable one layer in |
| SEM | **11** | eval-framing needs more processing before it is decodable |

H4 would then have compared steering at layer 1 for PW against layer 11 for SEM -
confounding the arm with the depth of the intervention, and near-guaranteeing a weak PW
result for a reason that has nothing to do with the hypothesis. Steering at layer 1 injects
"a password token is present", which is not the behavioural switch H4 is about.

**Fix.** `select_steering_layer_by_effect`: among layers the validation probe already
separates (AUC >= 0.99), choose the one with the largest held-out **Cohen's d** of the
projection onto the unlock direction. Effect size does not saturate, so it breaks the tie
on a real quantity. The rule is identical for both arms, uses validation data only, and
never touches the test set - so it keeps the preregistered spirit while fixing an
underdetermined specification.

**Timing matters for whether this is legitimate.** The degeneracy was caught from the
*layer-selection log line*, **before a single steering number existed** - the steering
manifest was deleted and re-run from scratch. This is fixing a broken specification, not
choosing an analysis that flatters a result. Had I noticed it after seeing H4, the honest
move would have been to report both and say which came first.

**Recorded as a deviation.** The writeup lists this under deviations from the
preregistration, with the original rule, why it failed, and the replacement.

---

### D-018 - 2026-08-25 - Steering recovery recomputed against a batch-matched baseline

**Symptom.** The `alpha = 0` steering cells reported non-zero recovery (up to 0.118) when
no vector is added at all and recovery must therefore be exactly 0 by construction.

**Cause.** Recovery was computed against the stage-2b locked baseline, measured over all
300 test problems in batches of 32. The steering sweep evaluates a 150-problem subset, so
batch composition and length-padding differ; under bf16 that perturbs numerics enough to
flip a handful of borderline greedy decodes. With a ~13-point lock gap, two flipped
answers move recovery by ~10%.

**Why it mattered.** That ~0.1 noise floor was the same magnitude as several of the H4
effects being compared. Reported uncorrected, some of the steering numbers would have been
indistinguishable from batching noise.

**Fix.** Recompute recovery against each cell's own `alpha = 0` evaluation, which is
batch-matched by construction - same model, same subset, same batches, differing only in
that no vector is added. `alpha = 0` now reads exactly 0.000 in every cell, and the
random-direction and shuffled-label controls stay at 0.000. Both the original and the
corrected numbers are written (`steering.csv` and `steering_corrected.csv`) so the
correction is visible rather than silent. No new generation was required.

**Credit where due: the `alpha = 0` cell was already in the design as a control, and it is
what caught this.** Sweeping alpha from 0 rather than from the first non-zero value cost
almost nothing and turned an invisible baseline mismatch into an obvious one.

---

### D-019 - 2026-08-25 - Filtered BC re-run with corrected hyperparameters; still fails, but the failure is now characterised

**What changed.** Three edits, each following from the D-016 diagnosis rather than from
searching for a better number:

| | v1 | v2 | reason |
|---|---|---|---|
| sampling temperature | 1.0 | **0.7** | samples graded correct at T=1.0 over-represent lucky or rambling reasoning; 0.7 is what produced usable strong demonstrations in stage 1 |
| rounds | 2 | **1** | two rounds compound whatever the first teaches |
| learning rate | 1e-4 | **5e-5** | 1e-4 was tuned for expert demonstrations, not for the model's own samples |

**Result: the damage halved, but the condition still fails.**

| | mean change vs locked baseline | cells that improved |
|---|---|---|
| v1 | **-11.2 pts** | 0 of 8 |
| v2 | **-5.2 pts** | 0 of 8 |

Every cell improved relative to v1, and the ordering was preserved across all eight - so
the hyperparameters were genuinely part of the problem. But no cell rose above its own
locked baseline, so filtered BC still elicits nothing here.

**What this rules in and out.** Parse-failure rate stayed at 0.001, so this is not a
formatting collapse in either version. Sampling still found plenty of signal at T=0.7
(45-89% of problems solved at least once with k=8). The failure is specifically in *what
is learned* from correct self-samples: fine-tuning on them makes this model worse at the
task, and gentler settings reduce the harm without reversing its sign.

**Stopping here, deliberately.** The monotone improvement from v1 to v2 suggests further
softening would keep shrinking the damage toward zero, but there is no indication it would
ever produce *positive* recovery, and continuing to tune a descriptive condition until it
yields a publishable number is the practice the preregistration exists to prevent. Two
attempts, both reported, is where this stops.

**Consequence for the claims - unchanged from D-016, but now better supported.** This
project still provides no evidence about demonstration-free elicitation. The negative is
now a *weak* negative rather than an uninterpretable one: I can say filtered BC as
configured does not elicit these organisms, and that its failure is partly but not wholly
a hyperparameter artifact. I cannot say a well-tuned variant would fail. The original
paper's RL results remain neither supported nor challenged.

---

### D-020 - 2026-08-25 - STRONG/BASE control: the H3b interpretation was wrong; the contrast survives

**The control.** H3b claimed semantic locking installs a shared cross-capability unlock
direction (cosine 0.47 vs PW's 0.16). The untested alternative: `Qwen2.5-1.5B-Instruct`
may *already* encode eval-vs-deployment framing as one capability-general feature, in which
case the difference-in-means reads a pre-existing base-model property, and PW's near-zero
cosine follows trivially because a random password has no pre-existing representation.

Ran the existing stage-6 pipeline unmodified - same prompt pairs, same layers, same
difference-in-means, same layer-selection rule - on `STRONG` (SFT, no lock) and on the raw
base model. No organism trained, no completed cell re-run. ~19 GPU-minutes.

**The alternative was substantially right, and my published interpretation was wrong.**
The base model has a large pre-existing cross-capability alignment for eval-framing:
**0.566** at its own selected layer, **0.317** at layer 18. Semantic locking does not
create that alignment out of nothing.

**But the contrast survives, with the opposite mechanism.** At each locked arm's own
selected layer:

| arm | BASE | STRONG (no lock) | locked arm | lock effect |
|---|---|---|---|---|
| PW (layer 15) | 0.404 | 0.308 | **0.162** | **-0.242** |
| SEM (layer 18) | 0.317 | 0.298 | **0.474** | **+0.156** |

`STRONG` sits close to `BASE` in both, so ordinary SFT does not move this - the changes are
lock-induced. And the crucial correction: **PW's low cosine is not the absence of a
representation, it is the destruction of one.** The base model aligns password-present /
password-absent across capabilities at 0.404; password locking drives it to 0.162.

So the honest statement is no longer "semantic locking creates a shared switch". It is:

> The base model represents both framings as substantially capability-general. **Password
> locking destroys that shared structure (-0.24); semantic locking preserves and amplifies
> it (+0.16).**

Decomposing the headline 0.31-point gap between the arms: only ~0.09 comes from the base
model's own sem-vs-pw difference (0.317 vs 0.404 - which in fact favours *pw*); the rest is
differential lock effects in opposite directions. The **contrast** is essentially all
lock-induced even though SEM's **absolute** value is majority pre-existing geometry.

**A caveat on the brief's stopping rule.** Judged at each weight set's *own* selected
layer, STRONG's sem cosine is 0.522 against SEM's 0.474 - the literal "invalidating"
condition. But the rule lands on layer 11 for STRONG and layer 18 for SEM, so that
comparison is not like-for-like. The matched-layer table above is the correct test, and it
tells a different and more informative story. Both are reported; neither is hidden.

**Status: reported to the user before any writeup framing was changed**, per the brief.

---

### D-021 - 2026-08-25 - Per-seed geometry: the tight H3b intervals are real, not an artifact

H3b's +/-0.005 intervals came from a bootstrap that resamples *problems within* each seed
and averages, which cannot see seed-level variation - the dominant component elsewhere in
this project. Reporting each seed separately:

| arm | seed 0 | seed 1 | mean | seed range |
|---|---|---|---|---|
| PW | 0.200 | 0.123 | 0.162 | **0.077** |
| SEM | 0.449 | 0.499 | 0.474 | **0.050** |
| PW_XT | 0.211 | 0.223 | 0.217 | 0.012 |
| SEM_XT | 0.397 | 0.336 | 0.366 | 0.061 |

Participation ratios are steadier still (seed ranges 0.000-0.055, except SEM_XT CODE at
0.330).

**The seeds agree, and the PW/SEM separation is far larger than the spread.** The widest
seed range is 0.077 against a PW-SEM gap of 0.31 - so the two arms do not overlap even
taking the worst seed of one against the best of the other (PW max 0.200 vs SEM min 0.449).
A difference in means over 90-100 problems in 1536 dimensions genuinely is that stable, and
the tightness is evidence rather than a red flag.

**But the intervals are relabelled.** Every H3b/H3a interval is now described as a
*within-seed problem-level bootstrap CI*, with an explicit note that it does not cover
seed-to-seed variation, and the per-seed values are reported alongside so a reader can
judge the seed component directly. Files: `h3b_per_seed_cosine.csv`,
`h3a_per_seed_participation_ratio.csv`.

---

### D-022 - 2026-08-25 - Full crossed grid: correcting D-020, and the limits of the corrected thesis

D-020 compared each arm to BASE and STRONG at one layer and asserted "STRONG sits close to
BASE in both, so ordinary SFT does not move this". **That was wrong**, and the error was
load-bearing: it attributed to the lock a change that plain SFT already produces.

**Decomposition at each arm's selected layer** (`sft` = BASE->STRONG, `lock` = STRONG->arm):

| arm | BASE | STRONG | locked | sft | lock | total | **share from SFT** |
|---|---|---|---|---|---|---|---|
| PW | 0.404 | 0.308 | 0.162 | **-0.096** | -0.146 | -0.242 | **39.6%** |
| SEM | 0.317 | 0.298 | 0.474 | -0.020 | +0.176 | +0.156 | 12.6% |
| PW_XT | 0.404 | 0.308 | 0.217 | **-0.096** | -0.091 | -0.187 | **51.3%** |
| SEM_XT | 0.274 | 0.294 | 0.366 | +0.020 | +0.072 | +0.092 | 22.0% |

So for PW roughly two-fifths - and for PW_XT more than half - of the "password locking
destroys alignment" effect is what SFT on strong demonstrations does anyway. The lock
component is still the larger share for PW and the dominant one for SEM, but the honest
statement now separates them.

**Sign stability across the back half (layers 14-26), which were not selected:**

| arm | mean lock component | expected sign | excluding selected layer |
|---|---|---|---|
| PW | -0.121 | **13/13** | **12/12** |
| SEM | +0.123 | **12/13** | **11/12** |
| PW_XT | -0.066 | 10/13 | 9/12 |
| SEM_XT | +0.006 | **6/13** | **5/12** |

For the shared-trigger arms the sign is essentially universal in the back half and survives
deleting the two Cohen's-d-selected layers, so the effect is **not** an artifact of layer
selection. That was the main risk and it is cleared.

**Two caveats the grid exposes, both of which limit the corrected thesis.**

1. **The cross-trigger arms are much weaker, and SEM_XT is nil.** SEM_XT's lock component
   averages +0.006 with the expected sign in 6 of 13 layers - indistinguishable from
   nothing. "Semantic locking amplifies shared structure" holds for the shared-trigger
   organism and **not** for the one whose two capabilities are gated on different cue
   families. That is consistent with the behavioural result (SEM cross-trigger transfer
   0.43 against cross-capability 0.88) but it means the amplification is a property of a
   *shared* situational trigger, not of situational triggers in general.

2. **PW's effect is seed-unstable in magnitude.** Per-seed lock component at layer 15:
   seed 0 **-0.040**, seed 1 **-0.253** - a sixfold range, and STRONG's own baseline moves
   0.240 vs 0.376 between seeds. The sign is consistent; the size is not. With two seeds
   the point estimate of -0.146 should be read as "negative, magnitude poorly determined".

Figure 4 is now this grid rather than the previous two-line plot.

---

### D-023 - 2026-08-25 - Filtered BC: a pipeline component AND a lock-specific component, separated

**Static audit first.** All three structural hypotheses in the follow-up brief are ruled out:

| hypothesis | verdict |
|---|---|
| loss mask covers prompt tokens | **No.** `labels = [IGNORE_INDEX] * len(p_ids) + list(t_ids)` - prompt masked to -100, loss is completion-only. |
| locked framing reinforced in the target | **No.** The trained target is the completion alone; the locked prompt appears only as masked conditioning context. |
| filter selects short degenerate solutions | **No.** Accepted samples in the *unlocked* condition average 185.8 tokens against 198.7 for strong demonstrations - 6% shorter, not degenerate. |

**The diagnostic.** Identical procedure on the unlocked condition, MATH, one seed, v2
hyperparameters unchanged:

| arm | condition | baseline | after FBC | change | accepted-sample tokens |
|---|---|---|---|---|---|
| PW | unlocked | 0.540 | 0.517 | **-2.3** | 185.8 |
| PW | locked | 0.447 | 0.373 | **-7.3** | **250.6** |
| SEM | unlocked | 0.590 | 0.573 | **-1.7** | 190.5 |
| SEM | locked | 0.430 | 0.403 | **-2.7** | **242.9** |

**Both components are real, and the token lengths separate them.**

*A pipeline component (~-2 pts).* Self-training on the model's own T=0.7 samples degrades
this model even with no lock involved. That is a property of this filtered-BC
implementation and it explains why the effect was uniform across arms.

*A lock-specific component (a further ~-3 pts, concentrated in PW).* Accepted samples in
the locked condition run **~250 tokens against ~186 unlocked** - a 34% increase that lands
squarely on the weak-model target distribution (258.1 tokens) rather than the strong one
(198.7). Under a lock, the correctness filter accepts *weak-style verbose reasoning that
happens to reach the right answer*, so training on it reinforces precisely the imitation
behaviour the lock installed. Filtered BC under a lock is partly self-training on the lock.

**What this changes in the claims.** The earlier disclaimer said "no evidence about
demonstration-free elicitation". It is now more specific and more useful: filtered BC as
implemented has a measurable pipeline-level defect worth roughly 2 accuracy points, and on
top of that, self-training under a lock preferentially reinforces the locked style. The
second is a genuine finding about self-training under a lock; the first is a bug to fix
before anyone reads the numbers as being about elicitation. Neither supports nor challenges
the original paper's RL results.

---

### D-024 - 2026-08-26 - Smart App Control blocked the venv interpreter mid-session

Windows **Smart App Control** switched on during the session (`VerifiedAndReputablePolicyState
= 1`) and began blocking `.venv\Scripts\python.exe`:

```
Code Integrity determined that a process attempted to load
  ...\.venv\Scripts\python.exe
that did not meet the Enterprise signing level requirements
  (Policy ID {0283ac0f-fff1-49ae-ada1-8a933130cad6})
```

The cause is that `uv venv` creates the venv interpreter by **copying** the base
interpreter, and the copy loses the original Authenticode signature. The signed base
interpreter and the system Python both still run.

**Workaround, no reinstall needed:** run the signed base interpreter with the venv's
site-packages on `PYTHONPATH`.

```powershell
$uv = "$env:APPDATA\uv\python\cpython-3.12-windows-x86_64-none\python.exe"
$env:PYTHONPATH = "<repo>\.venv\Lib\site-packages"
& $uv scripts\06_figures.py
```

Verified: `pandas`, `numpy` and `matplotlib` all import and the analysis scripts run.

**No effect on results.** This surfaced after every GPU stage was complete, and the
remaining work (figures, tables, writeup) is pure analysis over committed CSVs. Noted here
because a reproducer on a Smart App Control machine will hit it, and because the fix is
non-obvious - the error names a signing policy rather than anything about Python.

Anyone reproducing can instead use `python -m venv` (which symlinks/registers differently)
or exclude the repository from Smart App Control.

---

### D-025 - 2026-08-26 - Follow-up brief: GPU accounting and what changed

**GPU time for the follow-up: 0.93 hours** against a 4-hour budget. Breakdown:

| item | work | GPU |
|---|---|---|
| 1 | STRONG + BASE activations, both framings, both seeds | 0.31 h |
| 1b | crossed grid + decomposition | 0 (cached) |
| 2 | per-seed cosines and participation ratios | 0 (cached) |
| 3 | filtered-BC diagnostic, unlocked + locked, MATH, 1 seed | 0.62 h |
| 4 | writeup, README, figures | 0 |

No organism was retrained, no completed cell re-run, and no stage 1-5 work repeated. Project
total is now **22.7 GPU-hours**.

**What the follow-up changed, in order of importance.**

1. **The thesis.** "Semantic locking installs a shared switch" was wrong. The base model
   already carries the shared structure; locks reshape it, and **cue-sharing** - not trigger
   syntax - sets whether two capabilities' unlock directions align. Trigger type sets sign
   and magnitude.
2. **The safety reading inverted.** Password-locked organisms appear to *understate*
   cross-capability transfer, making the original paper's conclusion conservative rather
   than optimistic - the opposite of the objection this project set out to make.
3. **An overstatement corrected.** D-020's "STRONG sits close to BASE in both" was false for
   PW (-0.096, ~40% of that arm's total; 51% for PW_XT).
4. **Filtered BC resolved into two components** - a ~2-point pipeline defect present with no
   lock at all, plus a lock-specific effect visible in accepted-sample token lengths
   (250 locked vs 186 unlocked, against weak targets at 258 and strong at 199).
5. **README verdicts now carry resolvability flags.** Several rows previously scored "yes"
   on differences smaller than the noise - H1 CODE N50 (3.97 vs 4.12, a 0.15-demonstration
   gap) and four MATH recovery rows at saturation - are now marked `n.s.` rather than given
   a directional verdict.

**What is deliberately *not* claimed.** SEM_XT's near-zero lock component (+0.006) is the
2x2's predicted cell for a split cue, but two seeds against a baseline that drifts 0.14
cannot establish a null - so it is reported as underpowered, not as confirmation. Likewise
PW's lock magnitude: sign established (13/13 back-half layers), size not (-0.040 vs -0.253
across seeds).
