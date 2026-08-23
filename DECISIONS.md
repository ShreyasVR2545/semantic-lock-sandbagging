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
