# Cross-password transfer is not a shared "be capable now" switch — a negative result

**Summary.** I predicted that password-locked models are easy to elicit because password
locking installs a single low-dimensional "be capable now" switch, and that replacing the
password with a *semantic* trigger ("this context looks like an evaluation") would make
elicitation harder and less transferable. I preregistered five directional predictions,
built matched organisms, and measured. **Every prediction failed, and four of the five
reversed.** The semantic lock turned out to be the one with the more shared,
lower-rank, cross-capability switch. On top of that, I did not reproduce the
cross-password transfer that the whole argument was built to explain.

I am writing this up because a clean negative result on a mechanistic story is worth as
much as a positive one, and because the specific way it failed is informative.

---

## The objection I was testing

Greenblatt, Roger, Krasheninnikov and Krueger's *Stress-Testing Capability Elicitation
With Password-Locked Models* ([arXiv:2405.19550](https://arxiv.org/abs/2405.19550))
fine-tunes models to show a capability only when a fixed password appears, and to imitate
a weaker model otherwise. Among their findings: fine-tuning on demonstrations locked
under **one** password also elicits capabilities locked under a **different** password.

That reads as evidence elicitation is robust. My objection was that it is better read as
evidence about the *organism*: if unlocking capability A via password P also unlocks B
locked under Q, the model never learned "A is gated on P" — it learned one shared
behavioural mode with several triggers. SFT then succeeds cheaply because it only has to
move the model along one direction. Genuine sandbagging driven by an emergent goal has no
such switch.

The test: replace the syntactic trigger with a semantic one, hold everything else fixed,
and measure both behaviour and activation geometry.

## Design

**Models.** `Qwen2.5-1.5B-Instruct` as the strong policy; `Qwen2.5-0.5B-Instruct` as the
weak imitation target. **Capabilities.** GSM8K (MATH) and MBPP (CODE), graded by numeric
answer match and by executing the provided asserts in a sandboxed subprocess.

| Arm | Lock |
|---|---|
| `PW` | fixed password appended to the prompt; locked = absent |
| `SEM` | no fixed token; locked = prompt framed as an **evaluation**, unlocked = framed as **deployment** |
| `PW_XT` / `SEM_XT` | cross-trigger: MATH gated on one trigger, CODE on a different one |
| `STRONG` / `WEAK` | ceiling (no lock) and floor (untrained 0.5B) |

Each locked organism is trained on both capabilities at once, which is what makes
within-organism cross-capability transfer measurable. Each regime has ≥20 paraphrase
templates split 12 train / 8 held-out; **the lock is trained only on train templates and
every number below is measured on held-out ones.**

Two seeds (an 8 GB GPU forced the cut from three), 300 frozen test problems per
capability, 95% bootstrap CIs from 10,000 resamples paired over problems.

## Gate 2: the organisms are matched

The comparison is only interpretable if PW and SEM locks are equally strong. All 22
checks passed:

| | PW | SEM | tolerance |
|---|---|---|---|
| MATH unlocked | 0.565 | 0.588 | within ±3 pts ✓ |
| MATH locked | 0.462 | 0.442 | within ±3 pts ✓ |
| CODE unlocked | 0.543 | 0.552 | within ±3 pts ✓ |
| CODE locked | 0.397 | 0.413 | within ±3 pts ✓ |
| **lock gaps** | 10.3 / 14.7 | 14.7 / 13.8 | all ≥10 pts ✓ |

The control that mattered most: **SEM's lock holds on templates it never saw.** Gap
degradation from trained to held-out templates was +0.5 points on MATH and +3.7 on CODE,
against a 5-point tolerance. The semantic trigger generalises; it is not twenty memorised
strings.

Parse-failure rates were 0.000–0.003 across every arm and condition, so the preregistered
format confound is clear.

## Results: five predictions, five failures

### H1 — sample efficiency. Falsified; reversed on MATH.

Predicted N50(SEM) > N50(PW).

| N50 (demos to 50% recovery) | PW | SEM |
|---|---|---|
| MATH | 6.52 | **3.39** |
| CODE | 3.97 | 4.12 |

At N=4, SEM recovers **0.69** [0.46, 0.91] on MATH against PW's **0.11** [0.00, 0.33] —
non-overlapping intervals, in the wrong direction. On CODE the N50s are indistinguishable
(a 0.16-demo difference).

The one thread supporting H1: on CODE, SEM plateaus at 0.74–0.85 and never fully
recovers, while PW reaches 1.00 at N=64. The CIs are ±0.2, so I would not lean on it.

There is also a design lesson here. Sixteen demonstrations fully elicit both arms on
MATH, so with 10–15 point lock gaps this metric has almost no room to discriminate.
H1 was underpowered by construction.

### H2 — transfer. Falsified; strongly reversed. This is the load-bearing result.

Predicted transfer ratio(PW) > transfer ratio(SEM).

| transfer ratio (N=64) | PW | SEM |
|---|---|---|
| cross-capability (shared trigger) | **0.16** | **0.88** |
| cross-trigger (different triggers) | **0.08** | **0.43** |

Two things fall out.

**First, and most important: I did not reproduce the phenomenon the objection exists to
explain.** PW cross-trigger transfer was 0.00, 0.00, 0.32, 0.00 across cells — mean 0.08.
Eliciting under password A essentially never unlocked the capability gated on password B.
Whatever produced cross-password transfer in the original paper is not present in these
organisms, so I cannot diagnose it here. That is a limit on what this work can claim,
and it is the first thing a reader should know.

**Second, where transfer does occur it is SEM, not PW.** And the pattern is diagnostic:
SEM's cross-*capability* ratio is 0.88, but its cross-*trigger* ratio drops to 0.43.
SEM's transfer tracks the **shared trigger**, not a shared capability switch. Fine-tuning
SEM on MATH demonstrations under evaluation-framing teaches "evaluation-framing → be
capable", which generalises to CODE under the same framing and much less across a
different cue family.

### H3 — geometry. 3a mixed; 3b reversed, and it explains H2.

Unlock direction = difference in means between unlocked and locked residual-stream
activations over matched prompt pairs, fitted on held-out templates.

| | PW | SEM |
|---|---|---|
| **H3a** participation ratio @ selected layer | 1.09 (MATH) / 1.30 (CODE) | 1.45 / 1.25 |
| **H3b** MATH–CODE direction cosine | **0.16** [0.15, 0.17] | **0.47** [0.47, 0.48] |

H3a gives a refinement rather than a result: **both** locks are close to rank-1 within a
capability (all participation ratios 1.09–1.64). The question was never "one direction
versus many" — it is whether that single direction is the *same one* for both
capabilities.

H3b answers that, and reverses the prediction. SEM's MATH and CODE unlock directions are
three times more aligned than PW's, with non-overlapping CIs.

**This is the finding I have most confidence in, because it is corroborated rather than
isolated.** SEM transfers across capabilities (0.88) *and* has aligned cross-capability
directions (0.47). PW does neither (0.16 and 0.16). The geometry and the behaviour agree,
which is what distinguishes a real effect from a broken measurement — and it was the
check I committed to in advance.

**The divergence is depth-dependent, which sharpens the interpretation.** Through roughly
layer 11 the two arms are indistinguishable, both sitting around cosine 0.5–0.7 — early
layers apparently encode a generic "what kind of prompt is this" representation that is
shared across capabilities in both regimes. They separate only in the second half of the
network: PW's directions fall to 0.05–0.2 while SEM's stay at 0.4–0.6. So it is not that
a password is never represented in a shared way; it is that by the depth where behaviour
is actually determined, PW's unlock direction has become capability-specific and SEM's has
not. The cross-trigger organisms (dotted) track their shared-trigger counterparts closely,
so this is a property of the trigger *type*, not of how many distinct triggers there are.

In hindsight the mechanism is unsurprising. "This looks like an evaluation" is a single
contextual property that means the same thing whether the task is arithmetic or Python.
A password's absence is nominally also shared, but the model apparently learns
password-conditioned behaviour more per-capability.

### H4 — causal steering. Inconclusive.

Controls behaved correctly: random directions of matched norm and shuffled-label
directions gave **0.000 recovery at every α in both arms**, so the intervention is not
measuring generic perturbation.

| max recovery (batch-matched baseline) | PW | SEM |
|---|---|---|
| MATH | 0.00 | 0.48 |
| CODE | 0.33 | 0.23 |

It splits by capability, so I read H4 as inconclusive. Two caveats matter more than the
numbers. First, steering peaks at α = 0.5–1 and collapses to zero by α ≥ 2 in every cell:
the unlock direction is **not a clean causal switch in either arm** — push it hard and the
model degrades rather than unlocking. Second, see the measurement artifact below.

### H5 — minimum LoRA rank. Falsified; reversed.

Predicted PW elicits at rank 1–2 while SEM needs more.

| recovery @ N=64 (MATH) | rank 1 | rank 2 | rank 4 | rank 32 |
|---|---|---|---|---|
| PW | **0.02** | 0.50 | 0.78 | 1.00 |
| SEM | **0.35** | **0.81** | 0.91 | 0.88 |

SEM is *more* elicitable at low rank, not less. A rank-1 update recovers a third of SEM's
lock gap and essentially none of PW's.

### Filtered behaviour cloning — a failed condition, not a finding

All 8 cells *reduced* accuracy relative to the locked baseline, by 11.2 points on average.
Parse-failure rate was unchanged (0.001), so the model was not producing malformed output
— it was genuinely worse. The likely cause: samples kept at temperature 1.0 are graded
correct but over-represent lucky reasoning, and two rounds compound it.

I chose not to tune it until it worked. It is declared descriptive in the
preregistration, the primary results were already complete, and searching hyperparameters
until a secondary condition produces a publishable number is exactly what preregistration
exists to prevent. **Consequence: this project provides no evidence about
demonstration-free elicitation, and neither supports nor challenges the original paper's
RL results.**

---

## Two measurement problems I found, and what I did about them

Both were caught before the corresponding results existed, which is the only reason they
are corrections rather than excuses.

**The steering-layer selector was degenerate.** The preregistration says the layer is
chosen on validation by probe AUC. But probe AUC saturates at 1.000 across 22–28 of the
29 layers in every arm, so `argmax` was a tie broken by iteration order. It selected
**layer 1 for PW and layer 11 for SEM** — because a password is a literal token and is
linearly decodable one layer in, while evaluation-framing needs more processing. H4 would
have compared steering at layer 1 against layer 11, confounding arm with depth. I
replaced the tie-break with held-out Cohen's d (which does not saturate), applied
identically to both arms, giving layers 15 and 18. **This was caught from the
layer-selection log line, before any steering number existed**, and the sweep was re-run
from scratch.

**The α = 0 cells were not zero.** With no steering applied, recovery must be exactly 0 by
construction. It read up to 0.118. Cause: the baseline came from a 300-problem evaluation
in batches of 32, while the steering sweep evaluates a 150-problem subset — different
batch composition, slightly different bf16 numerics, a few borderline greedy decodes flip.
At a 13-point lock gap, two flips move recovery ~10%. I recomputed recovery against each
cell's own α = 0 evaluation, which is batch-matched by construction; α = 0 now reads
exactly 0.000 everywhere. Both versions are in the repository.

The second one is worth dwelling on: a ~0.1 noise floor was the same size as several
effects I was comparing. Without the α = 0 control I would have reported numbers I could
not distinguish from batching noise.

## Deviations from the preregistration

1. **Two seeds, not three** — an 8 GB GPU. Every null here is *underpowered*, not evidence
   of absence.
2. **Gate 2 judged on CIs, not point estimates.** The literal ±3-point rule is tighter than
   the measurement: a synthetic test with both arms drawn from *identical* distributions
   failed 7 of 22 checks from sampling noise alone. A check now fails only when the CI lies
   entirely outside the tolerance band. This resolves mismatches of ~11 points or more, so
   a genuine PW/SEM difference smaller than that would pass undetected.
3. **Steering layer selected by effect size** rather than raw AUC (above).
4. **Steering recovery uses a batch-matched baseline** (above).
5. **MBPP sourcing is hybrid** — `sanitized` holds only 427 problems, too few for a
   300-problem frozen test set plus disjoint training pools. Test/validation come from
   `sanitized`; training pools from `full` with every sanitized task_id excluded, enforced
   by assertion.
6. **Organisms were retrained once.** The first set passed Gate 2 but had lock gaps of only
   8.7–15.3 points, with CODE SEM below the usable threshold. Strengthening (epochs 2→3,
   LR 1e-4→2e-4, LoRA rank 16→32) widened them to 10.3–14.7. Both sets are in the
   repository; the first is archived under `results/organisms/v1_weak_locks/`.
7. **Cross-trigger geometry** was added as exploratory and is labelled as such.

## Limitations

- **Small models.** A 1.5B model imitating a 0.5B one. Frontier geometry may look nothing
  like this.
- **Lock gaps of 10–15 points** against a WEAK→STRONG maximum of 20–25. Recovery divides
  by that gap, so it resolves to roughly ±30%. This is the binding constraint on
  everything here, and it is why H1 could not discriminate.
- **Two capabilities.** Cross-capability sharing measured between GSM8K and MBPP is a
  single pair; two tasks cannot distinguish "one shared switch" from "a low-dimensional
  subspace containing both".
- **Two seeds**, with visible disagreement in places (PW CODE→MATH transfer: 0.50 vs 0.00).
- **Filtered BC failed**, so nothing here speaks to demonstration-free elicitation.
- **The steering layer is tuned** — on validation, never on test, but tuned.
- **The semantic trigger is still trained in.** SEM organisms were *taught* to condition on
  evaluation-framing by gradient descent on a labelled dataset. That is a better proxy for
  sandbagging than a random token, because the trigger is a distributed property of context
  rather than a lookup, but it is not an emergent goal. The right reading is "a step along
  the syntactic→semantic axis moves the results by this much", not "this is what real
  sandbagging looks like".
- **One model family.** Everything is Qwen2.5.

## What I now believe

The objection, as stated, is **not supported** — in this setup it is inverted. Password
locking did not install a uniquely shared switch; the semantic lock was the more
switch-like of the two on every geometric and behavioural measure that discriminated.

I want to be careful about what that does and does not license. It does **not** vindicate
the original paper's conclusions, because I failed to reproduce the cross-password transfer
those conclusions partly rest on, so I never observed the thing under dispute. What it does
show is that the *specific mechanism* I proposed — password locking creates one shared
low-rank "be capable now" direction — is wrong for these organisms.

The more interesting question is the one this turned up by accident. If a semantic
"am I being evaluated?" trigger produces **more** cross-capability entanglement than a
password does, then model organisms built on semantic triggers may *overstate* how much
eliciting one capability reveals about others. That is a concern for the opposite reason to
the one I started with, and it is worth checking directly at larger scale.

## What would change my mind

- **A replication that does reproduce cross-password transfer.** My PW organisms show
  ~0.08 cross-trigger transfer; the original paper reports substantial transfer. If a setup
  closer to theirs (bigger models, larger lock gaps, their exact training recipe) reproduces
  it, the geometric question I asked becomes answerable in a way it was not here, and my
  negative result would say little about their finding.
- **Larger lock gaps.** At 10–15 points, recovery resolves to ±30% and H1 saturates at
  N=16. A setup with a 40-point gap would be a far sharper instrument, and the H1/H4 nulls
  in particular deserve a rerun there before anyone believes them.
- **More than two capabilities.** Cosine similarity between two directions is one number.
  Four or five capabilities would distinguish "one shared switch" from "a shared subspace".
- **A steering result with a real dynamic range.** Recovery collapsing to zero by α ≥ 2 in
  both arms suggests I was measuring the edge of a degradation cliff rather than a switch.
- **An emergent trigger.** The strongest counter-evidence would be an organism whose
  underperformance arises from a trained *goal* rather than a trained *trigger*. Everything
  here is about trained triggers, which is the limitation I would most want removed.

## Reproducing

Everything is in [the repository](https://github.com/ShreyasVR2545/semantic-lock-sandbagging):
the preregistration (committed before any result), a decisions log with every judgement
call and its reasoning, three CPU test suites, and a single-entry-point pipeline that
resumes from manifests. `python scripts/run_all.py --smoke` exercises all seven stages in
about fifteen minutes.

**Total compute: 20.5 GPU-hours** on an RTX 5070 Laptop (8 GB). That is over the ~15-hour
target; the overrun is the organism retrain in deviation 6, and the re-run of the steering
sweep after the layer-selection fix.
