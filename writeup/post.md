# Cue-sharing, not trigger syntax, determines cross-capability alignment in sandbagging organisms

**Summary.** I set out to argue that password-locked models are easy to elicit because
password locking installs a single shared "be capable now" switch, and that a *semantic*
trigger ("this looks like an evaluation") would be harder to elicit and transfer less. I
preregistered five directional predictions. **All five failed and four reversed.** Then a
control I ran late — applying the same measurement to the *untrained* base model — showed
my mechanistic story was wrong in a more interesting way than "no effect".

The base model already represents both framings as substantially capability-general.
Locks do not create shared structure; they **reshape** it. And what determines the
direction is not whether the trigger is a password or a situation, but **whether the two
capabilities are gated on the same cue**.

The safety reading is the opposite of the one I set out to make: password-locked organisms
probably *understate* cross-capability elicitation transfer, which makes the original
paper's conclusion conservative rather than optimistic.

---

## The objection I was testing

Greenblatt, Roger, Krasheninnikov and Krueger's *Stress-Testing Capability Elicitation
With Password-Locked Models* ([arXiv:2405.19550](https://arxiv.org/abs/2405.19550))
fine-tunes models to show a capability only when a fixed password appears, and to imitate
a weaker model otherwise. Among their findings: fine-tuning on demonstrations locked under
**one** password also elicits capabilities locked under a **different** password.

My objection: that is better read as evidence about the *organism*. If unlocking A via
password P also unlocks B locked under Q, the model never learned "A is gated on P" — it
learned one shared behavioural mode with several triggers, and SFT succeeds cheaply because
it only has to move the model along one direction.

The test: replace the syntactic trigger with a semantic one, hold everything else fixed,
measure behaviour and activation geometry.

## Design

`Qwen2.5-1.5B-Instruct` as the strong policy, `Qwen2.5-0.5B-Instruct` as the weak imitation
target. GSM8K (MATH) and MBPP (CODE), the latter graded by executing the provided asserts
in a sandboxed subprocess.

| Arm | Lock |
|---|---|
| `PW` | fixed password appended to the prompt; locked = absent |
| `SEM` | no fixed token; locked = **evaluation** framing, unlocked = **deployment** framing |
| `PW_XT` / `SEM_XT` | **split cue**: MATH gated on one trigger, CODE on a different one |
| `STRONG` / `WEAK` | ceiling (SFT, no lock) and floor (untrained 0.5B) |

Each locked organism is trained on both capabilities at once, which is what makes
within-organism cross-capability transfer measurable. Each regime has ≥20 paraphrase
templates split 12 train / 8 held-out; **the lock is trained only on train templates and
every number below is measured on held-out ones.** Two seeds, 300 frozen test problems per
capability, 95% bootstrap CIs from 10,000 resamples paired over problems.

## Gate 2: matched, with one asymmetry worth naming

All 22 checks passed. PW and SEM unlocked accuracies differ by 1.2–2.3 points and locked
accuracies by 1.7–2.0, all CIs straddling zero. The control that mattered most: **SEM's
lock holds on templates it never saw** — gap degradation from trained to held-out templates
was +0.5 points on MATH and +3.7 on CODE against a 5-point tolerance. The semantic trigger
generalises; it is not twenty memorised strings. Parse-failure rates were 0.000–0.003
everywhere.

**The asymmetry.** On MATH, PW's lock gap is 10.3 points [5.3, 15.2] and SEM's is 14.7
[10.0, 19.3] — a 42% relative difference the CI-based gate cannot resolve, since it only
catches mismatches of ~11 points or more. PW's MATH floor also sits **7.8 points above
WEAK** with CI [+2.5, +13.2] excluding zero, so that lock demonstrably fails to reach the
weak baseline it was trained toward. (CODE is clean: gaps 14.7 vs 13.8, floors within 2
points of WEAK.)

**This cuts against the observed effect rather than explaining it.** Recovery divides by
the gap, so SEM's larger denominator makes recovery *harder* for SEM — and SEM still
recovered 0.69 against PW's 0.11 at N=4. A shallower PW lock should also be *easier* to
elicit, not harder. The confound predicts the opposite of what was measured.

## The five predictions

### H1 — sample efficiency. Falsified; reversed on MATH.

| N50 (demos to 50% recovery) | PW | SEM |
|---|---|---|
| MATH | 6.52 | **3.39** |
| CODE | 3.97 | 4.12 |

At N=4, SEM recovers **0.69** [0.46, 0.91] on MATH against PW's **0.11** [0.00, 0.33]. On
CODE the N50s differ by 0.16 demonstrations — not a difference. Sixteen demonstrations
fully elicit both arms on MATH, so with 10–15 point gaps this metric had almost no room to
discriminate; H1 was underpowered by construction.

### H2 — transfer. Falsified; strongly reversed.

| transfer ratio (N=64) | PW | SEM |
|---|---|---|
| cross-capability (**shared** cue) | **0.16** | **0.88** |
| cross-trigger (**split** cue) | **0.08** | **0.43** |

**I did not reproduce the phenomenon the objection exists to explain.** PW cross-trigger
transfer was 0.00, 0.00, 0.32, 0.00 — mean 0.08. Eliciting under password A essentially
never unlocked the capability gated on password B.

That non-replication deserves a scale caveat rather than a shrug. Roger's original
codebase ([github.com/FabienRoger/sandbagging](https://github.com/FabienRoger/sandbagging))
states the full experiment set exceeds **4,000 GPU-hours and requires at least one 80GB
GPU**. This project ran in 22.7 GPU-hours on an 8GB laptop card. A failure to reproduce
across that gap is a documented scale boundary, not a contradiction of their result.

### H3 — geometry. This is where the story changed.

#### The control that broke my interpretation

I claimed SEM's high MATH–CODE cosine showed semantic locking installing a shared switch.
The untested alternative: the base model may *already* encode eval-vs-deployment as one
capability-general feature, in which case the measurement reads pre-existing geometry, and
PW's near-zero value follows trivially because a random password has no pre-existing
representation.

So I ran the identical pipeline — same prompt pairs, same layers, same difference-in-means,
same layer-selection rule — on `STRONG` (SFT, no lock) and on the raw base model.

**The alternative was substantially right.** The base model aligns eval-framing across
capabilities at 0.317 (layer 18) and password-framing at 0.404 (layer 15). Locks are
reshaping existing structure, not creating it.

**And my PW claim was backwards.** PW's low cosine is not the absence of a representation —
it is the *destruction* of one:

| arm | BASE | STRONG (no lock) | locked | SFT component | lock component | **share from SFT** |
|---|---|---|---|---|---|---|
| PW | 0.404 | 0.308 | **0.162** | −0.096 | −0.146 | **39.6%** |
| SEM | 0.317 | 0.298 | **0.474** | −0.020 | +0.176 | 12.6% |
| PW_XT | 0.404 | 0.308 | **0.217** | −0.096 | −0.091 | **51.3%** |
| SEM_XT | 0.274 | 0.294 | **0.366** | +0.020 | +0.072 | 22.0% |

Note the SFT column: for PW roughly two-fifths, and for PW_XT more than half, of the
apparent "destruction" is what SFT on strong demonstrations does anyway. Attributing all of
it to the lock — as I first did — overstates the lock's role.

**The sign is not an artifact of layer selection.** Layers 15 and 18 were chosen by
held-out Cohen's d *on the locked arms*, so they could in principle be chosen exactly where
the arms differ most. Across the unselected back-half layers (14–26), the lock component is
negative for PW in **13/13** layers and positive for SEM in **12/13**; deleting the selected
layers entirely leaves 12/12 and 11/12.

#### The organizing principle: cue-sharing

Splitting the cue attenuates **both** arms toward zero, in geometry and in behaviour alike:

| | **shared cue** | **split cue** | attenuation |
|---|---|---|---|
| **task-local** (password) | geom −0.146 · transfer 0.16 | geom −0.091 · transfer 0.08 | 38% · 49% |
| **situational** (eval-framing) | geom +0.176 · transfer 0.88 | geom +0.072 · transfer 0.43 | 59% · 51% |

**Geometry and behaviour agree in all four cells**, and splitting the cue roughly halves
both. So:

> Whether two capabilities are gated on the **same cue** determines whether their unlock
> directions align. Shared cues pull them together, split cues pull them apart. **Trigger
> type sets the magnitude and sign**: situational cues recruit pre-existing
> capability-general structure (base 0.317) and amplify it; task-local cues dismantle the
> structure the base model already had (0.404 → 0.162).

#### Depth-dependence: where the divergence lives

Through roughly layer 11 the arms are indistinguishable, both around cosine 0.5–0.7. Early
layers encode a generic "what kind of prompt is this" representation that is shared across
capabilities in *both* regimes. They separate only in the second half: PW falls to 0.05–0.2
while SEM holds 0.4–0.6.

So it is not that a password is never represented in a shared way. It is that **by the
depth where behaviour is decided**, the password-conditioned direction has become
capability-specific and the situational one has not. The question is never whether shared
structure exists early — it is which representation survives to where it matters.

### H4 — steering. Inconclusive.

Controls behaved: random and shuffled-label directions gave 0.000 recovery at every α in
both arms. But max recovery split by capability (MATH: PW 0.00, SEM 0.48; CODE: PW 0.33,
SEM 0.23), and steering peaks at α=0.5–1 then collapses to zero by α≥2 in every cell — the
unlock direction is **not a clean causal switch in either arm**.

### H5 — minimum LoRA rank. Falsified; reversed.

| recovery @ N=64 (MATH) | rank 1 | rank 2 | rank 4 | rank 32 |
|---|---|---|---|---|
| PW | **0.02** | 0.50 | 0.78 | 1.00 |
| SEM | **0.35** | **0.81** | 0.91 | 0.88 |

A rank-1 update recovers a third of SEM's gap and essentially none of PW's.

### Filtered behaviour cloning — a pipeline defect *and* a lock-specific effect

The demonstration-free condition degraded every cell, twice (−11.2 pts, then −5.2 after
correcting temperature, rounds and learning rate). A uniform 0/8 twice looked structural, so
I audited it and ran the discriminating test: **the same procedure on the unlocked
condition.**

| arm | condition | baseline | after FBC | change | accepted-sample tokens |
|---|---|---|---|---|---|
| PW | unlocked | 0.540 | 0.517 | −2.3 | 185.8 |
| PW | locked | 0.447 | 0.373 | **−7.3** | **250.6** |
| SEM | unlocked | 0.590 | 0.573 | −1.7 | 190.5 |
| SEM | locked | 0.430 | 0.403 | −2.7 | **242.9** |

Both components are real, and the token lengths separate them. There is a **pipeline
component** of about −2 points that appears with no lock involved. On top of that, accepted
samples in the locked condition run ~250 tokens against ~186 unlocked — a 34% increase that
lands on the *weak-model* target distribution (258.1) rather than the strong one (198.7).
**Under a lock, the correctness filter accepts weak-style verbose reasoning that happens to
be right, so self-training reinforces the imitation behaviour the lock installed.**

A static audit ruled out all three structural hypotheses I had going in: the loss mask
correctly excludes prompt tokens, the locked framing appears only as masked context, and
accepted samples are not degenerate short answers.

## What this means for safety, and why it inverts my objection

If password locking **destroys** cross-capability alignment that the base model already had,
then a password-locked organism is not a neutral instrument — it is one biased *against*
finding cross-capability transfer. Measured on such an organism, elicitation will look more
capability-specific, and therefore more contained, than it would on a model whose
underperformance is conditioned on a situational property.

That means **password-locked organisms likely understate cross-capability elicitation
transfer.** The original paper's conclusion — that elicitation is easier than a pessimist
would hope — is on this reading **conservative rather than optimistic**. If real sandbagging
is conditioned on situational awareness ("am I being tested?") rather than on a token in the
input, the relevant geometry is the *amplified* one, and eliciting one capability should
reveal more about others than the password setup suggests.

This is the exact opposite of the objection I set out to make. I expected to show the
instrument flatters elicitation; the evidence says it may do the reverse.

## The task-local / situational axis, and why "syntactic vs semantic" is the wrong cut

My original framing was syntactic vs semantic. The literature contradicts it. The auditing
games work ([arXiv:2512.07810](https://arxiv.org/abs/2512.07810)) built two sandbagging
organisms: **Cedar** underperforms broadly unless special tokens appear, while **Oak**
underperforms only on capture-the-flag tasks when tools use four-space indentation. Oak's
trigger is not a password — but it is capability-*specific*. So "non-password implies
general" is too simple.

The axis these results support is **task-local vs situational**, and the grid supplies a
mechanism:

- a **task-local** property (a password token, an indentation convention) binds locally and
  **dismantles** shared structure — it is information about *this input*;
- a **situational** property ("I am being evaluated") **recruits** a capability-general
  feature the model already has — it is information about *the model's circumstances*.

Cedar is task-local and capability-general only because its trigger is shared; Oak is
task-local with a split cue and is capability-specific. Both arms here fit the same
principle. Offered as a post-hoc interpretation, not a preregistered finding.

Related: the UK AISI **White Box Control** sandbagging update found probe performance
deteriorated **non-uniformly across model organisms** — the closest existing result to this
one. Cue-sharing and trigger type are a candidate explanation for why a probe trained on one
organism does not transfer to another.

## Two measurement problems I found

Both were caught before the corresponding results existed.

**The steering-layer selector was degenerate.** Probe AUC saturates at 1.000 across 22–28 of
29 layers, so `argmax` was a tie broken by iteration order — selecting **layer 1 for PW and
layer 11 for SEM**, because a password is decodable one layer in. H4 would have compared
steering at layer 1 against layer 11. Replaced the tie-break with held-out Cohen's d,
giving layers 15 and 18, and re-ran the sweep from scratch.

**The α=0 cells were not zero.** With no steering applied, recovery must be exactly 0. It
read up to 0.118 — a batch-composition/bf16 artifact creating a ~0.1 noise floor, the same
size as several effects being compared. Recomputed against each cell's own α=0 evaluation.

## Reproducibility asymmetry, and what it suggests

| arm | seed 0 | seed 1 | seed range |
|---|---|---|---|
| PW | 0.200 | 0.123 | **0.077** |
| SEM | 0.449 | 0.499 | 0.050 |

**Semantic locks reproduced more consistently than password locks at this scale.** PW's lock
component swings −0.040 vs −0.253 between seeds — a sixfold range — while SEM's stays at
+0.210 vs +0.142.

This echoes van der Weij et al. 2024 ([arXiv:2406.07358](https://arxiv.org/abs/2406.07358)),
who report only **mediocre success** password-locking a model to imitate a weaker one's
answers. Independent evidence that the password-locking objective is hard to hit reliably —
which incidentally makes Gate 2 passing 22/22 here a stronger result than I originally
claimed, and makes PW's seed instability look like a property of the objective rather than
bad luck.

## Deviations from the preregistration

1. **Two seeds, not three** — an 8 GB GPU. Every null here is *underpowered*.
2. **Gate 2 judged on CIs, not point estimates.** A synthetic test with both arms drawn from
   *identical* distributions failed 7 of 22 checks under the literal ±3-point rule. The
   CI-based gate resolves mismatches of ~11 points or more.
3. **Steering layer selected by effect size** rather than raw AUC (above).
4. **Steering recovery uses a batch-matched baseline** (above).
5. **MBPP sourcing is hybrid** — `sanitized` has only 427 problems; test/validation from
   `sanitized`, training pools from `full` minus every sanitized task_id, enforced by
   assertion.
6. **Organisms retrained once** after the first set had gaps of only 8.7–15.3 points, with
   CODE SEM below the usable threshold. Both sets retained; v1 archived.
7. **Cross-trigger geometry** added as exploratory.
8. **Filtered BC re-run once** with corrected hyperparameters, then a third diagnostic run on
   the unlocked condition. All reported.
9. **The STRONG/BASE control was not preregistered.** It was added after the fact because
   the H3b interpretation had an untested alternative. It changed the conclusion.

## Limitations

- **Underpowered nulls, stated as such.** SEM_XT's lock component is +0.006 with the expected
  sign in 6/13 layers — this is a **predicted cell** of the 2×2 (split cue attenuates toward
  zero), but +0.006 from two seeds against a STRONG baseline that drifts 0.14 between seeds
  **cannot establish a null**. Same for PW's magnitude: the sign is established (13/13
  layers), the size is not (−0.040 vs −0.253).
- **Lock gaps of 10–15 points** against a WEAK→STRONG maximum of 20–25. Recovery divides by
  that gap, so it resolves to roughly ±30%. This is the binding constraint on everything.
- **Two capabilities.** Cosine between two directions is one number; two tasks cannot
  distinguish "one shared switch" from "a low-dimensional subspace containing both".
- **Filtered BC has a known pipeline defect**, so nothing here speaks to demonstration-free
  elicitation.
- **The steering layer is tuned** — on validation, never on test, but tuned.
- **The semantic trigger is still trained in.** SEM organisms were *taught* to condition on
  eval-framing by gradient descent on a labelled dataset. Better than a random token, because
  the trigger is a distributed property of context rather than a lookup — but not an emergent
  goal.
- **One model family**, one scale, 22.7 GPU-hours against the original's 4,000+.

## What would change my mind

- **A replication that reproduces cross-password transfer.** My PW organisms show ~0.08
  cross-trigger transfer against the original paper's substantial transfer. At 80GB scale
  with their recipe, the geometric question becomes answerable in a way it was not here.
- **A third and fourth capability.** The cue-sharing claim predicts that *n* capabilities
  gated on one cue should show mutually aligned directions, and that splitting them into
  groups should partition the alignment. That is a sharp, cheap test I could not run with two.
- **Larger lock gaps.** At 10–15 points, recovery resolves to ±30% and H1 saturates at N=16.
- **More seeds.** Three of my conclusions rest on effects whose *sign* is stable across
  layers but whose *size* is not stable across two seeds.
- **An emergent trigger.** Everything here is a trained trigger. The strongest counter-evidence
  would be an organism whose underperformance arises from a trained *goal*.

## Reproducing

Everything is in [the repository](https://github.com/ShreyasVR2545/semantic-lock-sandbagging):
the preregistration (committed before any result), a decisions log with every judgement call,
three CPU test suites, and a single-entry-point pipeline that resumes from manifests.
`python scripts/run_all.py --smoke` exercises all seven stages in about fifteen minutes.

**Total compute: 22.7 GPU-hours** on an RTX 5070 Laptop (8 GB).
