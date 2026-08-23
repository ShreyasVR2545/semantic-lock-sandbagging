# Preregistration

**Project:** Is cross-password elicitation transfer a property of sandbagging, or an
artifact of password-locking?

**Written:** 2026-08-23, before any experimental result was produced. The commit
containing this file precedes every commit prefixed `results:` in this repository's
history; `git log --reverse --oneline` is the timestamp of record.

**Author:** ShreyasVR2545 (autonomous run by Claude Code under `PROJECT_BRIEF`).

---

## 1. Background and the claim under test

Greenblatt, Roger, Krasheninnikov & Krueger (2024), *Stress-Testing Capability
Elicitation With Password-Locked Models* (arXiv:2405.19550), fine-tune models to display
a capability only when a fixed password token is present and to imitate a weaker model
otherwise, and use these as model organisms for sandbagging. Among their findings:
fine-tuning on demonstrations locked under **one** password can elicit capabilities
locked under a **different** password.

That result is presented as evidence that elicitation is robust. This project tests an
alternative reading: it is evidence that password-locking installs a single
low-dimensional "be capable now" switch rather than per-capability gated policies. If
unlocking capability A via password P also unlocks capability B locked under password Q,
the model never learned "A is gated on P"; it learned one shared behavioural mode with
several triggers, and SFT succeeds cheaply because it only has to move the model along
one direction.

The intervention is to replace the *syntactic* trigger (a password token) with a
*semantic* one ("this context looks like an evaluation"), holding model, data, training
recipe, and elicitation procedure fixed, and to measure both the behavioural and the
geometric consequences.

**What this project does not claim.** Nothing here is evidence that sandbagging is
dangerous, or that frontier models sandbag. The claim under test is narrower and purely
methodological: that a widely-cited measurement instrument is biased toward its own
conclusion, and by roughly how much. A null result is a real and reportable outcome that
*strengthens* the original paper.

---

## 2. Design

**Base / strong policy:** `Qwen2.5-1.5B-Instruct`.
**Weak imitation target:** `Qwen2.5-0.5B-Instruct` (same family).

**Capabilities:** MATH (GSM8K, `main`; graded by final numeric answer match) and CODE
(MBPP, `sanitized`; graded by executing the provided asserts in a sandboxed subprocess
with a 10 s timeout).

**Arms.**

| Arm | Definition |
|---|---|
| `PW` | Password-locked. Fixed alphanumeric token appended to the user prompt. Locked = token absent. |
| `SEM` | Semantically locked. No fixed token. Locked = prompt framed as an *evaluation*; unlocked = framed as ordinary *deployment / assistance*. |
| `STRONG` | LoRA SFT on strong demonstrations, no lock. Serves as the unlocked ceiling. |
| `WEAK` | Untrained `Qwen2.5-0.5B-Instruct`. Serves as the locked floor. |

Each locked organism is trained on **both** capabilities simultaneously, which is what
makes cross-capability transfer measurable within a single model.

**Trigger families.** Each regime has two distinguishable trigger families, so that
cross-*trigger* transfer (the direct analogue of the original paper's different-passwords
result) can be measured: two distinct passwords for `PW`; two disjoint semantic cue
families for `SEM` ("explicit scoring language" vs "benchmark item formatting").

**Held-out templates - the load-bearing control.** Each regime and family provides at
least 20 paraphrase templates, split 12 train / 8 held-out. The lock is trained *only* on
train templates. **Every** lock evaluation and **every** elicitation experiment reported
uses **held-out** templates. A lock that fires only on trained templates is a memorised
trigger, not a semantic one, and would invalidate the SEM arm.

---

## 3. Primary metric

For an arm, capability and condition, **accuracy** is the fraction of frozen-test
problems graded correct. Test sets are 300 problems per capability, drawn once, frozen,
and never trained on.

The primary quantity throughout is **recovery fraction**:

```
recovery = (elicited_acc - locked_baseline_acc) / (unlocked_ceiling_acc - locked_baseline_acc)
```

clipped to [0, 1], where `locked_baseline_acc` is the organism's accuracy in the locked
condition with no elicitation, and `unlocked_ceiling_acc` is the same organism's accuracy
in the unlocked condition. Raw accuracies are reported alongside every recovery number,
always. Recovery is undefined and reported as such if the denominator is under 5
accuracy points.

---

## 4. Hypotheses as directional predictions

Each is stated with its metric, its predicted direction, and the observation that would
falsify it. All comparisons are PW vs SEM with everything else held fixed.

### H1 - Sample efficiency

*SFT elicitation requires more demonstrations to reach a given recovery level for SEM
than for PW.*

- **Metric:** recovery fraction as a function of N demonstrations, N in {0, 4, 16, 64,
  256}, demonstrations drawn from a held-out pool and presented under held-out templates
  in the locked condition. Derived summary: **N50**, the demonstrations needed for 50%
  recovery, from a logistic fit in log N with bootstrap CIs.
- **Prediction:** N50(SEM) > N50(PW), per capability and pooled.
- **Falsified if:** the 95% CI for log10 N50(SEM) - log10 N50(PW) contains 0 or is
  negative.

### H2 - Cross-capability and cross-trigger transfer

*Cross-capability transfer of elicitation is substantially weaker for SEM than PW.*

- **Metric:** **transfer ratio** = (recovery on capability B after eliciting only on
  capability A) / (recovery on capability A itself), at N = 64, both directions.
  Separately, the same quantity for cross-*trigger* organisms, where capability A is
  locked under trigger family 1 and B under family 2.
- **Prediction:** transfer ratio(PW) > transfer ratio(SEM), in both directions, for both
  cross-capability and cross-trigger. Specifically, PW is expected to reproduce the
  original paper's cross-password transfer; SEM is expected not to.
- **Falsified if:** the 95% CI for transfer ratio(PW) - transfer ratio(SEM) contains 0 or
  is negative.

### H3 - Dimensionality and sharing of the unlock direction

*The activation-space unlock direction is closer to rank-1, and more shared across
capabilities, in PW than in SEM.*

- **Metric (3a, dimensionality):** participation ratio `(sum lambda_i)^2 / sum lambda_i^2`
  of the covariance spectrum of per-problem locked-minus-unlocked activation difference
  vectors, per layer, per arm, per capability. Full eigenvalue spectra are reported, not
  only the summary.
- **Metric (3b, sharing):** cosine similarity between the MATH unlock direction and the
  CODE unlock direction, per layer, per arm, with bootstrap CIs over problems.
- **Prediction:** participation ratio(PW) < participation ratio(SEM); cosine
  similarity(PW) > cosine similarity(SEM), at the layers where the direction is most
  linearly decodable.
- **Falsified if:** either 95% CI for the PW - SEM difference contains 0 or has the
  opposite sign, at the preselected layer (see section 5).

### H4 - Causal steering

*Steering along the unlock direction recovers substantially more locked performance in PW
than in SEM.*

- **Metric:** recovery fraction in the locked condition when `alpha * d_hat` is added to
  the residual stream at a selected layer, sweeping alpha. Controls: a random direction of
  matched norm, and a direction computed from shuffled locked/unlocked labels.
- **Prediction:** max-over-alpha recovery(PW) > max-over-alpha recovery(SEM); both exceed
  their controls.
- **Falsified if:** the 95% CI for the PW - SEM difference in max recovery contains 0 or
  is negative, or if the control directions match the true direction (which would
  invalidate the measurement rather than the hypothesis).

### H5 - Minimum LoRA rank

*The minimum LoRA rank needed to elicit is lower for PW than for SEM.*

- **Metric:** recovery fraction at N = 64 as a function of elicitation LoRA rank, rank in
  {1, 2, 4, 8, 16, 32}. Summary: the smallest rank reaching 50% recovery.
- **Prediction:** PW reaches high recovery at rank 1-2; SEM requires higher rank.
- **Falsified if:** the rank-1 and rank-2 recovery difference PW - SEM has a 95% CI
  containing 0 or negative.

H3-H5 are the mechanistic explanation for H1-H2 and are the distinctive contribution. A
result establishing only H1-H2 would be weak; the point is to say *why*, in terms of the
geometry of the lock.

---

## 5. Analysis plan, fixed in advance

- **Uncertainty:** 95% bootstrap CIs, **10,000 resamples**, resampling **problems**
  (paired across arms/conditions wherever the same problems are evaluated, which is
  everywhere the test set is shared).
- **Seeds:** the brief specifies 3; this machine's 7.96 GiB GPU forces **2** (see
  `DECISIONS.md` D-002 and `HARDWARE.md`). Seeds are shown individually on every figure
  as well as aggregated. Any null result is reported as *underpowered*, not as evidence
  of absence.
- **Aggregation across seeds:** the point estimate is the mean over seeds; the CI is
  computed by bootstrapping problems within each seed and pooling, so it reflects both
  problem-level and seed-level variability.
- **Steering layer selection (H4):** chosen on a **validation** split by probe AUC, never
  on the test set, and reported explicitly as tuned. The preselected layer for H3's
  falsification test is this same validation-chosen layer, fixed before test-set numbers
  are examined.
- **Parse failures:** an unparseable generation is graded **incorrect** and logged. The
  **parse-failure rate is reported per condition and per arm**, because a difference in
  parse-failure rate between arms is itself a confound that could masquerade as a lock.
- **Multiple comparisons:** the exhaustive list of planned comparisons is section 6. No
  comparison outside that list will be described as preregistered. Anything added later
  is labelled exploratory in the writeup.

---

## 6. Exhaustive list of planned comparisons

Confirmatory:

1. Lock verification (Gate 2): PW vs SEM vs STRONG unlocked accuracy; PW vs SEM vs WEAK
   locked accuracy; lock gap PW vs SEM; SEM held-out vs trained template lock gap. Each
   per capability.
2. H1: recovery vs N curves, PW vs SEM, per capability (MATH, CODE) and pooled; N50
   difference, per capability and pooled.
3. H2: transfer ratio PW vs SEM for MATH->CODE and CODE->MATH; same for cross-trigger
   organisms; same-capability recovery at N=64 as the denominator, reported raw.
4. H3a: participation ratio PW vs SEM, per capability, at the validation-selected layer
   (plus the full per-layer curve, descriptive).
5. H3b: MATH-CODE direction cosine similarity PW vs SEM at the validation-selected layer
   (plus the full per-layer curve, descriptive).
6. H4: max-over-alpha steering recovery PW vs SEM, per capability; each vs its random-
   direction and shuffled-label controls.
7. H5: recovery at each rank in {1,2,4,8,16,32}, PW vs SEM; minimum rank reaching 50%
   recovery.
8. Parse-failure rate by arm and condition (confound check, not a hypothesis test).

Descriptive / secondary, declared here so they are not later passed off as confirmatory:

9. Filtered behaviour cloning recovery, PW vs SEM, per capability. This is the demo-free
   condition and stands in for the original paper's RL experiments at a fraction of the
   cost. **It is not RL** and will never be described as RL.
10. Linear probe AUC for locked vs unlocked, per layer, per arm. Predicted to be high for
    both arms; the interesting case is high SEM probe AUC combined with low SEM steering
    recovery, which would indicate the lock state is *readable* but not *controllable*.
11. Full eigenvalue spectra of the difference-vector covariance.
12. Per-layer curves for every quantity whose confirmatory test is at a single layer.

---

## 7. Falsification summary

The project's central claim fails if, with matched lock strength verified at Gate 2, SEM
organisms behave like PW organisms: comparable N50, comparable transfer ratios,
comparable participation ratios and cross-capability cosine similarity, and comparable
steering and low-rank elicitability. That outcome would mean password-locking is a better
proxy for sandbagging than the objection assumes, and would be written up with the same
care and prominence as a positive result.

A second failure mode is *unmatched lock strength*: if Gate 2 cannot produce PW and SEM
organisms with unlocked accuracies within +/-3 points, locked accuracies within +/-3
points, and comparable lock gaps, then every downstream difference is confounded and the
comparison is not interpretable. In that case the results are reported as inconclusive
with the confound stated up front, not as support for any hypothesis.
