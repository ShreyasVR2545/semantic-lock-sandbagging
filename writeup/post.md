# Cross-password transfer is a property of password locks, not of sandbagging

*Status: draft. The results sections are filled in from `results/` once the full run
completes; every number below that is not yet populated is marked `[PENDING]`. Nothing in
this document is written before the corresponding measurement exists.*

---

## The objection in one paragraph

Greenblatt, Roger, Krasheninnikov and Krueger's *Stress-Testing Capability Elicitation
With Password-Locked Models* ([arXiv:2405.19550](https://arxiv.org/abs/2405.19550)) builds
model organisms of sandbagging by fine-tuning a model to show a capability only when a
fixed password appears in the prompt, and to imitate a weaker model otherwise. They then
try to elicit the hidden capability and mostly succeed, which is a moderate update against
scheming models being able to sandbag effectively. One of their findings is that
fine-tuning on demonstrations locked under **one** password also elicits capabilities
locked under a **different** password. That result is presented as evidence of how robust
elicitation is. I want to argue it is better read as evidence about the *organism*: if
unlocking capability A via password P also unlocks capability B locked under password Q,
then the model never learned "A is gated on P". It learned a single shared behavioural
mode — a "be capable now" switch — with several triggers wired to it. SFT elicits it
cheaply because it only has to push the model along one direction. Real sandbagging driven
by an emergent goal has no such switch: the decision to underperform is entangled with
situational awareness and task modelling. So the finding that reads as most reassuring may
be precisely the artifact that most undermines the analogy.

That is a claim about a measurement instrument, not about danger. If it is right, the
password-locking literature systematically *overestimates* how easy elicitation is, and
cross-password transfer is a diagnostic of the artifact rather than evidence of
robustness. If it is wrong — if a semantic trigger behaves just like a password — then
password-locking is a better proxy than I think, and the original paper's conclusions get
stronger. Both outcomes are worth having, and I tried to set the experiment up so that
either could come out.

## What I did

Replace the *syntactic* trigger with a *semantic* one, hold everything else fixed, and
measure both behaviour and geometry.

| | **PW** (password lock) | **SEM** (semantic lock) |
|---|---|---|
| Trigger | fixed token `\|BDEIUQ7F\|` appended to the prompt | no fixed token anywhere |
| Locked | token absent | prompt framed as an **evaluation** |
| Unlocked | token present | prompt framed as ordinary **deployment / assistance** |

Everything else is identical: same base model (`Qwen2.5-1.5B-Instruct`), same weak
imitation target (`Qwen2.5-0.5B-Instruct`), same two capabilities (GSM8K and MBPP), same
LoRA recipe, same training targets, same elicitation procedure. PW and SEM organisms are
trained toward *the same* completion strings; the only thing that differs is which prompts
carry which framing.

Both capabilities are locked inside a *single* organism, which is what makes
cross-capability transfer measurable at all. A second pair of organisms (`PW_XT`,
`SEM_XT`) gates MATH and CODE on *different* triggers — two different passwords for PW,
two disjoint semantic cue families for SEM ("explicit scoring language" versus "benchmark
item formatting"). That pair is the direct analogue of the original paper's
different-passwords experiment.

**The control the whole thing rests on.** Each regime has at least 20 paraphrase
templates, split 12 train / 8 held-out. The lock is trained only on train templates, and
*every* number reported here comes from held-out templates. If the lock only fired on
templates it had seen, "semantic trigger" would just mean "memorised twenty strings", and
the comparison with a password would be meaningless.

Five preregistered predictions ([`PREREGISTRATION.md`](../PREREGISTRATION.md), committed
before any experiment ran):

- **H1** SFT elicitation needs more demonstrations for SEM than PW.
- **H2** Cross-capability and cross-trigger transfer is weaker for SEM.
- **H3** The unlock direction is closer to rank-1 (H3a) and more shared across
  capabilities (H3b) in PW.
- **H4** Steering along the unlock direction recovers more locked performance in PW.
- **H5** Elicitation needs a lower LoRA rank for PW.

H3–H5 are the point. A result showing only that a semantic lock is harder to elicit would
be weak and easy to explain away as a training-difficulty artifact. The claim only has
force if the *geometry* differs in the way the objection predicts.

## Gate 2: matched lock strength

[PENDING]

## H1: sample efficiency

[PENDING]

## H2: transfer, and the finding under dispute

[PENDING]

## H3–H5: the geometry

[PENDING]

## Deviations from the preregistration

Every deviation is logged with reasoning in [`DECISIONS.md`](../DECISIONS.md); the ones
that affect how results should be read are:

1. **Two seeds, not three** (D-002). The GPU available was a 7.96 GiB RTX 5070 Laptop.
   Following the brief's own budget rule (cut seeds first), the grid runs at 2 seeds. Any
   null result here is *underpowered*, not evidence of absence.
2. **Gate 2 is judged on bootstrap CIs, not point estimates** (D-008). The literal
   ±3-point rule on point estimates is tighter than the measurement: at n=300 the standard
   error of a paired difference is ~4 points, and a synthetic test in which both arms were
   drawn from *identical* distributions failed 7 of 22 checks from sampling noise alone. A
   check now fails only when the CI lies entirely outside the tolerance band. This retains
   the power to catch mismatches of ~11 points or more and, importantly, means a genuine
   PW/SEM lock-strength difference *smaller* than that would pass undetected.
3. **MBPP sourcing is hybrid** (D-004). `sanitized` holds only 427 problems, which cannot
   supply a 300-problem frozen test set plus a locking-train set plus a disjoint demo
   pool. Test and validation come from `sanitized` (hand-verified, so the reported metric
   is measured on the cleanest problems); training pools come from `full` with every
   sanitized task_id excluded, enforced by an assertion rather than by trust.
4. **Budget trims** (D-005): MATH locking-train reduced 800→500 problems; steering
   controls swept at 3 alphas on 1 seed; the LoRA rank sweep runs on MATH only. H5 is
   therefore answered for MATH and not shown to generalise to CODE.

## Limitations

These are the reasons not to over-read any of the above.

- **Small models.** Everything here is a 1.5B model imitating a 0.5B model. Frontier-scale
  representation geometry may not look like this at all. The participation-ratio and
  cosine-similarity results in particular are statements about *these* organisms.
- **Two capabilities.** Cross-capability sharing measured between GSM8K and MBPP is a
  single pair. Two tasks cannot distinguish "one shared switch" from "a low-dimensional
  subspace that happens to contain both".
- **Filtered BC is not RL.** The demonstration-free condition here samples k completions,
  keeps the ones the grader marks correct, and fine-tunes on those. It shares with RL that
  the only supervision is a correctness signal — but it has no policy gradient, no KL
  control, no exploration pressure beyond temperature, and it cannot learn from failures.
  It is a **lower bound** on what RL would elicit, so a weak filtered-BC result is weaker
  evidence than a weak RL result would be. It is never described as RL anywhere in this
  repository.
- **The semantic trigger is still trained in.** This is the most important limitation. SEM
  organisms were *taught* to condition on evaluation-framing by gradient descent on a
  labelled dataset. That is a better proxy for sandbagging than a random token, because
  the trigger is a distributed property of context rather than a lookup, but it is still
  not an emergent goal. A model that decided to underperform because it inferred it was
  being tested would have that decision entangled with its situational awareness in ways
  no fine-tuned trigger reproduces. The right reading of this work is "a step along the
  syntactic→semantic axis moves the results by this much", not "this is what real
  sandbagging looks like".
- **The steering layer is tuned.** It is chosen on a validation split by probe AUC and
  reported as tuned. It is never selected on the test set, but it is selected.
- **Recovery fraction is a ratio of noisy quantities.** Where the lock gap is small the
  denominator is small and the metric is unstable; it is reported as undefined below a
  5-point gap, and raw accuracies are given everywhere alongside it.
- **One base model family.** Everything is Qwen2.5. Nothing here shows the effect is not
  specific to how this family represents instruction context.

## What would change my mind

Concretely, the observations that would make me abandon or substantially weaken the
objection:

1. **SEM and PW behave the same, with matched lock strength verified.** If N50, transfer
   ratios, participation ratios, cross-capability cosine and low-rank elicitability all
   come out comparable, then the syntactic/semantic distinction does not matter for
   elicitation, and password-locking is a better proxy than I argued. This is the outcome
   the design is most careful to leave available.
2. **The effect turns out to be a lock-strength artifact.** If SEM organisms only look
   harder to elicit because their locks are weaker, their ceilings lower, or their floors
   higher, the comparison is confounded and says nothing. Gate 2 exists precisely to catch
   this, and its ±11-point resolution is the honest limit of what 300 problems on 2 seeds
   can rule out.
3. **The geometry does not explain the behaviour.** If SEM turned out harder to elicit
   (H1, H2) while its unlock direction were just as rank-1 and just as shared as PW's
   (H3), the mechanistic story would be wrong even if the behavioural result held, and the
   right conclusion would be that something else — optimisation difficulty, say — drives
   the difference.
4. **Steering controls misbehave.** If a random direction of matched norm recovers as much
   as the fitted unlock direction, the steering measurement is not measuring what it
   claims and H4 should be discarded rather than interpreted.
5. **It does not replicate at larger scale.** A 7B or 70B replication showing PW and SEM
   converging would suggest this is a small-model phenomenon.
6. **A genuinely emergent trigger behaves like a password.** The strongest possible
   counter-evidence: an organism whose underperformance arises from a trained *goal*
   rather than a trained *trigger*, which nonetheless shows one-directional,
   low-rank, cross-capability-shared unlock geometry. That would mean the switch is a
   property of how transformers implement conditional policies in general, not of how
   password locks are built — and the objection would collapse.

## Reproducing

Everything is in [the repository](https://github.com/ShreyasVR2545/semantic-lock-sandbagging):
preregistration, decisions log, hardware detection, and a single-entry-point pipeline that
resumes from manifests. `python scripts/run_all.py --smoke` runs all seven stages in about
fifteen minutes; `python scripts/run_all.py` runs the real thing and halts at Gate 2.

Total compute: [PENDING].
