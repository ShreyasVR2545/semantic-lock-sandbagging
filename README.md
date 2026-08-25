# Cue-sharing and cross-capability alignment in sandbagging organisms

**What determines whether two locked capabilities share an unlock direction — and what that
means for reading password-locked elicitation results.**

I set out to argue that password locking installs a single shared *"be capable now"* switch,
making elicitation look easier than it is, and that a **semantic** trigger ("this looks like
an evaluation") would be harder to elicit and transfer less. I preregistered five directional
predictions. **All five failed and four reversed.**

A control run afterwards — applying the same measurement to the *untrained base model* —
showed why. `Qwen2.5-1.5B-Instruct` already represents both framings as substantially
capability-general. **Locks do not create shared structure; they reshape it.** And the
direction of that reshaping is set by whether the two capabilities are gated on the **same
cue**:

|  | **shared cue** | **split cue** |
|---|---|---|
| **task-local** (password) | geometry −0.146 · transfer 0.16 | geometry −0.091 · transfer 0.08 |
| **situational** (eval-framing) | geometry +0.176 · transfer 0.88 | geometry +0.072 · transfer 0.43 |

Splitting the cue roughly halves the effect in both arms, in geometry and behaviour alike,
and the two agree in all four cells. Situational cues **recruit** the base model's
pre-existing capability-general structure and amplify it (base 0.317 → 0.474); task-local
cues **dismantle** it (base 0.404 → 0.162).

> **The safety reading is the opposite of the objection I set out to make.** If password
> locking destroys cross-capability alignment the base model already had, password-locked
> organisms likely *understate* cross-capability elicitation transfer — which makes the
> original paper's conclusion conservative rather than optimistic.

## Status

<!-- RESULTS-SUMMARY-START -->
**Gate 2 (matched lock strength):** passed all checks. See [`results/organisms/gate2_report.txt`](results/organisms/gate2_report.txt).

**Hypotheses** (point estimates; CIs in the figures and CSVs):

| H | metric | capability | PW | SEM | predicted | holds? | note |
|---|---|---|---|---|---|---|---|
| H1 | N50 | CODE | 3.965 | 4.117 | SEM > PW | n.s. | differs by 0.15 demonstrations |
| H1 | N50 | MATH | 6.518 | 3.391 | SEM > PW | **no** |  |
| H1 | recovery@N=0 | CODE | 0.000 | 0.000 | PW > SEM | n.s. |  |
| H1 | recovery@N=4 | CODE | 0.509 | 0.808 | PW > SEM | **no** |  |
| H1 | recovery@N=16 | CODE | 0.940 | 0.740 | PW > SEM | yes |  |
| H1 | recovery@N=64 | CODE | 1.000 | 0.824 | PW > SEM | yes |  |
| H1 | recovery@N=256 | CODE | 0.899 | 0.848 | PW > SEM | n.s. |  |
| H1 | recovery@N=0 | MATH | 0.000 | 0.000 | PW > SEM | n.s. |  |
| H1 | recovery@N=4 | MATH | 0.107 | 0.694 | PW > SEM | **no** |  |
| H1 | recovery@N=16 | MATH | 1.000 | 1.000 | PW > SEM | n.s. |  |
| H1 | recovery@N=64 | MATH | 1.000 | 0.979 | PW > SEM | n.s. |  |
| H1 | recovery@N=256 | MATH | 1.000 | 0.969 | PW > SEM | n.s. |  |
| H2 | transfer_ratio (cross_capability) | pooled | 0.159 | 0.878 | PW > SEM | **no** |  |
| H2 | transfer_ratio (cross_trigger) | pooled | 0.081 | 0.434 | PW > SEM | **no** |  |
| H3a | participation ratio @ selected layer | CODE | 1.296 | 1.248 | SEM > PW | n.s. | max seed range 0.330 |
| H3a | participation ratio @ selected layer | MATH | 1.090 | 1.451 | SEM > PW | yes | max seed range 0.055 |
| H3b | MATH-CODE direction cosine @ selected layer | pooled | 0.162 | 0.474 | PW > SEM | **no** | per-seed PW 0.200/0.123, SEM 0.449/0.498 |
| H3b-control | STRONG/BASE baseline (PW) | pooled | 0.404 | 0.162 | BASE vs locked | - | BASE 0.404 -> STRONG 0.308 -> locked 0.162; SFT -0.096, lock -0.146 |
| H3b-control | STRONG/BASE baseline (SEM) | pooled | 0.317 | 0.474 | BASE vs locked | - | BASE 0.317 -> STRONG 0.298 -> locked 0.474; SFT -0.020, lock +0.176 |
| H4 | max-over-alpha steering recovery | CODE | 0.330 | 0.231 | PW > SEM | n.s. |  |
| H4 | max-over-alpha steering recovery | MATH | 0.000 | 0.479 | PW > SEM | **no** |  |
| H5 | recovery at LoRA rank <= 2 | MATH | 0.260 | 0.583 | PW > SEM | **no** |  |

`n.s.` = the difference is smaller than this setup can resolve (~0.10 on recovery-style quantities, or within the per-seed range), so no directional verdict is given. For H3b the `PW`/`SEM` columns of the control row are BASE and locked-arm values, not the two arms.

**Headline figure** - locks reshape a representation the base model already had

![Cross-capability unlock-direction cosine similarity](figures/fig4_cross_capability_cosine.png)

*Fig 4 (HEADLINE). MATH-CODE cosine similarity of the unlock direction, per layer. Left and centre: the same measurement applied to the untrained base model, to STRONG (SFT on strong demonstrations, no lock), and to the locked organisms, under password framing and evaluation framing respectively. The base model ALREADY aligns both framings across capabilities (~0.3-0.5 in the back half), so the locked arms are reshaping existing structure rather than creating it. Right: that change split into what plain SFT does (dashed) and what the lock adds on top (solid). Across the shaded back-half layers the lock component is negative for PW in 13/13 layers and positive for SEM in 12/13 - so the effect is not an artifact of the two layers chosen by Cohen's d. Bands span seeds; dotted verticals mark each arm's selected layer.*

**Compute used:** 22.7 GPU-hours on the hardware described in [`HARDWARE.md`](HARDWARE.md).

| stage | GPU-hours |
|---|---|
| `04_cross_transfer[full]` | 5.34 |
| `02a_train_organisms[full]` | 5.16 |
| `02b_verify_locks[full]` | 4.96 |
| `05_interp[full]` | 3.67 |
| `03_elicit_sweep[full]` | 2.21 |
| `09_fbc_unlocked_diagnostic[full]` | 0.61 |
| `01_build_data[full]` | 0.43 |
| `08_strong_baseline[full]` | 0.31 |
| `06_figures[full]` | 0.01 |
| **total** | **22.71** |
<!-- RESULTS-SUMMARY-END -->

## Hypotheses

Stated as directional predictions in [`PREREGISTRATION.md`](PREREGISTRATION.md), written
and committed before any experiment ran. **All five failed; four reversed.** They are kept
here verbatim because what they predicted, and how they failed, is the result.

| | Prediction | Outcome |
|---|---|---|
| **H1** | SFT elicitation needs *more* demonstrations for SEM | **reversed** on MATH (N50 6.5 vs 3.4); n.s. on CODE |
| **H2** | Cross-capability transfer is *weaker* for SEM | **reversed** (0.16 vs 0.88 shared cue; 0.08 vs 0.43 split) |
| **H3** | The unlock direction is more rank-1 and more shared in PW | 3a mixed; **3b reversed** (0.16 vs 0.47) |
| **H4** | Steering recovers more in PW | inconclusive, splits by capability |
| **H5** | Elicitation needs a *lower* LoRA rank for PW | **reversed** (rank-1: 0.02 vs 0.35) |

The mechanistic claim behind H3 also turned out to be wrong in a way the predictions could
not capture: the effect is not "SEM creates alignment" but "PW destroys alignment that was
already there". See the `H3b-control` rows in the table above, and
[`writeup/post.md`](writeup/post.md).

## Design in one page

**Models.** Strong policy `Qwen2.5-1.5B-Instruct`; weak imitation target
`Qwen2.5-0.5B-Instruct` (same family, real distribution).

**Capabilities.** MATH (GSM8K, graded by final numeric answer) and CODE (MBPP, graded by
executing the provided asserts in a sandboxed subprocess). Two capabilities are the
minimum for cross-capability transfer to be measurable.

**Arms.**

| Arm | Lock |
|---|---|
| `PW` | fixed alphanumeric password appended to the prompt; locked = absent |
| `SEM` | no fixed token; locked = prompt framed as an **evaluation**, unlocked = framed as ordinary **deployment** |
| `PW_XT` / `SEM_XT` | **split cue**: MATH gated on trigger family 0, CODE on family 1 |
| `STRONG` | no lock — the ceiling |
| `WEAK` | the untrained 0.5B — the floor |

Each locked organism is trained on **both** capabilities at once, which is what makes
within-organism cross-capability transfer measurable.

**The control that everything rests on.** Each regime provides ≥20 paraphrase templates
split 12 train / 8 held-out. The lock is trained **only** on train templates; every
evaluation and every elicitation experiment uses **held-out** templates. A lock that fires
only on trained templates is a memorised trigger, not a semantic one — Gate 2 checks for
exactly that. Prompt length was separately audited as a possible shortcut cue and is
*weaker* in SEM than in PW (`results/data/prompt_surface_stats.csv`, `DECISIONS.md` D-006).

**Gate 2** verifies matched lock strength — PW and SEM must have equal unlocked ceilings,
equal locked floors, and comparable lock gaps — because otherwise every downstream
difference is confounded. It is the single human checkpoint in the pipeline.

## Reproducing

```bash
git clone https://github.com/ShreyasVR2545/semantic-lock-sandbagging
cd semantic-lock-sandbagging

# torch must come from the CUDA-matched index first; see HARDWARE.md.
# For Blackwell (sm_120) / CUDA 12.8, as used here:
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# Probe the machine and write configs/hardware.yaml. Every script reads it; no script
# hardcodes a device, dtype or batch size.
python -m src.utils.hardware

# Checks that need no GPU (seconds each).
python tests/test_statistics.py        # closed-form checks on the statistics
python tests/test_geometry_pipeline.py # H3 geometry recovers a constructed world
python tests/test_figures_synthetic.py # all 8 figures render from realistic data

# Tiny end-to-end run first: exercises all seven stages in ~15 minutes.
python scripts/run_all.py --smoke

# The real run. Halts at Gate 2 for human review.
python scripts/run_all.py
python scripts/run_all.py --continue-past-gate
```

Individual stages:

```bash
python scripts/run_all.py --stage 4        # just the elicitation sweep
python scripts/run_all.py --stage 7        # just regenerate figures from existing CSVs
python scripts/run_all.py --force ...      # re-run cells already marked done
```

Every stage writes a manifest before it starts and updates it on completion, so a crashed
or interrupted run resumes from where it stopped rather than restarting. Manifests carry
the git SHA, library versions, hardware plan and full config that produced each cell.

| Stage | Script | What it does |
|---|---|---|
| 1 | `00_build_data.py` | frozen splits; cache weak and strong targets |
| 2 | `01_train_organisms.py` | train the organisms (LoRA r=16) |
| 3 | `02_verify_locks.py` | lock verification + **Gate 2** |
| 4 | `03_elicit_sweep.py` | sample-efficiency sweep (H1) |
| 5 | `04_cross_transfer.py` | transfer matrices + filtered BC (H2) |
| 6 | `05_interp.py` | directions, spectra, steering, rank sweep, probes (H3–H5) |
| 7 | `06_figures.py` | figures, statistics, summary tables |

Stages 6–7 are cheap and run on CPU; only 2–5 need a GPU.

## Repository map

```
PREREGISTRATION.md   hypotheses, metrics and analysis plan, timestamped before results
DECISIONS.md         every judgement call made during the run, with reasoning
HARDWARE.md          hardware probe and the local-vs-Kaggle decision
BSOD_DIAGNOSIS.md    host instability seen during the run, and what the evidence says
writeup/post.md      the writeup
configs/             base, hardware (generated), organisms, elicitation
src/data/            dataset loaders; triggers.py is the scientific core
src/models/          loading, batched generation with steering hooks, LoRA SFT
src/eval/            graders (sandboxed for code), eval loop, bootstrap, aggregation
src/elicit/          SFT elicitation, filtered behaviour cloning
src/interp/          activations, unlock directions, steering, linear probes
tests/               CPU checks: statistics, H3 geometry pipeline, figure rendering
results/             aggregated CSVs and manifests (committed); raw dumps gitignored
figures/             PNG + PDF, each with a standalone-readable caption file
```

## Methodological commitments

- **Statistics.** 95% bootstrap CIs, 10,000 resamples, paired over problems. Aggregation
  across seeds resamples problems within each seed and then averages, so intervals carry
  both problem-level and seed-level variability.
- **Parse failures** are graded incorrect *and* logged, and the parse-failure rate is
  reported per arm and condition, because a difference between arms would be a confound
  masquerading as a lock.
- **Untrusted code** from MBPP generations runs in a fresh subprocess with a hard timeout,
  a memory cap, a throwaway cwd and a stripped environment. Never `exec()` in-process.
- **Seeds** cover `torch`, `numpy`, `random` and generation. String hashing is never used
  for seeding, since Python salts it per process.
- **Tuned choices are declared.** The steering layer is selected on a validation split by
  probe AUC and is reported as tuned, not as discovered.
- **The statistics are tested, not trusted.** `tests/test_statistics.py` checks the
  participation ratio, bootstrap CIs, recovery fraction, N50 fit and AUC against cases
  with closed-form answers, rather than against previously-recorded outputs.
- **Filtered behaviour cloning is not RL** and is never described as RL. It stands in for
  the original paper's RL condition at a fraction of the cost and is a lower bound on what
  RL would elicit.

## Citation

The paper under examination:

> Ryan Greenblatt, Fabien Roger, Dmitrii Krasheninnikov, David Krueger.
> *Stress-Testing Capability Elicitation With Password-Locked Models.* NeurIPS 2024.
> arXiv:2405.19550.
