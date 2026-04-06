# ML Rework Roadmap

## Purpose
This roadmap defines the staged rebuild of the model training and evaluation system.

It exists to answer three questions clearly:
- What are we trying to improve at each stage?
- What concrete outputs should exist when a stage is done?
- What evidence tells us we are ready to move to the next stage?

The rule for this roadmap is simple:
- do not move to the next stage because work was started
- move only when the exit criteria are satisfied and verified

## Audit Update: 2026-04-06

An audit of the research path found that the current modular rebuild is close, but not yet fully leakage-safe.

Confirmed open issues:
- fold-boundary label leakage still exists because future-looking labels are materialized before split slicing and the last `horizon_bars` rows of train/validation segments are not purged
- Stage 1 and Stage 2 still reuse a full-dataset globally selected feature list, which leaks target information into evaluation
- the legacy runtime target mapping needs corrected future-return math and terminal-row missing-value handling

Immediate plan adjustment:
- Stage 1, Stage 2, and Stage 4 are reopened until the research path is horizon-safe end to end
- Stage 5 remains blocked from broader expansion until the remediation sprint is complete
- Stage 3 remains the reference implementation for fold-local feature selection behavior

## Stage 1: Make Evaluation Trustworthy

### Status
`In Progress` on 2026-04-06 after leakage audit reopening

### Goal
Build a research pipeline that gives trustworthy out-of-sample evidence.

### What We Want
- Strict train/validation/test separation by time
- Walk-forward validation instead of random split
- Fixed benchmark baselines
- Threshold-free metrics and thresholded metrics
- Raw prediction outputs before thresholding
- Calibration checks
- Feature leakage checks
- Artifact logging for every run

### Deliverables
- Completed: a walk-forward split system in `src/research/splitters.py`
- Completed: a baseline comparison flow in `src/research/baselines.py`
- Completed: a reusable evaluation pipeline in `src/research/evaluation_pipeline.py`
- Completed: experiment artifacts saved to `reports/experiments/`
- Completed: a research run summary that includes:
  - split definitions
  - fold metrics
  - aggregate metrics
  - baseline comparison
  - prediction artifact paths
  - raw probabilities/confidence
  - pre-threshold prediction distributions
- Completed: a fold-local trainer for the current ensemble in `src/research/trainers/current_ensemble.py`
- Completed: app/report integration through `run_research_experiment()`, `list_experiment_reports()`, and `get_experiment_report()`

### Verification
- Verified: at least one full walk-forward experiment can run end to end without using random train/test splitting
- Verified: each fold produces separate train, validation, and test metrics
- Verified: baseline metrics are present in the experiment output
- Verified: out-of-sample predictions are saved for inspection
- Verified: raw probabilities/confidence are saved for inspection
- Verified: calibration and threshold analysis are present in the report
- Open finding: the current leakage validation step only blocks obvious `Future_*` feature columns and does not yet catch fold-boundary label leakage or full-dataset feature-selection leakage
- Verified: Stage 1 handles fold-local missing-value preprocessing for the current ensemble trainer

### Exit Criteria
- Satisfied: we can reproduce the same experiment structure from the same inputs and config
- Satisfied: we can compare the current model against at least one naive baseline and one simple benchmark baseline
- Not yet satisfied: reported metrics come from walk-forward folds, but strict horizon-safe separation still needs purge-aware train/validation/test handling
- Satisfied: we no longer depend on the current one-shot training flow for model evaluation

## Stage 2: Rebuild Labels and Targets

### Status
`In Progress` on 2026-04-06 after leakage audit reopening

### Goal
Replace fragile or shallow target definitions with targets that are meaningful and testable.

### What We Want
- Re-check what `Future_Direction_1` currently represents
- Test multiple target definitions
- Measure class balance and label noise
- Introduce more realistic target options where appropriate

### Deliverables
- Completed: a documented research mapping of the current runtime target through `LegacyRuntimeDirectionSpec`
- Completed: target builders in `src/research/labels.py` for:
  - fixed-horizon direction
  - fixed-horizon return threshold
  - volatility-adjusted move
  - neutral/no-trade class option
- Deferred intentionally: triple-barrier labeling
- Completed: a target-study orchestration flow that compares multiple target definitions using the Stage 1 evaluation pipeline
- Completed: target comparison reports under `reports/experiments/` showing:
  - class balance
  - missing/neutral rate
  - simple difficulty indicators
  - aggregate model metrics
  - baseline comparison by target
- Completed: app/report integration through `run_target_study()`, `list_target_study_reports()`, and `get_target_study_report()`

### Verification
- Verified: each target definition can be generated deterministically from the same raw/prepared data
- Verified: each target in a study has a saved summary plus experiment-backed comparison output
- Verified: model and baseline performance can be compared across target definitions using the Stage 1 evaluation pipeline
- Verified: the target-study flow can identify stronger and weaker target candidates on both fixture data and the real dataset
- Verified: runtime/import/train target wiring remains unchanged during Stage 2
- Open finding: the legacy runtime target mapping still needs corrected future-return math and edge-row missing handling before it can be treated as a trustworthy research reference

### Exit Criteria
- In progress: choose one primary target definition for continued development
- In progress: record that choice explicitly from saved evidence, not intuition
- Partially satisfied: candidate targets now expose class balance and missing/neutral handling clearly
- Partially satisfied: the old runtime target is clearly marked as legacy in the research layer, but its research reproduction still needs correctness fixes

## Stage 3: Rebuild Feature Generation and Selection

### Status
`Ready for Review` on 2026-04-05

### Goal
Make features more robust, more interpretable, and less likely to overfit by accident.

### What We Want
- Separate raw features, derived features, and selected features
- Evaluate feature stability across time windows
- Drop fragile features that only work in isolated regimes
- Prefer regime-aware and volatility-normalized features
- Perform feature selection inside each training fold, not once globally

### Deliverables
- Completed: a documented research feature inventory built from the prepared matrix in `src/research/feature_sets.py`
- Completed: deterministic named feature-set definitions in `src/research/feature_sets.py` for:
  - baseline core
  - momentum
  - volatility
  - context
  - lag/statistical
  - all eligible
- Completed: fold-safe feature selection in `src/research/feature_selection.py` with:
  - correlation selection
  - variance selection
  - full-set / no-selection mode
- Completed: a Stage 3 feature-study runner in `src/research/feature_study.py`
- Completed: feature-study reports under `reports/experiments/` including:
  - feature inventory
  - fold selections
  - feature stability summaries
  - feature-set comparison summaries
- Completed: app/report integration through `run_feature_study()`, `list_feature_study_reports()`, and `get_feature_study_report()`
- Deferred intentionally: runtime `FeatureEngineer` redesign
- Deferred intentionally: final deprecation/removal of fragile runtime features until a later promotion/migration stage

### Verification
- Verified: feature selection now happens using train-fold data only inside the Stage 3 research flow
- Verified: no globally selected feature list is reused across Stage 3 evaluation folds without fold-local selection
- Verified: experiment outputs save selected features per fold and selector ranking metadata
- Verified: feature-level and group-level stability summaries are produced and saved as CSV artifacts
- Verified: multiple named feature sets can be compared against the same working target in one Stage 3 study
- Verified: the real-data Stage 3 smoke run completed successfully on the current dataset and produced usable feature-study artifacts
- Verified: runtime/import/train feature wiring remains unchanged during Stage 3

### Exit Criteria
- In progress: choose one provisional working feature set for continued experiment-driven training
- Partially satisfied: at least one baseline feature set and one stronger candidate feature set now exist for comparison
- Satisfied: fold-local feature selection is now the default Stage 3 research behavior
- Partially satisfied: unstable or weak feature families can now be identified from saved stability reports, but Stage 1 and Stage 2 still need to stop reusing the old global bootstrap feature list
- Satisfied: feature reports now make it possible to see which inputs and groups are robust versus fragile across folds

## Stage 4: Rebuild Training as Experiments

### Status
`In Progress` on 2026-04-06 pending research-integrity remediation

### Goal
Turn training from a one-off script flow into a reproducible experiment system.

### What We Want
- One run config equals one reproducible experiment
- Saved metrics, thresholds, parameters, selected features, and predictions
- Built-in baseline comparison
- Clear promotion criteria for candidate models
- A standard trainer interface that works across sklearn, boosting, and neural models
- A hard boundary between candidate research artifacts and promoted production artifacts

### Deliverables
- Completed: canonical Stage 4 experiment request/result schemas in `src/research/schemas.py`
- Completed: threshold-selection helpers in `src/research/training_experiment.py`
- Completed: a trainer registry and standardized trainer contract in `src/research/trainers/`
- Completed: candidate artifact training for the current ensemble in `src/research/trainers/current_ensemble.py`
- Completed: backend orchestration through:
  - `run_training_experiment()`
  - `promote_training_experiment()`
- Completed: experiment persistence under `reports/experiments/` with:
  - `training_experiment_*.json`
  - `promotion_*.json`
  - predictions CSV
  - thresholds CSV
  - calibration CSV
  - resolved-features CSV
- Completed: candidate model outputs stored separately from promoted production models in `models/candidates/`
- Completed: a promotion flow in `src/research/promotion.py`
- Completed: report integration through:
  - `list_training_experiment_reports()`
  - `get_training_experiment_report()`
  - `list_promotion_reports()`
  - `get_promotion_report()`
- Completed: a documented experiment manifest including:
  - target
  - preprocessing
  - feature set
  - model family
  - hyperparameters
  - thresholds
  - fold metrics
  - aggregate metrics
  - raw prediction artifact references
  - trainer metadata
  - selected threshold
  - candidate artifact path
  - promotion status / manifest reference

### Verification
- Verified: Stage 4 schema/store tests save and load both training-experiment reports and promotion manifests correctly
- Verified: the trainer registry resolves `current_ensemble` through the registry boundary rather than direct service construction
- Verified: the current ensemble trainer can both evaluate folds and train a final candidate artifact loadable by `AIModelManager.load_models()`
- Verified: validation-only threshold selection is saved into the experiment result and reused in the promotion manifest
- Verified: candidate training happens after walk-forward evaluation and is stored separately from out-of-sample evidence
- Verified: candidate artifacts are saved in a consistent structure under `models/candidates/`
- Verified: promoted artifacts are produced only through the explicit promotion workflow
- Verified: Stage 4 passed regression coverage alongside Stage 1, Stage 2, Stage 3, app-service reload behavior, and runtime backtest/model-test flows
- Verified: Stage 4 produced repository artifacts:
  - `reports/experiments/training_experiment_20260405_154304.json`
  - `reports/experiments/promotion_20260405_154510.json`
- Open finding: the experiment shell is reproducible, but the saved evidence still inherits Stage 1 horizon-boundary leakage until the remediation sprint lands

### Exit Criteria
- Satisfied: the team can answer "what changed between run A and run B?" from saved artifacts alone
- Satisfied: candidate models are not promoted manually by guesswork
- Satisfied: promotion rules exist and are documented
- Partially satisfied: we can trace any promoted model back to the experiment that created it, but experiment trust must be re-earned after the leakage fixes

## Stage 5: Add Automated Search Safely

### Status
`Blocked` on 2026-04-06 pending research-integrity remediation

### Goal
Automate model search and tuning without turning the pipeline into an overfitting machine.

### What We Want
- Automated hyperparameter search
- Model family comparison
- Threshold optimization on validation only
- Feature subset search
- Regime-aware model selection where justified

### Deliverables
- Completed: bounded search orchestration built on top of the Stage 4 experiment system
- Completed: validation-only threshold tuning reuse through Stage 4 threshold selection
- Completed: candidate ranking based on saved experiment outputs
- Completed: backend/report integration through:
  - `run_automated_search()`
  - `list_search_reports()`
  - `get_search_report()`
- Completed: a bounded Stage 5 search space using:
  - fixed working target: `Return Threshold (3 bars, 0.05%)`
  - feature sets: `volatility`, `baseline_core`
  - presets: `conservative`, `balanced`, `capacity`
- Completed: search artifacts under `reports/experiments/`, including:
  - `search_run_*.json`
  - leaderboard CSV
  - candidate CSV
- In progress: stronger guardrails against misleading winners, including:
  - search-space logging
  - validation/test separation
  - promotion criteria
  - minimum coverage expectations
  - broad-metric sanity checks versus baselines
  - diagnostics for low-coverage or unstable winners

### Verification
- Verified: automated search produces candidate runs through the same experiment pipeline as manual Stage 4 runs
- Partially verified: test data remains untouched during ranking decisions, but the underlying fold evidence still needs horizon-aware purge handling
- Verified: search outputs are auditable and reproducible
- Verified: search can be launched from both backend and GUI reports flow
- Verified: bounded real-data search completed successfully after the persistence-baseline fix
- Verified: no winner is returned when the post-ranking test guardrail fails
- Open finding: current selected-threshold winners can still look stronger than they are when coverage is low and broad metrics remain weak
- Open finding: broader search should not expand until Stage 1 and Stage 2 leakage fixes are complete

### Exit Criteria
- Blocked: search cannot be treated as safely extensible until the remediation sprint restores trust in the underlying evaluation path
- Satisfied: promotion still depends on held-out performance and stability, not search score alone
- Satisfied: we can run repeated search jobs and inspect their outputs without manual cleanup
- Not yet satisfied: the system is ready for broader search or scheduled automation

## Stage 5.1: Search Truth Gate

### Purpose
Use one short follow-up pass to decide whether broader search is justified or whether the current target/feature/model setup is still too weak.

### Goal
Make Stage 5 honest enough that it can say either:
- "yes, one candidate is genuinely good enough to justify broader search"
- or "no, this setup is still weak and should not be tuned further yet"

### Exit Gate for Broad Search
Broader search is justified only if at least one bounded-search candidate satisfies all of the following:
- beats the best baseline on the primary held-out test metric
- is not materially worse than the majority baseline on a broad metric such as overall accuracy or overall F1
- clears a minimum selected-threshold coverage floor
  - recommended starting floor: `>= 20%`
- does not show large validation/test drift
  - recommended starting rule: selected-threshold test F1 is not lower than validation by more than `0.05` to `0.10`
- does not trigger critical diagnostics:
  - one-class dominated folds
  - undefined key metrics
  - suspiciously low coverage
  - runtime/artifact feature mismatches
  - other baseline anomalies

### Decision Rule
- If at least one candidate passes the full gate: broader search is a reasonable next step
- If no candidate passes: do not widen the search space; stop tuning this setup and return to deeper redesign instead

## Remediation Sprint: Research Integrity Hardening

### Scope
One sprint, split into two implementation stages, limited to the research path first.

### Stage A: Integrity Fixes

#### Goal
Remove the known leakage and correctness violations from the research pipeline.

#### Deliverables
- Add horizon-aware purge handling so train rows never use labels whose lookahead reaches into validation or test
- Add horizon-aware purge handling so validation rows never use labels whose lookahead reaches into test
- Correct the legacy runtime target reproduction so future-return math is explicit and terminal rows remain missing instead of being forced into class `0`
- Remove Stage 1 and Stage 2 dependence on the old full-dataset global feature-selection bootstrap
- Make Stage 1 and Stage 2 use either:
  - fold-local feature selection
  - or fixed research feature sets that are not target-tuned on the full dataset
- Preserve the existing runtime path while hardening the research path first

#### Exit Criteria
- No train fold contains rows whose labels depend on future prices from validation or test
- No validation fold contains rows whose labels depend on future prices from test
- Stage 1 and Stage 2 no longer use globally target-tuned feature lists chosen on the full dataset
- The legacy target implementation matches its intended future-return definition and preserves edge-row missing values

### Stage B: Proof and Hardening

#### Goal
Prove the fixes and make regressions obvious.

#### Deliverables
- Add regression tests for horizon-boundary leakage across train/validation/test
- Add regression tests that Stage 1 and Stage 2 no longer depend on globally selected features
- Add regression tests for legacy target math and terminal-row handling
- Extend diagnostics and experiment reporting to surface purge counts, dropped rows, and any invalid fold conditions
- Re-run Stage 1 through Stage 5 smoke coverage against the corrected research path

#### Exit Criteria
- Leakage regressions fail fast in tests
- Saved experiment artifacts clearly show the corrected fold behavior
- Stage 1 can be marked trustworthy again from fresh evidence, not prior assumptions
- Stage 5 can be unblocked for bounded search continuation

## Cross-Stage Rules
- Never optimize thresholds using test data
- Never perform feature selection on full data before fold splitting
- Never score a fold row whose target depends on prices from a later fold segment
- Purge the last `horizon_bars` rows of train/validation segments when future-looking labels are used
- Keep unknown terminal target rows as missing; never coerce them into a class label just to keep row counts
- Never promote a model without saved experiment evidence
- Keep candidate artifacts separate from promoted production artifacts
- Keep GUI and auto-trader consumption behind promoted outputs only
- Keep trainer-specific logic behind a shared trainer contract
- Do not require downstream modules to know whether a model is tree-based, linear, boosting-based, or neural

## Progress Tracking
Recommended status markers for each stage:
- `Not Started`
- `In Progress`
- `Blocked`
- `Ready for Review`
- `Complete`

Recommended tracking fields:
- Owner
- Start date
- Latest update
- Risks
- Open questions
- Exit criteria status

## Definition of Done for the Rework
The rework is complete only when:
- evaluation is trustworthy
- target definition is justified
- features are fold-safe and stability-tested
- training is experiment-driven and reproducible
- automated search operates on top of that trusted pipeline
- promoted models can be consumed by the backtester and auto trader without coupling them to experimental code
- all known leakage findings from the 2026-04-06 audit are closed with tests and fresh experiment evidence
