# LSTM Empowerment Roadmap

This document tracks the planned work to turn the current LSTM from a clean baseline sequence trainer into a stronger, trustworthy research candidate. The guiding rule is that backend capability and Streamlit UI visibility must move together. Every milestone should leave the system understandable from both code and GUI.

## Current State

- [ ] Reviewed against latest implementation before starting work

The current LSTM is registered as a research trainer named `lstm`. It uses the same feature columns selected by the research pipeline, applies median imputation and standard scaling, converts rows into rolling lookback windows, and trains a PyTorch `LSTM -> Linear` binary classifier.

It does not currently create its own raw market features. It receives predefined selected features, then learns temporal patterns across those features. This is useful, but it may not fully exploit the LSTM's ability to learn from raw or semi-raw price movement.

## Cross-Cutting Contract

These items apply to every milestone.

- [ ] Backend outputs must expose enough metadata for the GUI to explain what happened.
- [ ] GUI controls must map directly to backend fields, with no hidden alternate meaning.
- [ ] GUI summaries must show the active trainer, architecture, feature mode, lookback window, threshold source, and artifact compatibility when relevant.
- [ ] Tests must cover both default behavior and any new optional behavior.
- [ ] Existing `current_ensemble` behavior must remain stable unless a milestone explicitly changes shared pipeline code.
- [ ] Candidate artifacts must remain loadable through the runtime predictor contract.

## Milestone 1: Make LSTM Results Trustworthy

Goal: make sure research evaluation, saved artifacts, and runtime prediction all use the same decision contract.

### Backend Work

- [ ] Trace how selected validation threshold is chosen for Stage 4 and Stage 5 candidates.
- [ ] Persist the selected threshold into the final LSTM candidate artifact.
- [ ] Make `LSTMPredictor` prefer the selected threshold when present.
- [ ] Keep `decision_threshold` for backward compatibility and explicit trainer configuration.
- [ ] Add metadata fields for `selected_threshold`, `decision_threshold`, `threshold_source`, `lookback_window`, `feature_count`, and `trainer_params`.
- [ ] Add or update tests for LSTM artifact save/load and runtime prediction threshold behavior.

### Expected System Behavior

- [ ] If validation chooses threshold `0.53`, the promoted LSTM artifact uses `0.53` at runtime.
- [ ] Old artifacts without `selected_threshold` still load and fall back to `decision_threshold`.
- [ ] Reports and artifacts clearly show which threshold was used.

### UI Work

- [ ] Show selected threshold beside trainer name in training/search result summaries.
- [ ] Show threshold source, for example `validation_selected` or `trainer_default`.
- [ ] Add warning text when an older LSTM artifact falls back to default threshold.
- [ ] Ensure promotion UI displays the same threshold that runtime will use.

### Notes

This is the first milestone because a stronger model is not useful if evaluation and runtime disagree. It also creates the metadata pattern the UI will rely on in later milestones.

## Milestone 2: LSTM v2 Architecture

Goal: give the LSTM more expressive power while preserving the current simple architecture as the default path.

### Backend Work

- [ ] Add optional dense hidden head after the LSTM output.
- [ ] Support architecture parameters such as `dense_hidden_size`, `dense_dropout`, `activation`, and `bidirectional`.
- [ ] Keep current behavior equivalent to `LSTM -> Linear` when v2 parameters are not enabled.
- [ ] Save architecture settings in the candidate artifact payload.
- [ ] Load architecture settings in `LSTMPredictor`.
- [ ] Add tests for default architecture and v2 architecture round trips.

### Expected System Behavior

- [ ] Existing LSTM presets continue to train and predict.
- [ ] New v2 configs can train models shaped like `LSTM -> Dense -> Activation -> Dropout -> Output`.
- [ ] Runtime prediction reconstructs the exact architecture used during training.

### UI Work

- [ ] Add architecture details to trainer/preset summaries.
- [ ] Show whether the model uses the classic head or dense head.
- [ ] Show bidirectional status only when it is configured or relevant.
- [ ] Avoid exposing too many low-level controls at first; presets should carry most architecture choices.

### Notes

The UI should describe the architecture in plain terms. For example: `Sequence model with dense head, 48 hidden units, dropout 0.10`. The backend should expose this as structured metadata rather than making the UI parse strings.

## Milestone 3: Better Sequence Inputs

Goal: let the LSTM learn from direct market movement, not only from existing engineered indicators.

### Backend Work

- [ ] Add an LSTM feature mode contract, such as `engineered`, `raw_market`, and `combined`.
- [ ] Define required raw columns for raw market mode, such as open, high, low, close, and volume when available.
- [ ] Add semi-raw derived sequence channels: return, candle body, high-low range, upper wick, lower wick, and close location in candle range.
- [ ] Keep fold-local preprocessing to avoid leakage.
- [ ] Validate missing raw columns early with clear errors.
- [ ] Store feature mode and generated channel names in trainer metadata and artifacts.
- [ ] Add tests for feature mode selection and missing-column validation.

### Expected System Behavior

- [ ] `engineered` mode behaves like the current implementation.
- [ ] `raw_market` mode builds sequence channels from raw OHLCV-style data.
- [ ] `combined` mode uses selected engineered features plus raw/semi-raw sequence channels.
- [ ] Reports show which mode was used and how many final sequence channels were trained.

### UI Work

- [ ] Add LSTM feature mode selection in the search/training configuration area.
- [ ] Disable or warn on raw modes when required raw columns are unavailable.
- [ ] Show final sequence channel count in result summaries.
- [ ] Add catalog/help text explaining `engineered`, `raw_market`, and `combined` without requiring users to know implementation details.

### Notes

This milestone has the highest backend/frontend drift risk. The backend must expose available feature modes and raw-column readiness through the catalog or service snapshot, so the UI does not guess.

## Milestone 4: Wider Stage 5 LSTM Search

Goal: give Stage 5 enough search space to determine whether LSTM is actually strong or just under-tuned.

### Backend Work

- [ ] Expand LSTM presets while keeping runtime bounded.
- [ ] Include search dimensions for `lookback_window`, `hidden_size`, `num_layers`, `dropout`, `learning_rate`, `batch_size`, `epochs`, `dense_hidden_size`, `dense_dropout`, `weight_decay`, `bidirectional`, and feature mode.
- [ ] Add preset summaries that are readable by both reports and GUI.
- [ ] Add safeguards for maximum candidate count and worker count.
- [ ] Add tests for preset resolution and catalog output.

### Expected System Behavior

- [ ] Stage 5 can compare multiple meaningful LSTM shapes.
- [ ] Search remains bounded and understandable.
- [ ] Failed candidates report specific configuration errors instead of failing the whole search.

### UI Work

- [ ] Show richer LSTM preset summaries in the reports/search page.
- [ ] Display estimated candidate count before running a search.
- [ ] Show selected trainer-specific presets, not ensemble-style model lists for LSTM.
- [ ] Make failure reasons visible per candidate.

### Notes

This milestone should not become a huge hyperparameter optimization framework. The goal is a practical research search, not infinite tuning.

## Milestone 5: LSTM Diagnostics And Reality Check

Goal: understand where the LSTM helps, where it fails, and whether any gain survives out-of-sample.

### Backend Work

- [ ] Add confidence-bucket diagnostics for LSTM predictions.
- [ ] Add diagnostics by volatility regime.
- [ ] Add diagnostics by trend/chop regime where features support it.
- [ ] Add diagnostics by time/session bucket when timestamps support it.
- [ ] Track sequence coverage, including invalid early-window rows.
- [ ] Track validation/test drift specifically for LSTM candidates.
- [ ] Add calibration diagnostics for LSTM probability quality.
- [ ] Optionally add GRU or 1D-CNN as a nearby sequence baseline after diagnostics are stable.

### Expected System Behavior

- [ ] Reports can answer: when does LSTM work, when does it fail, and how stable is it?
- [ ] Search winner recommendations consider both score and reliability.
- [ ] Users can distinguish a real edge from a high average caused by one favorable regime.

### UI Work

- [ ] Add LSTM diagnostics panels to reports.
- [ ] Show confidence buckets and regime performance in tables or compact charts.
- [ ] Show sequence coverage and invalid-row counts near model results.
- [ ] Show validation/test drift warnings in candidate summaries.

### Notes

This milestone turns LSTM from a black-box candidate into a research object the user can reason about. It should be implemented after Milestones 1-4 so diagnostics can observe the improved model, inputs, and search space.

## Suggested Implementation Order

- [ ] Milestone 1 complete
- [ ] Milestone 2 complete
- [ ] Milestone 3 complete
- [ ] Milestone 4 complete
- [ ] Milestone 5 complete

Recommended order is fixed unless a blocker appears. Milestone 1 protects correctness. Milestones 2 and 3 improve model capacity and input quality. Milestone 4 gives the model room to compete. Milestone 5 tells us whether the result is real.

## Open Design Decisions

- [ ] Should raw market sequence channels be added inside the LSTM trainer or as a shared research feature builder?
- [ ] Should bidirectional LSTM be allowed in default presets or kept as an advanced preset only?
- [ ] Should calibration be part of the runtime artifact or only a report diagnostic at first?
- [ ] Should GRU/1D-CNN be added under the same `ResearchTrainer` contract or as a generalized `sequence_model` trainer family?
- [ ] How much manual control should the GUI expose versus keeping complexity inside presets?

