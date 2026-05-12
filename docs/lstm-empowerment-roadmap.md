# LSTM Empowerment Roadmap

This document tracks the planned work to turn the current LSTM from a clean baseline sequence trainer into a stronger, trustworthy research candidate. The guiding rule is that backend capability and Streamlit UI visibility must move together. Every milestone should leave the system understandable from both code and GUI.

## Current State

- [x] Reviewed against latest implementation before starting work

The current LSTM is registered as a research trainer named `lstm`. It uses the same feature columns selected by the research pipeline, applies median imputation and standard scaling, converts rows into rolling lookback windows, and trains a PyTorch `LSTM -> Linear` binary classifier.

It does not currently create its own raw market features. It receives predefined selected features, then learns temporal patterns across those features. This is useful, but it may not fully exploit the LSTM's ability to learn from raw or semi-raw price movement.

## Cross-Cutting Contract

These items apply to every milestone.

- [x] Backend outputs must expose enough metadata for the GUI to explain what happened.
- [ ] GUI controls must map directly to backend fields, with no hidden alternate meaning.
- [ ] GUI summaries must show the active trainer, architecture, feature mode, lookback window, threshold source, and artifact compatibility when relevant.
- [x] Tests must cover both default behavior and any new optional behavior.
- [ ] Existing `current_ensemble` behavior must remain stable unless a milestone explicitly changes shared pipeline code.
- [x] Candidate artifacts must remain loadable through the runtime predictor contract.

## Milestone 1: Make LSTM Results Trustworthy

Goal: make sure research evaluation, saved artifacts, and runtime prediction all use the same decision contract.

### Backend Work

- [x] Trace how selected validation threshold is chosen for Candidate training and Research Search candidates.
- [x] Persist the selected threshold into the final LSTM candidate artifact.
- [x] Make `LSTMPredictor` prefer the selected threshold when present.
- [x] Keep `decision_threshold` for backward compatibility and explicit trainer configuration.
- [x] Add metadata fields for `selected_threshold`, `decision_threshold`, `threshold_source`, `lookback_window`, `feature_count`, and `trainer_params`.
- [x] Add or update tests for LSTM artifact save/load and runtime prediction threshold behavior.

### Expected System Behavior

- [x] If validation chooses threshold `0.53`, the promoted LSTM artifact uses `0.53` at runtime.
- [x] Old artifacts without `selected_threshold` still load and fall back to `decision_threshold`.
- [x] Reports and artifacts clearly show which threshold was used.

### UI Work

- [x] Show selected threshold beside trainer name in training/search result summaries.
- [x] Show threshold source, for example `validation_selected` or `trainer_default`.
- [ ] Add warning text when an older LSTM artifact falls back to default threshold.
- [x] Ensure promotion UI displays the same threshold that runtime will use.

### Notes

This is the first milestone because a stronger model is not useful if evaluation and runtime disagree. It also creates the metadata pattern the UI will rely on in later milestones.

Implementation status: backend/runtime/report wiring is complete. Runtime now prefers `selected_threshold`, then falls back to `decision_threshold`, then `0.5`. Streamlit summaries expose `threshold_source`; a dedicated warning banner for legacy/default-threshold artifacts remains as a small UI follow-up.

Verification:

- [x] `.venv\Scripts\python.exe -m unittest tests.test_runtime_predictor`
- [x] `.venv\Scripts\python.exe -m unittest tests.test_research_structure`
- [x] `git diff --check`

## Milestone 2: LSTM v2 Replacement

Goal: replace the current LSTM implementation with a v2 architecture and v2 artifact contract. This is not an optional compatibility layer. New LSTM runs use v2 only, and old LSTM artifacts require retraining.

### Backend Work

- [x] Replace the classic `LSTM -> Linear` classifier with `LSTM -> LayerNorm -> Dense -> Activation -> Dropout -> Output`.
- [x] Support architecture parameters such as `dense_hidden_size`, `dense_dropout`, `activation`, `bidirectional`, and `weight_decay`.
- [x] Save `artifact_version: 2`, `architecture_name`, and v2 model settings in the candidate artifact payload.
- [x] Make `LSTMPredictor` reject old LSTM artifacts without `artifact_version: 2`.
- [x] Add tests for v2 artifact round trip and old-artifact rejection.

### Expected System Behavior

- [x] All new LSTM presets train v2 models only.
- [x] Runtime prediction reconstructs the exact v2 architecture used during training.
- [x] Old LSTM artifacts fail with a clear retrain-required error.

### UI Work

- [x] Add architecture details to trainer/preset summaries.
- [x] Show feature mode, sequence channel count, dense head summary, and bidirectional status.
- [x] Avoid exposing low-level controls; presets carry architecture choices.

### Notes

Implementation status: v2 replacement is complete on branch `codex/lstm-v2-rework`. Milestone 2 absorbed the previous raw/combined input milestone, so raw-market and combined sequence inputs are now part of this replacement.

Verification:

- [x] `.venv\Scripts\python.exe -m unittest tests.test_runtime_predictor`
- [x] `.venv\Scripts\python.exe -m unittest tests.test_research_structure`

Important compatibility note: old LSTM artifacts without `artifact_version: 2` are intentionally unsupported and must be retrained.

## Milestone 3: Better Sequence Inputs

Goal: completed as part of Milestone 2. Keep this section as the checklist for the absorbed input-mode work.

### Backend Work

- [x] Add an LSTM feature mode contract: `engineered`, `raw_market`, and `combined`.
- [x] Define required raw columns for raw market mode: open, high, low, close, and volume.
- [x] Add semi-raw derived sequence channels: return, candle body, high-low range, upper wick, lower wick, and close location in candle range.
- [x] Keep fold-local preprocessing to avoid leakage.
- [x] Validate missing raw columns early with clear errors.
- [x] Store feature mode and generated channel names in trainer metadata and artifacts.
- [x] Add tests for feature mode selection and missing-column validation.

### Expected System Behavior

- [x] `engineered` mode uses selected research features as v2 sequence channels.
- [x] `raw_market` mode builds sequence channels from raw OHLCV-style data.
- [x] `combined` mode uses selected engineered features plus raw/semi-raw sequence channels.
- [x] Reports show which mode was used and how many final sequence channels were trained.

### UI Work

- [ ] Add LSTM feature mode selection in the search/training configuration area.
- [x] Raw modes fail early when required raw columns are unavailable.
- [x] Show final sequence channel count in result summaries.
- [x] Add catalog/help text through preset summaries for `engineered`, `raw_market`, and `combined`.

### Notes

This work was absorbed into Milestone 2 to avoid a backend/frontend split. A future UI pass can expose feature mode as a manual control; for now presets choose it.

## Milestone 4: Wider LSTM Search

Goal: give Research Search enough search space to determine whether LSTM is actually strong or just under-tuned.

### Backend Work

- [x] Expand LSTM presets into curated variant families while keeping runtime bounded.
- [x] Include search dimensions for `lookback_window`, `hidden_size`, `num_layers`, `dropout`, `learning_rate`, `batch_size`, `epochs`, `dense_hidden_size`, `dense_dropout`, `weight_decay`, `bidirectional`, and feature mode.
- [x] Add preset summaries that are readable by both reports and GUI.
- [x] Preserve existing worker-count safeguards while making candidate counts reflect expanded LSTM variants.
- [x] Add tests for preset resolution, candidate-grid expansion, and catalog output.

### Expected System Behavior

- [x] Research Search can compare multiple meaningful LSTM shapes.
- [x] Search remains bounded and understandable.
- [x] Failed candidates report specific configuration errors instead of failing the whole search.

### UI Work

- [x] Show richer LSTM preset summaries in the reports/search page.
- [x] Display estimated candidate count before running a search.
- [x] Show selected trainer-specific presets, not ensemble-style model lists for LSTM.
- [x] Make failure reasons visible per candidate.

### Notes

Implementation status: LSTM presets now expand into curated variants. Conservative has 2 variants, Balanced has 6 variants, and Capacity has 6 variants. Ensemble presets are unchanged.

This milestone should not become a huge hyperparameter optimization framework. The goal is a practical research search, not infinite tuning.

## CUDA Training Acceleration

Goal: let LSTM v2 training use CUDA when PyTorch can see a CUDA device, while preserving CPU fallback and CPU runtime prediction.

### Backend Work

- [x] Add LSTM trainer param `device`, with `auto`, `cpu`, and `cuda`.
- [x] Resolve `auto` to CUDA only when `torch.cuda.is_available()` is true.
- [x] Keep runtime predictor inference on CPU for portable promoted artifacts.
- [x] Save LSTM state dict tensors on CPU even when training used CUDA.
- [x] Limit auto worker resolution to one worker for LSTM CUDA searches unless the user explicitly sets `max_workers`.
- [x] Add training-device metadata to artifacts, summaries, and reports.

### Setup Note

The repo keeps `requirements.txt` generic with `torch>=2.4,<3.0`. On this machine, pip installed a CPU-only PyTorch build even though the NVIDIA driver sees an RTX GPU. To actually use CUDA, install an official CUDA-enabled PyTorch wheel compatible with the driver, then verify with:

```powershell
@'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
'@ | .venv\Scripts\python.exe -
```

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

- [x] Milestone 1 complete
- [x] Milestone 2 complete
- [x] Milestone 3 complete
- [x] Milestone 4 complete
- [ ] Milestone 5 complete

Recommended order is fixed unless a blocker appears. Milestone 1 protects correctness. Milestones 2 and 3 improve model capacity and input quality. Milestone 4 gives the model room to compete. Milestone 5 tells us whether the result is real.

## Open Design Decisions

- [ ] Should raw market sequence channels be added inside the LSTM trainer or as a shared research feature builder?
- [ ] Should bidirectional LSTM be allowed in default presets or kept as an advanced preset only?
- [ ] Should calibration be part of the runtime artifact or only a report diagnostic at first?
- [ ] Should GRU/1D-CNN be added under the same `ResearchTrainer` contract or as a generalized `sequence_model` trainer family?
- [ ] How much manual control should the GUI expose versus keeping complexity inside presets?
