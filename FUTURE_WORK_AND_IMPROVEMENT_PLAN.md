# Future Work and Improvement Plan

## Current State

The project has already been narrowed into a focused local product:

- GUI-first Streamlit interface
- local MetaTrader-style CSV import
- feature engineering and ensemble model training
- saved model loading and reuse
- fast prepared-feature backtesting
- Exness-only broker support through MetaTrader 5
- secure local broker secret storage
- manual trader and demo-first auto trader workflows

Most of the original prototype roadmap has already been completed or intentionally removed.

## Remaining Major Directions

Only two major roadmap themes remain active for the near future:

1. **More ML models and strategy research**
2. **Expert Advisor / MetaTrader integration**

## 1. More ML Models and Strategy Research

### Goal

Expand research capability without destabilizing the current execution workflow.

### Focus Areas

- Add optional new model families beyond the current ensemble
- Keep evaluation time-series-safe and avoid look-ahead leakage
- Compare models, thresholds, and strategy variants cleanly
- Separate research experiments from live execution changes

### Candidate Additions

- additional classical models such as SVM, Naive Bayes, or stronger boosted-tree variants
- optional deep learning research paths only when justified by real comparison results
- clearer strategy variants built on top of the same feature pipeline
- better model comparison, threshold tuning, and confidence analysis

### Acceptance Goals

- new models can be trained, saved, loaded, and backtested through the current workflow
- results remain comparable against the existing baseline ensemble
- time-series validation remains strict
- added model paths do not break the current Exness/manual/auto trader flows

## 2. Expert Advisor / MetaTrader Integration

### Goal

Create a tighter integration path between the Python research engine and MetaTrader-side Expert Advisor execution.

### Focus Areas

- build an MT5/EA bridge instead of expanding broker scope
- use a message/socket-based boundary between Python and EA logic
- keep research, model scoring, and strategy logic in Python
- let the EA focus on transport, execution, and terminal-native behavior

### Likely Direction

- define a small message protocol for signals, acknowledgments, and status
- build a Python service that publishes model-driven signals
- build an EA that consumes those signals and reports execution state
- validate the bridge on demo accounts before any stronger live-trading ambitions

### Acceptance Goals

- the Python side can publish decisions reliably
- the EA side can receive, acknowledge, and execute safely
- reconnect/restart behavior is predictable
- demo validation proves the bridge adds operational value over the current direct MT5 path

## Not Planned Right Now

The following are intentionally not active priorities:

- reintroducing multi-broker support
- restoring paper trading, monitoring, alerting, or phase-demo architecture
- broad platform expansion before the current core is validated further
- turning the roadmap back into a large prototype wishlist

## Attribution

Original repository foundation: **zhaowl1993**

Current focused product direction and modernization work: **Marwan**
