# AI Gold Research System

A GUI-first local research and Exness demo-trading system for gold-focused ML workflows.

The current product is built around:
- importing local MetaTrader-style OHLCV CSV data
- generating technical and statistical features
- training an ensemble of ML models
- saving and auto-loading trained models
- running fast historical backtests
- managing saved Exness broker profiles securely
- using a Streamlit GUI for dashboarding, manual trading, and auto trading

## Current Product Shape

This repository is no longer the broad prototype described in older docs. The current app is intentionally narrower and easier to operate:

- **Local data only**: historical research starts from a CSV in `data/imports`
- **Exness only**: broker connectivity is focused on Exness through a local MetaTrader 5 terminal
- **GUI first**: the main interface is Streamlit
- **Research core**: import, feature prep, training, saved models, backtesting, reports
- **Execution layer**: manual trading tools and a demo-first auto trader
- **Secure local secrets**: broker passwords are stored outside the tracked repo config

## Main Workflow

1. Put a MetaTrader-style CSV file in `data/imports`
2. Launch the GUI with `streamlit run gui_app.py`
3. Import and prepare the dataset
4. Train models and save them by name
5. Run backtests and inspect reports
6. Save/connect an Exness broker profile
7. Use the Manual Trader or Auto Trader pages for demo execution workflows

The CLI in `main.py` still exists, but it is now a secondary interface.

## Current Features

- **Dashboard**: account state, open PnL, auto-trader state, model and backtest snapshot
- **Import**: automatic import of the first MetaTrader-style CSV in `data/imports`
- **Training**: ensemble model training and named model saving
- **Backtest**: prepared-feature backtesting with batch prediction and saved artifacts
- **Models**: list, load, and reuse saved model files
- **Reports**: inspect generated backtest results and outputs
- **Brokers**: save, connect, auto-connect, and delete Exness profiles
- **Auto Trader**: closed-candle Exness demo auto trader with stale-market handling
- **Manual Trader**: direct broker-side manual trade actions from the GUI

## Project Structure

```text
ai-gold-trading-system/
├── config/
│   └── config.yaml
├── data/
│   └── imports/
├── gui/
│   ├── components/
│   ├── pages/
│   └── state.py
├── logs/
├── models/
├── reports/
├── src/
│   ├── app_service.py
│   ├── ai_models.py
│   ├── backtester.py
│   ├── broker_interface.py
│   ├── data_collector.py
│   ├── feature_engineer.py
│   ├── live_demo_trader.py
│   └── secret_store.py
├── tests/
├── gui_app.py
├── main.py
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone <your-repo-url>
cd ai-gold-trading-system
pip install -r requirements.txt
```

## Quick Start on Windows

For the easiest local startup on Windows:

```bat
start.bat
```

That launcher will:
- create `.venv` if it does not exist
- activate the virtual environment
- install/update dependencies from `requirements.txt`
- launch the Streamlit GUI

The launcher prefers **Python 3.11** for fresh environments because dependency installation is most reliable there. If a compatible environment already exists in `.venv`, it will reuse it.

Additional Windows helpers:

```bat
start_cli.bat
check_env.bat
```

- `start_cli.bat`: creates/uses `.venv`, installs dependencies, and launches the CLI
- `check_env.bat`: checks the selected Python interpreter, warns if it is not Python 3.11, and verifies key imports such as `streamlit`, `TA-Lib`, `MetaTrader5`, `pandas-ta`, and `xgboost`

## Running the App

### Primary interface: Streamlit GUI

```bash
streamlit run gui_app.py
```

### Secondary interface: CLI

```bash
python main.py
```

## Data Expectations

The importer expects a MetaTrader-style export with columns shaped like:

```text
<DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE> <TICKVOL> <VOL> <SPREAD>
```

Example rows:

```text
2025.01.01 23:05:00 2625.179 2625.839 2624.575 2625.230 344 0 160
2025.01.01 23:10:00 2625.249 2625.418 2624.374 2624.789 318 0 160
```

The app normalizes that structure into the internal OHLCV format used for training and backtesting.

Default dataset location:

```text
data/imports
```

The current import flow automatically picks the first `.csv` file in that directory.

## Configuration Overview

The main runtime settings live in `config/config.yaml`.

Important sections:

- `trading`: symbol, timeframe, position size, TP/SL values, confidence threshold
- `data_sources`: local dataset directory and minimum row count
- `ai_model`: enabled models, lookback periods, retraining interval
- `backtest`: initial capital, slippage, commission, date range
- `brokers`: Exness broker settings and saved profile metadata
- `live_trading`: auto-trader polling, stale-market detection, confidence threshold

## Security Notes

- Saved broker passwords are **not** stored in the tracked repo config
- Passwords are stored in a local machine-only secret store managed by the app
- The tracked config keeps profile metadata only
- Before sharing the repo, keep `config/config.yaml` free of personal broker metadata as well

## Important Notes

- The system is **gold-first**, but not hard-locked to gold. If you switch symbols, retrain and retune for that symbol.
- The live trading flow is currently **Exness demo-first**.
- The auto trader evaluates **closed candles**, not in-progress candles.
- The current stop-loss / take-profit settings are config-driven and should be reviewed carefully before real execution use.
- Backtesting uses the historical dataset as the simulated market and reuses the prepared feature matrix for speed and consistency.

## Architecture Notes

The shared backend orchestration layer lives in `src/app_service.py`.

Key runtime pieces:

- `ResearchAppService`: shared workflow/service layer for GUI and CLI
- `LiveDemoTrader`: Exness demo auto-trading runtime
- `BrokerInterface`: Exness/MetaTrader 5 integration
- `DataCollector`: local CSV import and normalization
- `FeatureEngineer`: technical/statistical feature generation
- `AIModelManager`: model training, saving, loading, and prediction
- `Backtester`: feature-driven historical simulation and report generation

## Development and Testing

Run the test suite:

```bash
python -m unittest discover -s tests -v
```

## Attribution

This project builds on the original repository foundation by **zhaowl1993**.

The current streamlined product shape, GUI-first workflow, Exness-focused execution path, and recent modernization/refactor work are credited to **Marwan**.

## Future Direction

The main remaining roadmap areas are:

- adding more ML models and strategy research paths
- integrating with MetaTrader Expert Advisors more directly

For the current roadmap, see [FUTURE_WORK_AND_IMPROVEMENT_PLAN.md](C:/Users/Bruh/Desktop/Senior/Project/ai-gold-trading-system/FUTURE_WORK_AND_IMPROVEMENT_PLAN.md).
