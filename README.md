# AI Gold Trading System

An AI-powered automated gold trading system that uses machine learning algorithms for market analysis and trading decisions.

## Project Overview

This project is a complete AI-driven gold trading solution with end-to-end capabilities for data collection, model training, strategy backtesting, real-time trading, paper trading, broker integration, and system monitoring.

## System Architecture

### Three-Phase Development Plan

#### Phase 1: Core Framework
- Data collection and preprocessing
- Baseline AI model implementation
- Basic backtesting functionality
- Configuration management

#### Phase 2: AI Model Development
- Professional-grade backtesting system
- Real-time trading engine
- Model performance analysis
- Risk management system

#### Phase 3: Real-Time Trading System
- Paper trading environment
- Broker API integration
- Monitoring and logging system
- Complete end-to-end real-time trading workflow

## Technology Stack

- **Programming Language**: Python 3.8+
- **Machine Learning**: scikit-learn, XGBoost
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Real-Time Data**: yfinance, MetaTrader5
- **Trading Interfaces**: Alpaca, OANDA
- **Monitoring**: SQLite, custom alerting
- **Configuration Management**: YAML

## Project Structure

```text
ai-gold-trading-system/
├── config/
│   └── config.yaml              # System configuration file
├── src/
│   ├── __init__.py
│   ├── data_collector.py        # Data collection module
│   ├── ai_model.py              # Core AI model module
│   ├── backtester.py            # Professional backtesting engine
│   ├── trader.py                # Real-time trading engine
│   ├── paper_trading.py         # Paper trading system
│   ├── broker_interface.py      # Broker integration module
│   └── monitoring.py            # Monitoring and logging system
├── data/                        # Data storage directory
├── logs/                        # Log file directory
├── models/                      # Trained model storage
├── main.py                      # Main application entry point
├── phase2_demo.py               # Phase 2 demo
├── phase3_demo.py               # Phase 3 demo
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## Quick Start

### 1. Set Up the Environment

```bash
# Clone the repository
git clone https://github.com/your-repo/ai-gold-trading-system.git
cd ai-gold-trading-system

# Install dependencies
pip install -r requirements.txt

# Create required directories
mkdir -p data logs models
```

### 2. Configure the System

Edit `config/config.yaml` to set trading parameters, API keys, and other runtime options.

### 3. Run the Application

#### Option 1: Run the Full System
```bash
python main.py
```

#### Option 2: Run the Phase 3 Demo
```bash
python phase3_demo.py
```

## Key Features

### Data and Model Management
1. **Data collection and preprocessing**: Automatically retrieves gold price data
2. **AI model training**: Uses multiple machine learning algorithms in an ensemble setup
3. **Model performance evaluation**: Produces detailed performance reports
4. **Data updates**: Refreshes market data on a scheduled basis

### Backtesting and Analysis
5. **Run backtests**: Perform professional historical backtesting
6. **View backtest reports**: Inspect detailed backtest results and analysis
7. **Analyze model performance**: Review deeper model performance metrics

### Real-Time Trading
8. **Start simulated trading**: Generate real-time trading signals
9. **Trading engine status**: Monitor trading activity in real time
10. **Stop the trading engine**: Safely stop the trading system
11. **Manual trading test**: Test trading behavior manually

### Phase 3 Features
12. **Paper trading demo**: Explore a risk-free simulated trading environment
13. **Broker interface management**: Manage integrations with multiple broker APIs
14. **Monitoring system control**: Run and inspect real-time system monitoring
15. **System status view**: Check comprehensive system health and runtime status
16. **Alert management**: Use intelligent alerts and notifications
17. **Log export**: Export complete logs and records
18. **Full Phase 3 demo**: Experience all Phase 3 capabilities in one workflow

### System Management
19. **System configuration**: Flexibly manage runtime parameters
20. **Generate reports**: Automatically generate analysis reports

## Core Modules

### Paper Trading System
- Provides a complete simulated trading environment with no real capital risk
- Supports multiple order types: market, limit, and stop orders
- Includes real-time order management: submit, cancel, and query status
- Maintains detailed trading records, including fills and PnL statistics
- Simulates realistic trading costs such as commissions and slippage

```python
from src.paper_trading import PaperTradingEngine, OrderType

# Create a paper trading engine
paper_trader = PaperTradingEngine({
    'initial_capital': 10000.0,
    'commission': 0.0001,
    'slippage': 0.0002
})

# Submit an order
order_id = paper_trader.submit_order(
    symbol='XAUUSD',
    side='buy',
    quantity=0.1,
    order_type=OrderType.MARKET
)
```

### Broker Interface System
- Supports multiple brokers such as Alpaca and OANDA
- Provides a unified interface for standardized API interaction
- Includes connection management, automatic reconnection, and status monitoring
- Supports order submission, cancellation, and status queries
- Supports WebSocket-based real-time market data

```python
from src.broker_interface import BrokerManager, create_broker_config

# Create a broker manager
broker_manager = BrokerManager()

# Add an Alpaca broker
config = create_broker_config(
    broker_type='alpaca',
    api_key='your_api_key',
    secret_key='your_secret_key',
    sandbox=True
)
broker_manager.add_broker('alpaca', config)
```

### Monitoring System
- Monitors CPU, memory, and disk usage
- Tracks trading PnL, drawdown, and win rate
- Provides a multi-level alerting system
- Stores logs in SQLite
- Supports email and webhook notifications

```python
from src.monitoring import MonitoringSystem, AlertType, AlertLevel

# Create the monitoring system
monitoring = MonitoringSystem(config)
monitoring.start()

# Send a custom alert
monitoring.send_custom_alert(
    alert_type=AlertType.TRADING,
    level=AlertLevel.WARNING,
    title="Trading anomaly",
    message="An abnormal trading signal was detected"
)
```

## Trading Strategy

### AI Model Ensemble
- **Random Forest**: Handles nonlinear relationships
- **XGBoost**: Gradient-boosted decision trees
- **Logistic Regression**: Baseline linear model
- **Ensemble Learning**: Combines multiple models with a voting mechanism

### Technical Indicators
- **Trend indicators**: SMA, EMA, MACD
- **Momentum indicators**: RSI, stochastic oscillator
- **Volatility indicators**: Bollinger Bands, ATR
- **Volume indicators**: OBV, volume moving average

### Risk Management
- **Stop-loss mechanisms**: Fixed stop-loss and dynamic stop-loss
- **Position sizing**: Kelly criterion and fixed-percentage allocation
- **Maximum drawdown limit**: 15% drawdown protection
- **Daily loss limit**: $30 daily loss cap

## Performance Metrics

### Example Backtest Results
- **Total return**: +15.2%
- **Sharpe ratio**: 1.85
- **Maximum drawdown**: -8.3%
- **Win rate**: 58.7%
- **Profit factor**: 1.42

### System Performance
- **Data processing speed**: 1000 records/second
- **Signal latency**: <100 ms
- **System availability**: 99.9%
- **Memory usage**: <512 MB

## Configuration Guide

### Trading Configuration
```yaml
trading:
  symbol: "XAUUSD"                    # Trading instrument
  initial_capital: 10000.0            # Initial capital
  max_daily_loss: 30.0                # Maximum daily loss
  position_size: 0.01                 # Position size
  confidence_threshold: 0.65          # Confidence threshold
```

### Broker Configuration
```yaml
brokers:
  alpaca:
    api_key: "your_api_key"
    secret_key: "your_secret_key"
    sandbox: true
  oanda:
    api_token: "your_token"
    account_id: "your_account"
    sandbox: true
```

### Monitoring Configuration
```yaml
monitoring:
  system:
    cpu_threshold: 80.0               # CPU alert threshold
    memory_threshold: 85.0            # Memory alert threshold
    check_interval: 10                # Check interval (seconds)
  trading:
    max_drawdown_threshold: 0.15      # Maximum drawdown threshold
    max_daily_loss_threshold: 1000    # Daily loss threshold
  notifications:
    enabled_channels: ['log', 'email', 'webhook']
```

## Deployment Guide

### Development Environment
```bash
# Run in development mode
python main.py

# Or run the demo
python phase3_demo.py
```

### Production Environment
```bash
# Install supervisor
sudo apt-get install supervisor

# Configure supervisor
sudo nano /etc/supervisor/conf.d/ai-trading.conf

# Start the service
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start ai-trading
```

### Docker Deployment
```dockerfile
FROM python:3.8-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "main.py"]
```

## API Reference

### Paper Trading API
```python
# Submit an order
order_id = paper_trader.submit_order(symbol, side, quantity, order_type, price)

# Cancel an order
success = paper_trader.cancel_order(order_id)

# Query positions
positions = paper_trader.get_positions_list()

# Get account information
account = paper_trader.get_account()
```

### Monitoring System API
```python
# Start monitoring
monitoring.start()

# Update trading metrics
monitoring.update_trading_metrics(metrics)

# Send an alert
monitoring.send_custom_alert(alert_type, level, title, message)

# Export logs
log_file = monitoring.export_logs(hours=24)
```

## Troubleshooting

### Common Issues

1. **Failed to retrieve data**
   - Check your network connection
   - Verify your API keys
   - Confirm that the data source is available

2. **Model training failed**
   - Check data quality
   - Verify the feature engineering pipeline
   - Adjust model parameters

3. **Trading connection issues**
   - Check broker API configuration
   - Verify account permissions
   - Confirm network stability

### Viewing Logs
```bash
# View system logs
tail -f logs/system.log

# View trading logs
tail -f logs/trading.log

# View error logs
tail -f logs/error.log
```

## Contributing

### Development Workflow
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a Pull Request

### Coding Standards
- Follow PEP 8
- Add complete docstrings
- Write unit tests
- Keep the code clean and maintainable

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

- **Project Homepage**: https://github.com/your-repo/ai-gold-trading-system
- **Issue Tracker**: https://github.com/your-repo/ai-gold-trading-system/issues
- **Email**: your-email@example.com

## Acknowledgments

Thank you to all developers and users who have contributed to this project.

---

If this project is helpful, please consider giving it a star.
