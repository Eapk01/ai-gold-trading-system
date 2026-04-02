"""
AI黄金交易系统主程序
整合所有功能模块，提供完整的交易系统
"""

import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger
import os
import sys
import time

# 添加源代码路径
sys.path.append('src')

from src.config_utils import (
    ConfigValidationError,
    ensure_runtime_directories,
    load_config as load_validated_config,
)
from src.data_collector import DataCollector
from src.feature_engineer import FeatureEngineer
from src.ai_models import AIModelManager
from src.risk_manager import RiskManager
from src.backtester import Backtester
from src.trader import TradingEngine


def load_config():
    """加载配置文件"""
    try:
        return load_validated_config('config/config.yaml')
    except ConfigValidationError as e:
        logger.error(f"配置验证失败: {e}")
        return None
    except Exception as e:
        logger.error(f"加载配置失败: {e}")
        return None


def setup_directories(config=None):
    """创建必要的目录"""
    if config:
        ensure_runtime_directories(config)
        return

    directories = ['data', 'logs', 'models', 'reports']
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)


def setup_logging():
    """设置日志系统"""
    logger.add(
        "logs/trading_system_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    )


class TradingSystem:
    """AI黄金交易系统主类"""
    
    def __init__(self):
        """初始化交易系统"""
        logger.info("=== AI Gold Trading System Startup ===")
        
        # 加载配置
        self.config = load_config()
        if not self.config:
            raise Exception("Failed to load configuration")
        
        # 初始化各个模块
        self.data_collector = DataCollector(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.ai_model_manager = AIModelManager(self.config)
        self.risk_manager = RiskManager(self.config)
        self.backtester = Backtester(self.config)
        self.trading_engine = TradingEngine(
            self.config, 
            self.data_collector,
            self.feature_engineer,
            self.ai_model_manager,
            self.risk_manager
        )
        
        # 存储数据
        self.current_data = None
        self.feature_data = None
        self.selected_features = []
        
        logger.info("Trading system initialization complete")
    
    def collect_and_prepare_data(self):
        """采集和准备数据"""
        logger.info("Starting data collection and preparation...")
        
        # 获取历史数据
        historical_data = self.data_collector.get_historical_data(
            period="6mo",  # 6个月数据
            interval="1m"   # 1分钟间隔
        )
        
        if historical_data.empty:
            logger.error("Failed to fetch historical data")
            return False
        
        # 数据质量验证
        is_valid, issues = self.data_collector.validate_data_quality(historical_data)
        if not is_valid:
            logger.warning(f"Data quality issues: {issues}")
        
        # 保存原始数据
        self.data_collector.save_data_to_db(historical_data, 'raw_data')
        
        # 创建特征矩阵
        self.feature_data = self.feature_engineer.create_feature_matrix(
            historical_data, include_targets=True
        )
        
        if self.feature_data.empty:
            logger.error("Failed to create features")
            return False
        
        # 特征选择
        self.selected_features = self.feature_engineer.select_features(
            self.feature_data,
            target_column='Future_Direction_1',
            method='correlation',
            max_features=30
        )
        
        if not self.selected_features:
            logger.error("Feature selection failed")
            return False
        
        logger.info(f"Data preparation complete - selected {len(self.selected_features)} features")
        return True
    
    def train_models(self):
        """训练AI模型"""
        if self.feature_data is None or not self.selected_features:
            logger.error("Please prepare the data first")
            return False
        
        logger.info("Starting AI model training...")
        
        # 训练集成模型
        training_results = self.ai_model_manager.train_ensemble_models(
            self.feature_data,
            self.selected_features,
            target_column='Future_Direction_1'
        )
        
        if not training_results:
            logger.error("Model training failed")
            return False
        
        # 保存模型
        model_path = f"models/ai_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        self.ai_model_manager.save_models(model_path)
        
        # 显示训练结果
        logger.info("=== Model Training Results ===")
        for model_name, performance in training_results.items():
            accuracy = performance.get('test_accuracy', 0)
            logger.info(f"{model_name}: test accuracy = {accuracy:.4f}")
        
        return True
    
    def run_backtest(self):
        """运行回测"""
        logger.info("Starting professional backtest...")
        
        if not self.ai_model_manager.models:
            logger.error("Please train the models first")
            return False
        
        if self.feature_data is None or not self.selected_features:
            logger.error("Please prepare the data first")
            return False
        
        try:
            # 使用专业回测器
            result = self.backtester.run_backtest(
                self.feature_data,
                self.ai_model_manager,
                self.feature_engineer,
                self.selected_features
            )
            
            # 显示回测结果
            logger.info("=== Backtest Result Summary ===")
            logger.info(f"Total trades: {result.total_trades}")
            logger.info(f"Winning trades: {result.winning_trades}")
            logger.info(f"Losing trades: {result.losing_trades}")
            logger.info(f"Win rate: {result.win_rate:.1%}")
            logger.info(f"Total PnL: ${result.total_pnl:.2f}")
            logger.info(f"Max drawdown: {result.max_drawdown:.2%}")
            logger.info(f"Sharpe ratio: {result.sharpe_ratio:.2f}")
            logger.info(f"Profit factor: {result.profit_factor:.2f}")
            
            # 保存回测结果
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            result_file = f"reports/backtest_result_{timestamp}.json"
            self.backtester.save_results(result, result_file)
            
            # 生成回测图表
            chart_file = f"reports/backtest_chart_{timestamp}.png"
            self.backtester.plot_results(result, chart_file)
            
            # 保存交易明细
            trade_summary = self.backtester.get_trade_summary()
            if not trade_summary.empty:
                summary_file = f"reports/trade_summary_{timestamp}.csv"
                trade_summary.to_csv(summary_file, index=False, encoding='utf-8-sig')
                logger.info(f"Trade summary saved: {summary_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Professional backtest failed: {e}")
            return False
    
    def show_menu(self):
        """显示主菜单"""
        print("\n" + "="*60)
        print("🏆 AI Gold Trading System - Full Menu")
        print("="*60)
        print("📊 Data and Model Management:")
        print("  1. 📈 Data collection and preprocessing")
        print("  2. 🤖 Train AI models")
        print("  3. 📊 Model performance evaluation")
        print("  4. 🔄 Update data")
        
        print("\n💼 Backtesting and Analysis:")
        print("  5. 🎯 Run backtest")
        print("  6. 📋 View backtest reports")
        print("  7. 📈 Analyze model performance")
        
        print("\n🔄 Real-Time Trading:")
        print("  8. 🚀 Start simulation trading")
        print("  9. 📱 Trading engine status")
        print(" 10. ⏹️  Stop trading engine")
        print(" 11. 🧪 Manual trade test")
        
        print("\n🆕 Phase 3 Features:")
        print(" 12. 📊 Paper trading demo")
        print(" 13. 🏦 Broker interface management")
        print(" 14. 📱 Monitoring system control")
        print(" 15. 🔍 View system status")
        print(" 16. 🚨 Alert management")
        print(" 17. 📋 Export logs")
        print(" 18. 🎯 Full Phase 3 demo")
        
        print("\n🛠️  System Management:")
        print(" 19. ⚙️  System configuration")
        print(" 20. 📄 Generate report")
        print("  0. 🚪 Exit system")
        print("="*60)
    
    def show_system_status(self):
        """显示系统状态"""
        print("\n=== System Status ===")
        
        # 数据状态
        if self.feature_data is not None:
            print(f"✓ Feature data: {len(self.feature_data)} records")
            print(f"✓ Selected features: {len(self.selected_features)}")
        else:
            print("✗ Feature data: not prepared")
        
        # 模型状态
        if self.ai_model_manager.models:
            print(f"✓ Trained models: {len(self.ai_model_manager.models)}")
            
            # 显示模型性能
            summary = self.ai_model_manager.get_models_summary()
            if not summary.empty:
                print("\nModel performance:")
                for _, row in summary.iterrows():
                    print(f"  {row['model_name']}: accuracy {row.get('test_accuracy', 0):.4f}")
        else:
            print("✗ AI models: not trained")
        
        # 风险管理状态
        risk_summary = self.risk_manager.get_risk_summary()
        print(f"\nRisk management:")
        print(f"  Daily PnL: ${risk_summary['daily_pnl']:.2f}")
        print(f"  Open positions: {risk_summary['current_positions']}/{risk_summary['max_positions']}")
        print(f"  Tradable: {'Yes' if risk_summary['can_trade'] else 'No'}")
    
    def test_risk_management(self):
        """测试风险管理功能"""
        print("\n=== Risk Management Test ===")
        
        # 模拟交易测试
        test_position = {
            'symbol': 'XAUUSD',
            'side': 'buy',
            'size': 0.1,
            'entry_price': 2000.0,
            'stop_loss': 1980.0,
            'take_profit': 2020.0
        }
        
        # 检查交易风险
        can_trade, risk_msg, adjusted_size = self.risk_manager.check_position_risk(
            test_position['symbol'],
            test_position['side'],
            test_position['entry_price'],
            test_position['stop_loss'],
            test_position['size']
        )
        
        print(f"Risk check result: {risk_msg}")
        print(f"Suggested position size: {adjusted_size:.4f}")
        
        if can_trade:
            # 添加测试持仓
            position_id = "test_001"
            self.risk_manager.add_position(position_id, test_position)
            print(f"Added test position: {position_id}")
            
            # 模拟价格变动
            self.risk_manager.update_position_pnl(position_id, 2010.0)
            print("Simulated price increased to 2010.0")
            
            # 平仓
            pnl = self.risk_manager.close_position(position_id, 2015.0)
            print(f"Closed position PnL: ${pnl:.2f}")
    
    def demo_realtime_monitoring(self):
        """演示实时数据监控"""
        print("\n=== Real-Time Data Monitoring Demo ===")
        print("Fetching the latest gold price...")
        
        try:
            # 获取实时数据
            realtime_data = self.data_collector.get_realtime_data()
            
            if realtime_data:
                print(f"Symbol: {realtime_data['symbol']}")
                print(f"Current price: ${realtime_data['close']:.2f}")
                print(f"Bid: ${realtime_data['bid']:.2f}")
                print(f"Ask: ${realtime_data['ask']:.2f}")
                print(f"Spread: ${realtime_data['spread']:.2f}")
                print(f"Updated at: {realtime_data['timestamp']}")
                
                # 获取市场情绪数据
                sentiment_data = self.data_collector.get_market_sentiment_data()
                if sentiment_data:
                    print(f"\nMarket sentiment indicators:")
                    if 'vix' in sentiment_data:
                        print(f"  VIX fear index: {sentiment_data['vix']:.2f}")
                    if 'dxy' in sentiment_data:
                        print(f"  US Dollar Index: {sentiment_data['dxy']:.2f}")
            else:
                print("Failed to fetch real-time data")
                
        except Exception as e:
            logger.error(f"Real-time monitoring demo failed: {e}")
    
    def show_backtest_reports(self):
        """查看回测报告"""
        print("\n=== Backtest Reports ===")
        
        try:
            import os
            import glob
            
            # 查找回测报告文件
            report_files = glob.glob("reports/backtest_result_*.json")
            
            if not report_files:
                print("No backtest report files found")
                return
            
            # 按时间排序，显示最新的几个
            report_files.sort(reverse=True)
            
            print("Available backtest reports:")
            for i, file in enumerate(report_files[:5]):  # 显示最新5个
                timestamp = file.split('_')[-1].replace('.json', '')
                date_str = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]} {timestamp[9:11]}:{timestamp[11:13]}"
                print(f"{i+1}. {date_str}")
            
            choice = input("\nSelect a report number (1-5): ").strip()
            
            if choice.isdigit() and 1 <= int(choice) <= min(5, len(report_files)):
                selected_file = report_files[int(choice)-1]
                
                # 读取并显示报告
                import json
                with open(selected_file, 'r', encoding='utf-8') as f:
                    report = json.load(f)
                
                summary = report['backtest_summary']
                print(f"\n=== Backtest Report Details ===")
                print(f"Total trades: {summary['total_trades']}")
                print(f"Winning trades: {summary['winning_trades']}")
                print(f"Losing trades: {summary['losing_trades']}")
                print(f"Win rate: {summary['win_rate']:.1%}")
                print(f"Total PnL: ${summary['total_pnl']:.2f}")
                print(f"Max drawdown: {summary['max_drawdown']:.2%}")
                print(f"Sharpe ratio: {summary['sharpe_ratio']:.2f}")
                print(f"Profit factor: {summary['profit_factor']:.2f}")
                
        except Exception as e:
            logger.error(f"Failed to view backtest reports: {e}")
    
    def analyze_model_performance(self):
        """分析模型性能"""
        print("\n=== Model Performance Analysis ===")
        
        if not self.ai_model_manager.models:
            print("Please train the models first")
            return
        
        try:
            # 显示模型汇总
            summary = self.ai_model_manager.get_models_summary()
            if not summary.empty:
                print("\nModel performance comparison:")
                print(summary[['model_name', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1']])
            
            # 显示特征重要性
            print("\n=== Feature Importance Analysis ===")
            for model_name in self.ai_model_manager.models:
                importance = self.ai_model_manager.get_feature_importance(model_name)
                if importance:
                    print(f"\n{model_name} most important features:")
                    for i, (feature, score) in enumerate(list(importance.items())[:10]):
                        print(f"{i+1:2d}. {feature}: {score:.4f}")
            
            # 模型预测一致性分析
            if len(self.ai_model_manager.models) > 1:
                print("\n=== Model Prediction Consistency ===")
                print("Analyzing prediction consistency across models...")
                # 这里可以添加更复杂的一致性分析
                
        except Exception as e:
            logger.error(f"Model performance analysis failed: {e}")
    
    def start_simulation_trading(self):
        """启动模拟交易"""
        print("\n=== Start Simulation Trading ===")
        
        if not self.ai_model_manager.models:
            print("Please train the models first")
            return
        
        try:
            status = self.trading_engine.get_trading_status()
            
            if status.get('engine_running', False):
                print("Trading engine is already running")
                return
            
            print("Starting simulation trading engine...")
            self.trading_engine.start_trading()
            
            print("✓ Simulation trading engine started")
            print("⚠️  This is simulated trading and does not affect real funds")
            print("💡 The trading engine will keep running; use the menu to view status")
            
        except Exception as e:
            logger.error(f"Failed to start simulation trading: {e}")
    
    def show_trading_engine_status(self):
        """显示交易引擎状态"""
        print("\n=== Trading Engine Status ===")
        
        try:
            status = self.trading_engine.get_trading_status()
            
            print(f"Engine status: {'running' if status.get('engine_running') else 'stopped'}")
            print(f"Trading enabled: {'yes' if status.get('trading_enabled') else 'no'}")
            print(f"Current price: ${status.get('current_price', 0):.2f}")
            print(f"Open positions: {status.get('current_positions', 0)}")
            print(f"Daily PnL: ${status.get('daily_pnl', 0):.2f}")
            print(f"Total PnL: ${status.get('total_pnl', 0):.2f}")
            print(f"Tradable: {'Yes' if status.get('can_trade') else 'No'}")
            print(f"Signal count: {status.get('signal_count', 0)}")
            
            # 显示最近信号
            recent_signals = self.trading_engine.get_recent_signals(5)
            if recent_signals:
                print("\nRecent trading signals:")
                for signal in recent_signals:
                    print(f"  {signal['timestamp'][:19]} | {signal['signal_type']} | "
                          f"confidence: {signal['confidence']:.3f} | price: ${signal['price']:.2f}")
            
            # 控制选项
            if status.get('engine_running'):
                choice = input("\n操作选项 (stop/disable/enable/emergency): ").strip().lower()
                if choice == 'stop':
                    self.trading_engine.stop_trading()
                    print("Trading engine stopped")
                elif choice == 'disable':
                    self.trading_engine.disable_trading()
                    print("Trading disabled")
                elif choice == 'enable':
                    self.trading_engine.enable_trading()
                    print("Trading enabled")
                elif choice == 'emergency':
                    self.trading_engine.emergency_stop()
                    print("Emergency stop completed")
            
        except Exception as e:
            logger.error(f"Failed to show trading engine status: {e}")
    
    def manual_trading_test(self):
        """手动交易测试"""
        print("\n=== Manual Trade Test ===")
        
        try:
            status = self.trading_engine.get_trading_status()
            
            if not status.get('trading_enabled'):
                print("Trading is disabled; manual trading is unavailable")
                return
            
            print(f"Current price: ${status.get('current_price', 0):.2f}")
            print(f"Open positions: {status.get('current_positions', 0)}")
            
            side = input("Enter trade side (buy/sell): ").strip().lower()
            
            if side not in ['buy', 'sell']:
                print("Invalid trade side")
                return
            
            confirm = input(f"Confirm {side} action? (y/n): ").strip().lower()
            
            if confirm == 'y':
                success = self.trading_engine.manual_trade(side)
                if success:
                    print("✓ Manual trade executed successfully")
                else:
                    print("✗ Manual trade execution failed")
            else:
                print("Trade canceled")
                
        except Exception as e:
            logger.error(f"Manual trade test failed: {e}")
    
    def paper_trading_demo(self):
        """纸上交易演示"""
        try:
            print("\n📊 纸上交易系统演示")
            print("="*50)
            
            from src.paper_trading import PaperTradingEngine, OrderType
            
            # 创建纸上交易引擎
            config = {
                'initial_capital': 10000.0,
                'commission': 0.0001,
                'slippage': 0.0002
            }
            
            paper_trader = PaperTradingEngine(config)
            print("✅ 纸上交易引擎已初始化")
            
            # 模拟市场数据和交易
            import numpy as np
            base_price = 2000.0
            
            print("\n📈 模拟交易操作...")
            
            for i in range(3):
                # 更新价格
                price = base_price + np.random.normal(0, 10)
                paper_trader.update_market_data('XAUUSD', price)
                print(f"价格更新: ${price:.2f}")
                
                # 随机交易
                if np.random.random() > 0.5:
                    side = 'buy' if np.random.random() > 0.5 else 'sell'
                    order_id = paper_trader.submit_order(
                        symbol='XAUUSD',
                        side=side,
                        quantity=0.01,
                        order_type=OrderType.MARKET,
                        tag=f'demo_{i+1}'
                    )
                    print(f"提交{side}单: {order_id}")
                
                time.sleep(1)
            
            # 显示结果
            summary = paper_trader.get_portfolio_summary()
            print(f"\n📊 交易结果:")
            print(f"账户权益: ${summary.get('equity', 0):.2f}")
            print(f"总盈亏: ${summary.get('total_pnl', 0):.2f}")
            print(f"交易次数: {summary.get('trade_count', 0)}")
            print(f"胜率: {summary.get('win_rate', 0):.1%}")
            
        except Exception as e:
            print(f"❌ 纸上交易演示失败: {e}")
            logger.error(f"纸上交易演示异常: {e}")

    def broker_interface_management(self):
        """券商接口管理"""
        try:
            print("\n🏦 Broker Interface Management")
            print("="*50)
            
            from src.broker_interface import BrokerManager, create_broker_config
            
            if not hasattr(self, 'broker_manager'):
                self.broker_manager = BrokerManager()
            
            while True:
                print("\nBroker interface options:")
                print("1. Add broker configuration")
                print("2. View broker status")
                print("3. Connect broker")
                print("4. Disconnect all brokers")
                print("5. Mock order test")
                print("0. Return to main menu")
                
                choice = input("Select an action: ").strip()
                
                if choice == '0':
                    break
                elif choice == '1':
                    # 添加券商配置
                    print("\nSupported broker types:")
                    print("1. Alpaca")
                    print("2. OANDA")
                    print("3. Exness (MetaTrader5)")
                    
                    broker_choice = input("Select broker type (1-3): ").strip()
                    name = input("Enter configuration name: ").strip()
                    
                    if broker_choice == '1':
                        config = create_broker_config(
                            broker_type='alpaca',
                            api_key=input("API Key: ").strip(),
                            secret_key=input("Secret Key: ").strip(),
                            sandbox=True
                        )
                    elif broker_choice == '2':
                        config = create_broker_config(
                            broker_type='oanda',
                            api_key=input("API Token: ").strip(),
                            account_id=input("Account ID: ").strip(),
                            sandbox=True
                        )
                    elif broker_choice == '3':
                        config = create_broker_config(
                            broker_type='exness',
                            login=input("MT5 Login: ").strip(),
                            password=input("MT5 Password: ").strip(),
                            server=input("MT5 Server: ").strip(),
                            terminal_path=input("MT5 Terminal Path (optional): ").strip(),
                            sandbox=False
                        )
                    else:
                        print("❌ Invalid selection")
                        continue
                    
                    success = self.broker_manager.add_broker(name, config)
                    print(f"{'✅' if success else '❌'} Broker configuration {'saved' if success else 'failed'}")
                    
                elif choice == '2':
                    # 查看券商状态
                    status = self.broker_manager.get_broker_status()
                    print("\n📋 Broker Status:")
                    for name, info in status.items():
                        if name != 'active_broker':
                            print(f"{name}: {info}")
                    print(f"Active broker: {status.get('active_broker', 'None')}")
                    
                elif choice == '3':
                    # 连接券商
                    status = self.broker_manager.get_broker_status()
                    brokers = [k for k in status.keys() if k != 'active_broker']
                    
                    if not brokers:
                        print("❌ No brokers configured")
                        continue
                    
                    print("Available brokers:")
                    for i, broker in enumerate(brokers, 1):
                        print(f"{i}. {broker}")
                    
                    try:
                        broker_idx = int(input("Select broker (number): ")) - 1
                        if 0 <= broker_idx < len(brokers):
                            broker_name = brokers[broker_idx]
                            success = self.broker_manager.connect_broker(broker_name)
                            print(f"{'✅' if success else '❌'} Connection {'successful' if success else 'failed'}")
                        else:
                            print("❌ Invalid selection")
                    except ValueError:
                        print("❌ Please enter a valid number")
                
                elif choice == '4':
                    # 断开所有连接
                    self.broker_manager.disconnect_all()
                    print("✅ Disconnected all broker connections")
                    
                elif choice == '5':
                    # 模拟订单测试
                    print("\n🧪 Mock Order Test")
                    mock_result = {
                        'success': True,
                        'order_id': f'demo_{int(time.time())}',
                        'status': 'filled',
                        'message': 'Mock order executed successfully'
                    }
                    print(f"Mock result: {mock_result}")
                
                else:
                    print("❌ Invalid selection")
                    
        except Exception as e:
            print(f"❌ Broker interface management failed: {e}")
            logger.error(f"Broker interface management error: {e}")

    def monitoring_system_control(self):
        """监控系统控制"""
        try:
            print("\n📱 监控系统控制")
            print("="*50)
            
            from src.monitoring import MonitoringSystem, AlertType, AlertLevel
            
            if not hasattr(self, 'monitoring_system'):
                monitoring_config = self.config.get('monitoring', {})
                self.monitoring_system = MonitoringSystem(monitoring_config)
            
            while True:
                print("\n监控系统控制选项:")
                print("1. 启动监控系统")
                print("2. 停止监控系统")
                print("3. 查看系统状态")
                print("4. 发送测试告警")
                print("5. 查看告警摘要")
                print("0. 返回主菜单")
                
                choice = input("请选择操作: ").strip()
                
                if choice == '0':
                    break
                elif choice == '1':
                    self.monitoring_system.start()
                    print("✅ 监控系统已启动")
                elif choice == '2':
                    self.monitoring_system.stop()
                    print("✅ 监控系统已停止")
                elif choice == '3':
                    status = self.monitoring_system.get_system_status()
                    print("\n📊 系统状态:")
                    for key, value in status.items():
                        if isinstance(value, dict):
                            print(f"{key}:")
                            for sub_key, sub_value in value.items():
                                print(f"  {sub_key}: {sub_value}")
                        else:
                            print(f"{key}: {value}")
                elif choice == '4':
                    self.monitoring_system.send_custom_alert(
                        alert_type=AlertType.SYSTEM,
                        level=AlertLevel.INFO,
                        title="测试告警",
                        message="这是一个手动发送的测试告警",
                        source="ManualTest"
                    )
                    print("✅ 测试告警已发送")
                elif choice == '5':
                    summary = self.monitoring_system.get_alert_summary(24)
                    print("\n📋 告警摘要 (最近24小时):")
                    for key, value in summary.items():
                        print(f"{key}: {value}")
                else:
                    print("❌ 无效选择")
                    
        except Exception as e:
            print(f"❌ 监控系统控制失败: {e}")
            logger.error(f"监控系统控制异常: {e}")

    def view_system_status(self):
        """查看系统状态"""
        try:
            print("\n🔍 系统状态概览")
            print("="*50)
            
            # 基本系统信息
            import psutil
            from datetime import datetime
            
            print("💻 系统资源:")
            print(f"CPU使用率: {psutil.cpu_percent(interval=1):.1f}%")
            print(f"内存使用率: {psutil.virtual_memory().percent:.1f}%")
            print(f"磁盘使用率: {psutil.disk_usage('.').used / psutil.disk_usage('.').total * 100:.1f}%")
            
            # 交易系统状态
            print(f"\n🤖 AI模型状态:")
            model_count = len(self.ai_model_manager.models)
            print(f"模型已加载: {'是' if model_count > 0 else '否'}")
            print(f"已训练模型数: {model_count}")
            if self.ai_model_manager.training_history:
                print(f"最后训练时间: {self.ai_model_manager.training_history[-1]['timestamp']}")
            else:
                print("最后训练时间: 未知")
            
            print(f"\n📊 交易引擎状态:")
            trading_status = self.trading_engine.get_trading_status()
            print(f"引擎运行中: {trading_status.get('engine_running', False)}")
            print(f"交易启用: {trading_status.get('trading_enabled', False)}")
            print(f"当前信号数: {trading_status.get('signal_count', 0)}")
            print(f"当前持仓数: {trading_status.get('current_positions', 0)}")
            
            # 监控系统状态
            if hasattr(self, 'monitoring_system'):
                status = self.monitoring_system.get_system_status()
                print(f"\n📱 监控系统:")
                print(f"监控状态: {status.get('monitoring_active', '未知')}")
                print(f"最近告警: {status.get('recent_alerts', 0)}")
            
            # 数据状态
            database_path = self.config.get('database', {}).get('path', 'data/trading_system.db')
            print(f"\n📈 数据状态:")
            print(f"数据库路径: {database_path}")
            print(f"特征数据已准备: {'是' if self.feature_data is not None else '否'}")
            if self.feature_data is not None:
                print(f"特征记录数: {len(self.feature_data)}")
            
            print(f"\n🕐 当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
        except Exception as e:
            print(f"❌ 查看系统状态失败: {e}")
            logger.error(f"查看系统状态异常: {e}")

    def alert_management(self):
        """告警管理"""
        try:
            print("\n🚨 告警管理")
            print("="*50)
            
            if not hasattr(self, 'monitoring_system'):
                print("❌ 监控系统未初始化")
                return
            
            while True:
                print("\n告警管理选项:")
                print("1. 查看最近告警")
                print("2. 告警统计")
                print("3. 发送自定义告警")
                print("4. 清理历史告警")
                print("0. 返回主菜单")
                
                choice = input("请选择操作: ").strip()
                
                if choice == '0':
                    break
                elif choice == '1':
                    # 查看最近告警
                    alerts = self.monitoring_system.db_logger.get_alerts(limit=10)
                    print(f"\n📋 最近 {len(alerts)} 条告警:")
                    for alert in alerts:
                        print(f"[{alert['level'].upper()}] {alert['title']}")
                        print(f"  时间: {alert['timestamp']}")
                        print(f"  来源: {alert['source']}")
                        print(f"  消息: {alert['message']}")
                        print("-" * 40)
                        
                elif choice == '2':
                    # 告警统计
                    summary = self.monitoring_system.get_alert_summary(24)
                    print("\n📊 告警统计 (最近24小时):")
                    for key, value in summary.items():
                        print(f"{key}: {value}")
                        
                elif choice == '3':
                    # 发送自定义告警
                    from src.monitoring import AlertType, AlertLevel
                    
                    print("\n告警类型:")
                    print("1. 系统告警")
                    print("2. 交易告警")
                    print("3. 风险告警")
                    
                    type_choice = input("选择类型 (1-3): ").strip()
                    type_map = {'1': AlertType.SYSTEM, '2': AlertType.TRADING, '3': AlertType.RISK}
                    
                    print("\n告警级别:")
                    print("1. 信息")
                    print("2. 警告")
                    print("3. 错误")
                    print("4. 严重")
                    
                    level_choice = input("选择级别 (1-4): ").strip()
                    level_map = {'1': AlertLevel.INFO, '2': AlertLevel.WARNING, 
                               '3': AlertLevel.ERROR, '4': AlertLevel.CRITICAL}
                    
                    if type_choice in type_map and level_choice in level_map:
                        title = input("告警标题: ").strip()
                        message = input("告警消息: ").strip()
                        
                        self.monitoring_system.send_custom_alert(
                            alert_type=type_map[type_choice],
                            level=level_map[level_choice],
                            title=title,
                            message=message,
                            source="Manual"
                        )
                        print("✅ 自定义告警已发送")
                    else:
                        print("❌ 无效选择")
                        
                elif choice == '4':
                    print("⚠️  清理功能需要手动实现数据库操作")
                    
                else:
                    print("❌ 无效选择")
                    
        except Exception as e:
            print(f"❌ 告警管理失败: {e}")
            logger.error(f"告警管理异常: {e}")

    def export_logs(self):
        """导出日志"""
        try:
            print("\n📋 日志导出")
            print("="*50)
            
            if not hasattr(self, 'monitoring_system'):
                print("❌ 监控系统未初始化")
                return
            
            print("导出时间范围:")
            print("1. 最近1小时")
            print("2. 最近24小时")
            print("3. 最近7天")
            print("4. 自定义")
            
            choice = input("请选择 (1-4): ").strip()
            
            hours_map = {'1': 1, '2': 24, '3': 168}
            
            if choice in hours_map:
                hours = hours_map[choice]
            elif choice == '4':
                try:
                    hours = int(input("输入小时数: "))
                except ValueError:
                    print("❌ 无效的小时数")
                    return
            else:
                print("❌ 无效选择")
                return
            
            print(f"\n📤 导出最近{hours}小时的日志...")
            
            output_file = self.monitoring_system.export_logs(hours)
            
            if output_file:
                print(f"✅ 日志已导出到: {output_file}")
                
                # 显示文件大小
                file_size = os.path.getsize(output_file)
                print(f"文件大小: {file_size / 1024:.1f} KB")
            else:
                print("❌ 日志导出失败")
                
        except Exception as e:
            print(f"❌ 日志导出失败: {e}")
            logger.error(f"日志导出异常: {e}")

    def run_phase3_demo(self):
        """运行第三阶段完整演示"""
        try:
            print("\n🎯 第三阶段完整演示")
            print("="*50)
            print("这将演示实时交易系统的所有功能...")
            
            confirm = input("确认运行完整演示？(y/N): ").strip().lower()
            if confirm != 'y':
                print("演示已取消")
                return
            
            # 运行第三阶段演示
            import subprocess
            import sys
            
            print("\n🚀 启动第三阶段演示...")
            result = subprocess.run([sys.executable, 'phase3_demo.py'], 
                                  capture_output=False, text=True)
            
            if result.returncode == 0:
                print("✅ 第三阶段演示完成")
            else:
                print("❌ 第三阶段演示失败")
                
        except Exception as e:
            print(f"❌ 运行第三阶段演示失败: {e}")
            logger.error(f"第三阶段演示异常: {e}")

    def run(self):
        """运行主程序"""
        self.welcome()
        
        while True:
            try:
                self.show_menu()
                choice = input("\n请选择功能 (0-20): ").strip()
                
                if choice == '0':
                    print("\n👋 感谢使用AI黄金交易系统！")
                    self.cleanup()
                    break
                elif choice == '1':
                    self.collect_and_prepare_data()
                elif choice == '2':
                    self.train_models()
                elif choice == '3':
                    self.analyze_model_performance()
                elif choice == '4':
                    self.update_data()
                elif choice == '5':
                    self.run_backtest()
                elif choice == '6':
                    self.show_backtest_reports()
                elif choice == '7':
                    self.analyze_model_performance()
                elif choice == '8':
                    self.start_simulation_trading()
                elif choice == '9':
                    self.show_trading_engine_status()
                elif choice == '10':
                    self.stop_trader()
                elif choice == '11':
                    self.manual_trading_test()
                elif choice == '12':
                    self.paper_trading_demo()
                elif choice == '13':
                    self.broker_interface_management()
                elif choice == '14':
                    self.monitoring_system_control()
                elif choice == '15':
                    self.view_system_status()
                elif choice == '16':
                    self.alert_management()
                elif choice == '17':
                    self.export_logs()
                elif choice == '18':
                    self.run_phase3_demo()
                elif choice == '19':
                    self.system_configuration()
                elif choice == '20':
                    self.generate_report()
                else:
                    print("❌ 无效选择，请重新输入！")
                    
                input("\n按Enter键继续...")
                
            except KeyboardInterrupt:
                print("\n\n⚠️  操作被用户中断")
                self.cleanup()
                break
            except Exception as e:
                print(f"\n❌ 操作失败: {e}")
                logger.error(f"菜单操作异常: {e}")


def main():
    """主函数"""
    # 设置环境
    bootstrap_config = load_config()
    if not bootstrap_config:
        print("系统启动失败: 配置无效，请检查 config/config.yaml")
        return False

    setup_directories(bootstrap_config)
    setup_logging()
    
    try:
        # 创建并运行交易系统
        trading_system = TradingSystem()
        trading_system.run()
        
    except Exception as e:
        logger.critical(f"系统启动失败: {e}")
        print(f"系统启动失败: {e}")
        return False
    
    return True


if __name__ == "__main__":
    main() 
