#!/usr/bin/env python3
"""
AI黄金交易系统 - 第三阶段演示
实时交易系统完整功能展示

功能包括：
1. 纸上交易环境演示
2. 券商API接口演示
3. 监控和日志系统演示
4. 实时交易流程演示
"""

import os
import sys
import time
import json
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.paper_trading import PaperTradingEngine, OrderType, OrderStatus
from src.broker_interface import BrokerManager, create_broker_config, BrokerType
from src.config_utils import ConfigValidationError, ensure_runtime_directories, load_config as load_validated_config
from src.monitoring import (MonitoringSystem, AlertType, AlertLevel, 
                           SystemMetrics, TradingMetrics)
from src.data_collector import DataCollector
from src.ai_models import AIModelManager
from loguru import logger


class Phase3Demo:
    """第三阶段演示类"""
    
    def __init__(self):
        """初始化演示系统"""
        print("🚀 初始化AI黄金交易系统 - 第三阶段")
        print("=" * 60)
        
        # 加载配置
        self.config = self._load_config()
        
        # 初始化组件
        self.paper_trading = None
        self.broker_manager = None
        self.monitoring_system = None
        self.data_collector = None
        self.ai_model = None
        self.trader = None
        
        # 演示状态
        self.demo_running = False
        self.demo_results = {}
        
        print("✅ 演示系统初始化完成\n")
    
    def _load_config(self) -> Dict:
        """加载配置"""
        try:
            config = load_validated_config('config/config.yaml')
            ensure_runtime_directories(config)
            return config
        except ConfigValidationError as e:
            logger.warning(f"配置验证失败，使用默认配置: {e}")
            return {
                'trading': {
                    'symbol': 'XAUUSD',
                    'initial_capital': 10000.0,
                    'max_daily_loss': 30.0,
                    'position_size': 0.01,
                    'confidence_threshold': 0.65
                },
                'monitoring': {
                    'system': {
                        'cpu_threshold': 80.0,
                        'memory_threshold': 85.0,
                        'check_interval': 10
                    },
                    'trading': {
                        'max_drawdown_threshold': 0.15,
                        'max_daily_loss_threshold': 1000,
                        'min_win_rate_threshold': 0.4
                    },
                    'notifications': {
                        'enabled_channels': ['log']
                    }
                }
            }
        except Exception as e:
            logger.warning(f"加载配置失败，使用默认配置: {e}")
            return {
                'trading': {
                    'symbol': 'XAUUSD',
                    'initial_capital': 10000.0,
                    'max_daily_loss': 30.0,
                    'position_size': 0.01,
                    'confidence_threshold': 0.65
                },
                'monitoring': {
                    'system': {
                        'cpu_threshold': 80.0,
                        'memory_threshold': 85.0,
                        'check_interval': 10
                    },
                    'trading': {
                        'max_drawdown_threshold': 0.15,
                        'max_daily_loss_threshold': 1000,
                        'min_win_rate_threshold': 0.4
                    },
                    'notifications': {
                        'enabled_channels': ['log']
                    }
                }
            }
    
    def run_demo(self):
        """运行完整演示"""
        try:
            print("🎯 启动第三阶段完整演示")
            print("=" * 60)
            
            # 演示步骤
            steps = [
                ("1️⃣  纸上交易系统演示", self._demo_paper_trading),
                ("2️⃣  券商接口系统演示", self._demo_broker_interface),
                ("3️⃣  监控和日志系统演示", self._demo_monitoring_system),
                ("4️⃣  实时数据采集演示", self._demo_data_collection),
                ("5️⃣  AI模型集成演示", self._demo_ai_integration),
                ("6️⃣  完整交易流程演示", self._demo_full_trading_flow),
                ("7️⃣  系统性能评估", self._evaluate_system_performance)
            ]
            
            self.demo_running = True
            
            for step_name, step_func in steps:
                print(f"\n{step_name}")
                print("-" * 50)
                
                try:
                    step_func()
                    print(f"✅ {step_name} 完成")
                    time.sleep(2)
                    
                except Exception as e:
                    print(f"❌ {step_name} 失败: {e}")
                    logger.error(f"{step_name} 执行失败: {e}")
            
            # 生成演示报告
            self._generate_demo_report()
            
            print("\n🎉 第三阶段演示完成！")
            print("=" * 60)
            
        except KeyboardInterrupt:
            print("\n\n⚠️  演示被用户中断")
            self._cleanup()
        except Exception as e:
            print(f"\n❌ 演示执行失败: {e}")
            logger.error(f"演示执行异常: {e}")
            self._cleanup()
    
    def _demo_paper_trading(self):
        """演示纸上交易系统"""
        print("📊 初始化纸上交易引擎...")
        
        # 创建纸上交易引擎
        config = {
            'initial_capital': 10000.0,
            'commission': 0.0001,
            'slippage': 0.0002,
            'margin_requirement': 0.05
        }
        
        self.paper_trading = PaperTradingEngine(config)
        
        # 模拟市场数据
        print("📈 模拟市场数据更新...")
        base_price = 2000.0
        
        for i in range(5):
            # 模拟价格变动
            price = base_price + np.random.normal(0, 5)
            self.paper_trading.update_market_data('XAUUSD', price)
            print(f"   价格更新: XAUUSD = ${price:.2f}")
            time.sleep(0.5)
        
        # 演示订单操作
        print("💼 演示订单操作...")
        
        # 提交买单
        order_id1 = self.paper_trading.submit_order(
            symbol='XAUUSD',
            side='buy',
            quantity=0.1,
            order_type=OrderType.MARKET,
            tag='demo_buy'
        )
        print(f"   ✅ 市价买单提交: {order_id1}")
        
        # 更新价格触发成交
        self.paper_trading.update_market_data('XAUUSD', base_price + 10)
        
        # 提交限价卖单
        order_id2 = self.paper_trading.submit_order(
            symbol='XAUUSD',
            side='sell',
            quantity=0.05,
            order_type=OrderType.LIMIT,
            price=base_price + 15,
            tag='demo_sell_limit'
        )
        print(f"   ✅ 限价卖单提交: {order_id2}")
        
        # 更新价格触发限价单
        self.paper_trading.update_market_data('XAUUSD', base_price + 20)
        
        # 显示账户状态
        summary = self.paper_trading.get_portfolio_summary()
        print(f"   📊 账户权益: ${summary.get('equity', 0):.2f}")
        print(f"   📊 总盈亏: ${summary.get('total_pnl', 0):.2f}")
        print(f"   📊 交易次数: {summary.get('trade_count', 0)}")
        print(f"   📊 胜率: {summary.get('win_rate', 0):.1%}")
        
        # 保存演示结果
        self.demo_results['paper_trading'] = summary
    
    def _demo_broker_interface(self):
        """演示券商接口系统"""
        print("🏦 初始化券商管理器...")
        
        self.broker_manager = BrokerManager()
        
        # 添加模拟券商
        print("➕ 添加券商配置...")
        
        # Alpaca模拟配置
        alpaca_config = create_broker_config(
            broker_type='alpaca',
            api_key='demo_key',
            secret_key='demo_secret',
            sandbox=True
        )
        
        success = self.broker_manager.add_broker('alpaca_demo', alpaca_config)
        print(f"   {'✅' if success else '❌'} Alpaca券商配置: {'成功' if success else '失败'}")
        
        # OANDA模拟配置
        oanda_config = create_broker_config(
            broker_type='oanda',
            api_key='demo_token',
            account_id='demo_account',
            sandbox=True
        )
        
        success = self.broker_manager.add_broker('oanda_demo', oanda_config)
        print(f"   {'✅' if success else '❌'} OANDA券商配置: {'成功' if success else '失败'}")
        
        # 显示券商状态
        status = self.broker_manager.get_broker_status()
        print("📋 券商状态:")
        for name, info in status.items():
            if name != 'active_broker':
                print(f"   {name}: {info.get('type', 'Unknown')} ({'沙盒' if info.get('sandbox') else '生产'})")
        
        # 演示模拟订单
        print("📝 模拟订单管理演示...")
        
        # 由于是演示环境，我们模拟订单响应
        mock_order_result = {
            'success': True,
            'order_id': 'demo_order_123',
            'status': 'pending',
            'message': '模拟订单已提交'
        }
        
        print(f"   ✅ 模拟订单结果: {mock_order_result}")
        
        # 保存演示结果
        self.demo_results['broker_interface'] = {
            'brokers_configured': len(status) - 1,
            'demo_order_submitted': True,
            'broker_status': status
        }
    
    def _demo_monitoring_system(self):
        """演示监控和日志系统"""
        print("📱 初始化监控系统...")
        
        # 创建监控系统
        monitoring_config = self.config.get('monitoring', {})
        self.monitoring_system = MonitoringSystem(monitoring_config)
        
        # 启动监控
        self.monitoring_system.start()
        print("   ✅ 系统监控已启动")
        
        # 演示告警功能
        print("🚨 演示告警系统...")
        
        # 发送测试告警
        self.monitoring_system.send_custom_alert(
            alert_type=AlertType.SYSTEM,
            level=AlertLevel.INFO,
            title="第三阶段演示启动",
            message="AI黄金交易系统第三阶段演示正在运行",
            source="Phase3Demo"
        )
        
        self.monitoring_system.send_custom_alert(
            alert_type=AlertType.TRADING,
            level=AlertLevel.WARNING,
            title="模拟交易告警",
            message="这是一个演示用的交易告警",
            source="Phase3Demo",
            data={"demo": True, "timestamp": datetime.now().isoformat()}
        )
        
        print("   ✅ 测试告警已发送")
        
        # 模拟交易指标更新
        print("📊 演示交易指标监控...")
        
        for i in range(3):
            metrics = TradingMetrics(
                total_trades=10 + i * 2,
                winning_trades=6 + i,
                losing_trades=4 + i,
                win_rate=(6 + i) / (10 + i * 2),
                total_pnl=150.0 + i * 50,
                unrealized_pnl=25.0 + i * 10,
                realized_pnl=125.0 + i * 40,
                max_drawdown=0.05 + i * 0.01,
                current_positions=1,
                active_orders=2,
                timestamp=datetime.now()
            )
            
            self.monitoring_system.update_trading_metrics(metrics)
            print(f"   📈 交易指标更新 #{i+1}: 总盈亏=${metrics.total_pnl:.2f}")
            time.sleep(1)
        
        # 获取系统状态
        status = self.monitoring_system.get_system_status()
        print("🔍 系统状态概览:")
        for key, value in status.items():
            if isinstance(value, dict) and value:
                print(f"   {key}:")
                for sub_key, sub_value in value.items():
                    print(f"     {sub_key}: {sub_value}")
            else:
                print(f"   {key}: {value}")
        
        # 保存演示结果
        self.demo_results['monitoring_system'] = {
            'system_status': status,
            'alerts_sent': 2,
            'metrics_updated': 3
        }
        
        time.sleep(2)
    
    def _demo_data_collection(self):
        """演示实时数据采集"""
        print("📡 初始化数据采集系统...")
        
        # 创建数据采集器
        self.data_collector = DataCollector(self.config)
        
        # 模拟数据采集
        print("📊 模拟实时数据采集...")
        
        # 生成模拟K线数据
        base_price = 2000.0
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(minutes=100),
            end=datetime.now(),
            freq='1min'
        )
        
        mock_data = []
        for i, ts in enumerate(timestamps):
            price = base_price + np.random.normal(0, 5) + np.sin(i * 0.1) * 10
            volume = np.random.randint(100, 1000)
            
            mock_data.append({
                'timestamp': ts,
                'open': price,
                'high': price + np.random.uniform(0, 3),
                'low': price - np.random.uniform(0, 3),
                'close': price + np.random.normal(0, 1),
                'volume': volume
            })
        
        # 模拟数据处理
        df = pd.DataFrame(mock_data)
        print(f"   📈 生成{len(df)}条模拟K线数据")
        print(f"   💰 价格范围: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
        print(f"   📊 平均成交量: {df['volume'].mean():.0f}")
        
        # 保存演示结果
        self.demo_results['data_collection'] = {
            'data_points': len(df),
            'price_range': f"${df['low'].min():.2f} - ${df['high'].max():.2f}",
            'avg_volume': int(df['volume'].mean()),
            'time_range': f"{timestamps[0].strftime('%H:%M')} - {timestamps[-1].strftime('%H:%M')}"
        }
    
    def _demo_ai_integration(self):
        """演示AI模型集成"""
        print("🤖 初始化AI模型系统...")
        
        # 创建AI模型
        self.ai_model = AIModelManager(self.config)
        
        # 模拟模型训练状态
        print("🧠 模拟AI模型状态...")
        
        # 生成模拟特征数据
        n_samples = 100
        n_features = 20
        
        # 模拟历史数据特征
        features = np.random.randn(n_samples, n_features)
        labels = np.random.choice([0, 1], n_samples, p=[0.4, 0.6])  # 60%上涨概率
        
        print(f"   📊 生成{n_samples}个样本，{n_features}个特征")
        
        # 模拟预测
        print("🔮 演示AI预测功能...")
        
        predictions = []
        confidences = []
        
        for i in range(5):
            # 模拟当前特征
            current_features = np.random.randn(1, n_features)
            
            # 模拟预测结果
            pred_prob = np.random.uniform(0.3, 0.9)
            prediction = 1 if pred_prob > 0.5 else 0
            confidence = pred_prob if prediction == 1 else 1 - pred_prob
            
            predictions.append(prediction)
            confidences.append(confidence)
            
            direction = "上涨" if prediction == 1 else "下跌"
            print(f"   🎯 预测#{i+1}: {direction} (置信度: {confidence:.1%})")
        
        # 统计预测结果
        avg_confidence = np.mean(confidences)
        bullish_ratio = np.mean(predictions)
        
        print(f"   📈 看涨信号比例: {bullish_ratio:.1%}")
        print(f"   🎯 平均置信度: {avg_confidence:.1%}")
        
        # 保存演示结果
        self.demo_results['ai_integration'] = {
            'samples_processed': n_samples,
            'features_count': n_features,
            'predictions_made': len(predictions),
            'avg_confidence': avg_confidence,
            'bullish_ratio': bullish_ratio
        }
    
    def _demo_full_trading_flow(self):
        """演示完整交易流程"""
        print("🔄 演示完整交易流程...")
        
        # 确保必要组件已初始化
        if not self.paper_trading:
            self.paper_trading = PaperTradingEngine({'initial_capital': 10000.0})
        
        # 模拟完整交易周期
        print("🎯 执行模拟交易周期...")
        
        trading_results = []
        base_price = 2000.0
        
        for cycle in range(3):
            print(f"\n   📊 交易周期 #{cycle + 1}")
            
            # 1. 市场数据更新
            current_price = base_price + np.random.normal(0, 10)
            self.paper_trading.update_market_data('XAUUSD', current_price)
            print(f"      💰 当前价格: ${current_price:.2f}")
            
            # 2. AI信号生成
            ai_signal = np.random.choice(['buy', 'sell', 'hold'], p=[0.3, 0.3, 0.4])
            ai_confidence = np.random.uniform(0.5, 0.9)
            print(f"      🤖 AI信号: {ai_signal} (置信度: {ai_confidence:.1%})")
            
            # 3. 交易决策
            if ai_signal != 'hold' and ai_confidence > 0.65:
                # 执行交易
                order_id = self.paper_trading.submit_order(
                    symbol='XAUUSD',
                    side=ai_signal,
                    quantity=0.01,
                    order_type=OrderType.MARKET,
                    tag=f'cycle_{cycle + 1}'
                )
                
                # 更新价格以触发成交
                fill_price = current_price * (1.001 if ai_signal == 'buy' else 0.999)
                self.paper_trading.update_market_data('XAUUSD', fill_price)
                
                order = self.paper_trading.get_order(order_id)
                if order and order.status == OrderStatus.FILLED:
                    print(f"      ✅ 交易执行: {ai_signal} 0.01手 @ ${order.filled_price:.2f}")
                    
                    trading_results.append({
                        'cycle': cycle + 1,
                        'signal': ai_signal,
                        'confidence': ai_confidence,
                        'price': order.filled_price,
                        'executed': True
                    })
                else:
                    print(f"      ❌ 交易失败")
                    trading_results.append({
                        'cycle': cycle + 1,
                        'signal': ai_signal,
                        'confidence': ai_confidence,
                        'executed': False
                    })
            else:
                print(f"      ⏸️  无交易 (信号: {ai_signal}, 置信度低)")
                trading_results.append({
                    'cycle': cycle + 1,
                    'signal': ai_signal,
                    'confidence': ai_confidence,
                    'executed': False
                })
            
            # 4. 风险监控
            portfolio = self.paper_trading.get_portfolio_summary()
            current_pnl = portfolio.get('total_pnl', 0)
            max_drawdown = portfolio.get('max_drawdown', 0)
            
            print(f"      📊 当前盈亏: ${current_pnl:.2f}")
            
            # 风险检查
            if current_pnl < -50:  # 损失超过50美元
                print(f"      ⚠️  风险警告: 损失过大")
                
                # 发送风险告警
                if self.monitoring_system:
                    self.monitoring_system.send_custom_alert(
                        alert_type=AlertType.RISK,
                        level=AlertLevel.WARNING,
                        title="交易损失告警",
                        message=f"当前损失: ${abs(current_pnl):.2f}",
                        source="TradingFlow"
                    )
            
            time.sleep(1)
        
        # 最终结果统计
        executed_trades = [r for r in trading_results if r['executed']]
        success_rate = len(executed_trades) / len(trading_results) if trading_results else 0
        
        final_portfolio = self.paper_trading.get_portfolio_summary()
        
        print(f"\n   📈 交易流程完成:")
        print(f"      总周期数: {len(trading_results)}")
        print(f"      执行交易: {len(executed_trades)}")
        print(f"      执行成功率: {success_rate:.1%}")
        print(f"      最终盈亏: ${final_portfolio.get('total_pnl', 0):.2f}")
        print(f"      总交易次数: {final_portfolio.get('trade_count', 0)}")
        
        # 保存演示结果
        self.demo_results['full_trading_flow'] = {
            'cycles_completed': len(trading_results),
            'trades_executed': len(executed_trades),
            'execution_rate': success_rate,
            'final_pnl': final_portfolio.get('total_pnl', 0),
            'total_trades': final_portfolio.get('trade_count', 0),
            'final_portfolio': final_portfolio
        }
    
    def _evaluate_system_performance(self):
        """评估系统性能"""
        print("📊 系统性能评估...")
        
        # 收集性能指标
        performance_metrics = {}
        
        # 纸上交易性能
        if 'paper_trading' in self.demo_results:
            pt_data = self.demo_results['paper_trading']
            performance_metrics['纸上交易'] = {
                '初始资金': f"${self.config['trading']['initial_capital']:.2f}",
                '当前权益': f"${pt_data.get('equity', 0):.2f}",
                '总盈亏': f"${pt_data.get('total_pnl', 0):.2f}",
                '交易次数': pt_data.get('trade_count', 0),
                '胜率': f"{pt_data.get('win_rate', 0):.1%}"
            }
        
        # 监控系统性能
        if 'monitoring_system' in self.demo_results:
            ms_data = self.demo_results['monitoring_system']
            performance_metrics['监控系统'] = {
                '系统状态': ms_data['system_status'].get('监控系统状态', '未知'),
                '发送告警': ms_data.get('alerts_sent', 0),
                '指标更新': ms_data.get('metrics_updated', 0)
            }
        
        # AI系统性能
        if 'ai_integration' in self.demo_results:
            ai_data = self.demo_results['ai_integration']
            performance_metrics['AI系统'] = {
                '处理样本': ai_data.get('samples_processed', 0),
                '特征维度': ai_data.get('features_count', 0),
                '预测次数': ai_data.get('predictions_made', 0),
                '平均置信度': f"{ai_data.get('avg_confidence', 0):.1%}",
                '看涨比例': f"{ai_data.get('bullish_ratio', 0):.1%}"
            }
        
        # 完整流程性能
        if 'full_trading_flow' in self.demo_results:
            flow_data = self.demo_results['full_trading_flow']
            performance_metrics['交易流程'] = {
                '完成周期': flow_data.get('cycles_completed', 0),
                '执行交易': flow_data.get('trades_executed', 0),
                '执行成功率': f"{flow_data.get('execution_rate', 0):.1%}",
                '流程盈亏': f"${flow_data.get('final_pnl', 0):.2f}"
            }
        
        # 显示性能报告
        print("\n📋 系统性能报告:")
        print("=" * 50)
        
        for category, metrics in performance_metrics.items():
            print(f"\n🔹 {category}:")
            for metric, value in metrics.items():
                print(f"   {metric}: {value}")
        
        # 系统评分
        score = self._calculate_system_score()
        print(f"\n🏆 系统综合评分: {score:.1f}/100")
        
        # 保存性能评估结果
        self.demo_results['performance_evaluation'] = {
            'metrics': performance_metrics,
            'system_score': score,
            'evaluation_time': datetime.now().isoformat()
        }
    
    def _calculate_system_score(self) -> float:
        """计算系统综合评分"""
        score = 0.0
        max_score = 100.0
        
        # 各模块权重
        weights = {
            'paper_trading': 20,
            'broker_interface': 15,
            'monitoring_system': 20,
            'data_collection': 15,
            'ai_integration': 15,
            'full_trading_flow': 15
        }
        
        for module, weight in weights.items():
            if module in self.demo_results:
                # 根据模块完成情况评分
                module_score = weight
                
                # 特殊评分逻辑
                if module == 'paper_trading':
                    pnl = self.demo_results[module].get('total_pnl', 0)
                    if pnl > 0:
                        module_score = weight
                    elif pnl > -50:
                        module_score = weight * 0.8
                    else:
                        module_score = weight * 0.5
                
                elif module == 'ai_integration':
                    confidence = self.demo_results[module].get('avg_confidence', 0)
                    if confidence > 0.7:
                        module_score = weight
                    elif confidence > 0.5:
                        module_score = weight * 0.8
                    else:
                        module_score = weight * 0.6
                
                elif module == 'full_trading_flow':
                    exec_rate = self.demo_results[module].get('execution_rate', 0)
                    if exec_rate > 0.6:
                        module_score = weight
                    elif exec_rate > 0.3:
                        module_score = weight * 0.8
                    else:
                        module_score = weight * 0.5
                
                score += module_score
        
        return min(score, max_score)
    
    def _generate_demo_report(self):
        """生成演示报告"""
        print("\n📄 生成演示报告...")
        
        # 创建报告
        report = {
            'demo_info': {
                'phase': '第三阶段 - 实时交易系统',
                'start_time': datetime.now().isoformat(),
                'version': '3.0.0',
                'components_tested': list(self.demo_results.keys())
            },
            'results': self.demo_results,
            'summary': {
                'total_components': len(self.demo_results),
                'successful_components': len([k for k in self.demo_results.keys()]),
                'system_score': self.demo_results.get('performance_evaluation', {}).get('system_score', 0)
            }
        }
        
        # 保存报告
        report_file = f"logs/phase3_demo_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs('logs', exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"   ✅ 报告已保存: {report_file}")
        
        # 显示摘要
        print(f"\n📊 演示摘要:")
        print(f"   测试组件: {report['summary']['total_components']}")
        print(f"   成功组件: {report['summary']['successful_components']}")
        print(f"   系统评分: {report['summary']['system_score']:.1f}/100")
        
        # 建议和下一步行动
        self._provide_recommendations()
    
    def _provide_recommendations(self):
        """提供建议和下一步行动"""
        print("\n💡 系统优化建议:")
        print("=" * 50)
        
        recommendations = []
        
        # 基于演示结果提供建议
        if 'full_trading_flow' in self.demo_results:
            flow_data = self.demo_results['full_trading_flow']
            final_pnl = flow_data.get('final_pnl', 0)
            exec_rate = flow_data.get('execution_rate', 0)
            
            if final_pnl < 0:
                recommendations.append("🔧 优化AI模型策略，提高预测准确性")
                recommendations.append("⚖️  调整风险管理参数，减少损失")
            
            if exec_rate < 0.5:
                recommendations.append("🎯 优化信号过滤机制，提高执行效率")
                recommendations.append("📊 调整置信度阈值设置")
        
        if 'ai_integration' in self.demo_results:
            ai_data = self.demo_results['ai_integration']
            avg_confidence = ai_data.get('avg_confidence', 0)
            
            if avg_confidence < 0.7:
                recommendations.append("🧠 增加训练数据，提高模型置信度")
                recommendations.append("🔍 优化特征工程，提升预测质量")
        
        # 通用建议
        recommendations.extend([
            "📈 实施更严格的回测验证",
            "🔄 设置自动化模型重训练机制",
            "📱 完善实时监控和告警系统",
            "🏦 集成更多券商API接口",
            "📊 增加更多技术指标和信号源",
            "💾 实施完整的数据备份策略"
        ])
        
        for i, rec in enumerate(recommendations[:6], 1):
            print(f"{i}. {rec}")
        
        print(f"\n🚀 下一步行动:")
        print("1. 🔧 根据演示结果优化系统参数")
        print("2. 📊 收集更多历史数据进行全面回测")
        print("3. 🏦 配置真实券商API进行小资金测试")
        print("4. 📱 部署生产环境监控系统")
        print("5. 🎯 开始小规模实盘交易验证")
    
    def _cleanup(self):
        """清理资源"""
        try:
            if self.monitoring_system:
                self.monitoring_system.stop()
            
            self.demo_running = False
            print("🧹 资源清理完成")
            
        except Exception as e:
            logger.error(f"清理资源失败: {e}")


def main():
    """主函数"""
    print("🎯 AI黄金交易系统 - 第三阶段演示")
    print("🚀 实时交易系统完整功能展示")
    print("=" * 60)
    
    try:
        # 创建并运行演示
        demo = Phase3Demo()
        demo.run_demo()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        logger.error(f"主程序异常: {e}")
    
    print("\n👋 感谢使用AI黄金交易系统！")


if __name__ == "__main__":
    main() 
