"""
第二阶段演示脚本：AI模型开发
展示完整的AI模型开发流程，包括训练、回测和模拟交易
"""

import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from loguru import logger
import os
import sys

# 添加源代码路径
sys.path.append('src')

from src.config_utils import ConfigValidationError, ensure_runtime_directories, load_config as load_validated_config
from src.data_collector import DataCollector
from src.feature_engineer import FeatureEngineer
from src.ai_models import AIModelManager
from src.risk_manager import RiskManager
from src.backtester import Backtester
from src.trader import TradingEngine


def setup_environment():
    """设置运行环境"""
    # 创建必要目录
    directories = ['data', 'logs', 'models', 'reports']
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)
    
    # 设置日志
    logger.add(
        "logs/phase2_demo_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="7 days",
        level="INFO"
    )


def load_config():
    """加载配置"""
    try:
        return load_validated_config('config/config.yaml')
    except ConfigValidationError as e:
        logger.error(f"配置验证失败: {e}")
        return None
    except Exception as e:
        logger.error(f"加载配置失败: {e}")
        return None


def demonstrate_ai_model_development():
    """演示AI模型开发完整流程"""
    
    print("🚀 AI黄金交易系统 - 第二阶段：AI模型开发演示")
    print("=" * 60)
    
    # 1. 环境设置
    print("\n📁 步骤1: 环境设置")
    setup_environment()
    config = load_config()
    if not config:
        print("❌ 配置加载失败")
        return
    ensure_runtime_directories(config)
    print("✅ 环境设置完成")
    
    # 2. 初始化模块
    print("\n🔧 步骤2: 初始化核心模块")
    try:
        data_collector = DataCollector(config)
        feature_engineer = FeatureEngineer(config)
        ai_model_manager = AIModelManager(config)
        risk_manager = RiskManager(config)
        backtester = Backtester(config)
        trading_engine = TradingEngine(
            config, data_collector, feature_engineer, 
            ai_model_manager, risk_manager
        )
        print("✅ 所有模块初始化完成")
    except Exception as e:
        print(f"❌ 模块初始化失败: {e}")
        return
    
    # 3. 数据采集与准备
    print("\n📊 步骤3: 数据采集与特征工程")
    try:
        print("   🔄 获取历史数据...")
        historical_data = data_collector.get_historical_data(
            period="3mo",  # 3个月数据足够演示
            interval="1m"
        )
        
        if historical_data.empty:
            print("❌ 历史数据获取失败")
            return
        
        print(f"   ✅ 获取到 {len(historical_data)} 条历史数据")
        
        print("   🔄 创建特征矩阵...")
        feature_data = feature_engineer.create_feature_matrix(
            historical_data, include_targets=True
        )
        
        if feature_data.empty:
            print("❌ 特征创建失败")
            return
        
        print(f"   ✅ 创建了 {len(feature_data.columns)} 个特征")
        
        print("   🔄 特征选择...")
        selected_features = feature_engineer.select_features(
            feature_data,
            target_column='Future_Direction_1',
            method='correlation',
            max_features=25
        )
        
        print(f"   ✅ 选择了 {len(selected_features)} 个最重要特征")
        
    except Exception as e:
        print(f"❌ 数据准备失败: {e}")
        return
    
    # 4. AI模型训练
    print("\n🤖 步骤4: AI模型训练")
    try:
        print("   🔄 训练集成模型...")
        training_results = ai_model_manager.train_ensemble_models(
            feature_data, selected_features, target_column='Future_Direction_1'
        )
        
        if not training_results:
            print("❌ 模型训练失败")
            return
        
        print("   ✅ 模型训练完成")
        print("\n   📈 模型性能:")
        for model_name, performance in training_results.items():
            accuracy = performance.get('test_accuracy', 0)
            f1 = performance.get('test_f1', 0)
            print(f"     • {model_name}: 准确率={accuracy:.3f}, F1={f1:.3f}")
        
        # 保存模型
        model_path = f"models/phase2_demo_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        ai_model_manager.save_models(model_path)
        print(f"   💾 模型已保存: {model_path}")
        
    except Exception as e:
        print(f"❌ 模型训练失败: {e}")
        return
    
    # 5. 专业回测
    print("\n📈 步骤5: 专业回测验证")
    try:
        print("   🔄 运行回测...")
        backtest_result = backtester.run_backtest(
            feature_data, ai_model_manager, feature_engineer, selected_features
        )
        
        print("   ✅ 回测完成")
        print("\n   📊 回测结果:")
        print(f"     • 总交易次数: {backtest_result.total_trades}")
        print(f"     • 胜率: {backtest_result.win_rate:.1%}")
        print(f"     • 总盈亏: ${backtest_result.total_pnl:.2f}")
        print(f"     • 最大回撤: {backtest_result.max_drawdown:.2%}")
        print(f"     • 夏普比率: {backtest_result.sharpe_ratio:.2f}")
        print(f"     • 盈利因子: {backtest_result.profit_factor:.2f}")
        
        # 保存回测结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file = f"reports/phase2_backtest_{timestamp}.json"
        backtester.save_results(backtest_result, result_file)
        print(f"   💾 回测结果已保存: {result_file}")
        
        # 检查是否达到预期标准
        if (backtest_result.win_rate >= 0.55 and 
            backtest_result.max_drawdown <= 0.15 and 
            backtest_result.profit_factor >= 1.2):
            print("   🎉 回测结果达到预期标准！")
        else:
            print("   ⚠️  回测结果未达到所有预期标准，需要优化")
        
    except Exception as e:
        print(f"❌ 回测失败: {e}")
        return
    
    # 6. 风险管理测试
    print("\n🛡️ 步骤6: 风险管理系统测试")
    try:
        print("   🔄 测试风险控制...")
        
        # 模拟交易测试
        test_position = {
            'symbol': 'XAUUSD',
            'side': 'buy',
            'size': 0.1,
            'entry_price': 2000.0,
            'stop_loss': 1980.0,
            'take_profit': 2020.0
        }
        
        can_trade, risk_msg, adjusted_size = risk_manager.check_position_risk(
            test_position['symbol'],
            test_position['side'],
            test_position['entry_price'],
            test_position['stop_loss'],
            test_position['size']
        )
        
        print(f"   📋 风险检查: {risk_msg}")
        print(f"   📐 建议仓位: {adjusted_size:.4f}")
        
        if can_trade:
            print("   ✅ 风险管理系统正常工作")
        else:
            print("   ⚠️  风险控制已生效")
        
        # 显示风险限制设置
        risk_summary = risk_manager.get_risk_summary()
        print(f"   💰 日损失限制: ${risk_manager.max_daily_loss}")
        print(f"   📊 最大持仓数: {risk_manager.max_positions}")
        print(f"   📉 回撤限制: {risk_manager.drawdown_limit:.1%}")
        
    except Exception as e:
        print(f"❌ 风险管理测试失败: {e}")
        return
    
    # 7. 模拟交易演示
    print("\n⚡ 步骤7: 模拟交易演示")
    try:
        print("   🔄 启动模拟交易引擎...")
        
        # 获取实时数据
        realtime_data = data_collector.get_realtime_data()
        if realtime_data:
            print(f"   💹 当前价格: ${realtime_data['close']:.2f}")
            print(f"   📊 买卖价差: ${realtime_data.get('spread', 0):.2f}")
        
        # 演示交易信号生成
        if feature_data is not None and len(selected_features) > 0:
            # 使用最新数据进行预测
            latest_features = feature_data[selected_features].dropna().iloc[-1].values
            
            if not np.isnan(latest_features).any():
                prediction, details = ai_model_manager.predict_ensemble(
                    latest_features, method='weighted'
                )
                
                if prediction is not None:
                    confidence = max([max(probs) for probs in details.get('individual_probabilities', {}).values()]) if details.get('individual_probabilities') else 0.5
                    
                    print(f"   🤖 AI预测: {'看涨' if prediction == 1 else '看跌'}")
                    print(f"   📈 置信度: {confidence:.1%}")
                    
                    if confidence >= 0.6:
                        print("   ✅ 信号强度足够，可以考虑交易")
                    else:
                        print("   ⚠️  信号强度不足，建议观望")
        
        print("   💡 模拟交易引擎准备就绪（未启动实时交易）")
        
    except Exception as e:
        print(f"❌ 模拟交易演示失败: {e}")
    
    # 8. 总结
    print("\n🎯 第二阶段完成总结")
    print("=" * 60)
    print("✅ 数据采集与特征工程")
    print("✅ AI模型训练与集成")
    print("✅ 专业回测验证")
    print("✅ 风险管理系统")
    print("✅ 模拟交易框架")
    print()
    print("📋 下一步建议:")
    print("1. 根据回测结果优化模型参数")
    print("2. 增加更多技术指标和特征")
    print("3. 实施更复杂的集成策略")
    print("4. 进行更长时间的模拟测试")
    print("5. 考虑加入情绪指标和基本面数据")
    print()
    print("⚠️  重要提醒:")
    print("• 本系统仅供学习研究，实盘前需充分测试")
    print("• 严格遵守30美元日损失限制")
    print("• 持续监控模型性能，及时调整")
    print()
    print("🎉 第二阶段演示完成！")


if __name__ == "__main__":
    demonstrate_ai_model_development() 
