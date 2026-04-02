"""
Backtesting module.
Implements strict time-series validation to avoid look-ahead leakage.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import json


@dataclass
class Trade:
    """Backtest trade record."""
    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    size: float = 0.01
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    pnl: float = 0.0
    status: str = 'open'  # 'open', 'closed', 'stopped'
    reason: str = ''  # 平仓原因


@dataclass
class BacktestResult:
    """Backtest result summary."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    avg_winning_trade: float = 0.0
    avg_losing_trade: float = 0.0
    profit_factor: float = 0.0
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    equity_curve: List[float] = None


class Backtester:
    """Backtester with a strict strategy validation workflow."""
    
    def __init__(self, config: Dict):
        """
        Initialize the backtester.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.backtest_config = config['backtest']
        self.initial_capital = self.backtest_config['initial_capital']
        self.commission = self.backtest_config['commission']
        self.slippage = self.backtest_config['slippage']
        
        # Trade tracking
        self.trades: List[Trade] = []
        self.current_positions: Dict[str, Trade] = {}
        self.equity_curve = []
        self.daily_returns = []
        
        # Performance tracking
        self.current_capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.max_drawdown = 0.0
        
        logger.info(f"Backtester initialized - initial capital: ${self.initial_capital}")
    
    def reset(self):
        """Reset backtest state."""
        self.trades.clear()
        self.current_positions.clear()
        self.equity_curve.clear()
        self.daily_returns.clear()
        self.current_capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.max_drawdown = 0.0
        
        logger.info("Backtester state reset")
    
    def open_position(self, signal: Dict, current_data: Dict) -> bool:
        """
        开仓
        
        Args:
            signal: 交易信号
            current_data: 当前市场数据
            
        Returns:
            是否成功开仓
        """
        try:
            # 生成交易ID
            trade_id = f"trade_{len(self.trades) + 1:06d}"
            
            # 获取交易参数
            symbol = signal.get('symbol', 'XAUUSD')
            side = signal.get('side', 'buy')
            size = signal.get('size', 0.01)
            entry_price = current_data.get('close', 0)
            
            # 应用滑点
            if side == 'buy':
                entry_price += entry_price * self.slippage
            else:
                entry_price -= entry_price * self.slippage
            
            # 计算止损止盈
            stop_loss = signal.get('stop_loss')
            take_profit = signal.get('take_profit')
            
            if not stop_loss:
                # 默认止损设置
                stop_loss_pips = self.config['trading']['stop_loss_pips']
                if side == 'buy':
                    stop_loss = entry_price - stop_loss_pips
                else:
                    stop_loss = entry_price + stop_loss_pips
            
            if not take_profit:
                # 默认止盈设置
                take_profit_pips = self.config['trading']['take_profit_pips']
                if side == 'buy':
                    take_profit = entry_price + take_profit_pips
                else:
                    take_profit = entry_price - take_profit_pips
            
            # 创建交易记录
            trade = Trade(
                id=trade_id,
                symbol=symbol,
                side=side,
                entry_time=current_data.get('timestamp'),
                entry_price=entry_price,
                size=size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                status='open'
            )
            
            # 检查资金充足
            required_margin = entry_price * size
            if required_margin > self.current_capital * 0.95:  # 保留5%缓冲
                logger.warning(f"Insufficient capital to open position: need ${required_margin:.2f}, available ${self.current_capital:.2f}")
                return False
            
            # 添加交易记录
            self.trades.append(trade)
            self.current_positions[trade_id] = trade
            
            # 扣除手续费
            commission_cost = entry_price * size * self.commission
            self.current_capital -= commission_cost
            
            logger.debug(f"Position opened: {trade_id} {side} {symbol} @ {entry_price:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to open position: {e}")
            return False
    
    def close_position(self, trade_id: str, current_data: Dict, reason: str = 'signal') -> bool:
        """
        平仓
        
        Args:
            trade_id: 交易ID
            current_data: 当前市场数据
            reason: 平仓原因
            
        Returns:
            是否成功平仓
        """
        try:
            if trade_id not in self.current_positions:
                return False
            
            trade = self.current_positions[trade_id]
            exit_price = current_data.get('close', 0)
            
            # 应用滑点
            if trade.side == 'buy':
                exit_price -= exit_price * self.slippage
            else:
                exit_price += exit_price * self.slippage
            
            # 计算盈亏
            if trade.side == 'buy':
                pnl = (exit_price - trade.entry_price) * trade.size
            else:
                pnl = (trade.entry_price - exit_price) * trade.size
            
            # 扣除手续费
            commission_cost = exit_price * trade.size * self.commission
            pnl -= commission_cost
            
            # 更新交易记录
            trade.exit_time = current_data.get('timestamp')
            trade.exit_price = exit_price
            trade.pnl = pnl
            trade.status = 'closed'
            trade.reason = reason
            
            # 更新资金
            self.current_capital += pnl
            
            # 移除持仓
            del self.current_positions[trade_id]
            
            logger.debug(f"Position closed: {trade_id} @ {exit_price:.2f}, PnL: ${pnl:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            return False
    
    def check_stop_conditions(self, current_data: Dict):
        """
        检查止损止盈条件
        
        Args:
            current_data: 当前市场数据
        """
        current_price = current_data.get('close', 0)
        positions_to_close = []
        
        for trade_id, trade in self.current_positions.items():
            should_close = False
            reason = ''
            
            if trade.side == 'buy':
                # 买入持仓检查
                if current_price <= trade.stop_loss:
                    should_close = True
                    reason = 'stop_loss'
                elif current_price >= trade.take_profit:
                    should_close = True
                    reason = 'take_profit'
            else:
                # 卖出持仓检查
                if current_price >= trade.stop_loss:
                    should_close = True
                    reason = 'stop_loss'
                elif current_price <= trade.take_profit:
                    should_close = True
                    reason = 'take_profit'
            
            if should_close:
                positions_to_close.append((trade_id, reason))
        
        # 执行平仓
        for trade_id, reason in positions_to_close:
            self.close_position(trade_id, current_data, reason)
    
    def update_equity(self, current_data: Dict):
        """
        更新权益曲线
        
        Args:
            current_data: 当前市场数据
        """
        current_price = current_data.get('close', 0)
        unrealized_pnl = 0.0
        
        # 计算未实现盈亏
        for trade in self.current_positions.values():
            if trade.side == 'buy':
                unrealized = (current_price - trade.entry_price) * trade.size
            else:
                unrealized = (trade.entry_price - current_price) * trade.size
            unrealized_pnl += unrealized
        
        # 总权益 = 现金 + 未实现盈亏
        total_equity = self.current_capital + unrealized_pnl
        self.equity_curve.append(total_equity)
        
        # 更新最大回撤
        if total_equity > self.peak_capital:
            self.peak_capital = total_equity
        else:
            drawdown = (self.peak_capital - total_equity) / self.peak_capital
            self.max_drawdown = max(self.max_drawdown, drawdown)
    
    def run_backtest(self, data: pd.DataFrame, ai_model_manager, 
                    feature_engineer, selected_features: List[str]) -> BacktestResult:
        """
        运行回测
        
        Args:
            data: 历史数据
            ai_model_manager: AI模型管理器
            feature_engineer: 特征工程器
            selected_features: 选择的特征
            
        Returns:
            回测结果
        """
        logger.info("开始运行回测...")
        
        # 重置状态
        self.reset()
        
        # 确保数据按时间排序
        data = data.sort_index()
        
        # 创建特征矩阵
        feature_data = feature_engineer.create_feature_matrix(data, include_targets=False)
        
        if feature_data.empty:
            logger.error("特征数据为空")
            return BacktestResult()
        
        # 回测主循环
        for i, (timestamp, row) in enumerate(feature_data.iterrows()):
            try:
                # 准备当前数据
                current_data = {
                    'timestamp': timestamp,
                    'open': row.get('Open', 0),
                    'high': row.get('High', 0),
                    'low': row.get('Low', 0),
                    'close': row.get('Close', 0),
                    'volume': row.get('Volume', 0)
                }
                
                # 检查止损止盈
                self.check_stop_conditions(current_data)
                
                # 获取特征向量
                try:
                    features = pd.to_numeric(row[selected_features], errors='coerce').to_numpy(dtype=np.float64)
                    
                    # 跳过包含NaN的数据
                    if np.isnan(features).any():
                        continue
                    
                    # AI预测
                    prediction, details = ai_model_manager.predict_ensemble(
                        features, method='voting'
                    )
                    
                    if prediction is not None:
                        # 生成交易信号
                        signal = self._generate_signal(prediction, current_data, details)
                        
                        if signal:
                            # 检查是否已有持仓
                            if len(self.current_positions) == 0:
                                self.open_position(signal, current_data)
                            elif signal.get('action') == 'close':
                                # 平仓信号
                                for trade_id in list(self.current_positions.keys()):
                                    self.close_position(trade_id, current_data, 'signal')
                
                except Exception as e:
                    logger.debug(f"预测出错: {e}")
                    continue
                
                # 更新权益曲线
                self.update_equity(current_data)
                
                # 进度显示
                if i % 1000 == 0:
                    progress = (i / len(feature_data)) * 100
                    logger.info(f"回测进度: {progress:.1f}% ({i}/{len(feature_data)})")
                
            except Exception as e:
                logger.debug(f"回测循环出错: {e}")
                continue
        
        # 关闭所有未平仓位
        final_data = {
            'timestamp': feature_data.index[-1],
            'close': feature_data['Close'].iloc[-1]
        }
        
        for trade_id in list(self.current_positions.keys()):
            self.close_position(trade_id, final_data, 'backtest_end')
        
        # 计算回测结果
        result = self._calculate_results()
        
        logger.info("回测完成")
        return result
    
    def _generate_signal(self, prediction: int, current_data: Dict, 
                        details: Dict) -> Optional[Dict]:
        """
        根据AI预测生成交易信号
        
        Args:
            prediction: AI预测结果 (0=跌, 1=涨)
            current_data: 当前数据
            details: 预测详情
            
        Returns:
            交易信号字典
        """
        try:
            # 获取预测置信度
            confidence = self._calculate_confidence(details)
            
            # 只有高置信度的信号才交易
            if confidence < 0.6:  # 60%置信度阈值
                return None
            
            current_price = current_data['close']
            
            if prediction == 1:  # 看涨信号
                return {
                    'action': 'open',
                    'side': 'buy',
                    'symbol': 'XAUUSD',
                    'size': 0.01,
                    'confidence': confidence,
                    'stop_loss': current_price - self.config['trading']['stop_loss_pips'],
                    'take_profit': current_price + self.config['trading']['take_profit_pips']
                }
            elif prediction == 0:  # 看跌信号
                return {
                    'action': 'open',
                    'side': 'sell',
                    'symbol': 'XAUUSD',
                    'size': 0.01,
                    'confidence': confidence,
                    'stop_loss': current_price + self.config['trading']['stop_loss_pips'],
                    'take_profit': current_price - self.config['trading']['take_profit_pips']
                }
            
            return None
            
        except Exception as e:
            logger.debug(f"生成信号失败: {e}")
            return None
    
    def _calculate_confidence(self, details: Dict) -> float:
        """
        计算预测置信度
        
        Args:
            details: 预测详情
            
        Returns:
            置信度 (0-1)
        """
        try:
            probabilities = details.get('individual_probabilities', {})
            
            if not probabilities:
                return 0.5  # 默认置信度
            
            # 计算所有模型的平均置信度
            confidences = []
            for model_probs in probabilities.values():
                if isinstance(model_probs, (list, np.ndarray)) and len(model_probs) > 1:
                    # 取最大概率作为置信度
                    confidences.append(max(model_probs))
            
            if confidences:
                return np.mean(confidences)
            else:
                return 0.5
                
        except Exception as e:
            logger.debug(f"计算置信度失败: {e}")
            return 0.5
    
    def _calculate_results(self) -> BacktestResult:
        """
        计算回测结果
        
        Returns:
            回测结果对象
        """
        try:
            if not self.trades:
                return BacktestResult()
            
            closed_trades = [t for t in self.trades if t.status == 'closed']
            
            if not closed_trades:
                return BacktestResult(total_trades=len(self.trades))
            
            # 基础统计
            total_trades = len(closed_trades)
            winning_trades = len([t for t in closed_trades if t.pnl > 0])
            losing_trades = len([t for t in closed_trades if t.pnl < 0])
            
            # 盈亏统计
            total_pnl = sum(t.pnl for t in closed_trades)
            winning_pnls = [t.pnl for t in closed_trades if t.pnl > 0]
            losing_pnls = [t.pnl for t in closed_trades if t.pnl < 0]
            
            # 胜率
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # 平均盈亏
            avg_winning_trade = np.mean(winning_pnls) if winning_pnls else 0
            avg_losing_trade = np.mean(losing_pnls) if losing_pnls else 0
            
            # 盈利因子
            gross_profit = sum(winning_pnls) if winning_pnls else 0
            gross_loss = abs(sum(losing_pnls)) if losing_pnls else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # 计算比率
            returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
            
            # 夏普比率
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
            else:
                sharpe_ratio = 0
            
            # 索提诺比率
            negative_returns = returns[returns < 0]
            if len(negative_returns) > 1 and np.std(negative_returns) > 0:
                sortino_ratio = np.mean(returns) / np.std(negative_returns) * np.sqrt(252)
            else:
                sortino_ratio = 0
            
            # 卡尔玛比率
            annual_return = ((self.equity_curve[-1] / self.equity_curve[0]) ** (252 / len(self.equity_curve))) - 1
            calmar_ratio = annual_return / self.max_drawdown if self.max_drawdown > 0 else 0
            
            # 连续盈亏
            consecutive_wins = 0
            consecutive_losses = 0
            max_consecutive_wins = 0
            max_consecutive_losses = 0
            
            for trade in closed_trades:
                if trade.pnl > 0:
                    consecutive_wins += 1
                    consecutive_losses = 0
                    max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
                else:
                    consecutive_losses += 1
                    consecutive_wins = 0
                    max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            
            # 时间范围
            start_date = closed_trades[0].entry_time
            end_date = closed_trades[-1].exit_time
            
            return BacktestResult(
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                win_rate=win_rate,
                total_pnl=total_pnl,
                max_drawdown=self.max_drawdown,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                max_consecutive_wins=max_consecutive_wins,
                max_consecutive_losses=max_consecutive_losses,
                avg_winning_trade=avg_winning_trade,
                avg_losing_trade=avg_losing_trade,
                profit_factor=profit_factor,
                start_date=start_date,
                end_date=end_date,
                equity_curve=self.equity_curve.copy()
            )
            
        except Exception as e:
            logger.error(f"计算回测结果失败: {e}")
            return BacktestResult()
    
    def plot_results(self, result: BacktestResult, save_path: str = None):
        """
        绘制回测结果图表
        
        Args:
            result: 回测结果
            save_path: 保存路径
        """
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # 权益曲线
            if result.equity_curve:
                ax1.plot(result.equity_curve, label='权益曲线', color='blue')
                ax1.axhline(y=self.initial_capital, color='red', linestyle='--', label='初始资金')
                ax1.set_title('权益曲线')
                ax1.set_ylabel('权益 ($)')
                ax1.legend()
                ax1.grid(True)
            
            # 回撤曲线
            if result.equity_curve:
                peak = np.maximum.accumulate(result.equity_curve)
                drawdown = (peak - result.equity_curve) / peak
                ax2.fill_between(range(len(drawdown)), drawdown, alpha=0.3, color='red')
                ax2.set_title(f'回撤曲线 (最大回撤: {result.max_drawdown:.2%})')
                ax2.set_ylabel('回撤比例')
                ax2.grid(True)
            
            # 交易统计
            labels = ['盈利交易', '亏损交易']
            sizes = [result.winning_trades, result.losing_trades]
            colors = ['green', 'red']
            ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax3.set_title(f'交易分布 (胜率: {result.win_rate:.1%})')
            
            # 性能指标
            metrics = [
                f'总交易次数: {result.total_trades}',
                f'总盈亏: ${result.total_pnl:.2f}',
                f'胜率: {result.win_rate:.1%}',
                f'盈利因子: {result.profit_factor:.2f}',
                f'夏普比率: {result.sharpe_ratio:.2f}',
                f'最大回撤: {result.max_drawdown:.2%}',
                f'平均盈利: ${result.avg_winning_trade:.2f}',
                f'平均亏损: ${result.avg_losing_trade:.2f}'
            ]
            
            ax4.axis('off')
            ax4.text(0.1, 0.9, '回测统计', fontsize=14, fontweight='bold', transform=ax4.transAxes)
            for i, metric in enumerate(metrics):
                ax4.text(0.1, 0.8 - i*0.08, metric, fontsize=10, transform=ax4.transAxes)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"回测图表已保存: {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"绘制回测结果失败: {e}")
    
    def save_results(self, result: BacktestResult, file_path: str):
        """
        保存回测结果
        
        Args:
            result: 回测结果
            file_path: 保存路径
        """
        try:
            # 转换为可序列化的格式
            result_dict = {
                'backtest_summary': {
                    'total_trades': result.total_trades,
                    'winning_trades': result.winning_trades,
                    'losing_trades': result.losing_trades,
                    'win_rate': result.win_rate,
                    'total_pnl': result.total_pnl,
                    'max_drawdown': result.max_drawdown,
                    'sharpe_ratio': result.sharpe_ratio,
                    'sortino_ratio': result.sortino_ratio,
                    'calmar_ratio': result.calmar_ratio,
                    'profit_factor': result.profit_factor,
                    'avg_winning_trade': result.avg_winning_trade,
                    'avg_losing_trade': result.avg_losing_trade,
                    'max_consecutive_wins': result.max_consecutive_wins,
                    'max_consecutive_losses': result.max_consecutive_losses,
                    'start_date': result.start_date.isoformat() if result.start_date else None,
                    'end_date': result.end_date.isoformat() if result.end_date else None
                },
                'trades': [
                    {
                        'id': trade.id,
                        'symbol': trade.symbol,
                        'side': trade.side,
                        'entry_time': trade.entry_time.isoformat() if trade.entry_time else None,
                        'entry_price': trade.entry_price,
                        'exit_time': trade.exit_time.isoformat() if trade.exit_time else None,
                        'exit_price': trade.exit_price,
                        'size': trade.size,
                        'pnl': trade.pnl,
                        'status': trade.status,
                        'reason': trade.reason
                    }
                    for trade in self.trades if trade.status == 'closed'
                ],
                'equity_curve': result.equity_curve,
                'config': self.config
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=2)
            
            logger.info(f"回测结果已保存: {file_path}")
            
        except Exception as e:
            logger.error(f"保存回测结果失败: {e}")
    
    def get_trade_summary(self) -> pd.DataFrame:
        """
        获取交易汇总表
        
        Returns:
            交易汇总DataFrame
        """
        try:
            closed_trades = [t for t in self.trades if t.status == 'closed']
            
            if not closed_trades:
                return pd.DataFrame()
            
            trade_data = []
            for trade in closed_trades:
                trade_data.append({
                    'ID': trade.id,
                    '品种': trade.symbol,
                    '方向': trade.side,
                    '开仓时间': trade.entry_time,
                    '开仓价格': trade.entry_price,
                    '平仓时间': trade.exit_time,
                    '平仓价格': trade.exit_price,
                    '仓位': trade.size,
                    '盈亏': trade.pnl,
                    '平仓原因': trade.reason
                })
            
            df = pd.DataFrame(trade_data)
            return df
            
        except Exception as e:
            logger.error(f"创建交易汇总失败: {e}")
            return pd.DataFrame() 
