"""
Backtesting module.
Implements strict time-series validation to avoid look-ahead leakage.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from loguru import logger
from src.config_utils import get_effective_confidence_threshold
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import seaborn as sns
except ImportError:
    sns = None
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
    exit_trigger_price: Optional[float] = None
    managed_stop_loss: Optional[float] = None
    protection_updated_on_exit_bar: bool = False
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
        self.trading_config = config['trading']
        self.initial_capital = self.backtest_config['initial_capital']
        self.commission = self.backtest_config['commission']
        self.slippage = self.backtest_config['slippage']
        self.signal_confidence_threshold = get_effective_confidence_threshold(config, "backtest")
        self.exit_management = self._load_exit_management_config()
        
        # Trade tracking
        self.trades: List[Trade] = []
        self.current_positions: Dict[str, Trade] = {}
        self.equity_curve = []
        self.daily_returns = []
        
        # Performance tracking
        self.current_capital = self.initial_capital
        self.peak_capital = self.initial_capital
        self.max_drawdown = 0.0
        
        logger.info(
            f"Backtester initialized - initial capital: ${self.initial_capital}, "
            f"signal threshold: {self.signal_confidence_threshold:.3f}"
        )
    
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
            
            if stop_loss is None:
                # 默认止损设置
                stop_loss_pips = self.config['trading']['stop_loss_pips']
                if side == 'buy':
                    stop_loss = entry_price - stop_loss_pips
                else:
                    stop_loss = entry_price + stop_loss_pips
            
            if take_profit is None:
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

            return self._close_position_at_price(
                trade_id,
                timestamp=current_data.get('timestamp'),
                exit_price=float(current_data.get('close', 0) or 0.0),
                reason=reason,
            )
            
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

    def _close_position_at_price(
        self,
        trade_id: str,
        *,
        timestamp: Any,
        exit_price: float,
        reason: str,
        managed_stop_loss: Optional[float] = None,
        protection_updated_on_exit_bar: bool = False,
    ) -> bool:
        try:
            if trade_id not in self.current_positions:
                return False

            trade = self.current_positions[trade_id]
            raw_exit_price = float(exit_price)
            adjusted_exit_price = raw_exit_price

            if trade.side == 'buy':
                adjusted_exit_price -= adjusted_exit_price * self.slippage
                pnl = (adjusted_exit_price - trade.entry_price) * trade.size
            else:
                adjusted_exit_price += adjusted_exit_price * self.slippage
                pnl = (trade.entry_price - adjusted_exit_price) * trade.size

            commission_cost = adjusted_exit_price * trade.size * self.commission
            pnl -= commission_cost

            trade.exit_time = timestamp
            trade.exit_price = adjusted_exit_price
            trade.exit_trigger_price = raw_exit_price
            trade.pnl = pnl
            trade.status = 'closed'
            trade.reason = reason
            trade.managed_stop_loss = managed_stop_loss
            trade.protection_updated_on_exit_bar = protection_updated_on_exit_bar

            self.current_capital += pnl
            del self.current_positions[trade_id]

            logger.debug(f"Position closed: {trade_id} @ {adjusted_exit_price:.2f}, PnL: ${pnl:.2f}")
            return True
        except Exception as e:
            logger.error(f"Failed to close position at price: {e}")
            return False
    
    def check_stop_conditions(self, current_data: Dict) -> bool:
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
        closed_any = False
        for trade_id, reason in positions_to_close:
            closed_any = self.close_position(trade_id, current_data, reason) or closed_any

        return closed_any

    def manage_open_positions(self, current_data: Dict) -> bool:
        """Tighten open-position stop losses using the configured exit-management rules."""
        if not bool(self.exit_management.get("enabled")):
            return False

        updated_any = False
        for trade in self.current_positions.values():
            favorable_price = self._get_favorable_price(current_data, trade.side)
            updated_stop_loss = self._calculate_managed_stop_loss(trade, favorable_price)
            if updated_stop_loss is None:
                continue
            if trade.stop_loss is None or not np.isclose(float(trade.stop_loss), float(updated_stop_loss)):
                trade.stop_loss = float(updated_stop_loss)
                updated_any = True

        return updated_any

    def _simulate_open_positions_on_bar(self, current_data: Dict) -> bool:
        closed_any = False
        for trade_id in list(self.current_positions.keys()):
            trade = self.current_positions.get(trade_id)
            if trade is None:
                continue

            pre_update_hit = self._resolve_intrabar_exit(trade, current_data)
            if pre_update_hit is not None:
                closed_any = self._close_position_at_price(
                    trade_id,
                    timestamp=current_data.get("timestamp"),
                    exit_price=float(pre_update_hit["exit_price"]),
                    reason=str(pre_update_hit["reason"]),
                    managed_stop_loss=float(trade.stop_loss) if trade.stop_loss is not None else None,
                    protection_updated_on_exit_bar=False,
                ) or closed_any
                continue

            updated_stop_loss = None
            if bool(self.exit_management.get("enabled")):
                favorable_price = self._get_favorable_price(current_data, trade.side)
                updated_stop_loss = self._calculate_managed_stop_loss(trade, favorable_price)
                if updated_stop_loss is not None:
                    trade.stop_loss = float(updated_stop_loss)

            post_update_hit = self._resolve_intrabar_exit(trade, current_data)
            if post_update_hit is not None:
                closed_any = self._close_position_at_price(
                    trade_id,
                    timestamp=current_data.get("timestamp"),
                    exit_price=float(post_update_hit["exit_price"]),
                    reason=str(post_update_hit["reason"]),
                    managed_stop_loss=float(trade.stop_loss) if trade.stop_loss is not None else None,
                    protection_updated_on_exit_bar=updated_stop_loss is not None,
                ) or closed_any

        return closed_any

    def _resolve_intrabar_exit(self, trade: Trade, current_data: Dict[str, Any]) -> Optional[Dict[str, float | str]]:
        high_price = self._safe_price(current_data.get("high"))
        low_price = self._safe_price(current_data.get("low"))
        stop_loss = self._safe_price(trade.stop_loss)
        take_profit = self._safe_price(trade.take_profit)
        if high_price is None or low_price is None:
            return None

        stop_hit = False
        take_profit_hit = False
        if trade.side == "buy":
            stop_hit = stop_loss is not None and low_price <= stop_loss
            take_profit_hit = take_profit is not None and high_price >= take_profit
        else:
            stop_hit = stop_loss is not None and high_price >= stop_loss
            take_profit_hit = take_profit is not None and low_price <= take_profit

        if stop_hit and take_profit_hit:
            stop_exit = float(stop_loss if stop_loss is not None else current_data.get("close", 0.0))
            take_profit_exit = float(take_profit if take_profit is not None else current_data.get("close", 0.0))
            return self._pick_conservative_intrabar_exit(trade, stop_exit, take_profit_exit)
        if stop_hit:
            return {"reason": "stop_loss", "exit_price": float(stop_loss)}
        if take_profit_hit:
            return {"reason": "take_profit", "exit_price": float(take_profit)}
        return None

    def _pick_conservative_intrabar_exit(self, trade: Trade, stop_exit: float, take_profit_exit: float) -> Dict[str, float | str]:
        if trade.side == "buy":
            stop_pnl = (stop_exit - trade.entry_price) * trade.size
            take_profit_pnl = (take_profit_exit - trade.entry_price) * trade.size
        else:
            stop_pnl = (trade.entry_price - stop_exit) * trade.size
            take_profit_pnl = (trade.entry_price - take_profit_exit) * trade.size
        if stop_pnl <= take_profit_pnl:
            return {"reason": "stop_loss", "exit_price": float(stop_exit)}
        return {"reason": "take_profit", "exit_price": float(take_profit_exit)}

    def _get_favorable_price(self, current_data: Dict[str, Any], side: str) -> float:
        default_price = float(current_data.get("close", 0) or 0.0)
        if side == "buy":
            return float(current_data.get("high", default_price) or default_price)
        return float(current_data.get("low", default_price) or default_price)

    def _safe_price(self, value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(parsed):
            return None
        return parsed
    
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
    
    def run_backtest(self, prepared_data: pd.DataFrame, runtime_predictor,
                    selected_features: List[str]) -> BacktestResult:
        """
        运行回测
        
        Args:
            prepared_data: Prepared feature matrix
            ai_model_manager: AI模型管理器
            selected_features: 选择的特征
            
        Returns:
            回测结果
        """
        logger.info("Starting backtest on prepared feature data...")
        
        # 重置状态
        self.reset()
        
        # 确保数据按时间排序
        feature_data = prepared_data.sort_index().copy()

        if feature_data.empty:
            logger.error("Prepared feature data is empty")
            return BacktestResult()

        missing_features = [feature for feature in selected_features if feature not in feature_data.columns]
        if missing_features:
            raise ValueError(f"Prepared feature data is missing selected features: {missing_features}")

        if hasattr(runtime_predictor, "predict_batch"):
            prediction_frame = runtime_predictor.predict_batch(feature_data)
        else:
            prediction_frame = runtime_predictor.predict_ensemble_batch(feature_data, feature_columns=selected_features)

        if int(prediction_frame['is_valid'].sum()) == 0:
            raise ValueError("No valid rows remain for backtesting after feature filtering")
        
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
                    'volume': row.get('Volume', 0),
                }
                
                # 检查止损止盈
                closed_this_bar = self._simulate_open_positions_on_bar(current_data)

                if closed_this_bar:
                    self.update_equity(current_data)
                    continue
                
                prediction_row = prediction_frame.loc[timestamp]
                if not bool(prediction_row['is_valid']):
                    continue

                prediction = int(prediction_row['prediction'])
                confidence = float(prediction_row['confidence'])

                signal = self._generate_signal(prediction, current_data, confidence)

                if signal:
                    if len(self.current_positions) == 0:
                        self.open_position(signal, current_data)
                    elif signal.get('action') == 'close':
                        for trade_id in list(self.current_positions.keys()):
                            self.close_position(trade_id, current_data, 'signal')
                
                # 更新权益曲线
                self.update_equity(current_data)
                
                # 进度显示
                if i % 1000 == 0:
                    progress = (i / len(feature_data)) * 100
                    logger.info(f"Backtest progress: {progress:.1f}% ({i}/{len(feature_data)})")
                
            except Exception as e:
                logger.debug(f"Backtest loop error: {e}")
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
        
        logger.info("Backtest complete")
        return result
    
    def _generate_signal(self, prediction: int, current_data: Dict,
                        confidence: float) -> Optional[Dict]:
        """
        根据AI预测生成交易信号
        
        Args:
            prediction: AI预测结果 (0=跌, 1=涨)
            current_data: 当前数据
            confidence: Prediction confidence (0-1)
            
        Returns:
            交易信号字典
        """
        try:
            # 只有高置信度的信号才交易
            if confidence < self.signal_confidence_threshold:
                return None
            
            current_price = current_data['close']
            stop_distance = float(self.trading_config['stop_loss_pips'])
            take_profit_distance = float(self.trading_config['take_profit_pips'])
            
            if prediction == 1:  # 看涨信号
                return {
                    'action': 'open',
                    'side': 'buy',
                    'symbol': self.trading_config.get('symbol', 'XAUUSD'),
                    'size': float(self.trading_config.get('position_size', 0.01)),
                    'confidence': confidence,
                    'stop_loss': current_price - stop_distance,
                    'take_profit': current_price + take_profit_distance
                }
            elif prediction == 0:  # 看跌信号
                return {
                    'action': 'open',
                    'side': 'sell',
                    'symbol': self.trading_config.get('symbol', 'XAUUSD'),
                    'size': float(self.trading_config.get('position_size', 0.01)),
                    'confidence': confidence,
                    'stop_loss': current_price + stop_distance,
                    'take_profit': current_price - take_profit_distance
                }
            
            return None
            
        except Exception as e:
            logger.debug(f"Failed to generate signal: {e}")
            return None

    def _load_exit_management_config(self) -> Dict[str, Any]:
        exit_config = dict(self.config.get("live_trading", {}).get("exit_management", {}) or {})
        mode = str(exit_config.get("mode", "disabled")).strip().lower() or "disabled"
        return {
            "enabled": mode == "trailing_stop",
            "mode": mode,
            "break_even_enabled": bool(exit_config.get("break_even_enabled", True)),
            "break_even_trigger_pips": float(exit_config.get("break_even_trigger_pips", 35.0)),
            "break_even_offset_pips": float(exit_config.get("break_even_offset_pips", 2.0)),
            "trailing_enabled": bool(exit_config.get("trailing_enabled", True)),
            "trailing_activation_pips": float(exit_config.get("trailing_activation_pips", 60.0)),
            "trailing_distance_pips": float(exit_config.get("trailing_distance_pips", 25.0)),
            "trailing_step_pips": float(exit_config.get("trailing_step_pips", 10.0)),
        }

    def _calculate_managed_stop_loss(self, trade: Trade, current_price: float) -> Optional[float]:
        if current_price <= 0:
            return None

        entry_price = float(trade.entry_price)
        current_stop_loss = None if trade.stop_loss is None else float(trade.stop_loss)
        favorable_move = (current_price - entry_price) if trade.side == 'buy' else (entry_price - current_price)
        if favorable_move <= 0:
            return None

        break_even_trigger = float(self.exit_management["break_even_trigger_pips"])
        break_even_offset = float(self.exit_management["break_even_offset_pips"])
        trailing_activation = float(self.exit_management["trailing_activation_pips"])
        trailing_distance = float(self.exit_management["trailing_distance_pips"])
        trailing_step = float(self.exit_management["trailing_step_pips"])

        candidate_stop_loss: Optional[float] = None
        if bool(self.exit_management.get("break_even_enabled")) and favorable_move >= break_even_trigger:
            offset = break_even_offset
            candidate_stop_loss = entry_price + offset if trade.side == 'buy' else entry_price - offset

        if bool(self.exit_management.get("trailing_enabled")) and favorable_move >= trailing_activation:
            distance = trailing_distance
            trailing_stop_loss = current_price - distance if trade.side == 'buy' else current_price + distance
            if candidate_stop_loss is None:
                candidate_stop_loss = trailing_stop_loss
            else:
                candidate_stop_loss = max(candidate_stop_loss, trailing_stop_loss) if trade.side == 'buy' else min(candidate_stop_loss, trailing_stop_loss)

        if candidate_stop_loss is None:
            return None

        if current_stop_loss is None:
            return float(candidate_stop_loss)

        if trade.side == 'buy':
            if candidate_stop_loss <= current_stop_loss:
                return None
        else:
            if candidate_stop_loss >= current_stop_loss:
                return None

        if abs(candidate_stop_loss - current_stop_loss) < float(trailing_step or 0.0):
            return None
        return float(candidate_stop_loss)

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
        Plot backtest result charts.
        
        Args:
            result: Backtest result
            save_path: Optional output path
        """
        try:
            if plt is None:
                raise RuntimeError("matplotlib is not installed; cannot plot backtest results")

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Equity curve
            if result.equity_curve:
                ax1.plot(result.equity_curve, label='Equity Curve', color='blue')
                ax1.axhline(y=self.initial_capital, color='red', linestyle='--', label='Initial Capital')
                ax1.set_title('Equity Curve')
                ax1.set_ylabel('Equity ($)')
                ax1.legend()
                ax1.grid(True)
            
            # Drawdown curve
            if result.equity_curve:
                peak = np.maximum.accumulate(result.equity_curve)
                drawdown = (peak - result.equity_curve) / peak
                ax2.fill_between(range(len(drawdown)), drawdown, alpha=0.3, color='red')
                ax2.set_title(f'Drawdown Curve (Max Drawdown: {result.max_drawdown:.2%})')
                ax2.set_ylabel('Drawdown Ratio')
                ax2.grid(True)
            
            # Trade distribution
            labels = ['Winning Trades', 'Losing Trades']
            sizes = [result.winning_trades, result.losing_trades]
            colors = ['green', 'red']
            ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax3.set_title(f'Trade Distribution (Win Rate: {result.win_rate:.1%})')
            
            # Performance metrics
            metrics = [
                f'Total Trades: {result.total_trades}',
                f'Total PnL: ${result.total_pnl:.2f}',
                f'Win Rate: {result.win_rate:.1%}',
                f'Profit Factor: {result.profit_factor:.2f}',
                f'Sharpe Ratio: {result.sharpe_ratio:.2f}',
                f'Max Drawdown: {result.max_drawdown:.2%}',
                f'Average Winner: ${result.avg_winning_trade:.2f}',
                f'Average Loser: ${result.avg_losing_trade:.2f}'
            ]
            
            ax4.axis('off')
            ax4.text(0.1, 0.9, 'Backtest Summary', fontsize=14, fontweight='bold', transform=ax4.transAxes)
            for i, metric in enumerate(metrics):
                ax4.text(0.1, 0.8 - i*0.08, metric, fontsize=10, transform=ax4.transAxes)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Backtest chart saved: {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Failed to plot backtest results: {e}")
    
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
                        'exit_trigger_price': trade.exit_trigger_price,
                        'size': trade.size,
                        'pnl': trade.pnl,
                        'status': trade.status,
                        'reason': trade.reason,
                        'managed_stop_loss': trade.managed_stop_loss,
                        'protection_updated_on_exit_bar': trade.protection_updated_on_exit_bar,
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
        Return a DataFrame summarizing closed trades.
        
        Returns:
            Trade summary DataFrame
        """
        try:
            closed_trades = [t for t in self.trades if t.status == 'closed']
            
            if not closed_trades:
                return pd.DataFrame()
            
            trade_data = []
            for trade in closed_trades:
                trade_data.append({
                    'ID': trade.id,
                    'Symbol': trade.symbol,
                    'Side': trade.side,
                    'Entry Time': trade.entry_time,
                    'Entry Price': trade.entry_price,
                    'Exit Time': trade.exit_time,
                    'Exit Price': trade.exit_price,
                    'Exit Trigger Price': trade.exit_trigger_price,
                    'Size': trade.size,
                    'PnL': trade.pnl,
                    'Close Reason': trade.reason,
                    'Managed Stop Loss': trade.managed_stop_loss,
                    'Protection Updated On Exit Bar': trade.protection_updated_on_exit_bar,
                })
            
            df = pd.DataFrame(trade_data)
            return df
            
        except Exception as e:
            logger.error(f"创建交易汇总失败: {e}")
            return pd.DataFrame() 
