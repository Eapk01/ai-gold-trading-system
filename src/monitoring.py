"""
监控和日志系统
实时监控交易系统的性能、风险、状态
提供告警、日志记录、性能分析功能
"""

import os
import json
import time
import threading
import psutil
import sqlite3
import uuid
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import pandas as pd
import numpy as np
from collections import deque, defaultdict


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """告警类型"""
    SYSTEM = "system"           # 系统告警
    TRADING = "trading"         # 交易告警
    PERFORMANCE = "performance" # 性能告警
    RISK = "risk"              # 风险告警
    CONNECTION = "connection"   # 连接告警


@dataclass
class Alert:
    """告警对象"""
    id: str = ""
    type: AlertType = AlertType.SYSTEM
    level: AlertLevel = AlertLevel.INFO
    title: str = ""
    message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    acknowledged: bool = False
    resolved: bool = False
    data: Dict = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """系统指标"""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    network_sent: float = 0.0
    network_received: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TradingMetrics:
    """交易指标"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    max_drawdown: float = 0.0
    current_positions: int = 0
    active_orders: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


class DatabaseLogger:
    """数据库日志记录器"""
    
    def __init__(self, db_path: str = "logs/trading_monitor.db"):
        """
        初始化数据库日志记录器
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_database()
        
    def _init_database(self):
        """初始化数据库表"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # 创建告警表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS alerts (
                        id TEXT PRIMARY KEY,
                        type TEXT NOT NULL,
                        level TEXT NOT NULL,
                        title TEXT NOT NULL,
                        message TEXT,
                        timestamp TEXT NOT NULL,
                        source TEXT,
                        acknowledged INTEGER DEFAULT 0,
                        resolved INTEGER DEFAULT 0,
                        data TEXT
                    )
                ''')
                
                # 创建系统指标表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS system_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        cpu_percent REAL,
                        memory_percent REAL,
                        disk_percent REAL,
                        network_sent REAL,
                        network_received REAL,
                        timestamp TEXT NOT NULL
                    )
                ''')
                
                # 创建交易指标表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS trading_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        total_trades INTEGER,
                        winning_trades INTEGER,
                        losing_trades INTEGER,
                        win_rate REAL,
                        total_pnl REAL,
                        unrealized_pnl REAL,
                        realized_pnl REAL,
                        max_drawdown REAL,
                        current_positions INTEGER,
                        active_orders INTEGER,
                        timestamp TEXT NOT NULL
                    )
                ''')
                
                # 创建事件日志表
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS event_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_type TEXT NOT NULL,
                        level TEXT NOT NULL,
                        message TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        source TEXT,
                        data TEXT
                    )
                ''')
                
                conn.commit()
                logger.info(f"数据库初始化完成: {self.db_path}")
                
        except Exception as e:
            logger.error(f"数据库初始化失败: {e}")
    
    def log_alert(self, alert: Alert):
        """记录告警"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO alerts (id, type, level, title, message, timestamp, source, acknowledged, resolved, data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    alert.id,
                    alert.type.value,
                    alert.level.value,
                    alert.title,
                    alert.message,
                    alert.timestamp.isoformat(),
                    alert.source,
                    int(alert.acknowledged),
                    int(alert.resolved),
                    json.dumps(alert.data)
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"记录告警失败: {e}")
    
    def log_system_metrics(self, metrics: SystemMetrics):
        """记录系统指标"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO system_metrics (cpu_percent, memory_percent, disk_percent, network_sent, network_received, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.cpu_percent,
                    metrics.memory_percent,
                    metrics.disk_percent,
                    metrics.network_sent,
                    metrics.network_received,
                    metrics.timestamp.isoformat()
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"记录系统指标失败: {e}")
    
    def log_trading_metrics(self, metrics: TradingMetrics):
        """记录交易指标"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO trading_metrics (total_trades, winning_trades, losing_trades, win_rate, total_pnl, 
                                               unrealized_pnl, realized_pnl, max_drawdown, current_positions, active_orders, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.total_trades,
                    metrics.winning_trades,
                    metrics.losing_trades,
                    metrics.win_rate,
                    metrics.total_pnl,
                    metrics.unrealized_pnl,
                    metrics.realized_pnl,
                    metrics.max_drawdown,
                    metrics.current_positions,
                    metrics.active_orders,
                    metrics.timestamp.isoformat()
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"记录交易指标失败: {e}")
    
    def log_event(self, event_type: str, level: str, message: str, source: str = "", data: Dict = None):
        """记录事件日志"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO event_logs (event_type, level, message, timestamp, source, data)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    event_type,
                    level,
                    message,
                    datetime.now().isoformat(),
                    source,
                    json.dumps(data or {})
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"记录事件日志失败: {e}")
    
    def get_alerts(self, level: Optional[AlertLevel] = None, 
                   alert_type: Optional[AlertType] = None,
                   limit: int = 100) -> List[Dict]:
        """获取告警记录"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM alerts"
                conditions = []
                params = []
                
                if level:
                    conditions.append("level = ?")
                    params.append(level.value)
                
                if alert_type:
                    conditions.append("type = ?")
                    params.append(alert_type.value)
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                columns = [desc[0] for desc in cursor.description]
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            logger.error(f"获取告警记录失败: {e}")
            return []


class NotificationManager:
    """通知管理器"""
    
    def __init__(self, config: Dict):
        """
        初始化通知管理器
        
        Args:
            config: 通知配置
        """
        self.config = config
        self.email_config = config.get('email', {})
        self.webhook_config = config.get('webhook', {})
        self.enabled_channels = config.get('enabled_channels', ['log'])
        
    def send_alert(self, alert: Alert):
        """发送告警通知"""
        try:
            # 根据配置发送到不同渠道
            if 'email' in self.enabled_channels:
                self._send_email_alert(alert)
                
            if 'webhook' in self.enabled_channels:
                self._send_webhook_alert(alert)
                
            if 'log' in self.enabled_channels:
                self._log_alert(alert)
                
        except Exception as e:
            logger.error(f"发送告警通知失败: {e}")
    
    def _send_email_alert(self, alert: Alert):
        """发送邮件告警"""
        try:
            if not self.email_config.get('enabled', False):
                return
            
            smtp_server = self.email_config.get('smtp_server')
            smtp_port = self.email_config.get('smtp_port', 587)
            username = self.email_config.get('username')
            password = self.email_config.get('password')
            recipients = self.email_config.get('recipients', [])
            
            if not all([smtp_server, username, password, recipients]):
                logger.warning("邮件配置不完整，跳过邮件通知")
                return
            
            # 创建邮件内容
            msg = MIMEMultipart()
            msg['From'] = username
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"[{alert.level.value.upper()}] {alert.title}"
            
            body = f"""
            告警类型: {alert.type.value}
            告警级别: {alert.level.value}
            告警时间: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
            告警来源: {alert.source}
            
            告警信息:
            {alert.message}
            
            ---
            AI黄金交易系统监控
            """
            
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            # 发送邮件
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(username, password)
                text = msg.as_string()
                server.sendmail(username, recipients, text)
            
            logger.info(f"邮件告警已发送: {alert.title}")
            
        except Exception as e:
            logger.error(f"发送邮件告警失败: {e}")
    
    def _send_webhook_alert(self, alert: Alert):
        """发送Webhook告警"""
        try:
            if not self.webhook_config.get('enabled', False):
                return
            
            webhook_url = self.webhook_config.get('url')
            if not webhook_url:
                return
            
            payload = {
                'alert_id': alert.id,
                'type': alert.type.value,
                'level': alert.level.value,
                'title': alert.title,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'source': alert.source,
                'data': alert.data
            }
            
            headers = {'Content-Type': 'application/json'}
            auth_token = self.webhook_config.get('auth_token')
            if auth_token:
                headers['Authorization'] = f'Bearer {auth_token}'
            
            response = requests.post(webhook_url, json=payload, headers=headers, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"Webhook告警已发送: {alert.title}")
            else:
                logger.warning(f"Webhook告警发送失败: {response.status_code}")
                
        except Exception as e:
            logger.error(f"发送Webhook告警失败: {e}")
    
    def _log_alert(self, alert: Alert):
        """记录告警日志"""
        log_level_map = {
            AlertLevel.INFO: "info",
            AlertLevel.WARNING: "warning",
            AlertLevel.ERROR: "error",
            AlertLevel.CRITICAL: "critical"
        }
        
        log_func = getattr(logger, log_level_map.get(alert.level, "info"))
        log_func(f"[{alert.type.value.upper()}] {alert.title}: {alert.message}")


class SystemMonitor:
    """系统监控器"""
    
    def __init__(self, config: Dict):
        """
        初始化系统监控器
        
        Args:
            config: 监控配置
        """
        self.config = config
        self.is_running = False
        self.monitoring_thread = None
        self.metrics_history = deque(maxlen=1000)
        self.callbacks: List[Callable] = []
        
        # 监控阈值
        self.cpu_threshold = config.get('cpu_threshold', 80.0)
        self.memory_threshold = config.get('memory_threshold', 85.0)
        self.disk_threshold = config.get('disk_threshold', 90.0)
        self.check_interval = config.get('check_interval', 10)
        
    def start(self):
        """启动系统监控"""
        try:
            if self.is_running:
                logger.warning("系统监控已在运行")
                return
            
            self.is_running = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            
            logger.info("系统监控已启动")
            
        except Exception as e:
            logger.error(f"启动系统监控失败: {e}")
    
    def stop(self):
        """停止系统监控"""
        try:
            self.is_running = False
            
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5)
            
            logger.info("系统监控已停止")
            
        except Exception as e:
            logger.error(f"停止系统监控失败: {e}")
    
    def add_callback(self, callback: Callable):
        """添加监控回调函数"""
        self.callbacks.append(callback)
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                # 收集系统指标
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)
                
                # 检查告警条件
                self._check_alerts(metrics)
                
                # 通知回调函数
                for callback in self.callbacks:
                    try:
                        callback(metrics)
                    except Exception as e:
                        logger.error(f"监控回调执行失败: {e}")
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"系统监控循环异常: {e}")
                time.sleep(self.check_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """收集系统指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 内存使用率
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # 磁盘使用率
            disk = psutil.disk_usage('.')
            disk_percent = (disk.used / disk.total) * 100
            
            # 网络IO
            network = psutil.net_io_counters()
            network_sent = network.bytes_sent
            network_received = network.bytes_recv
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_percent=disk_percent,
                network_sent=network_sent,
                network_received=network_received,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"收集系统指标失败: {e}")
            return SystemMetrics()
    
    def _check_alerts(self, metrics: SystemMetrics):
        """检查告警条件"""
        try:
            alerts = []
            
            # CPU告警
            if metrics.cpu_percent > self.cpu_threshold:
                alerts.append(Alert(
                    id=f"cpu_high_{int(time.time())}",
                    type=AlertType.SYSTEM,
                    level=AlertLevel.WARNING,
                    title="CPU使用率过高",
                    message=f"当前CPU使用率: {metrics.cpu_percent:.1f}%, 阈值: {self.cpu_threshold}%",
                    source="SystemMonitor",
                    data={"cpu_percent": metrics.cpu_percent, "threshold": self.cpu_threshold}
                ))
            
            # 内存告警
            if metrics.memory_percent > self.memory_threshold:
                alerts.append(Alert(
                    id=f"memory_high_{int(time.time())}",
                    type=AlertType.SYSTEM,
                    level=AlertLevel.WARNING,
                    title="内存使用率过高",
                    message=f"当前内存使用率: {metrics.memory_percent:.1f}%, 阈值: {self.memory_threshold}%",
                    source="SystemMonitor",
                    data={"memory_percent": metrics.memory_percent, "threshold": self.memory_threshold}
                ))
            
            # 磁盘告警
            if metrics.disk_percent > self.disk_threshold:
                alerts.append(Alert(
                    id=f"disk_high_{int(time.time())}",
                    type=AlertType.SYSTEM,
                    level=AlertLevel.ERROR,
                    title="磁盘使用率过高",
                    message=f"当前磁盘使用率: {metrics.disk_percent:.1f}%, 阈值: {self.disk_threshold}%",
                    source="SystemMonitor",
                    data={"disk_percent": metrics.disk_percent, "threshold": self.disk_threshold}
                ))
            
            # 触发告警
            for alert in alerts:
                for callback in self.callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        logger.error(f"告警回调执行失败: {e}")
                        
        except Exception as e:
            logger.error(f"检查系统告警失败: {e}")
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """获取当前系统指标"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def get_metrics_history(self, minutes: int = 60) -> List[SystemMetrics]:
        """获取历史指标"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]


class TradingMonitor:
    """交易监控器"""
    
    def __init__(self, config: Dict):
        """
        初始化交易监控器
        
        Args:
            config: 监控配置
        """
        self.config = config
        self.metrics_history = deque(maxlen=1000)
        self.callbacks: List[Callable] = []
        
        # 风险阈值
        self.max_drawdown_threshold = config.get('max_drawdown_threshold', 0.15)  # 15%
        self.max_daily_loss_threshold = config.get('max_daily_loss_threshold', 1000)  # $1000
        self.min_win_rate_threshold = config.get('min_win_rate_threshold', 0.4)  # 40%
        
    def add_callback(self, callback: Callable):
        """添加监控回调函数"""
        self.callbacks.append(callback)
    
    def update_metrics(self, metrics: TradingMetrics):
        """更新交易指标"""
        try:
            self.metrics_history.append(metrics)
            
            # 检查风险告警
            self._check_risk_alerts(metrics)
            
            # 通知回调函数
            for callback in self.callbacks:
                try:
                    callback(metrics)
                except Exception as e:
                    logger.error(f"交易监控回调执行失败: {e}")
                    
        except Exception as e:
            logger.error(f"更新交易指标失败: {e}")
    
    def _check_risk_alerts(self, metrics: TradingMetrics):
        """检查风险告警"""
        try:
            alerts = []
            
            # 最大回撤告警
            if metrics.max_drawdown > self.max_drawdown_threshold:
                alerts.append(Alert(
                    id=f"drawdown_high_{int(time.time())}",
                    type=AlertType.RISK,
                    level=AlertLevel.ERROR,
                    title="最大回撤超过阈值",
                    message=f"当前最大回撤: {metrics.max_drawdown:.2%}, 阈值: {self.max_drawdown_threshold:.2%}",
                    source="TradingMonitor",
                    data={"max_drawdown": metrics.max_drawdown, "threshold": self.max_drawdown_threshold}
                ))
            
            # 胜率告警
            if metrics.win_rate < self.min_win_rate_threshold and metrics.total_trades >= 10:
                alerts.append(Alert(
                    id=f"win_rate_low_{int(time.time())}",
                    type=AlertType.TRADING,
                    level=AlertLevel.WARNING,
                    title="胜率低于阈值",
                    message=f"当前胜率: {metrics.win_rate:.2%}, 阈值: {self.min_win_rate_threshold:.2%}",
                    source="TradingMonitor",
                    data={"win_rate": metrics.win_rate, "threshold": self.min_win_rate_threshold}
                ))
            
            # 日损失告警
            daily_pnl = self._calculate_daily_pnl()
            if daily_pnl < -self.max_daily_loss_threshold:
                alerts.append(Alert(
                    id=f"daily_loss_high_{int(time.time())}",
                    type=AlertType.RISK,
                    level=AlertLevel.CRITICAL,
                    title="日损失超过阈值",
                    message=f"今日损失: ${abs(daily_pnl):.2f}, 阈值: ${self.max_daily_loss_threshold:.2f}",
                    source="TradingMonitor",
                    data={"daily_pnl": daily_pnl, "threshold": -self.max_daily_loss_threshold}
                ))
            
            # 触发告警
            for alert in alerts:
                for callback in self.callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        logger.error(f"风险告警回调执行失败: {e}")
                        
        except Exception as e:
            logger.error(f"检查风险告警失败: {e}")
    
    def _calculate_daily_pnl(self) -> float:
        """计算今日盈亏"""
        try:
            today = datetime.now().date()
            today_metrics = [m for m in self.metrics_history if m.timestamp.date() == today]
            
            if len(today_metrics) >= 2:
                # 今日总盈亏变化
                first_pnl = today_metrics[0].total_pnl
                latest_pnl = today_metrics[-1].total_pnl
                return latest_pnl - first_pnl
            
            return 0.0
            
        except Exception as e:
            logger.error(f"计算今日盈亏失败: {e}")
            return 0.0
    
    def get_current_metrics(self) -> Optional[TradingMetrics]:
        """获取当前交易指标"""
        if self.metrics_history:
            return self.metrics_history[-1]
        return None
    
    def get_performance_summary(self) -> Dict:
        """获取性能摘要"""
        try:
            if not self.metrics_history:
                return {}
            
            latest_metrics = self.metrics_history[-1]
            
            # 计算统计数据
            daily_pnl = self._calculate_daily_pnl()
            
            return {
                "总交易数": latest_metrics.total_trades,
                "获胜交易": latest_metrics.winning_trades,
                "失败交易": latest_metrics.losing_trades,
                "胜率": f"{latest_metrics.win_rate:.2%}",
                "总盈亏": f"${latest_metrics.total_pnl:.2f}",
                "未实现盈亏": f"${latest_metrics.unrealized_pnl:.2f}",
                "已实现盈亏": f"${latest_metrics.realized_pnl:.2f}",
                "最大回撤": f"{latest_metrics.max_drawdown:.2%}",
                "今日盈亏": f"${daily_pnl:.2f}",
                "当前持仓": latest_metrics.current_positions,
                "活跃订单": latest_metrics.active_orders,
                "更新时间": latest_metrics.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            }
            
        except Exception as e:
            logger.error(f"获取性能摘要失败: {e}")
            return {}


class MonitoringSystem:
    """完整监控系统"""
    
    def __init__(self, config: Dict):
        """
        初始化监控系统
        
        Args:
            config: 系统配置
        """
        self.config = config
        
        # 初始化子组件
        self.db_logger = DatabaseLogger(config.get('db_path', 'logs/trading_monitor.db'))
        self.notification_manager = NotificationManager(config.get('notifications', {}))
        self.system_monitor = SystemMonitor(config.get('system', {}))
        self.trading_monitor = TradingMonitor(config.get('trading', {}))
        
        # 注册回调函数
        self.system_monitor.add_callback(self._on_system_metrics)
        self.system_monitor.add_callback(self._on_system_alert)
        self.trading_monitor.add_callback(self._on_trading_metrics)
        self.trading_monitor.add_callback(self._on_trading_alert)
        
        # 告警历史
        self.alert_history = deque(maxlen=1000)
        
        logger.info("监控系统初始化完成")
    
    def start(self):
        """启动监控系统"""
        try:
            self.system_monitor.start()
            logger.info("监控系统已启动")
            
        except Exception as e:
            logger.error(f"启动监控系统失败: {e}")
    
    def stop(self):
        """停止监控系统"""
        try:
            self.system_monitor.stop()
            logger.info("监控系统已停止")
            
        except Exception as e:
            logger.error(f"停止监控系统失败: {e}")
    
    def update_trading_metrics(self, metrics: TradingMetrics):
        """更新交易指标"""
        self.trading_monitor.update_metrics(metrics)
    
    def send_custom_alert(self, alert_type: AlertType, level: AlertLevel, 
                         title: str, message: str, source: str = "", data: Dict = None):
        """发送自定义告警"""
        try:
            alert = Alert(
                id=f"custom_{int(time.time())}",
                type=alert_type,
                level=level,
                title=title,
                message=message,
                source=source,
                data=data or {}
            )
            
            self._handle_alert(alert)
            
        except Exception as e:
            logger.error(f"发送自定义告警失败: {e}")
    
    def _on_system_metrics(self, metrics: SystemMetrics):
        """处理系统指标"""
        try:
            self.db_logger.log_system_metrics(metrics)
        except Exception as e:
            logger.error(f"处理系统指标失败: {e}")
    
    def _on_system_alert(self, alert: Alert):
        """处理系统告警"""
        self._handle_alert(alert)
    
    def _on_trading_metrics(self, metrics: TradingMetrics):
        """处理交易指标"""
        try:
            self.db_logger.log_trading_metrics(metrics)
        except Exception as e:
            logger.error(f"处理交易指标失败: {e}")
    
    def _on_trading_alert(self, alert: Alert):
        """处理交易告警"""
        self._handle_alert(alert)
    
    def _handle_alert(self, alert: Alert):
        """统一处理告警"""
        try:
            # 记录到数据库
            self.db_logger.log_alert(alert)
            
            # 记录到历史
            self.alert_history.append(alert)
            
            # 发送通知
            self.notification_manager.send_alert(alert)
            
            # 记录事件日志
            self.db_logger.log_event(
                event_type="alert",
                level=alert.level.value,
                message=f"{alert.title}: {alert.message}",
                source=alert.source,
                data=alert.data
            )
            
        except Exception as e:
            logger.error(f"处理告警失败: {e}")
    
    def get_system_status(self) -> Dict:
        """获取系统状态"""
        try:
            system_metrics = self.system_monitor.get_current_metrics()
            trading_metrics = self.trading_monitor.get_current_metrics()
            
            status = {
                "监控系统状态": "运行中" if self.system_monitor.is_running else "已停止",
                "系统指标": {},
                "交易指标": {},
                "最近告警": len([a for a in self.alert_history if a.timestamp > datetime.now() - timedelta(hours=1)]),
                "总告警数": len(self.alert_history),
                "更新时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            if system_metrics:
                status["系统指标"] = {
                    "CPU使用率": f"{system_metrics.cpu_percent:.1f}%",
                    "内存使用率": f"{system_metrics.memory_percent:.1f}%",
                    "磁盘使用率": f"{system_metrics.disk_percent:.1f}%",
                    "网络发送": f"{system_metrics.network_sent / 1024 / 1024:.1f} MB",
                    "网络接收": f"{system_metrics.network_received / 1024 / 1024:.1f} MB"
                }
            
            if trading_metrics:
                status["交易指标"] = self.trading_monitor.get_performance_summary()
            
            return status
            
        except Exception as e:
            logger.error(f"获取系统状态失败: {e}")
            return {"错误": str(e)}
    
    def get_alert_summary(self, hours: int = 24) -> Dict:
        """获取告警摘要"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_alerts = [a for a in self.alert_history if a.timestamp >= cutoff_time]
            
            # 按级别统计
            level_counts = defaultdict(int)
            for alert in recent_alerts:
                level_counts[alert.level.value] += 1
            
            # 按类型统计
            type_counts = defaultdict(int)
            for alert in recent_alerts:
                type_counts[alert.type.value] += 1
            
            return {
                "时间范围": f"最近{hours}小时",
                "总告警数": len(recent_alerts),
                "按级别统计": dict(level_counts),
                "按类型统计": dict(type_counts),
                "未确认告警": len([a for a in recent_alerts if not a.acknowledged]),
                "未解决告警": len([a for a in recent_alerts if not a.resolved])
            }
            
        except Exception as e:
            logger.error(f"获取告警摘要失败: {e}")
            return {}
    
    def export_logs(self, hours: int = 24, output_file: str = None) -> str:
        """导出日志"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # 获取告警记录
            alerts = self.db_logger.get_alerts(limit=1000)
            recent_alerts = [a for a in alerts if datetime.fromisoformat(a['timestamp']) >= cutoff_time]
            
            # 生成报告
            report = {
                "导出时间": datetime.now().isoformat(),
                "时间范围": f"最近{hours}小时",
                "系统状态": self.get_system_status(),
                "告警摘要": self.get_alert_summary(hours),
                "详细告警": recent_alerts
            }
            
            # 保存到文件
            if output_file is None:
                output_file = f"logs/monitor_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"监控日志已导出: {output_file}")
            return output_file
            
        except Exception as e:
            logger.error(f"导出日志失败: {e}")
            return "" 
