"""
特征工程模块
负责计算技术指标和提取用于机器学习的特征
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import talib
import pandas_ta as ta
from loguru import logger
from scipy import stats
from src.config_utils import get_target_column


class FeatureEngineer:
    """特征工程器 - 计算技术指标和特征"""
    
    def __init__(self, config: Dict):
        """
        初始化特征工程器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.features_config = config.get('features', {})
        self.lookback_periods = config['ai_model']['lookback_periods']
        
        logger.info("Feature engineer initialized")
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        
        Args:
            data: 包含OHLCV数据的DataFrame
            
        Returns:
            包含技术指标的DataFrame
        """
        try:
            df = data.copy()

            # TA-Lib expects double-precision numeric arrays.
            for column in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[column] = pd.to_numeric(df[column], errors='coerce').astype('float64')
            
            # 基础价格数据
            high = df['High'].to_numpy(dtype=np.float64)
            low = df['Low'].to_numpy(dtype=np.float64)
            close = df['Close'].to_numpy(dtype=np.float64)
            volume = df['Volume'].to_numpy(dtype=np.float64)
            open_price = df['Open'].to_numpy(dtype=np.float64)
            
            # === 移动平均线 ===
            df['SMA_5'] = talib.SMA(close, timeperiod=5)
            df['SMA_10'] = talib.SMA(close, timeperiod=10)
            df['SMA_20'] = talib.SMA(close, timeperiod=20)
            df['SMA_50'] = talib.SMA(close, timeperiod=50)
            
            df['EMA_5'] = talib.EMA(close, timeperiod=5)
            df['EMA_10'] = talib.EMA(close, timeperiod=10)
            df['EMA_20'] = talib.EMA(close, timeperiod=20)
            
            # === 动量指标 ===
            df['RSI_14'] = talib.RSI(close, timeperiod=14)
            df['RSI_7'] = talib.RSI(close, timeperiod=7)
            
            # MACD
            macd, macdsignal, macdhist = talib.MACD(close, 
                                                   fastperiod=12, 
                                                   slowperiod=26, 
                                                   signalperiod=9)
            df['MACD'] = macd
            df['MACD_Signal'] = macdsignal
            df['MACD_Hist'] = macdhist
            
            # Stochastic
            slowk, slowd = talib.STOCH(high, low, close, 
                                      fastk_period=14, 
                                      slowk_period=3, 
                                      slowd_period=3)
            df['STOCH_K'] = slowk
            df['STOCH_D'] = slowd
            
            # Williams %R
            df['WILLIAMS_R'] = talib.WILLR(high, low, close, timeperiod=14)
            
            # === 波动率指标 ===
            # 布林带
            bb_upper, bb_middle, bb_lower = talib.BBANDS(close, 
                                                        timeperiod=20, 
                                                        nbdevup=2, 
                                                        nbdevdn=2)
            df['BB_Upper'] = bb_upper
            df['BB_Middle'] = bb_middle
            df['BB_Lower'] = bb_lower
            df['BB_Width'] = (bb_upper - bb_lower) / bb_middle
            df['BB_Position'] = (close - bb_lower) / (bb_upper - bb_lower)
            
            # ATR (平均真实波幅)
            df['ATR_14'] = talib.ATR(high, low, close, timeperiod=14)
            df['ATR_7'] = talib.ATR(high, low, close, timeperiod=7)
            
            # === 成交量指标 ===
            df['Volume_SMA_10'] = talib.SMA(volume, timeperiod=10)
            df['Volume_Ratio'] = volume / df['Volume_SMA_10']
            
            # On Balance Volume
            df['OBV'] = talib.OBV(close, volume)
            
            # === 价格形态指标 ===
            # 价格变化
            df['Price_Change'] = close - open_price
            df['Price_Change_Pct'] = (close - open_price) / open_price * 100
            
            # 高低点差异
            df['High_Low_Ratio'] = (high - low) / close
            
            # 收盘价相对位置
            df['Close_Position'] = (close - low) / (high - low)
            
            # === 趋势指标 ===
            # ADX (平均趋向指数)
            df['ADX'] = talib.ADX(high, low, close, timeperiod=14)
            
            # Parabolic SAR
            df['SAR'] = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
            
            # === 支撑阻力指标 ===
            # 枢轴点
            df['Pivot'] = (high + low + close) / 3
            df['R1'] = 2 * df['Pivot'] - low
            df['S1'] = 2 * df['Pivot'] - high
            
            logger.info(f"计算了 {len([col for col in df.columns if col not in data.columns])} 个技术指标")
            
            return df
            
        except Exception as e:
            logger.error(f"计算技术指标失败: {e}")
            return data
    
    def calculate_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算统计特征
        
        Args:
            data: 输入数据
            
        Returns:
            包含统计特征的DataFrame
        """
        try:
            df = data.copy()
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce').astype('float64')
            close = df['Close']
            
            # 滚动统计特征
            for window in [5, 10, 20]:
                # 收益率统计
                returns = close.pct_change()
                df[f'Returns_Mean_{window}'] = returns.rolling(window).mean()
                df[f'Returns_Std_{window}'] = returns.rolling(window).std()
                df[f'Returns_Skew_{window}'] = returns.rolling(window).skew()
                df[f'Returns_Kurt_{window}'] = returns.rolling(window).kurt()
                
                # 价格统计
                df[f'Price_Mean_{window}'] = close.rolling(window).mean()
                df[f'Price_Std_{window}'] = close.rolling(window).std()
                df[f'Price_Min_{window}'] = close.rolling(window).min()
                df[f'Price_Max_{window}'] = close.rolling(window).max()
                
                # Z-Score
                df[f'Z_Score_{window}'] = (close - df[f'Price_Mean_{window}']) / df[f'Price_Std_{window}']
            
            # 价格分位数特征
            df['Price_Percentile_252'] = close.rolling(252).rank(pct=True)
            df['Price_Percentile_63'] = close.rolling(63).rank(pct=True)
            
            logger.info("统计特征计算完成")
            return df
            
        except Exception as e:
            logger.error(f"计算统计特征失败: {e}")
            return data
    
    def calculate_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算时间特征
        
        Args:
            data: 输入数据
            
        Returns:
            包含时间特征的DataFrame
        """
        try:
            df = data.copy()
            
            # 确保索引是datetime类型
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # 基础时间特征
            df['Hour'] = df.index.hour
            df['DayOfWeek'] = df.index.dayofweek
            df['DayOfMonth'] = df.index.day
            df['Month'] = df.index.month
            df['Quarter'] = df.index.quarter
            
            # 是否为交易时间
            df['Is_Market_Hours'] = ((df['Hour'] >= 0) & (df['Hour'] <= 23)).astype(int)
            
            # 是否为周末
            df['Is_Weekend'] = (df['DayOfWeek'] >= 5).astype(int)
            
            # 月初月末标识
            df['Is_Month_Start'] = (df['DayOfMonth'] <= 5).astype(int)
            df['Is_Month_End'] = (df['DayOfMonth'] >= 25).astype(int)
            
            logger.info("时间特征计算完成")
            return df
            
        except Exception as e:
            logger.error(f"计算时间特征失败: {e}")
            return data
    
    def calculate_lag_features(self, data: pd.DataFrame, 
                             columns: List[str] = None,
                             lags: List[int] = None) -> pd.DataFrame:
        """
        计算滞后特征
        
        Args:
            data: 输入数据
            columns: 要创建滞后特征的列名列表
            lags: 滞后期数列表
            
        Returns:
            包含滞后特征的DataFrame
        """
        try:
            df = data.copy()
            
            if columns is None:
                columns = ['Close', 'Volume', 'RSI_14', 'MACD']
            
            if lags is None:
                lags = [1, 2, 3, 5, 10]
            
            for col in columns:
                if col in df.columns:
                    for lag in lags:
                        df[f'{col}_Lag_{lag}'] = df[col].shift(lag)
            
            logger.info(f"计算了滞后特征: {len(columns)} 列 × {len(lags)} 期")
            return df
            
        except Exception as e:
            logger.error(f"计算滞后特征失败: {e}")
            return data
    
    def calculate_target_variables(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算目标变量
        
        Args:
            data: 输入数据
            
        Returns:
            包含目标变量的DataFrame
        """
        try:
            df = data.copy()
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce').astype('float64')
            close = df['Close']
            
            # 未来收益率 (用于回归模型)
            for horizon in [1, 3, 5, 10]:
                shifted_close = close.shift(-horizon)
                df[f'Future_Return_{horizon}'] = shifted_close.pct_change(fill_method=None) * 100
                
                # 未来价格方向 (用于分类模型)
                df[f'Future_Direction_{horizon}'] = (df[f'Future_Return_{horizon}'] > 0).astype(int)
                
                # 三分类目标 (涨/跌/盘整)
                df[f'Future_Direction_3Class_{horizon}'] = pd.cut(
                    df[f'Future_Return_{horizon}'], 
                    bins=[-np.inf, -0.1, 0.1, np.inf], 
                    labels=[0, 1, 2]  # 0:跌, 1:盘整, 2:涨
                ).astype(float)
            
            # 波动率目标
            df['Future_Volatility_5'] = close.pct_change(fill_method=None).rolling(5).std().shift(-5)
            
            logger.info("目标变量计算完成")
            return df
            
        except Exception as e:
            logger.error(f"计算目标变量失败: {e}")
            return data
    
    def create_feature_matrix(self, data: pd.DataFrame, 
                            include_targets: bool = True) -> pd.DataFrame:
        """
        创建完整的特征矩阵
        
        Args:
            data: 原始OHLCV数据
            include_targets: 是否包含目标变量
            
        Returns:
            完整的特征矩阵
        """
        logger.info("开始创建特征矩阵...")
        
        # 按顺序计算所有特征
        df = data.copy()
        
        # 1. 技术指标
        df = self.calculate_technical_indicators(df)
        
        # 2. 统计特征
        df = self.calculate_statistical_features(df)
        
        # 3. 时间特征
        df = self.calculate_time_features(df)
        
        # 4. 滞后特征
        key_columns = ['Close', 'Volume', 'RSI_14', 'MACD', 'ATR_14']
        df = self.calculate_lag_features(df, key_columns)
        
        # 5. 目标变量
        if include_targets:
            df = self.calculate_target_variables(df)
        
        # 移除无穷大和极大值
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 获取特征列名
        original_columns = set(data.columns)
        feature_columns = [col for col in df.columns if col not in original_columns]
        
        logger.info(f"特征矩阵创建完成 - 总特征数: {len(feature_columns)}")
        
        return df
    
    def select_features(self, data: pd.DataFrame, 
                       target_column: Optional[str] = None,
                       method: str = 'correlation',
                       max_features: int = 50) -> List[str]:
        """
        特征选择
        
        Args:
            data: 特征数据
            target_column: 目标列
            method: 选择方法 ('correlation', 'mutual_info', 'variance')
            max_features: 最大特征数
            
        Returns:
            选择的特征列表
        """
        try:
            target_column = target_column or get_target_column(self.config)
            if target_column not in data.columns:
                logger.warning(f"目标列 {target_column} 不存在")
                return []
            
            # 移除原始OHLCV列和目标列
            exclude_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'Timestamp']
            target_columns = [col for col in data.columns if col.startswith('Future_')]
            exclude_columns.extend(target_columns)
            
            feature_columns = [col for col in data.columns 
                             if col not in exclude_columns and not data[col].isna().all()]
            
            if not feature_columns:
                logger.warning("没有可用的特征列")
                return []
            
            # 创建特征数据
            X = data[feature_columns].ffill().fillna(0)
            y = data[target_column].fillna(0)
            
            # 移除常数特征
            constant_features = X.columns[X.nunique() <= 1].tolist()
            if constant_features:
                X = X.drop(columns=constant_features)
                logger.info(f"移除常数特征: {len(constant_features)} 个")
            
            if method == 'correlation':
                # 基于相关性的特征选择
                correlations = abs(X.corrwith(y)).sort_values(ascending=False)
                selected_features = correlations.head(max_features).index.tolist()
                
            elif method == 'variance':
                # 基于方差的特征选择
                variances = X.var().sort_values(ascending=False)
                selected_features = variances.head(max_features).index.tolist()
            
            else:
                # 默认使用相关性
                correlations = abs(X.corrwith(y)).sort_values(ascending=False)
                selected_features = correlations.head(max_features).index.tolist()
            
            logger.info(f"特征选择完成 - 选择了 {len(selected_features)} 个特征")
            return selected_features
            
        except Exception as e:
            logger.error(f"特征选择失败: {e}")
            return []
    
    def get_feature_importance_summary(self, data: pd.DataFrame, 
                                     selected_features: List[str]) -> pd.DataFrame:
        """
        获取特征重要性汇总
        
        Args:
            data: 特征数据
            selected_features: 选择的特征列表
            
        Returns:
            特征重要性汇总DataFrame
        """
        try:
            summary_data = []
            
            for feature in selected_features:
                if feature in data.columns:
                    feature_data = data[feature].dropna()
                    
                    summary_data.append({
                        'Feature': feature,
                        'Count': len(feature_data),
                        'Mean': feature_data.mean(),
                        'Std': feature_data.std(),
                        'Min': feature_data.min(),
                        'Max': feature_data.max(),
                        'Null_Count': data[feature].isna().sum(),
                        'Null_Percentage': data[feature].isna().sum() / len(data) * 100
                    })
            
            summary_df = pd.DataFrame(summary_data)
            logger.info("特征重要性汇总完成")
            return summary_df
            
        except Exception as e:
            logger.error(f"创建特征重要性汇总失败: {e}")
            return pd.DataFrame() 
