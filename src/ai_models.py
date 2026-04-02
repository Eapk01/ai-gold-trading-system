"""
AI模型管理模块
负责机器学习模型的训练、预测和管理
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import joblib
import pickle
from datetime import datetime, timedelta
from loguru import logger

# 机器学习模型
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb


class AIModelManager:
    """AI模型管理器"""
    
    def __init__(self, config: Dict):
        """
        初始化AI模型管理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.model_config = config['ai_model']
        self.model_type = self.model_config['type']
        self.model_names = self.model_config['models']
        
        # 模型存储
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.target_column = 'Future_Direction_1'
        
        # 模型性能记录
        self.model_performance = {}
        self.training_history = []
        
        logger.info(f"AI model manager initialized - model type: {self.model_type}")
    
    def _get_model_instance(self, model_name: str, task_type: str = 'classification'):
        """
        获取模型实例
        
        Args:
            model_name: 模型名称
            task_type: 任务类型 ('classification' or 'regression')
            
        Returns:
            模型实例
        """
        try:
            if task_type == 'classification':
                if model_name == 'random_forest':
                    return RandomForestClassifier(
                        n_estimators=100,
                        max_depth=10,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        random_state=42
                    )
                elif model_name == 'logistic_regression':
                    return LogisticRegression(
                        random_state=42,
                        max_iter=1000
                    )
                elif model_name == 'xgboost':
                    return xgb.XGBClassifier(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=42
                    )
                elif model_name == 'lightgbm':
                    return lgb.LGBMClassifier(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=42,
                        verbose=-1
                    )
                elif model_name == 'svm':
                    return SVC(
                        kernel='rbf',
                        probability=True,
                        random_state=42
                    )
                elif model_name == 'naive_bayes':
                    return GaussianNB()
            
            else:  # regression
                if model_name == 'random_forest':
                    return RandomForestRegressor(
                        n_estimators=100,
                        max_depth=10,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        random_state=42
                    )
                elif model_name == 'linear_regression':
                    return LinearRegression()
                elif model_name == 'xgboost':
                    return xgb.XGBRegressor(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=42
                    )
                elif model_name == 'lightgbm':
                    return lgb.LGBMRegressor(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=42,
                        verbose=-1
                    )
                elif model_name == 'svr':
                    return SVR(kernel='rbf')
            
            logger.warning(f"未知模型名称: {model_name}")
            return None
            
        except Exception as e:
            logger.error(f"创建模型实例失败: {e}")
            return None
    
    def prepare_training_data(self, data: pd.DataFrame, 
                            feature_columns: List[str],
                            target_column: str = 'Future_Direction_1',
                            test_size: float = 0.2) -> Tuple[Any, Any, Any, Any]:
        """
        准备训练数据
        
        Args:
            data: 特征数据
            feature_columns: 特征列名列表
            target_column: 目标列名
            test_size: 测试集比例
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        try:
            # 移除包含NaN的行
            clean_data = data[feature_columns + [target_column]].dropna()
            
            if clean_data.empty:
                logger.error("清理后的数据为空")
                return None, None, None, None
            
            # 分离特征和目标
            X = clean_data[feature_columns]
            y = clean_data[target_column]
            
            # 时间序列数据按时间顺序分割
            split_idx = int(len(clean_data) * (1 - test_size))
            
            X_train = X.iloc[:split_idx]
            X_test = X.iloc[split_idx:]
            y_train = y.iloc[:split_idx]
            y_test = y.iloc[split_idx:]
            
            # 特征标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 保存特征列名和缩放器
            self.feature_columns = feature_columns
            self.target_column = target_column
            self.scalers[target_column] = scaler
            
            logger.info(f"训练数据准备完成 - 训练集: {len(X_train)}, 测试集: {len(X_test)}")
            
            return X_train_scaled, X_test_scaled, y_train, y_test
            
        except Exception as e:
            logger.error(f"准备训练数据失败: {e}")
            return None, None, None, None
    
    def train_single_model(self, model_name: str, 
                          X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray,
                          task_type: str = 'classification') -> Dict:
        """
        训练单个模型
        
        Args:
            model_name: 模型名称
            X_train, y_train: 训练数据
            X_test, y_test: 测试数据
            task_type: 任务类型
            
        Returns:
            模型性能字典
        """
        try:
            # 获取模型实例
            model = self._get_model_instance(model_name, task_type)
            if model is None:
                return {}
            
            # 训练模型
            logger.info(f"开始训练模型: {model_name}")
            start_time = datetime.now()
            
            model.fit(X_train, y_train)
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # 预测
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # 计算性能指标
            if task_type == 'classification':
                # 分类指标
                train_accuracy = accuracy_score(y_train, y_train_pred)
                test_accuracy = accuracy_score(y_test, y_test_pred)
                
                train_precision = precision_score(y_train, y_train_pred, average='weighted', zero_division=0)
                test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
                
                train_recall = recall_score(y_train, y_train_pred, average='weighted', zero_division=0)
                test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
                
                train_f1 = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
                test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
                
                performance = {
                    'model_name': model_name,
                    'task_type': task_type,
                    'training_time': training_time,
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'train_precision': train_precision,
                    'test_precision': test_precision,
                    'train_recall': train_recall,
                    'test_recall': test_recall,
                    'train_f1': train_f1,
                    'test_f1': test_f1
                }
                
            else:
                # 回归指标
                train_mse = mean_squared_error(y_train, y_train_pred)
                test_mse = mean_squared_error(y_test, y_test_pred)
                
                train_mae = mean_absolute_error(y_train, y_train_pred)
                test_mae = mean_absolute_error(y_test, y_test_pred)
                
                performance = {
                    'model_name': model_name,
                    'task_type': task_type,
                    'training_time': training_time,
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'train_mae': train_mae,
                    'test_mae': test_mae,
                    'train_rmse': np.sqrt(train_mse),
                    'test_rmse': np.sqrt(test_mse)
                }
            
            # 保存模型
            self.models[model_name] = model
            self.model_performance[model_name] = performance
            
            logger.info(f"模型 {model_name} 训练完成 - 测试准确率: {performance.get('test_accuracy', 'N/A')}")
            
            return performance
            
        except Exception as e:
            logger.error(f"训练模型 {model_name} 失败: {e}")
            return {}
    
    def train_ensemble_models(self, data: pd.DataFrame, 
                            feature_columns: List[str],
                            target_column: str = 'Future_Direction_1') -> Dict:
        """
        训练集成模型
        
        Args:
            data: 训练数据
            feature_columns: 特征列
            target_column: 目标列
            
        Returns:
            训练结果汇总
        """
        logger.info("开始训练集成模型...")
        
        # 准备数据
        X_train, X_test, y_train, y_test = self.prepare_training_data(
            data, feature_columns, target_column
        )
        
        if X_train is None:
            logger.error("数据准备失败")
            return {}
        
        # 训练各个模型
        training_results = {}
        
        for model_name in self.model_names:
            try:
                performance = self.train_single_model(
                    model_name, X_train, y_train, X_test, y_test
                )
                if performance:
                    training_results[model_name] = performance
                    
            except Exception as e:
                logger.error(f"训练模型 {model_name} 时出错: {e}")
        
        # 记录训练历史
        training_record = {
            'timestamp': datetime.now().isoformat(),
            'target_column': target_column,
            'feature_count': len(feature_columns),
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'models_trained': list(training_results.keys()),
            'results': training_results
        }
        
        self.training_history.append(training_record)
        
        logger.info(f"集成模型训练完成 - 成功训练 {len(training_results)} 个模型")
        
        return training_results
    
    def predict_single(self, features: np.ndarray, model_name: str) -> Tuple[Any, Any]:
        """
        使用单个模型进行预测
        
        Args:
            features: 特征数组
            model_name: 模型名称
            
        Returns:
            (预测结果, 预测概率)
        """
        try:
            if model_name not in self.models:
                logger.warning(f"模型 {model_name} 不存在")
                return None, None
            
            model = self.models[model_name]
            scaler = self.scalers.get(self.target_column)
            features = np.asarray(features, dtype=np.float64)
            feature_frame = pd.DataFrame([features], columns=self.feature_columns)
            
            # 特征标准化
            if scaler:
                features_scaled = scaler.transform(feature_frame)
            else:
                features_scaled = feature_frame.to_numpy(dtype=np.float64)
            
            # 预测
            prediction = model.predict(features_scaled)[0]
            
            # 获取预测概率
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features_scaled)[0]
            else:
                probabilities = None
            
            return prediction, probabilities
            
        except Exception as e:
            logger.error(f"模型 {model_name} 预测失败: {e}")
            return None, None
    
    def predict_ensemble(self, features: np.ndarray, 
                        method: str = 'voting') -> Tuple[Any, Dict]:
        """
        集成预测
        
        Args:
            features: 特征数组
            method: 集成方法 ('voting', 'weighted', 'averaging')
            
        Returns:
            (最终预测, 各模型预测详情)
        """
        try:
            predictions = {}
            probabilities = {}
            
            # 获取各模型预测
            for model_name in self.models:
                pred, prob = self.predict_single(features, model_name)
                if pred is not None:
                    predictions[model_name] = pred
                    if prob is not None:
                        probabilities[model_name] = prob
            
            if not predictions:
                logger.warning("没有可用的模型预测")
                return None, {}
            
            # 集成方法
            if method == 'voting':
                # 多数投票
                pred_values = list(predictions.values())
                final_prediction = max(set(pred_values), key=pred_values.count)
                
            elif method == 'weighted':
                # 加权平均 (基于模型性能)
                weighted_sum = 0
                total_weight = 0
                
                for model_name, pred in predictions.items():
                    # 使用测试准确率作为权重
                    weight = self.model_performance.get(model_name, {}).get('test_accuracy', 0.5)
                    weighted_sum += pred * weight
                    total_weight += weight
                
                final_prediction = int(weighted_sum / total_weight) if total_weight > 0 else 0
                
            else:  # averaging
                # 简单平均
                final_prediction = int(np.mean(list(predictions.values())))
            
            prediction_details = {
                'method': method,
                'individual_predictions': predictions,
                'individual_probabilities': probabilities,
                'final_prediction': final_prediction
            }
            
            return final_prediction, prediction_details
            
        except Exception as e:
            logger.error(f"集成预测失败: {e}")
            return None, {}
    
    def get_feature_importance(self, model_name: str) -> Optional[Dict]:
        """
        获取特征重要性
        
        Args:
            model_name: 模型名称
            
        Returns:
            特征重要性字典
        """
        try:
            if model_name not in self.models:
                return None
            
            model = self.models[model_name]
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_importance = dict(zip(self.feature_columns, importances))
                
                # 按重要性排序
                sorted_importance = dict(sorted(feature_importance.items(), 
                                              key=lambda x: x[1], reverse=True))
                
                return sorted_importance
            
            return None
            
        except Exception as e:
            logger.error(f"获取特征重要性失败: {e}")
            return None
    
    def save_models(self, save_path: str):
        """
        保存模型
        
        Args:
            save_path: 保存路径
        """
        try:
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'feature_columns': self.feature_columns,
                'target_column': self.target_column,
                'model_performance': self.model_performance,
                'training_history': self.training_history,
                'timestamp': datetime.now().isoformat()
            }
            
            joblib.dump(model_data, save_path)
            logger.info(f"模型已保存至: {save_path}")
            
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
    
    def load_models(self, load_path: str) -> bool:
        """
        加载模型
        
        Args:
            load_path: 模型文件路径
            
        Returns:
            是否加载成功
        """
        try:
            model_data = joblib.load(load_path)
            
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.feature_columns = model_data['feature_columns']
            self.target_column = model_data['target_column']
            self.model_performance = model_data['model_performance']
            self.training_history = model_data.get('training_history', [])
            
            logger.info(f"模型已从 {load_path} 加载")
            return True
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False
    
    def get_models_summary(self) -> pd.DataFrame:
        """
        获取模型性能汇总
        
        Returns:
            模型性能汇总DataFrame
        """
        try:
            if not self.model_performance:
                return pd.DataFrame()
            
            summary_data = []
            for model_name, performance in self.model_performance.items():
                summary_data.append(performance)
            
            summary_df = pd.DataFrame(summary_data)
            return summary_df
            
        except Exception as e:
            logger.error(f"创建模型汇总失败: {e}")
            return pd.DataFrame() 
