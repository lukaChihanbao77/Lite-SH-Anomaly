"""
轻量化孤立森林模型
核心参数：决策树数量=50、树深度≤8、采样量=256
目标：推理延迟≤100ms、内存占用≤30MB、F1≥0.85
"""

import numpy as np
import joblib
import time
import psutil
import os
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, accuracy_score
)
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LightweightIsolationForest:
    """轻量化孤立森林异常检测模型"""
    
    def __init__(self, 
                 n_estimators: int = 50,
                 max_depth: int = 8,
                 max_samples: int = 256,
                 contamination: float = 0.2,
                 random_state: int = 42):
        """
        初始化轻量化孤立森林
        
        Args:
            n_estimators: 决策树数量（默认50，轻量化）
            max_depth: 最大树深度（默认8，限制复杂度）
            max_samples: 采样数量（默认256，加速训练）
            contamination: 异常比例（默认0.2，即20%异常样本）
            random_state: 随机种子
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_samples = max_samples
        self.contamination = contamination
        self.random_state = random_state
        
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            n_jobs=1,  # 单线程，适配边缘设备
            warm_start=False
        )
        
        self.is_trained = False
        self.training_time = 0
        self.feature_names = None
        
    def train(self, X_train: np.ndarray, feature_names: list = None):
        """
        训练模型
        
        Args:
            X_train: 训练数据（仅使用正常样本或混合样本）
            feature_names: 特征名称列表
        """
        logger.info(f"开始训练孤立森林模型...")
        logger.info(f"参数: n_estimators={self.n_estimators}, max_depth={self.max_depth}, max_samples={self.max_samples}")
        
        self.feature_names = feature_names
        
        start_time = time.time()
        self.model.fit(X_train)
        self.training_time = time.time() - start_time
        
        self.is_trained = True
        logger.info(f"训练完成，耗时: {self.training_time:.3f}s")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测异常
        
        Args:
            X: 待预测数据
            
        Returns:
            预测结果（0=正常，1=异常）
        """
        if not self.is_trained:
            raise ValueError("模型未训练")
            
        # IsolationForest返回1=正常，-1=异常，需要转换
        predictions = self.model.predict(X)
        # 转换为0=正常，1=异常
        return np.where(predictions == -1, 1, 0)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测异常分数（置信度）
        
        Args:
            X: 待预测数据
            
        Returns:
            异常分数（越高越可能是异常）
        """
        if not self.is_trained:
            raise ValueError("模型未训练")
            
        # decision_function返回负值表示异常，正值表示正常
        scores = self.model.decision_function(X)
        # 转换为0-1范围的异常概率
        # 使用sigmoid转换
        proba = 1 / (1 + np.exp(scores * 5))
        return proba
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        评估模型性能
        
        Args:
            X_test: 测试数据
            y_test: 真实标签（0=正常，非0=异常）
            
        Returns:
            评估指标字典
        """
        # 将多分类标签转换为二分类（0=正常，1=异常）
        y_binary = np.where(y_test > 0, 1, 0)
        
        # 预测
        y_pred = self.predict(X_test)
        
        # 计算指标
        metrics = {
            'accuracy': accuracy_score(y_binary, y_pred),
            'precision': precision_score(y_binary, y_pred, zero_division=0),
            'recall': recall_score(y_binary, y_pred, zero_division=0),
            'f1_score': f1_score(y_binary, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_binary, y_pred).tolist()
        }
        
        # 计算误报率和漏报率
        tn, fp, fn, tp = confusion_matrix(y_binary, y_pred).ravel()
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0  # 误报率
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0  # 漏报率
        
        logger.info(f"评估结果:")
        logger.info(f"  准确率: {metrics['accuracy']:.4f}")
        logger.info(f"  精确率: {metrics['precision']:.4f}")
        logger.info(f"  召回率: {metrics['recall']:.4f}")
        logger.info(f"  F1分数: {metrics['f1_score']:.4f}")
        logger.info(f"  误报率: {metrics['false_positive_rate']:.4f}")
        logger.info(f"  漏报率: {metrics['false_negative_rate']:.4f}")
        
        return metrics
    
    def benchmark_inference(self, X_sample: np.ndarray, n_iterations: int = 100) -> dict:
        """
        推理性能基准测试
        
        Args:
            X_sample: 测试样本
            n_iterations: 迭代次数
            
        Returns:
            性能指标
        """
        if not self.is_trained:
            raise ValueError("模型未训练")
            
        # 单条样本推理时间
        single_sample = X_sample[0:1]
        times = []
        
        for _ in range(n_iterations):
            start = time.time()
            self.predict(single_sample)
            times.append((time.time() - start) * 1000)  # 转换为毫秒
            
        # 内存占用
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        benchmark = {
            'avg_inference_time_ms': np.mean(times),
            'max_inference_time_ms': np.max(times),
            'min_inference_time_ms': np.min(times),
            'std_inference_time_ms': np.std(times),
            'memory_usage_mb': memory_mb
        }
        
        logger.info(f"推理性能基准测试:")
        logger.info(f"  平均推理时间: {benchmark['avg_inference_time_ms']:.3f}ms")
        logger.info(f"  最大推理时间: {benchmark['max_inference_time_ms']:.3f}ms")
        logger.info(f"  内存占用: {benchmark['memory_usage_mb']:.2f}MB")
        
        return benchmark
    
    def save_model(self, filepath: str):
        """保存模型"""
        if not self.is_trained:
            raise ValueError("模型未训练")
            
        model_data = {
            'model': self.model,
            'params': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'max_samples': self.max_samples,
                'contamination': self.contamination
            },
            'feature_names': self.feature_names,
            'training_time': self.training_time
        }
        
        joblib.dump(model_data, filepath, compress=3)
        
        # 检查文件大小
        file_size_mb = os.path.getsize(filepath) / 1024 / 1024
        logger.info(f"模型已保存: {filepath}")
        logger.info(f"模型文件大小: {file_size_mb:.2f}MB")
        
        return file_size_mb
    
    @classmethod
    def load_model(cls, filepath: str) -> 'LightweightIsolationForest':
        """加载模型"""
        model_data = joblib.load(filepath)
        
        instance = cls(
            n_estimators=model_data['params']['n_estimators'],
            max_depth=model_data['params']['max_depth'],
            max_samples=model_data['params']['max_samples'],
            contamination=model_data['params']['contamination']
        )
        
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.training_time = model_data['training_time']
        instance.is_trained = True
        
        logger.info(f"模型已加载: {filepath}")
        return instance


if __name__ == '__main__':
    # 测试代码
    print("轻量化孤立森林模型模块已就绪")
    print("使用方法:")
    print("  model = LightweightIsolationForest()")
    print("  model.train(X_train)")
    print("  predictions = model.predict(X_test)")
    print("  metrics = model.evaluate(X_test, y_test)")
