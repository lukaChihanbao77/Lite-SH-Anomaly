"""
对比基准模型：KNN、SVM
用于验证轻量化算法的优势
"""

import numpy as np
import joblib
import time
import psutil
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix
)
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BaselineKNN:
    """KNN基准模型"""
    
    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=1)
        self.is_trained = False
        self.training_time = 0
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """训练模型"""
        logger.info(f"开始训练KNN模型 (k={self.n_neighbors})...")
        
        # 转换为二分类
        y_binary = np.where(y_train > 0, 1, 0)
        
        start_time = time.time()
        self.model.fit(X_train, y_binary)
        self.training_time = time.time() - start_time
        
        self.is_trained = True
        logger.info(f"训练完成，耗时: {self.training_time:.3f}s")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba = self.model.predict_proba(X)
        return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        y_binary = np.where(y_test > 0, 1, 0)
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_binary, y_pred),
            'precision': precision_score(y_binary, y_pred, zero_division=0),
            'recall': recall_score(y_binary, y_pred, zero_division=0),
            'f1_score': f1_score(y_binary, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_binary, y_pred).tolist()
        }
        
        tn, fp, fn, tp = confusion_matrix(y_binary, y_pred).ravel()
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        logger.info(f"KNN评估结果: F1={metrics['f1_score']:.4f}")
        return metrics
    
    def benchmark_inference(self, X_sample: np.ndarray, n_iterations: int = 100) -> dict:
        single_sample = X_sample[0:1]
        times = []
        
        for _ in range(n_iterations):
            start = time.time()
            self.predict(single_sample)
            times.append((time.time() - start) * 1000)
            
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        return {
            'avg_inference_time_ms': np.mean(times),
            'max_inference_time_ms': np.max(times),
            'memory_usage_mb': memory_mb
        }
    
    def save_model(self, filepath: str):
        joblib.dump({'model': self.model, 'n_neighbors': self.n_neighbors}, filepath, compress=3)
        return os.path.getsize(filepath) / 1024 / 1024


class BaselineSVM:
    """SVM基准模型"""
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0):
        self.kernel = kernel
        self.C = C
        self.model = SVC(kernel=kernel, C=C, probability=True, random_state=42)
        self.is_trained = False
        self.training_time = 0
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """训练模型"""
        logger.info(f"开始训练SVM模型 (kernel={self.kernel})...")
        
        y_binary = np.where(y_train > 0, 1, 0)
        
        start_time = time.time()
        self.model.fit(X_train, y_binary)
        self.training_time = time.time() - start_time
        
        self.is_trained = True
        logger.info(f"训练完成，耗时: {self.training_time:.3f}s")
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        proba = self.model.predict_proba(X)
        return proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        y_binary = np.where(y_test > 0, 1, 0)
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_binary, y_pred),
            'precision': precision_score(y_binary, y_pred, zero_division=0),
            'recall': recall_score(y_binary, y_pred, zero_division=0),
            'f1_score': f1_score(y_binary, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_binary, y_pred).tolist()
        }
        
        tn, fp, fn, tp = confusion_matrix(y_binary, y_pred).ravel()
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        logger.info(f"SVM评估结果: F1={metrics['f1_score']:.4f}")
        return metrics
    
    def benchmark_inference(self, X_sample: np.ndarray, n_iterations: int = 100) -> dict:
        single_sample = X_sample[0:1]
        times = []
        
        for _ in range(n_iterations):
            start = time.time()
            self.predict(single_sample)
            times.append((time.time() - start) * 1000)
            
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        return {
            'avg_inference_time_ms': np.mean(times),
            'max_inference_time_ms': np.max(times),
            'memory_usage_mb': memory_mb
        }
    
    def save_model(self, filepath: str):
        joblib.dump({'model': self.model, 'kernel': self.kernel, 'C': self.C}, filepath, compress=3)
        return os.path.getsize(filepath) / 1024 / 1024


if __name__ == '__main__':
    print("基准模型模块已就绪")
    print("可用模型: BaselineKNN, BaselineSVM")
