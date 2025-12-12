"""
简化自编码器模型
网络结构：输入→4神经元→2神经元→4神经元→输出
目标：参数量<100，适配低算力设备
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import joblib
import time
import psutil
import os
from pathlib import Path
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    accuracy_score, confusion_matrix
)
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimpleAutoEncoder(nn.Module):
    """简化自编码器网络结构"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 4, latent_dim: int = 2):
        """
        初始化自编码器
        
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度（默认4）
            latent_dim: 潜在空间维度（默认2）
        """
        super(SimpleAutoEncoder, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)


class LightweightAutoEncoder:
    """轻量化自编码器异常检测模型"""
    
    def __init__(self,
                 input_dim: int = 14,
                 hidden_dim: int = 4,
                 latent_dim: int = 2,
                 learning_rate: float = 0.001,
                 epochs: int = 50,
                 batch_size: int = 64,
                 threshold_percentile: float = 95):
        """
        初始化轻量化自编码器
        
        Args:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层神经元数（默认4，轻量化）
            latent_dim: 潜在空间维度（默认2，极简压缩）
            learning_rate: 学习率
            epochs: 训练轮数
            batch_size: 批次大小
            threshold_percentile: 异常阈值百分位数
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.threshold_percentile = threshold_percentile
        
        self.device = torch.device('cpu')  # 边缘设备使用CPU
        self.model = SimpleAutoEncoder(input_dim, hidden_dim, latent_dim).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        self.threshold = None
        self.is_trained = False
        self.training_time = 0
        self.feature_names = None
        self.train_losses = []
        
        # 计算参数量
        self.param_count = sum(p.numel() for p in self.model.parameters())
        logger.info(f"模型参数量: {self.param_count}")
        
    def train(self, X_train: np.ndarray, X_val: np.ndarray = None, feature_names: list = None):
        """
        训练模型（仅使用正常样本）
        
        Args:
            X_train: 训练数据（正常样本）
            X_val: 验证数据
            feature_names: 特征名称
        """
        logger.info(f"开始训练自编码器模型...")
        logger.info(f"网络结构: {self.input_dim}→{self.hidden_dim}→{self.latent_dim}→{self.hidden_dim}→{self.input_dim}")
        logger.info(f"参数量: {self.param_count}")
        
        self.feature_names = feature_names
        
        # 准备数据
        X_tensor = torch.FloatTensor(X_train).to(self.device)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        start_time = time.time()
        self.model.train()
        
        for epoch in range(self.epochs):
            epoch_loss = 0
            for batch_x, _ in dataloader:
                self.optimizer.zero_grad()
                output = self.model(batch_x)
                loss = self.criterion(output, batch_x)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            self.train_losses.append(avg_loss)
            
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.6f}")
        
        self.training_time = time.time() - start_time
        
        # 计算异常阈值（基于训练数据的重构误差）
        self._calculate_threshold(X_train)
        
        self.is_trained = True
        logger.info(f"训练完成，耗时: {self.training_time:.3f}s")
        logger.info(f"异常阈值: {self.threshold:.6f}")
        
    def _calculate_threshold(self, X: np.ndarray):
        """计算异常检测阈值"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            reconstructed = self.model(X_tensor)
            mse = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            self.threshold = np.percentile(mse.cpu().numpy(), self.threshold_percentile)
    
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
            
        reconstruction_errors = self._get_reconstruction_error(X)
        return np.where(reconstruction_errors > self.threshold, 1, 0)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测异常概率
        
        Args:
            X: 待预测数据
            
        Returns:
            异常概率（0-1）
        """
        if not self.is_trained:
            raise ValueError("模型未训练")
            
        reconstruction_errors = self._get_reconstruction_error(X)
        # 使用sigmoid将重构误差转换为概率
        proba = 1 / (1 + np.exp(-(reconstruction_errors - self.threshold) / self.threshold))
        return np.clip(proba, 0, 1)
    
    def _get_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """计算重构误差"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            reconstructed = self.model(X_tensor)
            mse = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            return mse.cpu().numpy()
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        评估模型性能
        
        Args:
            X_test: 测试数据
            y_test: 真实标签
            
        Returns:
            评估指标字典
        """
        y_binary = np.where(y_test > 0, 1, 0)
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_binary, y_pred),
            'precision': precision_score(y_binary, y_pred, zero_division=0),
            'recall': recall_score(y_binary, y_pred, zero_division=0),
            'f1_score': f1_score(y_binary, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y_binary, y_pred).tolist(),
            'param_count': self.param_count
        }
        
        tn, fp, fn, tp = confusion_matrix(y_binary, y_pred).ravel()
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        logger.info(f"评估结果:")
        logger.info(f"  准确率: {metrics['accuracy']:.4f}")
        logger.info(f"  精确率: {metrics['precision']:.4f}")
        logger.info(f"  召回率: {metrics['recall']:.4f}")
        logger.info(f"  F1分数: {metrics['f1_score']:.4f}")
        logger.info(f"  误报率: {metrics['false_positive_rate']:.4f}")
        logger.info(f"  漏报率: {metrics['false_negative_rate']:.4f}")
        
        return metrics
    
    def benchmark_inference(self, X_sample: np.ndarray, n_iterations: int = 100) -> dict:
        """推理性能基准测试"""
        if not self.is_trained:
            raise ValueError("模型未训练")
            
        single_sample = X_sample[0:1]
        times = []
        
        self.model.eval()
        for _ in range(n_iterations):
            start = time.time()
            self.predict(single_sample)
            times.append((time.time() - start) * 1000)
            
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        benchmark = {
            'avg_inference_time_ms': np.mean(times),
            'max_inference_time_ms': np.max(times),
            'min_inference_time_ms': np.min(times),
            'std_inference_time_ms': np.std(times),
            'memory_usage_mb': memory_mb,
            'param_count': self.param_count
        }
        
        logger.info(f"推理性能基准测试:")
        logger.info(f"  平均推理时间: {benchmark['avg_inference_time_ms']:.3f}ms")
        logger.info(f"  内存占用: {benchmark['memory_usage_mb']:.2f}MB")
        logger.info(f"  参数量: {benchmark['param_count']}")
        
        return benchmark
    
    def save_model(self, filepath: str):
        """保存模型"""
        if not self.is_trained:
            raise ValueError("模型未训练")
            
        model_data = {
            'state_dict': self.model.state_dict(),
            'params': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'latent_dim': self.latent_dim,
                'threshold': self.threshold,
                'threshold_percentile': self.threshold_percentile
            },
            'feature_names': self.feature_names,
            'training_time': self.training_time,
            'train_losses': self.train_losses
        }
        
        torch.save(model_data, filepath)
        
        file_size_mb = os.path.getsize(filepath) / 1024 / 1024
        logger.info(f"模型已保存: {filepath}")
        logger.info(f"模型文件大小: {file_size_mb:.4f}MB")
        
        return file_size_mb
    
    @classmethod
    def load_model(cls, filepath: str) -> 'LightweightAutoEncoder':
        """加载模型"""
        model_data = torch.load(filepath, map_location='cpu')
        
        params = model_data['params']
        instance = cls(
            input_dim=params['input_dim'],
            hidden_dim=params['hidden_dim'],
            latent_dim=params['latent_dim'],
            threshold_percentile=params['threshold_percentile']
        )
        
        instance.model.load_state_dict(model_data['state_dict'])
        instance.threshold = params['threshold']
        instance.feature_names = model_data['feature_names']
        instance.training_time = model_data['training_time']
        instance.train_losses = model_data.get('train_losses', [])
        instance.is_trained = True
        
        logger.info(f"模型已加载: {filepath}")
        return instance


if __name__ == '__main__':
    print("轻量化自编码器模型模块已就绪")
    print("使用方法:")
    print("  model = LightweightAutoEncoder(input_dim=14)")
    print("  model.train(X_train_normal)")
    print("  predictions = model.predict(X_test)")
