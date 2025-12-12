"""
模型服务层
负责加载和调用检测模型
"""

import os
import time
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

import numpy as np
import joblib
from django.conf import settings

from common.exceptions import ModelNotLoadedException

logger = logging.getLogger(__name__)


class ModelService:
    """模型管理服务（单例模式）"""
    
    _instance = None
    _model = None
    _model_version = 'v1.0'
    _is_loaded = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """加载模型"""
        if model_path is None:
            model_path = settings.MODEL_DIR / 'isolation_forest_lite.joblib'
        
        try:
            if os.path.exists(model_path):
                self._model = joblib.load(model_path)
                self._is_loaded = True
                logger.info(f'模型加载成功: {model_path}')
                return True
            else:
                logger.warning(f'模型文件不存在: {model_path}，使用模拟模式')
                self._is_loaded = False
                return False
        except Exception as e:
            logger.error(f'模型加载失败: {e}')
            self._is_loaded = False
            return False
    
    def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """
        单条样本预测
        
        Args:
            features: 特征向量 shape=(n_features,) 或 (1, n_features)
        
        Returns:
            预测结果字典
        """
        start_time = time.time()
        
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        if self._is_loaded and self._model is not None:
            # 使用真实模型预测
            prediction = self._model.predict(features)[0]
            score = self._model.decision_function(features)[0]
            is_anomaly = prediction == -1
            anomaly_score = -score  # 转换为正值，越大越异常
        else:
            # 模拟预测（用于开发测试）
            is_anomaly, anomaly_score = self._simulate_predict(features[0])
        
        inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
        
        # 确定攻击类型
        attack_type = self._classify_attack(features[0], anomaly_score) if is_anomaly else 'normal'
        
        # 计算置信度
        confidence = min(abs(anomaly_score) / 0.5, 1.0) if is_anomaly else 1.0 - min(abs(anomaly_score) / 0.5, 0.5)
        
        return {
            'is_anomaly': bool(is_anomaly),
            'attack_type': attack_type,
            'confidence': round(float(confidence), 4),
            'anomaly_score': round(float(anomaly_score), 4),
            'inference_time': round(inference_time, 2),
            'model_version': self._model_version
        }
    
    def predict_batch(self, features_batch: np.ndarray) -> List[Dict[str, Any]]:
        """批量预测"""
        results = []
        for features in features_batch:
            result = self.predict(features)
            results.append(result)
        return results
    
    def _simulate_predict(self, features: np.ndarray) -> tuple:
        """模拟预测（开发测试用）"""
        # 基于简单规则模拟异常检测
        # 特征顺序: duration, orig_bytes, resp_bytes, orig_pkts, resp_pkts, proto, bytes_ratio, pkts_ratio
        
        anomaly_score = 0.0
        
        # 规则1: 高流量可能是DDoS
        if len(features) > 1 and features[1] > 10000:
            anomaly_score += 0.3
        
        # 规则2: 包数异常
        if len(features) > 3 and features[3] > 100:
            anomaly_score += 0.2
        
        # 规则3: 持续时间异常短但流量大
        if len(features) > 0 and features[0] < 0.1 and len(features) > 1 and features[1] > 1000:
            anomaly_score += 0.3
        
        # 添加随机扰动
        anomaly_score += np.random.uniform(-0.1, 0.1)
        
        is_anomaly = anomaly_score > 0.3
        return is_anomaly, anomaly_score
    
    def _classify_attack(self, features: np.ndarray, score: float) -> str:
        """根据特征分类攻击类型"""
        # 简化的攻击分类逻辑
        if len(features) < 4:
            return 'unknown'
        
        orig_bytes = features[1] if len(features) > 1 else 0
        orig_pkts = features[3] if len(features) > 3 else 0
        duration = features[0] if len(features) > 0 else 0
        
        # DDoS: 短时间大量包
        if orig_pkts > 50 and duration < 1:
            return 'ddos'
        
        # 端口扫描: 多个短连接
        if duration < 0.01 and orig_bytes < 100:
            return 'port_scan'
        
        # 越权访问: 异常大的响应
        if len(features) > 2 and features[2] > 50000:
            return 'unauthorized'
        
        # 异常指令
        if score > 0.5:
            return 'malformed'
        
        return 'unknown'
    
    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
    
    @property
    def model_version(self) -> str:
        return self._model_version


# 全局模型服务实例
model_service = ModelService()
