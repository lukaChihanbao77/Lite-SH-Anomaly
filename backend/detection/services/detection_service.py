"""
检测业务逻辑层
"""

import uuid
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from django.db.models import Count, Q, Avg
from django.utils import timezone

from detection.models import DetectionRecord, DetectionTask
from detection.services.model_service import model_service
from devices.services import DeviceService
from logs.services import AlertService
from common.exceptions import InvalidDataException

logger = logging.getLogger(__name__)


class DetectionService:
    """异常检测服务"""
    
    # 特征字段映射
    FEATURE_COLUMNS = [
        'duration', 'orig_bytes', 'resp_bytes', 'orig_pkts', 
        'resp_pkts', 'proto_encoded', 'bytes_ratio', 'pkts_ratio'
    ]
    
    @staticmethod
    def detect_single(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        单条数据检测
        
        Args:
            data: 包含设备信息和流量特征的字典
        
        Returns:
            检测结果
        """
        # 提取特征
        features = DetectionService._extract_features(data)
        
        # 模型预测
        result = model_service.predict(features)
        
        # 保存检测记录
        record = DetectionService._save_record(data, result)
        
        # 更新设备统计
        device_id = data.get('device_id', 'unknown')
        DeviceService.update_device_stats(device_id, result['is_anomaly'])
        
        # 如果是异常，创建告警
        if result['is_anomaly']:
            DeviceService.update_device_status(device_id, 'warning')
            AlertService.create_alert_from_detection(record, result)
        
        return {
            'record_id': record.id,
            **result
        }
    
    @staticmethod
    def detect_batch(data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批量数据检测"""
        results = []
        anomaly_count = 0
        
        for data in data_list:
            result = DetectionService.detect_single(data)
            results.append(result)
            if result['is_anomaly']:
                anomaly_count += 1
        
        return {
            'total': len(results),
            'anomaly_count': anomaly_count,
            'normal_count': len(results) - anomaly_count,
            'results': results
        }
    
    @staticmethod
    def detect_from_csv(file_path: str, task_id: str = None) -> Dict[str, Any]:
        """从CSV文件批量检测"""
        if task_id is None:
            task_id = str(uuid.uuid4())[:8]
        
        # 创建任务记录
        task = DetectionTask.objects.create(
            task_id=task_id,
            file_name=file_path,
            status=DetectionTask.TaskStatus.RUNNING,
            started_at=timezone.now()
        )
        
        try:
            # 读取CSV
            df = pd.read_csv(file_path)
            task.total_count = len(df)
            task.save()
            
            anomaly_count = 0
            
            for idx, row in df.iterrows():
                data = row.to_dict()
                result = DetectionService.detect_single(data)
                
                if result['is_anomaly']:
                    anomaly_count += 1
                
                task.processed_count = idx + 1
                task.anomaly_count = anomaly_count
                task.save()
            
            task.status = DetectionTask.TaskStatus.COMPLETED
            task.completed_at = timezone.now()
            task.save()
            
            return {
                'task_id': task_id,
                'status': 'completed',
                'total': task.total_count,
                'anomaly_count': anomaly_count
            }
            
        except Exception as e:
            task.status = DetectionTask.TaskStatus.FAILED
            task.error_message = str(e)
            task.save()
            raise InvalidDataException(f'CSV处理失败: {e}')
    
    @staticmethod
    def get_detection_records(
        device_id: Optional[str] = None,
        attack_type: Optional[str] = None,
        is_anomaly: Optional[bool] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ):
        """查询检测记录"""
        queryset = DetectionRecord.objects.all()
        
        if device_id:
            queryset = queryset.filter(device_id=device_id)
        if attack_type:
            queryset = queryset.filter(attack_type=attack_type)
        if is_anomaly is not None:
            queryset = queryset.filter(is_anomaly=is_anomaly)
        if start_time:
            queryset = queryset.filter(timestamp__gte=start_time)
        if end_time:
            queryset = queryset.filter(timestamp__lte=end_time)
        
        return queryset
    
    @staticmethod
    def get_statistics(days: int = 7) -> Dict[str, Any]:
        """获取检测统计数据"""
        end_time = timezone.now()
        start_time = end_time - timedelta(days=days)
        
        records = DetectionRecord.objects.filter(timestamp__gte=start_time)
        
        total = records.count()
        anomaly = records.filter(is_anomaly=True).count()
        normal = total - anomaly
        
        # 按攻击类型统计
        attack_stats = records.filter(is_anomaly=True).values('attack_type').annotate(
            count=Count('id')
        )
        
        # 按天统计趋势
        daily_stats = records.extra(
            select={'date': 'date(timestamp)'}
        ).values('date').annotate(
            total=Count('id'),
            anomaly=Count('id', filter=Q(is_anomaly=True))
        ).order_by('date')
        
        # 平均推理时间
        avg_inference_time = records.aggregate(avg=Avg('inference_time'))['avg'] or 0
        
        return {
            'total': total,
            'anomaly': anomaly,
            'normal': normal,
            'anomaly_rate': round(anomaly / total * 100, 2) if total > 0 else 0,
            'attack_distribution': list(attack_stats),
            'daily_trend': list(daily_stats),
            'avg_inference_time': round(avg_inference_time, 2)
        }
    
    @staticmethod
    def _extract_features(data: Dict[str, Any]) -> np.ndarray:
        """从数据中提取特征向量"""
        features = []
        
        # 基础特征
        features.append(float(data.get('duration', 0)))
        features.append(float(data.get('orig_bytes', 0)))
        features.append(float(data.get('resp_bytes', 0)))
        features.append(float(data.get('orig_pkts', 0)))
        features.append(float(data.get('resp_pkts', 0)))
        
        # 协议编码
        proto = data.get('protocol', 'tcp').lower()
        proto_map = {'tcp': 0, 'udp': 1, 'icmp': 2}
        features.append(proto_map.get(proto, 0))
        
        # 衍生特征
        orig_bytes = float(data.get('orig_bytes', 0))
        resp_bytes = float(data.get('resp_bytes', 0))
        orig_pkts = float(data.get('orig_pkts', 0))
        resp_pkts = float(data.get('resp_pkts', 0))
        
        bytes_ratio = orig_bytes / (resp_bytes + 1)
        pkts_ratio = orig_pkts / (resp_pkts + 1)
        
        features.append(bytes_ratio)
        features.append(pkts_ratio)
        
        return np.array(features, dtype=np.float32)
    
    @staticmethod
    def _save_record(data: Dict[str, Any], result: Dict[str, Any]) -> DetectionRecord:
        """保存检测记录"""
        record = DetectionRecord.objects.create(
            device_id=data.get('device_id', 'unknown'),
            timestamp=data.get('timestamp', timezone.now()),
            src_ip=data.get('src_ip', '0.0.0.0'),
            dst_ip=data.get('dst_ip', '0.0.0.0'),
            src_port=data.get('src_port'),
            dst_port=data.get('dst_port'),
            protocol=data.get('protocol', 'TCP'),
            duration=data.get('duration', 0),
            orig_bytes=data.get('orig_bytes', 0),
            resp_bytes=data.get('resp_bytes', 0),
            orig_pkts=data.get('orig_pkts', 0),
            resp_pkts=data.get('resp_pkts', 0),
            is_anomaly=result['is_anomaly'],
            attack_type=result['attack_type'],
            confidence=result['confidence'],
            anomaly_score=result['anomaly_score'],
            model_version=result['model_version'],
            inference_time=result['inference_time']
        )
        return record
