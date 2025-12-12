"""
检测视图层 - RESTful API
"""

import os
import uuid
from datetime import datetime

from django.utils import timezone
from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.parsers import MultiPartParser, JSONParser
from drf_spectacular.utils import extend_schema, OpenApiParameter
from drf_spectacular.types import OpenApiTypes

from detection.models import DetectionRecord, DetectionTask
from detection.serializers import (
    DetectionRecordSerializer, DetectionInputSerializer,
    DetectionBatchInputSerializer, DetectionResultSerializer,
    DetectionTaskSerializer, DetectionStatsSerializer
)
from detection.services import DetectionService, ModelService
from common.response import APIResponse


class DetectionViewSet(viewsets.ViewSet):
    """
    异常检测API
    
    RESTful接口:
    - POST   /api/v1/detection/detect/       单条检测
    - POST   /api/v1/detection/batch/        批量检测
    - POST   /api/v1/detection/upload/       上传CSV检测
    - GET    /api/v1/detection/records/      检测记录列表
    - GET    /api/v1/detection/records/{id}/ 检测记录详情
    - GET    /api/v1/detection/stats/        检测统计
    - GET    /api/v1/detection/tasks/        任务列表
    - GET    /api/v1/detection/model-info/   模型信息
    """
    
    parser_classes = [JSONParser, MultiPartParser]
    
    @extend_schema(
        tags=['detection'],
        summary='单条数据检测',
        description='对单条网络流量数据进行异常检测，返回检测结果',
        request=DetectionInputSerializer,
        responses={200: DetectionResultSerializer},
    )
    @action(detail=False, methods=['post'])
    def detect(self, request):
        """单条数据检测"""
        serializer = DetectionInputSerializer(data=request.data)
        if not serializer.is_valid():
            return APIResponse.error(
                message='参数验证失败',
                data=serializer.errors
            )
        
        result = DetectionService.detect_single(serializer.validated_data)
        return APIResponse.success(
            data=result,
            message='检测完成'
        )
    
    @extend_schema(
        tags=['detection'],
        summary='批量数据检测',
        description='对多条网络流量数据进行批量异常检测',
        request=DetectionBatchInputSerializer,
    )
    @action(detail=False, methods=['post'])
    def batch(self, request):
        """批量数据检测"""
        serializer = DetectionBatchInputSerializer(data=request.data)
        if not serializer.is_valid():
            return APIResponse.error(
                message='参数验证失败',
                data=serializer.errors
            )
        
        result = DetectionService.detect_batch(serializer.validated_data['data'])
        return APIResponse.success(
            data=result,
            message=f'批量检测完成，共{result["total"]}条'
        )
    
    @extend_schema(
        tags=['detection'],
        summary='上传CSV文件检测',
        description='上传CSV格式的流量数据文件进行批量检测',
        request={'multipart/form-data': {'type': 'object', 'properties': {'file': {'type': 'string', 'format': 'binary'}}}},
    )
    @action(detail=False, methods=['post'], parser_classes=[MultiPartParser])
    def upload(self, request):
        """上传CSV文件进行检测"""
        file = request.FILES.get('file')
        if not file:
            return APIResponse.error(message='请上传CSV文件')
        
        if not file.name.endswith('.csv'):
            return APIResponse.error(message='仅支持CSV格式文件')
        
        # 保存临时文件
        task_id = str(uuid.uuid4())[:8]
        temp_dir = '/tmp/detection_uploads'
        os.makedirs(temp_dir, exist_ok=True)
        file_path = os.path.join(temp_dir, f'{task_id}_{file.name}')
        
        with open(file_path, 'wb+') as dest:
            for chunk in file.chunks():
                dest.write(chunk)
        
        try:
            result = DetectionService.detect_from_csv(file_path, task_id)
            return APIResponse.success(
                data=result,
                message='CSV检测完成'
            )
        except Exception as e:
            return APIResponse.error(message=str(e))
        finally:
            # 清理临时文件
            if os.path.exists(file_path):
                os.remove(file_path)
    
    @extend_schema(
        tags=['detection'],
        summary='获取检测记录列表',
        description='查询历史检测记录，支持按设备、攻击类型、时间范围筛选',
        parameters=[
            OpenApiParameter(name='device_id', description='设备ID', required=False, type=str),
            OpenApiParameter(name='attack_type', description='攻击类型', required=False, type=str),
            OpenApiParameter(name='is_anomaly', description='是否异常', required=False, type=bool),
            OpenApiParameter(name='start_time', description='开始时间(ISO格式)', required=False, type=str),
            OpenApiParameter(name='end_time', description='结束时间(ISO格式)', required=False, type=str),
        ],
        responses={200: DetectionRecordSerializer(many=True)},
    )
    @action(detail=False, methods=['get'])
    def records(self, request):
        """获取检测记录列表"""
        device_id = request.query_params.get('device_id')
        attack_type = request.query_params.get('attack_type')
        is_anomaly = request.query_params.get('is_anomaly')
        start_time = request.query_params.get('start_time')
        end_time = request.query_params.get('end_time')
        
        # 处理布尔参数
        if is_anomaly is not None:
            is_anomaly = is_anomaly.lower() == 'true'
        
        # 处理时间参数
        if start_time:
            start_time = datetime.fromisoformat(start_time)
        if end_time:
            end_time = datetime.fromisoformat(end_time)
        
        records = DetectionService.get_detection_records(
            device_id=device_id,
            attack_type=attack_type,
            is_anomaly=is_anomaly,
            start_time=start_time,
            end_time=end_time
        )
        
        return APIResponse.paginated(records, DetectionRecordSerializer, request)
    
    @extend_schema(
        tags=['detection'],
        summary='获取检测记录详情',
        description='根据记录ID获取检测记录详细信息',
        parameters=[
            OpenApiParameter(name='record_id', description='记录ID', required=True, type=int, location=OpenApiParameter.PATH),
        ],
    )
    @action(detail=False, methods=['get'], url_path='records/(?P<record_id>[^/.]+)')
    def record_detail(self, request, record_id=None):
        """获取检测记录详情"""
        try:
            record = DetectionRecord.objects.get(id=record_id)
            return APIResponse.success(data=DetectionRecordSerializer(record).data)
        except DetectionRecord.DoesNotExist:
            return APIResponse.not_found('检测记录不存在')
    
    @extend_schema(
        tags=['detection'],
        summary='获取检测统计数据',
        description='获取指定天数内的检测统计数据，包括异常率、攻击分布、趋势等',
        parameters=[
            OpenApiParameter(name='days', description='统计天数', required=False, type=int, default=7),
        ],
        responses={200: DetectionStatsSerializer},
    )
    @action(detail=False, methods=['get'])
    def stats(self, request):
        """获取检测统计数据"""
        days = int(request.query_params.get('days', 7))
        stats = DetectionService.get_statistics(days=days)
        return APIResponse.success(data=stats)
    
    @extend_schema(
        tags=['detection'],
        summary='获取检测任务列表',
        description='获取CSV批量检测任务列表及其状态',
        responses={200: DetectionTaskSerializer(many=True)},
    )
    @action(detail=False, methods=['get'])
    def tasks(self, request):
        """获取检测任务列表"""
        tasks = DetectionTask.objects.all()[:20]
        serializer = DetectionTaskSerializer(tasks, many=True)
        return APIResponse.success(data=serializer.data)
    
    @extend_schema(
        tags=['detection'],
        summary='获取模型信息',
        description='获取当前加载的检测模型信息，包括版本、特征列表、支持的攻击类型',
    )
    @action(detail=False, methods=['get'], url_path='model-info')
    def model_info(self, request):
        """获取模型信息"""
        model_service = ModelService()
        return APIResponse.success(data={
            'is_loaded': model_service.is_loaded,
            'model_version': model_service.model_version,
            'features': [
                'duration', 'orig_bytes', 'resp_bytes', 'orig_pkts',
                'resp_pkts', 'proto_encoded', 'bytes_ratio', 'pkts_ratio'
            ],
            'attack_types': ['normal', 'ddos', 'port_scan', 'unauthorized', 'malformed', 'unknown']
        })
