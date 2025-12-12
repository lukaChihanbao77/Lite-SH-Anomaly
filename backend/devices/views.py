"""
设备视图层 - RESTful API
"""

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from drf_spectacular.utils import extend_schema, extend_schema_view, OpenApiParameter

from devices.models import Device
from devices.serializers import (
    DeviceSerializer, DeviceCreateSerializer, 
    DeviceUpdateSerializer, DeviceOverviewSerializer
)
from devices.services import DeviceService
from common.response import APIResponse


@extend_schema_view(
    list=extend_schema(
        tags=['devices'],
        summary='获取设备列表',
        description='获取所有设备列表，支持按类型、状态、关键词筛选',
        parameters=[
            OpenApiParameter(name='device_type', description='设备类型', required=False, type=str),
            OpenApiParameter(name='status', description='设备状态', required=False, type=str),
            OpenApiParameter(name='keyword', description='搜索关键词', required=False, type=str),
        ]
    ),
    create=extend_schema(
        tags=['devices'],
        summary='创建设备',
        description='创建新的智能家居设备',
        request=DeviceCreateSerializer,
    ),
    retrieve=extend_schema(
        tags=['devices'],
        summary='获取设备详情',
        description='根据设备ID获取设备详细信息',
    ),
    update=extend_schema(
        tags=['devices'],
        summary='更新设备',
        description='更新设备信息',
        request=DeviceUpdateSerializer,
    ),
    destroy=extend_schema(
        tags=['devices'],
        summary='删除设备',
        description='删除指定设备',
    ),
)
class DeviceViewSet(viewsets.ViewSet):
    """
    设备管理API
    
    RESTful接口:
    - GET    /api/v1/devices/           获取设备列表
    - POST   /api/v1/devices/           创建设备
    - GET    /api/v1/devices/{id}/      获取设备详情
    - PUT    /api/v1/devices/{id}/      更新设备
    - DELETE /api/v1/devices/{id}/      删除设备
    - GET    /api/v1/devices/overview/  获取设备概览
    """
    
    def list(self, request):
        """获取设备列表"""
        device_type = request.query_params.get('device_type')
        device_status = request.query_params.get('status')
        keyword = request.query_params.get('keyword')
        
        devices = DeviceService.get_device_list(
            device_type=device_type,
            status=device_status,
            keyword=keyword
        )
        
        return APIResponse.paginated(devices, DeviceSerializer, request)
    
    def create(self, request):
        """创建设备"""
        serializer = DeviceCreateSerializer(data=request.data)
        if not serializer.is_valid():
            return APIResponse.error(
                message='参数验证失败',
                data=serializer.errors
            )
        
        device = DeviceService.create_device(serializer.validated_data)
        return APIResponse.created(
            data=DeviceSerializer(device).data,
            message='设备创建成功'
        )
    
    def retrieve(self, request, pk=None):
        """获取设备详情"""
        try:
            device = DeviceService.get_device_by_id(pk)
            return APIResponse.success(data=DeviceSerializer(device).data)
        except Exception as e:
            return APIResponse.not_found(str(e))
    
    def update(self, request, pk=None):
        """更新设备"""
        serializer = DeviceUpdateSerializer(data=request.data, partial=True)
        if not serializer.is_valid():
            return APIResponse.error(
                message='参数验证失败',
                data=serializer.errors
            )
        
        try:
            device = DeviceService.update_device(pk, serializer.validated_data)
            return APIResponse.success(
                data=DeviceSerializer(device).data,
                message='设备更新成功'
            )
        except Exception as e:
            return APIResponse.not_found(str(e))
    
    def destroy(self, request, pk=None):
        """删除设备"""
        try:
            DeviceService.delete_device(pk)
            return APIResponse.success(message='设备删除成功')
        except Exception as e:
            return APIResponse.not_found(str(e))
    
    @extend_schema(
        tags=['devices'],
        summary='获取设备概览',
        description='获取设备统计概览，包括在线/离线/告警数量及类型分布',
    )
    @action(detail=False, methods=['get'])
    def overview(self, request):
        """获取设备概览统计"""
        data = DeviceService.get_device_overview()
        return APIResponse.success(data=data)
    
    @extend_schema(
        tags=['devices'],
        summary='更新设备状态',
        description='更新指定设备的在线状态',
    )
    @action(detail=True, methods=['post'])
    def update_status(self, request, pk=None):
        """更新设备状态"""
        new_status = request.data.get('status')
        if not new_status:
            return APIResponse.error(message='请提供状态参数')
        
        try:
            device = DeviceService.update_device_status(pk, new_status)
            return APIResponse.success(
                data=DeviceSerializer(device).data,
                message='状态更新成功'
            )
        except Exception as e:
            return APIResponse.not_found(str(e))
