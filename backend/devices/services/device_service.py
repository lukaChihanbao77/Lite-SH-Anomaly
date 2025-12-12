"""
设备业务逻辑层
"""

from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from django.db.models import Count, Q
from django.utils import timezone

from devices.models import Device, DeviceStats
from common.exceptions import DeviceNotFoundException


class DeviceService:
    """设备管理服务"""
    
    @staticmethod
    def get_device_list(
        device_type: Optional[str] = None,
        status: Optional[str] = None,
        keyword: Optional[str] = None
    ) -> List[Device]:
        """获取设备列表"""
        queryset = Device.objects.all()
        
        if device_type:
            queryset = queryset.filter(device_type=device_type)
        if status:
            queryset = queryset.filter(status=status)
        if keyword:
            queryset = queryset.filter(
                Q(name__icontains=keyword) | 
                Q(device_id__icontains=keyword) |
                Q(ip_address__icontains=keyword)
            )
        
        return queryset
    
    @staticmethod
    def get_device_by_id(device_id: str) -> Device:
        """根据设备ID获取设备"""
        try:
            return Device.objects.get(device_id=device_id)
        except Device.DoesNotExist:
            raise DeviceNotFoundException(f'设备 {device_id} 不存在')
    
    @staticmethod
    def create_device(data: Dict[str, Any]) -> Device:
        """创建设备"""
        device = Device.objects.create(**data)
        return device
    
    @staticmethod
    def update_device(device_id: str, data: Dict[str, Any]) -> Device:
        """更新设备信息"""
        device = DeviceService.get_device_by_id(device_id)
        for key, value in data.items():
            if hasattr(device, key):
                setattr(device, key, value)
        device.save()
        return device
    
    @staticmethod
    def delete_device(device_id: str) -> bool:
        """删除设备"""
        device = DeviceService.get_device_by_id(device_id)
        device.delete()
        return True
    
    @staticmethod
    def update_device_status(device_id: str, status: str) -> Device:
        """更新设备状态"""
        device = DeviceService.get_device_by_id(device_id)
        device.status = status
        device.last_seen = timezone.now()
        device.save()
        return device
    
    @staticmethod
    def get_or_create_device(device_id: str, defaults: Dict[str, Any] = None) -> Device:
        """获取或创建设备"""
        defaults = defaults or {}
        device, created = Device.objects.get_or_create(
            device_id=device_id,
            defaults={
                'name': defaults.get('name', f'设备_{device_id[:8]}'),
                'device_type': defaults.get('device_type', Device.DeviceType.OTHER),
                'ip_address': defaults.get('ip_address'),
                'status': Device.Status.ONLINE,
                'last_seen': timezone.now(),
            }
        )
        if not created:
            device.last_seen = timezone.now()
            device.status = Device.Status.ONLINE
            device.save()
        return device
    
    @staticmethod
    def get_device_overview() -> Dict[str, Any]:
        """获取设备概览统计"""
        total = Device.objects.count()
        online = Device.objects.filter(status=Device.Status.ONLINE).count()
        offline = Device.objects.filter(status=Device.Status.OFFLINE).count()
        warning = Device.objects.filter(status=Device.Status.WARNING).count()
        
        # 按类型统计
        type_stats = Device.objects.values('device_type').annotate(
            count=Count('id')
        )
        
        return {
            'total': total,
            'online': online,
            'offline': offline,
            'warning': warning,
            'type_distribution': list(type_stats)
        }
    
    @staticmethod
    def update_device_stats(device_id: str, is_anomaly: bool) -> None:
        """更新设备统计信息"""
        today = timezone.now().date()
        try:
            device = Device.objects.get(device_id=device_id)
        except Device.DoesNotExist:
            return
        
        stats, _ = DeviceStats.objects.get_or_create(
            device=device,
            date=today,
            defaults={'total_requests': 0, 'normal_count': 0, 'anomaly_count': 0}
        )
        
        stats.total_requests += 1
        if is_anomaly:
            stats.anomaly_count += 1
        else:
            stats.normal_count += 1
        stats.save()
