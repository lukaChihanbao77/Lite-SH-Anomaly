"""
设备序列化器
"""

from rest_framework import serializers
from devices.models import Device, DeviceStats


class DeviceSerializer(serializers.ModelSerializer):
    """设备序列化器"""
    
    device_type_display = serializers.CharField(source='get_device_type_display', read_only=True)
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    
    class Meta:
        model = Device
        fields = [
            'id', 'device_id', 'name', 'device_type', 'device_type_display',
            'ip_address', 'mac_address', 'status', 'status_display',
            'location', 'is_trusted', 'last_seen', 'created_at', 'updated_at'
        ]
        read_only_fields = ['id', 'created_at', 'updated_at']


class DeviceCreateSerializer(serializers.ModelSerializer):
    """设备创建序列化器"""
    
    class Meta:
        model = Device
        fields = ['device_id', 'name', 'device_type', 'ip_address', 'mac_address', 'location']
    
    def validate_device_id(self, value):
        if Device.objects.filter(device_id=value).exists():
            raise serializers.ValidationError('设备ID已存在')
        return value


class DeviceUpdateSerializer(serializers.ModelSerializer):
    """设备更新序列化器"""
    
    class Meta:
        model = Device
        fields = ['name', 'device_type', 'ip_address', 'mac_address', 'location', 'is_trusted']


class DeviceStatsSerializer(serializers.ModelSerializer):
    """设备统计序列化器"""
    
    device_id = serializers.CharField(source='device.device_id', read_only=True)
    device_name = serializers.CharField(source='device.name', read_only=True)
    
    class Meta:
        model = DeviceStats
        fields = ['device_id', 'device_name', 'date', 'total_requests', 'normal_count', 'anomaly_count']


class DeviceOverviewSerializer(serializers.Serializer):
    """设备概览序列化器"""
    
    total = serializers.IntegerField()
    online = serializers.IntegerField()
    offline = serializers.IntegerField()
    warning = serializers.IntegerField()
    type_distribution = serializers.ListField()
