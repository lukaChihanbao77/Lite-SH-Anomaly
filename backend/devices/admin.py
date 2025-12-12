"""
设备管理后台配置
"""

from django.contrib import admin
from devices.models import Device, DeviceStats


@admin.register(Device)
class DeviceAdmin(admin.ModelAdmin):
    list_display = ['device_id', 'name', 'device_type', 'status', 'ip_address', 'last_seen', 'created_at']
    list_filter = ['device_type', 'status', 'is_trusted']
    search_fields = ['device_id', 'name', 'ip_address']
    ordering = ['-updated_at']


@admin.register(DeviceStats)
class DeviceStatsAdmin(admin.ModelAdmin):
    list_display = ['device', 'date', 'total_requests', 'normal_count', 'anomaly_count']
    list_filter = ['date']
    ordering = ['-date']
