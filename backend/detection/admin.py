"""
检测管理后台配置
"""

from django.contrib import admin
from detection.models import DetectionRecord, DetectionTask


@admin.register(DetectionRecord)
class DetectionRecordAdmin(admin.ModelAdmin):
    list_display = ['id', 'device_id', 'is_anomaly', 'attack_type', 'confidence', 'timestamp', 'inference_time']
    list_filter = ['is_anomaly', 'attack_type', 'protocol']
    search_fields = ['device_id', 'src_ip', 'dst_ip']
    ordering = ['-timestamp']
    date_hierarchy = 'timestamp'


@admin.register(DetectionTask)
class DetectionTaskAdmin(admin.ModelAdmin):
    list_display = ['task_id', 'file_name', 'status', 'total_count', 'processed_count', 'anomaly_count', 'created_at']
    list_filter = ['status']
    ordering = ['-created_at']
