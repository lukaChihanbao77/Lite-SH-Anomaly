"""
检测序列化器
"""

from rest_framework import serializers
from detection.models import DetectionRecord, DetectionTask


class DetectionRecordSerializer(serializers.ModelSerializer):
    """检测记录序列化器"""
    
    attack_type_display = serializers.CharField(source='get_attack_type_display', read_only=True)
    
    class Meta:
        model = DetectionRecord
        fields = [
            'id', 'device_id', 'timestamp', 'src_ip', 'dst_ip',
            'src_port', 'dst_port', 'protocol', 'duration',
            'orig_bytes', 'resp_bytes', 'orig_pkts', 'resp_pkts',
            'is_anomaly', 'attack_type', 'attack_type_display',
            'confidence', 'anomaly_score', 'model_version',
            'inference_time', 'created_at'
        ]


class DetectionInputSerializer(serializers.Serializer):
    """检测输入序列化器"""
    
    device_id = serializers.CharField(max_length=64)
    timestamp = serializers.DateTimeField(required=False)
    src_ip = serializers.IPAddressField()
    dst_ip = serializers.IPAddressField()
    src_port = serializers.IntegerField(required=False, min_value=0, max_value=65535)
    dst_port = serializers.IntegerField(required=False, min_value=0, max_value=65535)
    protocol = serializers.CharField(max_length=10, default='TCP')
    duration = serializers.FloatField(default=0)
    orig_bytes = serializers.IntegerField(default=0)
    resp_bytes = serializers.IntegerField(default=0)
    orig_pkts = serializers.IntegerField(default=0)
    resp_pkts = serializers.IntegerField(default=0)


class DetectionBatchInputSerializer(serializers.Serializer):
    """批量检测输入序列化器"""
    
    data = DetectionInputSerializer(many=True)


class DetectionResultSerializer(serializers.Serializer):
    """检测结果序列化器"""
    
    record_id = serializers.IntegerField()
    is_anomaly = serializers.BooleanField()
    attack_type = serializers.CharField()
    confidence = serializers.FloatField()
    anomaly_score = serializers.FloatField()
    inference_time = serializers.FloatField()
    model_version = serializers.CharField()


class DetectionTaskSerializer(serializers.ModelSerializer):
    """检测任务序列化器"""
    
    status_display = serializers.CharField(source='get_status_display', read_only=True)
    progress = serializers.SerializerMethodField()
    
    class Meta:
        model = DetectionTask
        fields = [
            'task_id', 'file_name', 'status', 'status_display',
            'total_count', 'processed_count', 'anomaly_count',
            'progress', 'error_message', 'started_at', 
            'completed_at', 'created_at'
        ]
    
    def get_progress(self, obj):
        if obj.total_count > 0:
            return round(obj.processed_count / obj.total_count * 100, 2)
        return 0


class DetectionStatsSerializer(serializers.Serializer):
    """检测统计序列化器"""
    
    total = serializers.IntegerField()
    anomaly = serializers.IntegerField()
    normal = serializers.IntegerField()
    anomaly_rate = serializers.FloatField()
    attack_distribution = serializers.ListField()
    daily_trend = serializers.ListField()
    avg_inference_time = serializers.FloatField()
