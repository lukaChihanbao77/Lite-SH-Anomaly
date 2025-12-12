"""
设备模型层
定义智能家居设备相关数据模型
"""

from django.db import models


class Device(models.Model):
    """智能家居设备模型"""
    
    class DeviceType(models.TextChoices):
        CAMERA = 'camera', '智能摄像头'
        DOOR_LOCK = 'door_lock', '智能门锁'
        SENSOR = 'sensor', '传感器'
        GATEWAY = 'gateway', '网关'
        OTHER = 'other', '其他'
    
    class Status(models.TextChoices):
        ONLINE = 'online', '在线'
        OFFLINE = 'offline', '离线'
        WARNING = 'warning', '告警'
    
    device_id = models.CharField('设备ID', max_length=64, unique=True, db_index=True)
    name = models.CharField('设备名称', max_length=128)
    device_type = models.CharField('设备类型', max_length=20, choices=DeviceType.choices, default=DeviceType.OTHER)
    ip_address = models.GenericIPAddressField('IP地址', null=True, blank=True)
    mac_address = models.CharField('MAC地址', max_length=17, null=True, blank=True)
    status = models.CharField('状态', max_length=20, choices=Status.choices, default=Status.OFFLINE)
    location = models.CharField('位置', max_length=128, null=True, blank=True)
    is_trusted = models.BooleanField('是否可信', default=True)
    last_seen = models.DateTimeField('最后在线时间', null=True, blank=True)
    created_at = models.DateTimeField('创建时间', auto_now_add=True)
    updated_at = models.DateTimeField('更新时间', auto_now=True)
    
    class Meta:
        db_table = 'devices'
        verbose_name = '设备'
        verbose_name_plural = '设备'
        ordering = ['-updated_at']
    
    def __str__(self):
        return f'{self.name} ({self.device_id})'


class DeviceStats(models.Model):
    """设备统计信息（每日汇总）"""
    
    device = models.ForeignKey(Device, on_delete=models.CASCADE, related_name='stats')
    date = models.DateField('统计日期', db_index=True)
    total_requests = models.IntegerField('总请求数', default=0)
    normal_count = models.IntegerField('正常请求数', default=0)
    anomaly_count = models.IntegerField('异常请求数', default=0)
    
    class Meta:
        db_table = 'device_stats'
        verbose_name = '设备统计'
        verbose_name_plural = '设备统计'
        unique_together = ['device', 'date']
        ordering = ['-date']
