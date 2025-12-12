#!/usr/bin/env python
"""
初始化测试数据脚本
用于开发和测试环境
"""

import os
import sys
import random
from datetime import datetime, timedelta

# 设置Django环境
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')

import django
django.setup()

from django.utils import timezone
from devices.models import Device, DeviceStats
from detection.models import DetectionRecord
from logs.models import AlertLog


def create_test_devices():
    """创建测试设备"""
    devices_data = [
        {'device_id': 'CAM-001', 'name': '客厅摄像头', 'device_type': 'camera', 'ip_address': '192.168.1.101'},
        {'device_id': 'CAM-002', 'name': '门口摄像头', 'device_type': 'camera', 'ip_address': '192.168.1.102'},
        {'device_id': 'LOCK-001', 'name': '大门智能锁', 'device_type': 'door_lock', 'ip_address': '192.168.1.103'},
        {'device_id': 'LOCK-002', 'name': '后门智能锁', 'device_type': 'door_lock', 'ip_address': '192.168.1.104'},
        {'device_id': 'SENSOR-001', 'name': '温湿度传感器', 'device_type': 'sensor', 'ip_address': '192.168.1.105'},
        {'device_id': 'SENSOR-002', 'name': '烟雾传感器', 'device_type': 'sensor', 'ip_address': '192.168.1.106'},
        {'device_id': 'GW-001', 'name': '主网关', 'device_type': 'gateway', 'ip_address': '192.168.1.1'},
    ]
    
    created_count = 0
    for data in devices_data:
        device, created = Device.objects.get_or_create(
            device_id=data['device_id'],
            defaults={
                'name': data['name'],
                'device_type': data['device_type'],
                'ip_address': data['ip_address'],
                'status': random.choice(['online', 'online', 'online', 'offline']),
                'is_trusted': True,
                'last_seen': timezone.now() - timedelta(minutes=random.randint(0, 60))
            }
        )
        if created:
            created_count += 1
            print(f'创建设备: {device.name}')
    
    print(f'设备创建完成，新增 {created_count} 个')
    return Device.objects.all()


def create_test_detection_records(devices, count=100):
    """创建测试检测记录"""
    attack_types = ['normal', 'normal', 'normal', 'normal', 'ddos', 'port_scan', 'unauthorized', 'malformed']
    protocols = ['TCP', 'UDP', 'ICMP']
    
    records = []
    for i in range(count):
        device = random.choice(devices)
        is_anomaly = random.random() < 0.2  # 20%异常
        attack_type = 'normal' if not is_anomaly else random.choice(['ddos', 'port_scan', 'unauthorized', 'malformed'])
        
        record = DetectionRecord(
            device_id=device.device_id,
            timestamp=timezone.now() - timedelta(hours=random.randint(0, 168)),
            src_ip=f'192.168.1.{random.randint(100, 200)}',
            dst_ip=device.ip_address or '192.168.1.1',
            src_port=random.randint(1024, 65535),
            dst_port=random.choice([80, 443, 8080, 22, 3306]),
            protocol=random.choice(protocols),
            duration=random.uniform(0.001, 10),
            orig_bytes=random.randint(64, 10000),
            resp_bytes=random.randint(64, 50000),
            orig_pkts=random.randint(1, 100),
            resp_pkts=random.randint(1, 100),
            is_anomaly=is_anomaly,
            attack_type=attack_type,
            confidence=random.uniform(0.7, 0.99) if is_anomaly else random.uniform(0.9, 0.99),
            anomaly_score=random.uniform(0.3, 0.8) if is_anomaly else random.uniform(-0.5, 0.2),
            inference_time=random.uniform(1, 10)
        )
        records.append(record)
    
    DetectionRecord.objects.bulk_create(records)
    print(f'检测记录创建完成，共 {count} 条')


def create_test_alerts(count=20):
    """创建测试告警"""
    devices = Device.objects.all()
    attack_types = ['ddos', 'port_scan', 'unauthorized', 'malformed']
    levels = ['warning', 'danger', 'critical']
    
    for i in range(count):
        device = random.choice(devices)
        attack_type = random.choice(attack_types)
        level = random.choice(levels)
        
        AlertLog.objects.create(
            device_id=device.device_id,
            level=level,
            status=random.choice(['pending', 'pending', 'confirmed', 'resolved']),
            attack_type=attack_type,
            title=f'检测到{attack_type}攻击',
            description=f'设备 {device.name} 检测到疑似{attack_type}攻击行为',
            confidence=random.uniform(0.7, 0.99),
            src_ip=f'192.168.1.{random.randint(100, 200)}',
            dst_ip=device.ip_address,
            alert_time=timezone.now() - timedelta(hours=random.randint(0, 72))
        )
    
    print(f'告警创建完成，共 {count} 条')


def main():
    print('=' * 50)
    print('初始化测试数据')
    print('=' * 50)
    
    devices = create_test_devices()
    create_test_detection_records(list(devices), count=100)
    create_test_alerts(count=20)
    
    print('=' * 50)
    print('测试数据初始化完成！')
    print('=' * 50)


if __name__ == '__main__':
    main()
