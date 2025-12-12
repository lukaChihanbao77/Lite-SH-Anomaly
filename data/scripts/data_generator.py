"""
数据模拟生成模块
功能：生成模拟的智能家居网络流量数据（正常流量 + 攻击流量）
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SmartHomeDataGenerator:
    """智能家居数据生成器"""
    
    # 模拟设备配置
    DEVICES = {
        'camera_01': {'type': 'smart_camera', 'ip': '192.168.1.101'},
        'camera_02': {'type': 'smart_camera', 'ip': '192.168.1.102'},
        'doorlock_01': {'type': 'smart_doorlock', 'ip': '192.168.1.103'},
        'doorlock_02': {'type': 'smart_doorlock', 'ip': '192.168.1.104'},
        'thermostat_01': {'type': 'smart_thermostat', 'ip': '192.168.1.105'},
        'light_01': {'type': 'smart_light', 'ip': '192.168.1.106'},
        'light_02': {'type': 'smart_light', 'ip': '192.168.1.107'},
        'speaker_01': {'type': 'smart_speaker', 'ip': '192.168.1.108'},
    }
    
    # 网关配置
    GATEWAY_IP = '192.168.1.1'
    
    # 外部IP池（用于模拟攻击）
    EXTERNAL_IPS = [
        '203.0.113.1', '203.0.113.2', '198.51.100.1', '198.51.100.2',
        '192.0.2.1', '192.0.2.2', '45.33.32.156', '104.16.123.96'
    ]
    
    # 协议类型
    PROTOCOLS = ['tcp', 'udp', 'icmp']
    
    # 服务类型
    SERVICES = ['http', 'https', 'mqtt', 'dns', 'ntp', 'ssh', '-']
    
    # 连接状态
    CONN_STATES = ['SF', 'S0', 'REJ', 'RSTO', 'RSTOS0', 'SH', 'SHR', 'OTH']
    
    def __init__(self, output_path: str = None):
        """
        初始化生成器
        
        Args:
            output_path: 输出路径
        """
        self.output_path = Path(output_path) if output_path else Path(__file__).parent.parent / 'raw'
        self.output_path.mkdir(parents=True, exist_ok=True)
        
    def _generate_timestamp(self, base_time: datetime, offset_seconds: float) -> str:
        """生成时间戳"""
        return (base_time + timedelta(seconds=offset_seconds)).strftime('%Y-%m-%d %H:%M:%S')
    
    def _generate_normal_traffic(self, num_samples: int, base_time: datetime) -> list:
        """
        生成正常流量数据
        
        Args:
            num_samples: 样本数量
            base_time: 基准时间
            
        Returns:
            正常流量数据列表
        """
        records = []
        
        for i in range(num_samples):
            device_id = random.choice(list(self.DEVICES.keys()))
            device = self.DEVICES[device_id]
            
            record = {
                'timestamp': self._generate_timestamp(base_time, i * random.randint(1, 10)),
                'device_id': device_id,
                'device_type': device['type'],
                'src_ip': device['ip'],
                'dst_ip': self.GATEWAY_IP,
                'proto': random.choice(['tcp', 'udp']),
                'service': random.choice(['mqtt', 'https', 'http']),
                'conn_state': 'SF',
                'duration': round(random.uniform(0.1, 30.0), 3),
                'orig_bytes': random.randint(64, 2048),
                'resp_bytes': random.randint(64, 4096),
                'orig_pkts': random.randint(1, 20),
                'resp_pkts': random.randint(1, 25),
                'orig_ip_bytes': random.randint(100, 2500),
                'resp_ip_bytes': random.randint(100, 5000),
                'label': 'benign'
            }
            records.append(record)
            
        return records
    
    def _generate_ddos_attack(self, num_samples: int, base_time: datetime) -> list:
        """
        生成DDoS攻击流量
        
        特征：高频率请求、短连接、大量小包
        """
        records = []
        target_device = random.choice(list(self.DEVICES.keys()))
        target_ip = self.DEVICES[target_device]['ip']
        
        for i in range(num_samples):
            record = {
                'timestamp': self._generate_timestamp(base_time, i * 0.1),
                'device_id': target_device,
                'device_type': self.DEVICES[target_device]['type'],
                'src_ip': random.choice(self.EXTERNAL_IPS),
                'dst_ip': target_ip,
                'proto': random.choice(['tcp', 'udp']),
                'service': '-',
                'conn_state': random.choice(['S0', 'REJ', 'RSTOS0']),
                'duration': round(random.uniform(0.0, 0.5), 3),
                'orig_bytes': random.randint(40, 200),
                'resp_bytes': random.randint(0, 100),
                'orig_pkts': random.randint(1, 5),
                'resp_pkts': random.randint(0, 2),
                'orig_ip_bytes': random.randint(60, 300),
                'resp_ip_bytes': random.randint(0, 150),
                'label': 'ddos'
            }
            records.append(record)
            
        return records
    
    def _generate_port_scan(self, num_samples: int, base_time: datetime) -> list:
        """
        生成端口扫描流量
        
        特征：探测多个端口、短连接、无响应或被拒绝
        """
        records = []
        attacker_ip = random.choice(self.EXTERNAL_IPS)
        
        for i in range(num_samples):
            target_device = random.choice(list(self.DEVICES.keys()))
            
            record = {
                'timestamp': self._generate_timestamp(base_time, i * random.randint(1, 3)),
                'device_id': target_device,
                'device_type': self.DEVICES[target_device]['type'],
                'src_ip': attacker_ip,
                'dst_ip': self.DEVICES[target_device]['ip'],
                'proto': 'tcp',
                'service': '-',
                'conn_state': random.choice(['S0', 'REJ', 'RSTO']),
                'duration': round(random.uniform(0.0, 0.1), 3),
                'orig_bytes': random.randint(40, 80),
                'resp_bytes': random.randint(0, 60),
                'orig_pkts': random.randint(1, 3),
                'resp_pkts': random.randint(0, 2),
                'orig_ip_bytes': random.randint(60, 120),
                'resp_ip_bytes': random.randint(0, 100),
                'label': 'scan'
            }
            records.append(record)
            
        return records
    
    def _generate_botnet_traffic(self, num_samples: int, base_time: datetime) -> list:
        """
        生成僵尸网络流量（如Mirai）
        
        特征：设备向外部C2服务器通信、周期性心跳、异常端口
        """
        records = []
        infected_device = random.choice(list(self.DEVICES.keys()))
        c2_server = random.choice(self.EXTERNAL_IPS)
        
        for i in range(num_samples):
            record = {
                'timestamp': self._generate_timestamp(base_time, i * random.randint(30, 120)),
                'device_id': infected_device,
                'device_type': self.DEVICES[infected_device]['type'],
                'src_ip': self.DEVICES[infected_device]['ip'],
                'dst_ip': c2_server,
                'proto': 'tcp',
                'service': '-',
                'conn_state': 'SF',
                'duration': round(random.uniform(0.5, 5.0), 3),
                'orig_bytes': random.randint(100, 500),
                'resp_bytes': random.randint(50, 300),
                'orig_pkts': random.randint(3, 10),
                'resp_pkts': random.randint(2, 8),
                'orig_ip_bytes': random.randint(150, 600),
                'resp_ip_bytes': random.randint(100, 400),
                'label': 'mirai'
            }
            records.append(record)
            
        return records
    
    def _generate_unauthorized_access(self, num_samples: int, base_time: datetime) -> list:
        """
        生成越权访问流量
        
        特征：尝试访问敏感设备（门锁、摄像头）、异常时间、多次失败尝试
        """
        records = []
        attacker_ip = random.choice(self.EXTERNAL_IPS)
        sensitive_devices = [d for d, info in self.DEVICES.items() 
                           if info['type'] in ['smart_doorlock', 'smart_camera']]
        
        for i in range(num_samples):
            target_device = random.choice(sensitive_devices)
            
            record = {
                'timestamp': self._generate_timestamp(base_time, i * random.randint(5, 30)),
                'device_id': target_device,
                'device_type': self.DEVICES[target_device]['type'],
                'src_ip': attacker_ip,
                'dst_ip': self.DEVICES[target_device]['ip'],
                'proto': 'tcp',
                'service': random.choice(['ssh', 'http', 'https']),
                'conn_state': random.choice(['SF', 'REJ', 'RSTO']),
                'duration': round(random.uniform(1.0, 60.0), 3),
                'orig_bytes': random.randint(200, 2000),
                'resp_bytes': random.randint(100, 1000),
                'orig_pkts': random.randint(5, 30),
                'resp_pkts': random.randint(3, 20),
                'orig_ip_bytes': random.randint(300, 2500),
                'resp_ip_bytes': random.randint(150, 1500),
                'label': 'unauthorized'
            }
            records.append(record)
            
        return records
    
    def generate_dataset(self, 
                        total_samples: int = 10000,
                        normal_ratio: float = 0.8,
                        attack_distribution: dict = None) -> pd.DataFrame:
        """
        生成完整数据集
        
        Args:
            total_samples: 总样本数
            normal_ratio: 正常样本比例
            attack_distribution: 攻击类型分布
            
        Returns:
            生成的DataFrame
        """
        logger.info(f"开始生成数据集，总样本数: {total_samples}")
        
        base_time = datetime.now() - timedelta(days=7)
        
        normal_samples = int(total_samples * normal_ratio)
        attack_samples = total_samples - normal_samples
        
        if attack_distribution is None:
            attack_distribution = {
                'ddos': 0.35,
                'scan': 0.25,
                'botnet': 0.20,
                'unauthorized': 0.20
            }
        
        all_records = []
        
        # 生成正常流量
        logger.info(f"生成正常流量: {normal_samples} 条")
        all_records.extend(self._generate_normal_traffic(normal_samples, base_time))
        
        # 生成各类攻击流量
        for attack_type, ratio in attack_distribution.items():
            num = int(attack_samples * ratio)
            logger.info(f"生成 {attack_type} 攻击流量: {num} 条")
            
            if attack_type == 'ddos':
                all_records.extend(self._generate_ddos_attack(num, base_time))
            elif attack_type == 'scan':
                all_records.extend(self._generate_port_scan(num, base_time))
            elif attack_type == 'botnet':
                all_records.extend(self._generate_botnet_traffic(num, base_time))
            elif attack_type == 'unauthorized':
                all_records.extend(self._generate_unauthorized_access(num, base_time))
        
        # 转换为DataFrame并打乱顺序
        df = pd.DataFrame(all_records)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        logger.info(f"数据集生成完成，共 {len(df)} 条记录")
        logger.info(f"标签分布:\n{df['label'].value_counts()}")
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, filename: str = 'simulated_iot_data.csv'):
        """保存数据集"""
        output_file = self.output_path / filename
        df.to_csv(output_file, index=False)
        logger.info(f"数据集已保存至: {output_file}")
        return output_file
    
    def generate_and_save(self, 
                         total_samples: int = 10000,
                         filename: str = 'simulated_iot_data.csv') -> str:
        """生成并保存数据集"""
        df = self.generate_dataset(total_samples)
        return self.save_dataset(df, filename)


if __name__ == '__main__':
    # 生成模拟数据集
    generator = SmartHomeDataGenerator()
    output_path = generator.generate_and_save(total_samples=10000)
    print(f"数据集已生成: {output_path}")
