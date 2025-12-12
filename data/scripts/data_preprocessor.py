"""
数据预处理模块
功能：数据清洗、特征工程、数据标准化
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """数据预处理器"""
    
    # 核心特征列（基于IoT-23数据集）
    CORE_FEATURES = [
        'duration',           # 连接持续时间
        'orig_bytes',         # 源端发送字节数
        'resp_bytes',         # 目的端响应字节数
        'orig_pkts',          # 源端发送包数
        'resp_pkts',          # 目的端响应包数
        'orig_ip_bytes',      # 源端IP层字节数
        'resp_ip_bytes',      # 目的端IP层字节数
    ]
    
    # 分类特征
    CATEGORICAL_FEATURES = [
        'proto',              # 协议类型 (TCP/UDP/ICMP)
        'service',            # 服务类型
        'conn_state',         # 连接状态
    ]
    
    # 标签列
    LABEL_COLUMN = 'label'
    
    # 攻击类型映射
    ATTACK_TYPES = {
        'benign': 0,          # 正常流量
        'ddos': 1,            # DDoS攻击
        'dos': 1,             # DoS攻击
        'scan': 2,            # 端口扫描
        'mirai': 3,           # Mirai僵尸网络
        'okiru': 3,           # Okiru僵尸网络
        'torii': 3,           # Torii僵尸网络
        'unauthorized': 4,    # 越权访问
    }
    
    def __init__(self, raw_data_path: str = None, output_path: str = None):
        """
        初始化预处理器
        
        Args:
            raw_data_path: 原始数据路径
            output_path: 输出路径
        """
        self.raw_data_path = Path(raw_data_path) if raw_data_path else None
        self.output_path = Path(output_path) if output_path else Path(__file__).parent.parent / 'processed'
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.df = None
        
    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """
        加载原始数据
        
        Args:
            file_path: 数据文件路径（支持CSV格式）
            
        Returns:
            加载的DataFrame
        """
        path = Path(file_path) if file_path else self.raw_data_path
        
        if path is None:
            raise ValueError("请提供数据文件路径")
            
        logger.info(f"正在加载数据: {path}")
        
        if path.suffix == '.csv':
            self.df = pd.read_csv(path, low_memory=False)
        else:
            raise ValueError(f"不支持的文件格式: {path.suffix}")
            
        logger.info(f"数据加载完成，共 {len(self.df)} 条记录")
        return self.df
    
    def clean_data(self) -> pd.DataFrame:
        """
        数据清洗
        - 处理缺失值
        - 去除重复值
        - 处理异常值
        
        Returns:
            清洗后的DataFrame
        """
        if self.df is None:
            raise ValueError("请先加载数据")
            
        logger.info("开始数据清洗...")
        original_count = len(self.df)
        
        # 1. 去除完全重复的行
        self.df = self.df.drop_duplicates()
        logger.info(f"去除重复行: {original_count - len(self.df)} 条")
        
        # 2. 处理缺失值
        # 数值型特征用中位数填充
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].isnull().sum() > 0:
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
                
        # 分类特征用众数填充
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.df[col].isnull().sum() > 0:
                mode_val = self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else 'unknown'
                self.df[col].fillna(mode_val, inplace=True)
        
        # 3. 处理异常值（使用IQR方法处理数值型特征的极端异常值）
        for col in numeric_cols:
            if col in self.CORE_FEATURES:
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                # 将极端异常值裁剪到边界
                self.df[col] = self.df[col].clip(lower=max(0, lower_bound), upper=upper_bound)
        
        # 4. 处理无穷值
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.dropna(inplace=True)
        
        logger.info(f"数据清洗完成，剩余 {len(self.df)} 条记录")
        return self.df
    
    def encode_categorical(self) -> pd.DataFrame:
        """
        编码分类特征
        
        Returns:
            编码后的DataFrame
        """
        if self.df is None:
            raise ValueError("请先加载数据")
            
        logger.info("开始编码分类特征...")
        
        for col in self.CATEGORICAL_FEATURES:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[col + '_encoded'] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
                logger.info(f"编码特征 {col}: {len(le.classes_)} 个类别")
                
        return self.df
    
    def extract_features(self) -> pd.DataFrame:
        """
        特征工程：提取和构造特征
        
        Returns:
            特征工程后的DataFrame
        """
        if self.df is None:
            raise ValueError("请先加载数据")
            
        logger.info("开始特征工程...")
        
        # 1. 构造派生特征
        if 'orig_bytes' in self.df.columns and 'resp_bytes' in self.df.columns:
            # 字节比率
            self.df['bytes_ratio'] = self.df['orig_bytes'] / (self.df['resp_bytes'] + 1)
            
        if 'orig_pkts' in self.df.columns and 'resp_pkts' in self.df.columns:
            # 包数比率
            self.df['pkts_ratio'] = self.df['orig_pkts'] / (self.df['resp_pkts'] + 1)
            
        if 'duration' in self.df.columns and 'orig_bytes' in self.df.columns:
            # 传输速率
            self.df['bytes_per_second'] = self.df['orig_bytes'] / (self.df['duration'] + 0.001)
            
        if 'duration' in self.df.columns and 'orig_pkts' in self.df.columns:
            # 包速率
            self.df['pkts_per_second'] = self.df['orig_pkts'] / (self.df['duration'] + 0.001)
        
        logger.info(f"特征工程完成，当前特征数: {len(self.df.columns)}")
        return self.df
    
    def normalize_labels(self) -> pd.DataFrame:
        """
        标准化标签（将多种攻击类型映射为统一标签）
        
        Returns:
            标签标准化后的DataFrame
        """
        if self.df is None:
            raise ValueError("请先加载数据")
            
        if self.LABEL_COLUMN not in self.df.columns:
            logger.warning(f"未找到标签列 '{self.LABEL_COLUMN}'")
            return self.df
            
        logger.info("开始标准化标签...")
        
        def map_label(label):
            label_lower = str(label).lower()
            for attack_type, code in self.ATTACK_TYPES.items():
                if attack_type in label_lower:
                    return code
            return 0  # 默认为正常
            
        self.df['label_encoded'] = self.df[self.LABEL_COLUMN].apply(map_label)
        
        # 统计各类别数量
        label_counts = self.df['label_encoded'].value_counts()
        logger.info(f"标签分布:\n{label_counts}")
        
        return self.df
    
    def scale_features(self, feature_columns: list = None) -> np.ndarray:
        """
        特征标准化
        
        Args:
            feature_columns: 需要标准化的特征列
            
        Returns:
            标准化后的特征数组
        """
        if self.df is None:
            raise ValueError("请先加载数据")
            
        if feature_columns is None:
            # 使用核心特征 + 编码后的分类特征 + 派生特征
            feature_columns = [col for col in self.CORE_FEATURES if col in self.df.columns]
            feature_columns += [col + '_encoded' for col in self.CATEGORICAL_FEATURES 
                              if col + '_encoded' in self.df.columns]
            feature_columns += ['bytes_ratio', 'pkts_ratio', 'bytes_per_second', 'pkts_per_second']
            feature_columns = [col for col in feature_columns if col in self.df.columns]
            
        logger.info(f"标准化特征: {feature_columns}")
        
        X = self.df[feature_columns].values
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, feature_columns
    
    def prepare_dataset(self, test_size: float = 0.2, random_state: int = 42) -> tuple:
        """
        准备训练和测试数据集
        
        Args:
            test_size: 测试集比例
            random_state: 随机种子
            
        Returns:
            (X_train, X_test, y_train, y_test, feature_columns)
        """
        logger.info("准备数据集...")
        
        X_scaled, feature_columns = self.scale_features()
        
        if 'label_encoded' in self.df.columns:
            y = self.df['label_encoded'].values
        else:
            raise ValueError("请先执行标签标准化")
            
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"训练集: {len(X_train)} 条，测试集: {len(X_test)} 条")
        
        return X_train, X_test, y_train, y_test, feature_columns
    
    def save_processed_data(self, filename: str = 'processed_data.csv'):
        """
        保存预处理后的数据
        
        Args:
            filename: 输出文件名
        """
        if self.df is None:
            raise ValueError("请先加载并处理数据")
            
        output_file = self.output_path / filename
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.df.to_csv(output_file, index=False)
        logger.info(f"数据已保存至: {output_file}")
        
    def run_pipeline(self, input_file: str, output_file: str = 'processed_data.csv') -> tuple:
        """
        运行完整预处理流水线
        
        Args:
            input_file: 输入文件路径
            output_file: 输出文件名
            
        Returns:
            (X_train, X_test, y_train, y_test, feature_columns)
        """
        logger.info("=" * 50)
        logger.info("开始数据预处理流水线")
        logger.info("=" * 50)
        
        # 1. 加载数据
        self.load_data(input_file)
        
        # 2. 数据清洗
        self.clean_data()
        
        # 3. 编码分类特征
        self.encode_categorical()
        
        # 4. 特征工程
        self.extract_features()
        
        # 5. 标签标准化
        self.normalize_labels()
        
        # 6. 保存处理后的数据
        self.save_processed_data(output_file)
        
        # 7. 准备数据集
        result = self.prepare_dataset()
        
        logger.info("=" * 50)
        logger.info("数据预处理流水线完成")
        logger.info("=" * 50)
        
        return result


if __name__ == '__main__':
    # 示例用法
    preprocessor = DataPreprocessor()
    
    # 如果有原始数据文件，运行预处理流水线
    # result = preprocessor.run_pipeline('path/to/raw_data.csv')
    
    print("数据预处理模块已就绪")
    print("使用方法:")
    print("  preprocessor = DataPreprocessor()")
    print("  X_train, X_test, y_train, y_test, features = preprocessor.run_pipeline('data.csv')")
