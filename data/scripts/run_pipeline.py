"""
数据处理流水线运行脚本
功能：一键执行数据生成、预处理、特征分析全流程
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data.scripts.data_generator import SmartHomeDataGenerator
from data.scripts.data_preprocessor import DataPreprocessor
from data.scripts.feature_analyzer import FeatureAnalyzer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_full_pipeline(total_samples: int = 10000, use_simulated: bool = True):
    """
    运行完整数据处理流水线
    
    Args:
        total_samples: 总样本数
        use_simulated: 是否使用模拟数据（True=生成模拟数据，False=使用已有数据）
    """
    logger.info("=" * 60)
    logger.info("开始运行数据处理流水线")
    logger.info("=" * 60)
    
    raw_data_path = Path(__file__).parent.parent / 'raw'
    processed_data_path = Path(__file__).parent.parent / 'processed'
    
    # Step 1: 生成或加载数据
    if use_simulated:
        logger.info("\n[Step 1/4] 生成模拟数据...")
        generator = SmartHomeDataGenerator(output_path=str(raw_data_path))
        raw_file = generator.generate_and_save(total_samples=total_samples)
    else:
        raw_file = raw_data_path / 'simulated_iot_data.csv'
        if not raw_file.exists():
            raise FileNotFoundError(f"数据文件不存在: {raw_file}")
        logger.info(f"\n[Step 1/4] 使用已有数据: {raw_file}")
    
    # Step 2: 数据预处理
    logger.info("\n[Step 2/4] 数据预处理...")
    preprocessor = DataPreprocessor(output_path=str(processed_data_path))
    preprocessor.load_data(str(raw_file))
    preprocessor.clean_data()
    preprocessor.encode_categorical()
    preprocessor.extract_features()
    preprocessor.normalize_labels()
    preprocessor.save_processed_data('processed_data.csv')
    
    # Step 3: 特征分析
    logger.info("\n[Step 3/4] 特征分析...")
    analyzer = FeatureAnalyzer(preprocessor.df)
    
    # 获取数值型特征
    numeric_features = [
        'duration', 'orig_bytes', 'resp_bytes', 'orig_pkts', 'resp_pkts',
        'orig_ip_bytes', 'resp_ip_bytes', 'bytes_ratio', 'pkts_ratio',
        'bytes_per_second', 'pkts_per_second'
    ]
    numeric_features = [f for f in numeric_features if f in preprocessor.df.columns]
    
    # 添加编码后的分类特征
    encoded_features = [col for col in preprocessor.df.columns if col.endswith('_encoded') and col != 'label_encoded']
    all_features = numeric_features + encoded_features
    
    # 计算特征重要性
    if 'label_encoded' in preprocessor.df.columns:
        importance = analyzer.calculate_feature_importance(all_features, 'label_encoded')
        
        # 选择Top特征
        selected_features = analyzer.select_top_features(n_features=8, min_importance=0.05)
        
        # 计算相关性
        analyzer.calculate_correlation(numeric_features)
        high_corr = analyzer.get_high_correlation_pairs(threshold=0.8)
        
        # 保存分析报告
        analyzer.save_analysis_report(str(processed_data_path))
    
    # Step 4: 准备训练数据集
    logger.info("\n[Step 4/4] 准备训练数据集...")
    X_train, X_test, y_train, y_test, feature_cols = preprocessor.prepare_dataset()
    
    # 保存数据集信息
    import numpy as np
    np.savez(
        processed_data_path / 'train_test_data.npz',
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )
    
    # 保存特征列表
    with open(processed_data_path / 'feature_columns.txt', 'w') as f:
        f.write('\n'.join(feature_cols))
    
    logger.info("=" * 60)
    logger.info("数据处理流水线完成！")
    logger.info("=" * 60)
    logger.info(f"\n输出文件:")
    logger.info(f"  - 原始数据: {raw_file}")
    logger.info(f"  - 处理后数据: {processed_data_path / 'processed_data.csv'}")
    logger.info(f"  - 训练测试数据: {processed_data_path / 'train_test_data.npz'}")
    logger.info(f"  - 特征重要性: {processed_data_path / 'feature_importance.csv'}")
    logger.info(f"  - 相关性矩阵: {processed_data_path / 'correlation_matrix.csv'}")
    logger.info(f"\n数据集统计:")
    logger.info(f"  - 训练集: {len(X_train)} 条")
    logger.info(f"  - 测试集: {len(X_test)} 条")
    logger.info(f"  - 特征数: {len(feature_cols)}")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'features': feature_cols
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='运行数据处理流水线')
    parser.add_argument('--samples', type=int, default=10000, help='样本数量')
    parser.add_argument('--no-simulate', action='store_true', help='使用已有数据而非生成模拟数据')
    
    args = parser.parse_args()
    
    run_full_pipeline(
        total_samples=args.samples,
        use_simulated=not args.no_simulate
    )
