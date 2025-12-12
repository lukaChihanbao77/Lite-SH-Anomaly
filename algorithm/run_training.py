"""
模型训练与评估运行脚本
功能：训练所有模型、评估性能、生成对比报告
"""

import sys
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from algorithm.training.isolation_forest import LightweightIsolationForest
from algorithm.training.autoencoder import LightweightAutoEncoder
from algorithm.training.baseline_models import BaselineKNN, BaselineSVM
from algorithm.evaluation.evaluator import ModelEvaluator
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data():
    """加载预处理后的数据"""
    data_path = project_root / 'data' / 'processed' / 'train_test_data.npz'
    
    if not data_path.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_path}\n请先运行数据预处理流水线")
        
    data = np.load(data_path)
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    
    logger.info(f"数据加载完成:")
    logger.info(f"  训练集: {X_train.shape}")
    logger.info(f"  测试集: {X_test.shape}")
    
    # 加载特征名称
    feature_file = project_root / 'data' / 'processed' / 'feature_columns.txt'
    if feature_file.exists():
        with open(feature_file, 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
    else:
        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
        
    return X_train, X_test, y_train, y_test, feature_names


def train_isolation_forest(X_train, y_train, feature_names):
    """训练轻量化孤立森林"""
    logger.info("\n" + "=" * 60)
    logger.info("训练轻量化孤立森林")
    logger.info("=" * 60)
    
    # 孤立森林是无监督算法，仅使用正常样本训练效果更好
    normal_mask = y_train == 0
    X_train_normal = X_train[normal_mask]
    
    logger.info(f"使用正常样本训练: {len(X_train_normal)} 条")
    
    model = LightweightIsolationForest(
        n_estimators=100,      # 增加树数量提升性能
        max_depth=10,          # 适当增加深度
        max_samples=512,       # 增加采样量
        contamination=0.2
    )
    
    model.train(X_train_normal, feature_names)
    return model


def train_autoencoder(X_train, y_train, feature_names):
    """训练轻量化自编码器"""
    logger.info("\n" + "=" * 60)
    logger.info("训练轻量化自编码器")
    logger.info("=" * 60)
    
    # 仅使用正常样本训练自编码器
    normal_mask = y_train == 0
    X_train_normal = X_train[normal_mask]
    
    logger.info(f"使用正常样本训练: {len(X_train_normal)} 条")
    
    model = LightweightAutoEncoder(
        input_dim=X_train.shape[1],
        hidden_dim=4,
        latent_dim=2,
        epochs=50,
        batch_size=64
    )
    
    model.train(X_train_normal, feature_names=feature_names)
    return model


def train_baseline_models(X_train, y_train):
    """训练基准模型"""
    logger.info("\n" + "=" * 60)
    logger.info("训练基准模型")
    logger.info("=" * 60)
    
    # KNN
    knn = BaselineKNN(n_neighbors=5)
    knn.train(X_train, y_train)
    
    # SVM
    svm = BaselineSVM(kernel='rbf', C=1.0)
    svm.train(X_train, y_train)
    
    return knn, svm


def save_models(models: dict, output_path: Path):
    """保存所有模型"""
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n保存模型...")
    
    for name, model in models.items():
        if name == 'LightweightIsolationForest':
            filepath = output_path / 'isolation_forest.joblib'
            size = model.save_model(str(filepath))
        elif name == 'LightweightAutoEncoder':
            filepath = output_path / 'autoencoder.pt'
            size = model.save_model(str(filepath))
        elif name == 'KNN':
            filepath = output_path / 'knn.joblib'
            size = model.save_model(str(filepath))
        elif name == 'SVM':
            filepath = output_path / 'svm.joblib'
            size = model.save_model(str(filepath))
            
        logger.info(f"  {name}: {filepath} ({size:.2f}MB)")


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("智能家居异常检测 - 模型训练与评估")
    logger.info("=" * 60)
    
    # 1. 加载数据
    X_train, X_test, y_train, y_test, feature_names = load_data()
    
    # 2. 训练模型
    if_model = train_isolation_forest(X_train, y_train, feature_names)
    ae_model = train_autoencoder(X_train, y_train, feature_names)
    knn_model, svm_model = train_baseline_models(X_train, y_train)
    
    # 3. 模型评估与对比
    logger.info("\n" + "=" * 60)
    logger.info("模型评估与对比")
    logger.info("=" * 60)
    
    models = {
        'LightweightIsolationForest': if_model,
        'LightweightAutoEncoder': ae_model,
        'KNN': knn_model,
        'SVM': svm_model
    }
    
    evaluator = ModelEvaluator(output_path=str(project_root / 'algorithm' / 'evaluation'))
    comparison_df = evaluator.compare_models(models, X_test, y_test)
    
    # 4. 检查是否满足要求
    logger.info("\n" + "=" * 60)
    logger.info("指标要求检查")
    logger.info("=" * 60)
    
    check_results = evaluator.check_requirements()
    
    # 5. 保存模型
    save_models(models, project_root / 'algorithm' / 'models')
    
    # 6. 保存评估报告
    evaluator.save_report('evaluation_report.json')
    evaluator.save_comparison_csv(comparison_df, 'model_comparison.csv')
    
    # 7. 输出总结
    logger.info("\n" + "=" * 60)
    logger.info("训练完成总结")
    logger.info("=" * 60)
    
    logger.info("\n模型性能排名（按F1分数）:")
    for i, row in comparison_df.iterrows():
        logger.info(f"  {row['模型']}: F1={row['F1分数']:.4f}, 推理={row['推理时间(ms)']:.2f}ms")
        
    # 找出最佳轻量化模型
    lightweight_models = ['LightweightIsolationForest', 'LightweightAutoEncoder']
    best_lightweight = comparison_df[comparison_df['模型'].isin(lightweight_models)].iloc[0]
    
    logger.info(f"\n推荐部署模型: {best_lightweight['模型']}")
    logger.info(f"  F1分数: {best_lightweight['F1分数']:.4f}")
    logger.info(f"  推理时间: {best_lightweight['推理时间(ms)']:.2f}ms")
    logger.info(f"  内存占用: {best_lightweight['内存占用(MB)']:.2f}MB")
    
    return models, evaluator


if __name__ == '__main__':
    models, evaluator = main()
