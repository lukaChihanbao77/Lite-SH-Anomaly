"""
特征分析模块
功能：特征相关性分析、特征重要性评估、特征可视化
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureAnalyzer:
    """特征分析器"""
    
    def __init__(self, df: pd.DataFrame = None):
        """
        初始化分析器
        
        Args:
            df: 数据集DataFrame
        """
        self.df = df
        self.feature_importance = None
        self.correlation_matrix = None
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """加载数据"""
        self.df = pd.read_csv(file_path)
        logger.info(f"数据加载完成，共 {len(self.df)} 条记录，{len(self.df.columns)} 个特征")
        return self.df
    
    def calculate_correlation(self, numeric_features: list = None) -> pd.DataFrame:
        """
        计算特征相关性矩阵
        
        Args:
            numeric_features: 数值型特征列表
            
        Returns:
            相关性矩阵
        """
        if self.df is None:
            raise ValueError("请先加载数据")
            
        if numeric_features is None:
            numeric_features = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
        self.correlation_matrix = self.df[numeric_features].corr()
        logger.info(f"相关性矩阵计算完成，特征数: {len(numeric_features)}")
        
        return self.correlation_matrix
    
    def get_high_correlation_pairs(self, threshold: float = 0.8) -> list:
        """
        获取高相关性特征对
        
        Args:
            threshold: 相关性阈值
            
        Returns:
            高相关性特征对列表
        """
        if self.correlation_matrix is None:
            self.calculate_correlation()
            
        high_corr_pairs = []
        cols = self.correlation_matrix.columns
        
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                corr_value = abs(self.correlation_matrix.iloc[i, j])
                if corr_value >= threshold:
                    high_corr_pairs.append({
                        'feature_1': cols[i],
                        'feature_2': cols[j],
                        'correlation': round(corr_value, 4)
                    })
                    
        logger.info(f"发现 {len(high_corr_pairs)} 对高相关性特征（阈值: {threshold}）")
        return high_corr_pairs
    
    def calculate_feature_importance(self, 
                                    feature_columns: list,
                                    label_column: str = 'label_encoded',
                                    method: str = 'random_forest') -> pd.DataFrame:
        """
        计算特征重要性
        
        Args:
            feature_columns: 特征列
            label_column: 标签列
            method: 计算方法 ('random_forest', 'mutual_info', 'f_classif')
            
        Returns:
            特征重要性DataFrame
        """
        if self.df is None:
            raise ValueError("请先加载数据")
            
        X = self.df[feature_columns].values
        y = self.df[label_column].values
        
        if method == 'random_forest':
            rf = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            importance = rf.feature_importances_
            
        elif method == 'mutual_info':
            importance = mutual_info_classif(X, y, random_state=42)
            
        elif method == 'f_classif':
            selector = SelectKBest(f_classif, k='all')
            selector.fit(X, y)
            importance = selector.scores_
            importance = importance / importance.max()  # 归一化
            
        else:
            raise ValueError(f"不支持的方法: {method}")
            
        self.feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        logger.info(f"特征重要性计算完成（方法: {method}）")
        logger.info(f"Top 5 特征:\n{self.feature_importance.head()}")
        
        return self.feature_importance
    
    def select_top_features(self, n_features: int = 6, min_importance: float = 0.1) -> list:
        """
        选择Top特征
        
        Args:
            n_features: 选择特征数量
            min_importance: 最小重要性阈值
            
        Returns:
            选中的特征列表
        """
        if self.feature_importance is None:
            raise ValueError("请先计算特征重要性")
            
        # 筛选重要性大于阈值的特征
        filtered = self.feature_importance[self.feature_importance['importance'] >= min_importance]
        
        # 取前n个
        selected = filtered.head(n_features)['feature'].tolist()
        
        logger.info(f"选择 {len(selected)} 个特征: {selected}")
        return selected
    
    def remove_redundant_features(self, 
                                  feature_columns: list,
                                  correlation_threshold: float = 0.9) -> list:
        """
        移除冗余特征（高相关性）
        
        Args:
            feature_columns: 特征列
            correlation_threshold: 相关性阈值
            
        Returns:
            去除冗余后的特征列表
        """
        if self.correlation_matrix is None:
            self.calculate_correlation(feature_columns)
            
        # 找出需要移除的特征
        features_to_remove = set()
        
        for i in range(len(feature_columns)):
            if feature_columns[i] in features_to_remove:
                continue
            for j in range(i + 1, len(feature_columns)):
                if feature_columns[j] in features_to_remove:
                    continue
                if abs(self.correlation_matrix.loc[feature_columns[i], feature_columns[j]]) > correlation_threshold:
                    # 移除重要性较低的特征
                    if self.feature_importance is not None:
                        imp_i = self.feature_importance[self.feature_importance['feature'] == feature_columns[i]]['importance'].values
                        imp_j = self.feature_importance[self.feature_importance['feature'] == feature_columns[j]]['importance'].values
                        if len(imp_i) > 0 and len(imp_j) > 0:
                            if imp_i[0] < imp_j[0]:
                                features_to_remove.add(feature_columns[i])
                            else:
                                features_to_remove.add(feature_columns[j])
                    else:
                        features_to_remove.add(feature_columns[j])
                        
        selected_features = [f for f in feature_columns if f not in features_to_remove]
        logger.info(f"移除 {len(features_to_remove)} 个冗余特征，保留 {len(selected_features)} 个")
        
        return selected_features
    
    def get_feature_statistics(self, feature_columns: list = None) -> pd.DataFrame:
        """
        获取特征统计信息
        
        Args:
            feature_columns: 特征列
            
        Returns:
            统计信息DataFrame
        """
        if self.df is None:
            raise ValueError("请先加载数据")
            
        if feature_columns is None:
            feature_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
            
        stats = self.df[feature_columns].describe().T
        stats['missing'] = self.df[feature_columns].isnull().sum()
        stats['missing_ratio'] = stats['missing'] / len(self.df)
        
        return stats
    
    def analyze_by_label(self, feature_columns: list, label_column: str = 'label') -> dict:
        """
        按标签分组分析特征
        
        Args:
            feature_columns: 特征列
            label_column: 标签列
            
        Returns:
            各标签的特征统计
        """
        if self.df is None:
            raise ValueError("请先加载数据")
            
        result = {}
        for label in self.df[label_column].unique():
            subset = self.df[self.df[label_column] == label]
            result[label] = subset[feature_columns].describe().T
            
        return result
    
    def save_analysis_report(self, output_path: str = None):
        """保存分析报告"""
        if output_path is None:
            output_path = Path(__file__).parent.parent / 'processed'
            
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.feature_importance is not None:
            self.feature_importance.to_csv(output_path / 'feature_importance.csv', index=False)
            
        if self.correlation_matrix is not None:
            self.correlation_matrix.to_csv(output_path / 'correlation_matrix.csv')
            
        logger.info(f"分析报告已保存至: {output_path}")


if __name__ == '__main__':
    print("特征分析模块已就绪")
    print("使用方法:")
    print("  analyzer = FeatureAnalyzer()")
    print("  analyzer.load_data('processed_data.csv')")
    print("  importance = analyzer.calculate_feature_importance(features, 'label_encoded')")
    print("  selected = analyzer.select_top_features(n_features=6)")
