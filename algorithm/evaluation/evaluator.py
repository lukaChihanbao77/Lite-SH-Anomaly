"""
模型评估模块
功能：统一评估所有模型、生成对比报告、性能基准测试
"""

import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, output_path: str = None):
        """
        初始化评估器
        
        Args:
            output_path: 评估报告输出路径
        """
        self.output_path = Path(output_path) if output_path else Path(__file__).parent.parent / 'evaluation'
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.results = {}
        
    def evaluate_model(self, 
                      model, 
                      model_name: str,
                      X_test: np.ndarray, 
                      y_test: np.ndarray,
                      run_benchmark: bool = True) -> dict:
        """
        评估单个模型
        
        Args:
            model: 模型实例
            model_name: 模型名称
            X_test: 测试数据
            y_test: 测试标签
            run_benchmark: 是否运行性能基准测试
            
        Returns:
            评估结果字典
        """
        logger.info(f"=" * 50)
        logger.info(f"评估模型: {model_name}")
        logger.info(f"=" * 50)
        
        result = {
            'model_name': model_name,
            'metrics': {},
            'benchmark': {},
            'model_size_mb': None
        }
        
        # 性能指标评估
        metrics = model.evaluate(X_test, y_test)
        result['metrics'] = metrics
        
        # 性能基准测试
        if run_benchmark:
            benchmark = model.benchmark_inference(X_test)
            result['benchmark'] = benchmark
            
        self.results[model_name] = result
        return result
    
    def compare_models(self, models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        对比多个模型
        
        Args:
            models: 模型字典 {名称: 模型实例}
            X_test: 测试数据
            y_test: 测试标签
            
        Returns:
            对比结果DataFrame
        """
        logger.info("开始模型对比评估...")
        
        for name, model in models.items():
            self.evaluate_model(model, name, X_test, y_test)
            
        # 生成对比表格
        comparison_data = []
        for name, result in self.results.items():
            row = {
                '模型': name,
                '准确率': result['metrics'].get('accuracy', 0),
                '精确率': result['metrics'].get('precision', 0),
                '召回率': result['metrics'].get('recall', 0),
                'F1分数': result['metrics'].get('f1_score', 0),
                '误报率': result['metrics'].get('false_positive_rate', 0),
                '漏报率': result['metrics'].get('false_negative_rate', 0),
                '推理时间(ms)': result['benchmark'].get('avg_inference_time_ms', 0),
                '内存占用(MB)': result['benchmark'].get('memory_usage_mb', 0)
            }
            comparison_data.append(row)
            
        df = pd.DataFrame(comparison_data)
        
        # 按F1分数排序
        df = df.sort_values('F1分数', ascending=False)
        
        logger.info("\n模型对比结果:")
        logger.info(f"\n{df.to_string(index=False)}")
        
        return df
    
    def check_requirements(self, model_name: str = None) -> dict:
        """
        检查是否满足项目要求
        
        要求：
        - F1分数 ≥ 0.85
        - 误报率 ≤ 5%
        - 漏报率 ≤ 3%
        - 推理延迟 ≤ 100ms
        - 内存占用 ≤ 30MB
        """
        requirements = {
            'f1_score': {'min': 0.85, 'unit': ''},
            'false_positive_rate': {'max': 0.05, 'unit': ''},
            'false_negative_rate': {'max': 0.03, 'unit': ''},
            'avg_inference_time_ms': {'max': 100, 'unit': 'ms'},
            'memory_usage_mb': {'max': 30, 'unit': 'MB'}
        }
        
        results_to_check = {model_name: self.results[model_name]} if model_name else self.results
        
        check_results = {}
        for name, result in results_to_check.items():
            checks = {}
            
            # F1分数
            f1 = result['metrics'].get('f1_score', 0)
            checks['F1分数 ≥ 0.85'] = {
                'value': f1,
                'passed': f1 >= 0.85,
                'target': '≥ 0.85'
            }
            
            # 误报率
            fpr = result['metrics'].get('false_positive_rate', 1)
            checks['误报率 ≤ 5%'] = {
                'value': fpr,
                'passed': fpr <= 0.05,
                'target': '≤ 5%'
            }
            
            # 漏报率
            fnr = result['metrics'].get('false_negative_rate', 1)
            checks['漏报率 ≤ 3%'] = {
                'value': fnr,
                'passed': fnr <= 0.03,
                'target': '≤ 3%'
            }
            
            # 推理延迟
            latency = result['benchmark'].get('avg_inference_time_ms', float('inf'))
            checks['推理延迟 ≤ 100ms'] = {
                'value': latency,
                'passed': latency <= 100,
                'target': '≤ 100ms'
            }
            
            # 内存占用
            memory = result['benchmark'].get('memory_usage_mb', float('inf'))
            checks['内存占用 ≤ 30MB'] = {
                'value': memory,
                'passed': memory <= 30,
                'target': '≤ 30MB'
            }
            
            all_passed = all(c['passed'] for c in checks.values())
            check_results[name] = {
                'checks': checks,
                'all_passed': all_passed
            }
            
            logger.info(f"\n{name} 指标检查:")
            for check_name, check_info in checks.items():
                status = "✓" if check_info['passed'] else "✗"
                logger.info(f"  {status} {check_name}: {check_info['value']:.4f}")
            logger.info(f"  总体: {'全部通过' if all_passed else '未全部通过'}")
            
        return check_results
    
    def save_report(self, filename: str = 'evaluation_report.json'):
        """保存评估报告"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': {}
        }
        
        for name, result in self.results.items():
            # 转换numpy类型为Python原生类型
            report['results'][name] = {
                'metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                           for k, v in result['metrics'].items()},
                'benchmark': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                             for k, v in result['benchmark'].items()}
            }
            
        output_file = self.output_path / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        logger.info(f"评估报告已保存: {output_file}")
        
    def save_comparison_csv(self, df: pd.DataFrame, filename: str = 'model_comparison.csv'):
        """保存对比结果CSV"""
        output_file = self.output_path / filename
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        logger.info(f"对比结果已保存: {output_file}")


if __name__ == '__main__':
    print("模型评估模块已就绪")
