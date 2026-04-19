"""
评估器模块

实现模型评估功能：
- 评估循环
- 指标计算（准确率、EER、FAR、FRR）
- 结果保存和可视化
"""

import os
import json
from typing import Optional, Dict, Any, List
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm

from utils.metrics import (
    calculate_eer,
    calculate_accuracy,
    calculate_far_frr,
    calculate_metrics_at_threshold,
    plot_roc_curve,
    plot_det_curve
)
from utils.logger import get_logger


class Evaluator:
    """评估器类"""

    def __init__(
        self,
        model: tf.keras.Model,
        test_dataset: tf.data.Dataset,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            model: 待评估的模型
            test_dataset: 测试数据集
            config: 配置字典
        """
        self.model = model
        self.test_dataset = test_dataset
        self.config = config or {}
        self.logger = get_logger(__name__)

    def evaluate(
        self,
        steps: Optional[int] = None,
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        执行评估

        Args:
            steps: 评估步数
            save_dir: 结果保存目录

        Returns:
            results: 评估结果字典
        """
        self.logger.info("Starting evaluation")

        # 收集预测结果
        y_true_list = []
        y_scores_list = []

        dataset = self.test_dataset
        if steps is not None:
            dataset = dataset.take(steps)

        progress_bar = tqdm(dataset, desc="Evaluating")

        for batch in progress_bar:
            sig1, sig2, labels = batch

            # 预测
            predictions = self.model((sig1, sig2), training=False)

            # 收集结果
            y_true_list.append(labels.numpy())
            y_scores_list.append(predictions.numpy().flatten())

        # 合并结果
        y_true = np.concatenate(y_true_list)
        y_scores = np.concatenate(y_scores_list)

        # 计算EER和最优阈值
        eer, eer_threshold = calculate_eer(y_true, y_scores)
        self.logger.info(f"EER: {eer:.4f} at threshold {eer_threshold:.4f}")

        # 计算EER阈值下的指标
        eer_metrics = calculate_metrics_at_threshold(y_true, y_scores, eer_threshold)

        # 计算0.5阈值下的指标
        default_metrics = calculate_metrics_at_threshold(y_true, y_scores, 0.5)

        # 汇总结果
        results = {
            'eer': eer,
            'eer_threshold': eer_threshold,
            'eer_metrics': eer_metrics,
            'default_metrics': default_metrics,
            'num_samples': len(y_true),
            'num_positives': int(y_true.sum()),
            'num_negatives': int((1 - y_true).sum())
        }

        # 打印结果
        self._print_results(results)

        # 保存结果
        if save_dir is not None:
            self._save_results(results, y_true, y_scores, save_dir)

        return results

    def _print_results(self, results: Dict[str, Any]) -> None:
        """打印评估结果"""
        print("\n" + "="*60)
        print("Evaluation Results")
        print("="*60)

        print(f"\nDataset Statistics:")
        print(f"  Total samples: {results['num_samples']}")
        print(f"  Positive samples (genuine): {results['num_positives']}")
        print(f"  Negative samples (forgery): {results['num_negatives']}")

        print(f"\nEER Metrics (threshold={results['eer_threshold']:.4f}):")
        print(f"  EER: {results['eer']:.4f}")
        eer_m = results['eer_metrics']
        print(f"  Accuracy: {eer_m['accuracy']:.4f}")
        print(f"  FAR: {eer_m['far']:.4f}")
        print(f"  FRR: {eer_m['frr']:.4f}")
        print(f"  Precision: {eer_m['precision']:.4f}")
        print(f"  Recall: {eer_m['recall']:.4f}")
        print(f"  F1-Score: {eer_m['f1_score']:.4f}")

        print(f"\nDefault Threshold Metrics (threshold=0.5):")
        def_m = results['default_metrics']
        print(f"  Accuracy: {def_m['accuracy']:.4f}")
        print(f"  FAR: {def_m['far']:.4f}")
        print(f"  FRR: {def_m['frr']:.4f}")
        print(f"  Precision: {def_m['precision']:.4f}")
        print(f"  Recall: {def_m['recall']:.4f}")
        print(f"  F1-Score: {def_m['f1_score']:.4f}")

        print("\n" + "="*60)

    def _save_results(
        self,
        results: Dict[str, Any],
        y_true: np.ndarray,
        y_scores: np.ndarray,
        save_dir: str
    ) -> None:
        """保存评估结果"""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # 保存JSON结果
        json_path = save_path / 'evaluation_results.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Results saved to {json_path}")

        # 保存预测结果
        predictions_path = save_path / 'predictions.npz'
        np.savez(
            predictions_path,
            y_true=y_true,
            y_scores=y_scores
        )
        self.logger.info(f"Predictions saved to {predictions_path}")

        # 绘制ROC曲线
        roc_path = save_path / 'roc_curve.png'
        plot_roc_curve(y_true, y_scores, save_path=str(roc_path), show=False)
        self.logger.info(f"ROC curve saved to {roc_path}")

        # 绘制DET曲线
        det_path = save_path / 'det_curve.png'
        plot_det_curve(y_true, y_scores, save_path=str(det_path), show=False)
        self.logger.info(f"DET curve saved to {det_path}")

    def evaluate_per_user(
        self,
        user_ids: Optional[List[str]] = None,
        save_dir: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        按用户评估

        Args:
            user_ids: 用户ID列表
            save_dir: 结果保存目录

        Returns:
            per_user_results: 每个用户的评估结果
        """
        self.logger.info("Starting per-user evaluation")

        # 这里需要数据集提供用户ID信息
        # 简化实现，假设数据集已经按用户分组
        # 实际使用时需要根据数据集结构调整

        per_user_results = {}

        # TODO: 实现按用户评估逻辑
        # 需要数据集支持按用户迭代

        self.logger.warning("Per-user evaluation not fully implemented")

        return per_user_results

    def compute_confusion_matrix(
        self,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        计算混淆矩阵

        Args:
            threshold: 判定阈值

        Returns:
            confusion_matrix: 2x2混淆矩阵
        """
        y_true_list = []
        y_scores_list = []

        for batch in self.test_dataset:
            sig1, sig2, labels = batch
            predictions = self.model((sig1, sig2), training=False)

            y_true_list.append(labels.numpy())
            y_scores_list.append(predictions.numpy().flatten())

        y_true = np.concatenate(y_true_list)
        y_scores = np.concatenate(y_scores_list)
        y_pred = (y_scores >= threshold).astype(int)

        # 计算混淆矩阵
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)

        return cm

    @tf.function
    def _predict_batch(self, sig1, sig2):
        """批量预测"""
        return self.model((sig1, sig2), training=False)
