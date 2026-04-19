"""
评估指标计算模块

实现签名验证的核心评估指标：
- EER (Equal Error Rate): 等错误率
- Accuracy: 准确率
- FAR/FRR: 错误接受率/错误拒绝率
- ROC/DET曲线绘制
"""

import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix


def calculate_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """
    计算等错误率 (Equal Error Rate)

    Args:
        y_true: 真实标签 (0=伪签名, 1=真签名)
        y_scores: 预测分数 (0-1之间的概率值)

    Returns:
        eer: 等错误率
        eer_threshold: EER对应的阈值
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr

    # 找到FAR=FRR的点
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[eer_idx]
    eer_threshold = thresholds[eer_idx]

    return float(eer), float(eer_threshold)


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    计算准确率

    Args:
        y_true: 真实标签
        y_pred: 预测标签

    Returns:
        accuracy: 准确率 (0-1)
    """
    return float(np.mean(y_true == y_pred))


def calculate_far_frr(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float
) -> Tuple[float, float]:
    """
    计算FAR和FRR

    Args:
        y_true: 真实标签 (0=伪签名, 1=真签名)
        y_scores: 预测分数
        threshold: 判定阈值

    Returns:
        far: False Acceptance Rate (错误接受率)
        frr: False Rejection Rate (错误拒绝率)
    """
    y_pred = (y_scores >= threshold).astype(int)

    # FAR: 伪签名被错误接受的比例
    negatives = y_true == 0
    if negatives.sum() > 0:
        far = float((y_pred[negatives] == 1).sum() / negatives.sum())
    else:
        far = 0.0

    # FRR: 真签名被错误拒绝的比例
    positives = y_true == 1
    if positives.sum() > 0:
        frr = float((y_pred[positives] == 0).sum() / positives.sum())
    else:
        frr = 0.0

    return far, frr


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True
) -> float:
    """
    绘制ROC曲线

    Args:
        y_true: 真实标签
        y_scores: 预测分数
        save_path: 保存路径
        show: 是否显示图像

    Returns:
        auc_score: ROC曲线下面积
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FAR)')
    plt.ylabel('True Positive Rate (1 - FRR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()

    return float(roc_auc)


def plot_det_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    绘制DET曲线 (Detection Error Tradeoff)

    Args:
        y_true: 真实标签
        y_scores: 预测分数
        save_path: 保存路径
        show: 是否显示图像
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr

    plt.figure(figsize=(8, 6))
    plt.plot(fpr * 100, fnr * 100, color='blue', lw=2)
    plt.xlabel('False Acceptance Rate (%)')
    plt.ylabel('False Rejection Rate (%)')
    plt.title('Detection Error Tradeoff (DET) Curve')
    plt.grid(alpha=0.3)
    plt.axis('equal')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def calculate_metrics_at_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float
) -> dict:
    """
    计算给定阈值下的所有指标

    Args:
        y_true: 真实标签
        y_scores: 预测分数
        threshold: 判定阈值

    Returns:
        metrics: 包含所有指标的字典
    """
    y_pred = (y_scores >= threshold).astype(int)

    accuracy = calculate_accuracy(y_true, y_pred)
    far, frr = calculate_far_frr(y_true, y_scores, threshold)

    # 混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # 精确率和召回率
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'accuracy': float(accuracy),
        'far': float(far),
        'frr': float(frr),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'threshold': float(threshold)
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    metric: str = 'eer'
) -> float:
    """
    寻找最优阈值

    Args:
        y_true: 真实标签
        y_scores: 预测分数
        metric: 优化目标 ('eer', 'accuracy', 'f1')

    Returns:
        optimal_threshold: 最优阈值
    """
    if metric == 'eer':
        _, threshold = calculate_eer(y_true, y_scores)
        return threshold

    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)

    if metric == 'accuracy':
        # 最大化准确率
        accuracies = []
        for thresh in thresholds:
            y_pred = (y_scores >= thresh).astype(int)
            acc = calculate_accuracy(y_true, y_pred)
            accuracies.append(acc)
        best_idx = np.argmax(accuracies)
        return float(thresholds[best_idx])

    elif metric == 'f1':
        # 最大化F1分数
        f1_scores = []
        for thresh in thresholds:
            metrics = calculate_metrics_at_threshold(y_true, y_scores, thresh)
            f1_scores.append(metrics['f1_score'])
        best_idx = np.argmax(f1_scores)
        return float(thresholds[best_idx])

    else:
        raise ValueError(f"Unknown metric: {metric}")
