"""
可视化工具模块

实现训练和评估过程的可视化功能：
- 训练曲线
- 注意力权重
- 签名对比
- 混淆矩阵
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Any
from pathlib import Path
import seaborn as sns


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    绘制训练曲线

    Args:
        history: 训练历史字典
        save_path: 保存路径
        show: 是否显示图像
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 损失曲线
    if 'train_loss' in history:
        axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history:
        axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 准确率曲线
    if 'train_accuracy' in history:
        axes[1].plot(history['train_accuracy'], label='Train Accuracy', linewidth=2)
    if 'val_accuracy' in history:
        axes[1].plot(history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_attention_weights(
    attention_weights: np.ndarray,
    signature_length: int,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    绘制注意力权重热图

    Args:
        attention_weights: 注意力权重 (seq_len,) 或 (batch, seq_len)
        signature_length: 签名长度
        save_path: 保存路径
        show: 是否显示图像
    """
    if attention_weights.ndim == 2:
        # 取第一个样本
        attention_weights = attention_weights[0]

    # 只显示有效长度
    attention_weights = attention_weights[:signature_length]

    plt.figure(figsize=(12, 3))
    plt.bar(range(len(attention_weights)), attention_weights, color='steelblue')
    plt.xlabel('Time Step')
    plt.ylabel('Attention Weight')
    plt.title('Attention Weights over Time')
    plt.grid(alpha=0.3, axis='y')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_signature_comparison(
    sig1: np.ndarray,
    sig2: np.ndarray,
    label: int,
    prediction: float,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    绘制签名对比图

    Args:
        sig1: 第一个签名 (seq_len, features)
        sig2: 第二个签名 (seq_len, features)
        label: 真实标签 (0=伪签名对, 1=真签名对)
        prediction: 预测分数
        save_path: 保存路径
        show: 是否显示图像
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 提取x, y坐标（假设在前两列）
    x1, y1 = sig1[:, 0], sig1[:, 1]
    x2, y2 = sig2[:, 0], sig2[:, 1]

    # 签名1轨迹
    axes[0, 0].plot(x1, y1, 'b-', linewidth=2)
    axes[0, 0].scatter(x1[0], y1[0], c='green', s=100, marker='o', label='Start')
    axes[0, 0].scatter(x1[-1], y1[-1], c='red', s=100, marker='x', label='End')
    axes[0, 0].set_title('Signature 1 Trajectory')
    axes[0, 0].set_xlabel('X')
    axes[0, 0].set_ylabel('Y')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].invert_yaxis()

    # 签名2轨迹
    axes[0, 1].plot(x2, y2, 'r-', linewidth=2)
    axes[0, 1].scatter(x2[0], y2[0], c='green', s=100, marker='o', label='Start')
    axes[0, 1].scatter(x2[-1], y2[-1], c='red', s=100, marker='x', label='End')
    axes[0, 1].set_title('Signature 2 Trajectory')
    axes[0, 1].set_xlabel('X')
    axes[0, 1].set_ylabel('Y')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].invert_yaxis()

    # 速度对比（假设速度在第3列）
    if sig1.shape[1] > 2:
        v1 = sig1[:, 2]
        v2 = sig2[:, 2]
        axes[1, 0].plot(v1, label='Signature 1', linewidth=2)
        axes[1, 0].plot(v2, label='Signature 2', linewidth=2)
        axes[1, 0].set_title('Velocity Comparison')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('Velocity')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

    # 预测结果
    label_text = 'Genuine Pair' if label == 1 else 'Forgery Pair'
    pred_text = 'Genuine' if prediction >= 0.5 else 'Forgery'
    correct = (label == 1 and prediction >= 0.5) or (label == 0 and prediction < 0.5)
    result_text = 'Correct' if correct else 'Incorrect'

    axes[1, 1].text(0.5, 0.7, f'Ground Truth: {label_text}',
                    ha='center', va='center', fontsize=14, weight='bold')
    axes[1, 1].text(0.5, 0.5, f'Prediction: {pred_text} ({prediction:.4f})',
                    ha='center', va='center', fontsize=14)
    axes[1, 1].text(0.5, 0.3, f'Result: {result_text}',
                    ha='center', va='center', fontsize=14,
                    color='green' if correct else 'red', weight='bold')
    axes[1, 1].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    绘制混淆矩阵

    Args:
        cm: 混淆矩阵
        class_names: 类别名称
        normalize: 是否归一化
        save_path: 保存路径
        show: 是否显示图像
    """
    if class_names is None:
        class_names = ['Forgery', 'Genuine']

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_feature_distribution(
    features: np.ndarray,
    feature_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    绘制特征分布

    Args:
        features: 特征矩阵 (n_samples, n_features)
        feature_names: 特征名称列表
        save_path: 保存路径
        show: 是否显示图像
    """
    n_features = features.shape[1]
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    axes = axes.flatten()

    for i in range(n_features):
        axes[i].hist(features[:, i], bins=50, alpha=0.7, edgecolor='black')
        if feature_names and i < len(feature_names):
            axes[i].set_title(feature_names[i])
        else:
            axes[i].set_title(f'Feature {i}')
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(alpha=0.3)

    # 隐藏多余的子图
    for i in range(n_features, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


def plot_score_distribution(
    genuine_scores: np.ndarray,
    forgery_scores: np.ndarray,
    threshold: Optional[float] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    绘制真伪签名分数分布

    Args:
        genuine_scores: 真签名分数
        forgery_scores: 伪签名分数
        threshold: 判定阈值
        save_path: 保存路径
        show: 是否显示图像
    """
    plt.figure(figsize=(10, 6))

    plt.hist(forgery_scores, bins=50, alpha=0.6, label='Forgery', color='red', edgecolor='black')
    plt.hist(genuine_scores, bins=50, alpha=0.6, label='Genuine', color='green', edgecolor='black')

    if threshold is not None:
        plt.axvline(threshold, color='blue', linestyle='--', linewidth=2, label=f'Threshold={threshold:.3f}')

    plt.xlabel('Prediction Score')
    plt.ylabel('Frequency')
    plt.title('Score Distribution: Genuine vs Forgery')
    plt.legend()
    plt.grid(alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()
