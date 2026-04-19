"""
数据工具函数
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, List
import json


def split_train_val_test(
    file_list: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    划分训练/验证/测试集

    Args:
        file_list: 文件路径列表
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子

    Returns:
        (train_files, val_files, test_files)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"

    np.random.seed(seed)
    indices = np.random.permutation(len(file_list))

    n_train = int(len(file_list) * train_ratio)
    n_val = int(len(file_list) * val_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    train_files = [file_list[i] for i in train_indices]
    val_files = [file_list[i] for i in val_indices]
    test_files = [file_list[i] for i in test_indices]

    return train_files, val_files, test_files


def compute_dataset_statistics(features_list: List[np.ndarray]) -> Dict:
    """
    计算数据集统计信息

    Args:
        features_list: 特征数组列表

    Returns:
        统计信息字典
    """
    lengths = [len(f) for f in features_list]
    all_features = np.concatenate(features_list, axis=0)

    stats = {
        'num_samples': len(features_list),
        'total_points': len(all_features),
        'length_mean': np.mean(lengths),
        'length_std': np.std(lengths),
        'length_min': np.min(lengths),
        'length_max': np.max(lengths),
        'length_median': np.median(lengths),
        'feature_mean': all_features.mean(axis=0).tolist(),
        'feature_std': all_features.std(axis=0).tolist(),
        'feature_min': all_features.min(axis=0).tolist(),
        'feature_max': all_features.max(axis=0).tolist(),
    }

    return stats


def visualize_signature(
    features: np.ndarray,
    save_path: str = None,
    show: bool = True
):
    """
    可视化签名

    Args:
        features: 形状为 (N, 23) 的特征数组
        save_path: 保存路径
        show: 是否显示
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 轨迹图
    x_norm = features[:, 2]
    y_norm = features[:, 3]
    axes[0, 0].plot(x_norm, y_norm, 'b-', linewidth=1)
    axes[0, 0].scatter(x_norm[0], y_norm[0], c='g', s=100, marker='o', label='Start')
    axes[0, 0].scatter(x_norm[-1], y_norm[-1], c='r', s=100, marker='x', label='End')
    axes[0, 0].set_xlabel('X (normalized)')
    axes[0, 0].set_ylabel('Y (normalized)')
    axes[0, 0].set_title('Signature Trajectory')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_aspect('equal')

    # 2. 压力曲线
    p_norm = features[:, 4]
    axes[0, 1].plot(p_norm, 'r-', linewidth=1)
    axes[0, 1].set_xlabel('Point Index')
    axes[0, 1].set_ylabel('Pressure (normalized)')
    axes[0, 1].set_title('Pressure Profile')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. 速度曲线
    v_abs = features[:, 11]
    axes[1, 0].plot(v_abs, 'g-', linewidth=1)
    axes[1, 0].set_xlabel('Point Index')
    axes[1, 0].set_ylabel('Velocity (absolute)')
    axes[1, 0].set_title('Velocity Profile')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 笔画标记
    stroke_mark = features[:, 0]
    axes[1, 1].plot(stroke_mark, 'k-', linewidth=2)
    axes[1, 1].set_xlabel('Point Index')
    axes[1, 1].set_ylabel('Stroke Mark')
    axes[1, 1].set_title('Stroke Mark (1=pen down, 0=pen up)')
    axes[1, 1].set_ylim([-0.1, 1.1])
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close()


def save_file_list(file_list: List[str], output_path: str):
    """
    保存文件列表

    Args:
        file_list: 文件路径列表
        output_path: 输出路径
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for filepath in file_list:
            f.write(f"{filepath}\n")


def load_file_list(input_path: str) -> List[str]:
    """
    加载文件列表

    Args:
        input_path: 输入路径

    Returns:
        文件路径列表
    """
    with open(input_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def save_statistics(stats: Dict, output_path: str):
    """
    保存统计信息

    Args:
        stats: 统计信息字典
        output_path: 输出路径
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
