"""
数据工具函数
"""
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import json

from data.pair_sampler import group_by_user


def split_train_val_test(
    file_list: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    train_users: Optional[List[int]] = None,
    val_users: Optional[List[int]] = None,
    test_users: Optional[List[int]] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Writer-independent 划分：按 user_id 分组，同一 user 的全部文件落入同一 split。

    SVC2004 Task 2 共 40 user，推荐默认划分（28/6/6 = 70%/15%/15%）：
        train: user 1-28   val: user 29-34   test: user 35-40

    这样可保证训练/验证/测试 user 集合严格不相交，不会出现 writer 泄漏。

    Args:
        file_list: 文件路径列表（应来自 SVC2004 目录，命名形如 UxSy.TXT）
        train_ratio, val_ratio, test_ratio: 仅当未指定 *_users 时使用；按 user 数量 round
        seed: 随机种子（仅当未指定 *_users 且比例需要随机分配多余 user 时使用）
        train_users/val_users/test_users: 显式指定 user_id 列表（优先级最高）

    Returns:
        (train_files, val_files, test_files)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"

    user_groups = group_by_user(file_list)
    all_user_ids = sorted(user_groups.keys())
    n_users = len(all_user_ids)

    # 显式指定 user 划分（优先）
    if train_users is not None or val_users is not None or test_users is not None:
        train_u = set(train_users or [])
        val_u = set(val_users or [])
        test_u = set(test_users or [])
        overlap = (train_u & val_u) | (train_u & test_u) | (val_u & test_u)
        if overlap:
            raise ValueError(f"train/val/test user 列表有重叠: {sorted(overlap)}")
    else:
        # 按比例划分 user_id（固定 seed 下可复现）
        import random as _random
        _rng = _random.Random(seed)
        shuffled = list(all_user_ids)
        _rng.shuffle(shuffled)

        n_train = int(round(n_users * train_ratio))
        n_val = int(round(n_users * val_ratio))
        # 剩余全部给 test，保证三者之和等于 n_users
        n_train = max(1, n_train)
        n_val = max(1, n_val)
        if n_train + n_val >= n_users:
            n_val = max(1, n_users - n_train - 1)

        train_u = set(shuffled[:n_train])
        val_u = set(shuffled[n_train:n_train + n_val])
        test_u = set(shuffled[n_train + n_val:])

    def _collect(users: set) -> List[str]:
        out = []
        for uid in sorted(users):
            out.extend(user_groups.get(uid, []))
        return out

    train_files = _collect(train_u)
    val_files = _collect(val_u)
    test_files = _collect(test_u)

    return train_files, val_files, test_files


def compute_dataset_statistics(features_list: List) -> Dict:
    """
    计算数据集统计信息

    Args:
        features_list: 特征数组列表

    Returns:
        统计信息字典
    """
    import numpy as np
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
    features,
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
    import matplotlib.pyplot as plt
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
