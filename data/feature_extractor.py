"""
特征提取模块
从原始签名数据提取23维时序特征
"""
import numpy as np
import warnings
from typing import Tuple, Optional


def mark_stroke(pressure: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """
    根据压力值标记笔画

    Args:
        pressure: 压力序列
        threshold: 笔抬起/按下的判断阈值

    Returns:
        笔画标记序列，1表示笔按下，0表示笔抬起
    """
    return (pressure >= threshold).astype(np.float32)


def normalize_coordinates(
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    time: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    归一化坐标、压力和时间到[0, 1]范围

    Args:
        x: X坐标序列
        y: Y坐标序列
        p: 压力序列
        time: 时间戳序列（可选）

    Returns:
        归一化后的 (x_norm, y_norm, p_norm, time_norm)
    """
    def normalize(arr):
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max != arr_min:
            return (arr - arr_min) / (arr_max - arr_min)
        return np.zeros_like(arr)

    x_norm = normalize(x).astype(np.float32)
    y_norm = normalize(y).astype(np.float32)
    p_norm = normalize(p).astype(np.float32)

    time_norm = None
    if time is not None:
        time_norm = normalize(time).astype(np.float32)

    return x_norm, y_norm, p_norm, time_norm


def compute_velocity(arr: np.ndarray) -> np.ndarray:
    """
    计算速度（一阶差分）

    Args:
        arr: 输入序列

    Returns:
        速度序列
    """
    return np.diff(arr, prepend=arr[0]).astype(np.float32)


def compute_acceleration(velocity: np.ndarray) -> np.ndarray:
    """
    计算加速度（速度的一阶差分）

    Args:
        velocity: 速度序列

    Returns:
        加速度序列
    """
    return np.diff(velocity, prepend=velocity[0]).astype(np.float32)


def compute_curvature(v_x: np.ndarray, v_y: np.ndarray, a_x: np.ndarray, a_y: np.ndarray) -> np.ndarray:
    """
    计算曲率

    Args:
        v_x: X方向速度
        v_y: Y方向速度
        a_x: X方向加速度
        a_y: Y方向加速度

    Returns:
        曲率序列
    """
    v_abs = np.sqrt(v_x ** 2 + v_y ** 2)
    curvature = np.divide(
        a_x * v_y - a_y * v_x,
        v_abs ** 3 + 1e-8,
        out=np.zeros_like(v_abs),
        where=(v_abs ** 3 + 1e-8) != 0
    )
    return curvature.astype(np.float32)


def compute_angle(v_x: np.ndarray, v_y: np.ndarray) -> np.ndarray:
    """
    计算角度

    Args:
        v_x: X方向速度
        v_y: Y方向速度

    Returns:
        角度序列（弧度）
    """
    return np.arctan2(v_y, v_x).astype(np.float32)


def compute_arc_length(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算弧长和累积弧长

    Args:
        x: X坐标序列
        y: Y坐标序列

    Returns:
        (累积弧长, 归一化累积弧长)
    """
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    arc_lengths = np.sqrt(dx ** 2 + dy ** 2)
    cumulative_arc_length = np.cumsum(arc_lengths)

    # 归一化累积弧长
    if cumulative_arc_length.max() != cumulative_arc_length.min():
        arc_length_norm = (cumulative_arc_length - cumulative_arc_length.min()) / \
                         (cumulative_arc_length.max() - cumulative_arc_length.min())
    else:
        arc_length_norm = np.zeros_like(cumulative_arc_length)

    return cumulative_arc_length.astype(np.float32), arc_length_norm.astype(np.float32)


def extract_temporal_features(
    x: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
    time: Optional[np.ndarray] = None,
    threshold: float = 0.01
) -> np.ndarray:
    """
    提取23维时序特征

    Args:
        x: X坐标序列
        y: Y坐标序列
        p: 压力序列
        time: 时间戳序列（可选）
        threshold: 笔画标记阈值

    Returns:
        形状为 (N, 23) 的特征数组，其中 N 为序列长度
        特征顺序：
        0. stroke_mark - 笔画标记
        1. time_norm - 归一化时间
        2. x_norm - 归一化X坐标
        3. y_norm - 归一化Y坐标
        4. p_norm - 归一化压力
        5. v_x - X方向速度
        6. v_y - Y方向速度
        7. v_p - 压力变化率
        8. a_x - X方向加速度
        9. a_y - Y方向加速度
        10. a_p - 压力加速度
        11. v_abs - 笔尖速度（绝对值）
        12. a_abs - 笔尖加速度（绝对值）
        13. curvature - 曲率
        14. angle - 角度
        15. cumulative_arc_length - 累积弧长
        16. arc_length_norm - 归一化弧长
        17. v_t - 时间速度
        18. a_t - 时间加速度
        19-22. 零填充
    """
    # 转换为numpy数组
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    p = np.array(p, dtype=np.float32)

    if len(x) < 2:
        warnings.warn("Signature has less than 2 points")
        return np.zeros((len(x), 23), dtype=np.float32)

    # 1. 笔画标记
    stroke_mark = mark_stroke(p, threshold)

    # 2. 归一化坐标
    x_norm, y_norm, p_norm, time_norm = normalize_coordinates(x, y, p, time)

    # 3. 速度
    v_x = compute_velocity(x)
    v_y = compute_velocity(y)
    v_p = compute_velocity(p)
    v_t = compute_velocity(time_norm) if time_norm is not None else np.zeros_like(x)

    # 4. 加速度
    a_x = compute_acceleration(v_x)
    a_y = compute_acceleration(v_y)
    a_p = compute_acceleration(v_p)
    a_t = compute_acceleration(v_t)

    # 5. 笔尖速度和加速度
    v_abs = np.sqrt(v_x ** 2 + v_y ** 2).astype(np.float32)
    a_abs = np.sqrt(a_x ** 2 + a_y ** 2).astype(np.float32)

    # 6. 曲率
    curvature = compute_curvature(v_x, v_y, a_x, a_y)

    # 7. 角度
    angle = compute_angle(v_x, v_y)

    # 8. 弧长
    cumulative_arc_length, arc_length_norm = compute_arc_length(x, y)

    # 组装特征（19个核心特征 + 4个零填充 = 23维）
    features = [
        stroke_mark,           # 0
        time_norm if time_norm is not None else np.zeros_like(x),  # 1
        x_norm,                # 2
        y_norm,                # 3
        p_norm,                # 4
        v_x,                   # 5
        v_y,                   # 6
        v_p,                   # 7
        a_x,                   # 8
        a_y,                   # 9
        a_p,                   # 10
        v_abs,                 # 11
        a_abs,                 # 12
        curvature,             # 13
        angle,                 # 14
        cumulative_arc_length, # 15
        arc_length_norm,       # 16
        v_t,                   # 17
        a_t,                   # 18
    ]

    # 填充到23维
    while len(features) < 23:
        features.append(np.zeros_like(x, dtype=np.float32))

    # 转换为 (N, 23) 数组
    features_array = np.stack(features, axis=1)

    return features_array.astype(np.float32)


def load_signature_txt(filepath: str) -> np.ndarray:
    """
    从SVC2004的.txt文件加载签名数据

    Args:
        filepath: .txt文件路径

    Returns:
        形状为 (N, 4) 的数组，列为 [x, y, time, pressure]
    """
    points = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            try:
                # SVC2004格式: ID x y time pressure pen_status
                x, y, t, p = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                points.append([x, y, t, p])
            except ValueError:
                continue

    if not points:
        warnings.warn(f"No valid points found in {filepath}")
        return np.array([]).reshape(0, 4)

    return np.array(points, dtype=np.float32)
