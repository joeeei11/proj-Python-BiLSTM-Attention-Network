"""
数据增强模块
"""
import numpy as np
from typing import Optional


def rotate_signature(features: np.ndarray, angle_degrees: float) -> np.ndarray:
    """
    旋转签名

    Args:
        features: 形状为 (N, 23) 的特征数组
        angle_degrees: 旋转角度（度）

    Returns:
        旋转后的特征数组
    """
    features = features.copy()
    angle_rad = np.deg2rad(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # 旋转归一化坐标 (索引2和3是x_norm和y_norm)
    x_norm = features[:, 2]
    y_norm = features[:, 3]

    features[:, 2] = cos_a * x_norm - sin_a * y_norm
    features[:, 3] = sin_a * x_norm + cos_a * y_norm

    # 旋转速度 (索引5和6是v_x和v_y)
    v_x = features[:, 5]
    v_y = features[:, 6]

    features[:, 5] = cos_a * v_x - sin_a * v_y
    features[:, 6] = sin_a * v_x + cos_a * v_y

    # 旋转加速度 (索引8和9是a_x和a_y)
    a_x = features[:, 8]
    a_y = features[:, 9]

    features[:, 8] = cos_a * a_x - sin_a * a_y
    features[:, 9] = sin_a * a_x + cos_a * a_y

    # 更新角度 (索引14)
    features[:, 14] = features[:, 14] + angle_rad

    return features


def scale_signature(features: np.ndarray, scale_factor: float) -> np.ndarray:
    """
    缩放签名

    Args:
        features: 形状为 (N, 23) 的特征数组
        scale_factor: 缩放因子

    Returns:
        缩放后的特征数组
    """
    features = features.copy()

    # 缩放坐标相关特征
    # 索引2,3: x_norm, y_norm
    features[:, 2:4] *= scale_factor

    # 索引5,6: v_x, v_y
    features[:, 5:7] *= scale_factor

    # 索引8,9: a_x, a_y
    features[:, 8:10] *= scale_factor

    # 索引11: v_abs
    features[:, 11] *= scale_factor

    # 索引12: a_abs
    features[:, 12] *= scale_factor

    # 索引15,16: cumulative_arc_length, arc_length_norm
    features[:, 15:17] *= scale_factor

    return features


def add_noise(features: np.ndarray, noise_std: float = 0.01) -> np.ndarray:
    """
    添加高斯噪声

    Args:
        features: 形状为 (N, 23) 的特征数组
        noise_std: 噪声标准差

    Returns:
        添加噪声后的特征数组
    """
    features = features.copy()
    noise = np.random.normal(0, noise_std, features.shape).astype(np.float32)

    # 只对部分特征添加噪声（不对笔画标记添加噪声）
    features[:, 1:] += noise[:, 1:]

    return features


class SignatureAugmentation:
    """
    签名数据增强类
    """

    def __init__(
        self,
        rotation_range: float = 5.0,
        scale_range: tuple = (0.9, 1.1),
        noise_std: float = 0.01,
        probability: float = 0.5
    ):
        """
        初始化数据增强

        Args:
            rotation_range: 旋转角度范围（±度）
            scale_range: 缩放范围 (min, max)
            noise_std: 噪声标准差
            probability: 应用增强的概率
        """
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.probability = probability

    def __call__(self, features: np.ndarray) -> np.ndarray:
        """
        应用数据增强

        Args:
            features: 形状为 (N, 23) 的特征数组

        Returns:
            增强后的特征数组
        """
        if np.random.rand() > self.probability:
            return features

        # 随机旋转
        if self.rotation_range > 0:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            features = rotate_signature(features, angle)

        # 随机缩放
        if self.scale_range[0] != self.scale_range[1]:
            scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
            features = scale_signature(features, scale)

        # 添加噪声
        if self.noise_std > 0:
            features = add_noise(features, self.noise_std)

        return features


def create_augmentation_from_config(config: dict) -> Optional[SignatureAugmentation]:
    """
    从配置创建数据增强对象

    Args:
        config: 配置字典

    Returns:
        SignatureAugmentation对象，如果未启用则返回None
    """
    if not config.get('enabled', False):
        return None

    return SignatureAugmentation(
        rotation_range=config.get('rotation_range', 5.0),
        scale_range=config.get('scale_range', [0.9, 1.1]),
        noise_std=config.get('noise_std', 0.01),
        probability=config.get('probability', 0.5)
    )
