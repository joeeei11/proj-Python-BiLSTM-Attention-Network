"""
特征提取测试
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.feature_extractor import (
    mark_stroke,
    normalize_coordinates,
    compute_velocity,
    compute_acceleration,
    compute_curvature,
    compute_angle,
    compute_arc_length,
    extract_temporal_features
)


class TestFeatureExtractor:
    """特征提取器测试"""

    def test_mark_stroke(self):
        """测试笔画标记"""
        pressure = np.array([0.0, 0.5, 1.0, 0.005, 0.8])
        stroke_mark = mark_stroke(pressure, threshold=0.01)

        assert stroke_mark.shape == pressure.shape
        assert stroke_mark.dtype == np.float32
        assert stroke_mark[0] == 0  # 0.0 < 0.01
        assert stroke_mark[1] == 1  # 0.5 >= 0.01
        assert stroke_mark[3] == 0  # 0.005 < 0.01

    def test_normalize_coordinates(self):
        """测试坐标归一化"""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 2.0, 4.0, 6.0])
        p = np.array([0.5, 0.6, 0.7, 0.8])

        x_norm, y_norm, p_norm, _ = normalize_coordinates(x, y, p)

        # 检查归一化范围
        assert x_norm.min() == 0.0
        assert x_norm.max() == 1.0
        assert y_norm.min() == 0.0
        assert y_norm.max() == 1.0
        assert p_norm.min() == 0.0
        assert p_norm.max() == 1.0

    def test_compute_velocity(self):
        """测试速度计算"""
        arr = np.array([0.0, 1.0, 3.0, 6.0])
        velocity = compute_velocity(arr)

        assert velocity.shape == arr.shape
        assert velocity.dtype == np.float32
        assert velocity[0] == 0.0  # 第一个点速度为0
        assert velocity[1] == 1.0  # 1.0 - 0.0
        assert velocity[2] == 2.0  # 3.0 - 1.0

    def test_compute_acceleration(self):
        """测试加速度计算"""
        velocity = np.array([0.0, 1.0, 2.0, 3.0])
        acceleration = compute_acceleration(velocity)

        assert acceleration.shape == velocity.shape
        assert acceleration.dtype == np.float32

    def test_compute_curvature(self):
        """测试曲率计算"""
        v_x = np.array([1.0, 1.0, 0.0, -1.0])
        v_y = np.array([0.0, 1.0, 1.0, 0.0])
        a_x = np.array([0.0, -1.0, -1.0, 0.0])
        a_y = np.array([1.0, 0.0, -1.0, -1.0])

        curvature = compute_curvature(v_x, v_y, a_x, a_y)

        assert curvature.shape == v_x.shape
        assert curvature.dtype == np.float32
        assert not np.any(np.isnan(curvature))

    def test_compute_angle(self):
        """测试角度计算"""
        v_x = np.array([1.0, 0.0, -1.0, 0.0])
        v_y = np.array([0.0, 1.0, 0.0, -1.0])

        angle = compute_angle(v_x, v_y)

        assert angle.shape == v_x.shape
        assert angle.dtype == np.float32
        assert np.isclose(angle[0], 0.0)  # arctan2(0, 1) = 0
        assert np.isclose(angle[1], np.pi / 2)  # arctan2(1, 0) = π/2

    def test_compute_arc_length(self):
        """测试弧长计算"""
        x = np.array([0.0, 1.0, 1.0, 0.0])
        y = np.array([0.0, 0.0, 1.0, 1.0])

        cumulative, normalized = compute_arc_length(x, y)

        assert cumulative.shape == x.shape
        assert normalized.shape == x.shape
        assert cumulative.dtype == np.float32
        assert normalized.dtype == np.float32
        assert cumulative[0] == 0.0
        assert normalized.min() == 0.0
        assert normalized.max() == 1.0

    def test_extract_temporal_features(self):
        """测试完整特征提取"""
        # 创建简单的测试数据
        n_points = 10
        x = np.linspace(0, 1, n_points)
        y = np.linspace(0, 1, n_points)
        p = np.ones(n_points) * 0.5
        time = np.linspace(0, 1, n_points)

        features = extract_temporal_features(x, y, p, time)

        # 检查形状
        assert features.shape == (n_points, 23)
        assert features.dtype == np.float32

        # 检查没有NaN或Inf
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))

        # 检查笔画标记
        assert np.all(features[:, 0] == 1)  # 所有点压力都>=0.01

    def test_extract_temporal_features_no_time(self):
        """测试无时间戳的特征提取"""
        n_points = 10
        x = np.linspace(0, 1, n_points)
        y = np.linspace(0, 1, n_points)
        p = np.ones(n_points) * 0.5

        features = extract_temporal_features(x, y, p, time=None)

        assert features.shape == (n_points, 23)
        assert features.dtype == np.float32

    def test_extract_temporal_features_edge_cases(self):
        """测试边界情况"""
        # 单点
        x = np.array([0.5])
        y = np.array([0.5])
        p = np.array([0.5])

        features = extract_temporal_features(x, y, p)
        assert features.shape == (1, 23)

        # 两点
        x = np.array([0.0, 1.0])
        y = np.array([0.0, 1.0])
        p = np.array([0.5, 0.5])

        features = extract_temporal_features(x, y, p)
        assert features.shape == (2, 23)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
