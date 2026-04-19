"""
数据增强测试
"""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.augmentation import (
    rotate_signature,
    scale_signature,
    add_noise,
    SignatureAugmentation
)


class TestAugmentation:
    """数据增强测试"""

    @pytest.fixture
    def sample_features(self):
        """创建示例特征"""
        n_points = 10
        features = np.random.randn(n_points, 23).astype(np.float32)
        # 设置笔画标记为1
        features[:, 0] = 1.0
        return features

    def test_rotate_signature(self, sample_features):
        """测试旋转"""
        angle = 45.0
        rotated = rotate_signature(sample_features, angle)

        assert rotated.shape == sample_features.shape
        assert rotated.dtype == np.float32
        # 检查笔画标记未改变
        np.testing.assert_array_equal(rotated[:, 0], sample_features[:, 0])

    def test_scale_signature(self, sample_features):
        """测试缩放"""
        scale = 1.5
        scaled = scale_signature(sample_features, scale)

        assert scaled.shape == sample_features.shape
        assert scaled.dtype == np.float32
        # 检查笔画标记未改变
        np.testing.assert_array_equal(scaled[:, 0], sample_features[:, 0])

    def test_add_noise(self, sample_features):
        """测试添加噪声"""
        noisy = add_noise(sample_features, noise_std=0.01)

        assert noisy.shape == sample_features.shape
        assert noisy.dtype == np.float32
        # 检查笔画标记未改变
        np.testing.assert_array_equal(noisy[:, 0], sample_features[:, 0])
        # 检查其他特征有变化
        assert not np.allclose(noisy[:, 1:], sample_features[:, 1:])

    def test_signature_augmentation(self, sample_features):
        """测试数据增强类"""
        aug = SignatureAugmentation(
            rotation_range=5.0,
            scale_range=(0.9, 1.1),
            noise_std=0.01,
            probability=1.0  # 总是应用
        )

        augmented = aug(sample_features)

        assert augmented.shape == sample_features.shape
        assert augmented.dtype == np.float32

    def test_signature_augmentation_probability(self, sample_features):
        """测试增强概率"""
        # 概率为0，不应用增强
        aug = SignatureAugmentation(probability=0.0)
        augmented = aug(sample_features)
        np.testing.assert_array_equal(augmented, sample_features)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
