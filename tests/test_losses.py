"""
损失函数测试
"""

import pytest
import numpy as np

try:
    import tensorflow as tf
    from models.losses import (
        WeightedBinaryCrossentropy,
        ContrastiveLoss,
        focal_loss,
        get_loss_function
    )
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow未安装")


class TestWeightedBinaryCrossentropy:
    """加权二元交叉熵测试"""

    def test_loss_creation(self):
        """测试损失函数创建"""
        loss_fn = WeightedBinaryCrossentropy(pos_weight=2.0, from_logits=False)
        assert loss_fn.pos_weight == 2.0
        assert loss_fn.from_logits is False

    def test_loss_computation(self):
        """测试损失计算"""
        loss_fn = WeightedBinaryCrossentropy(pos_weight=1.0, from_logits=False)

        y_true = tf.constant([1.0, 0.0, 1.0, 0.0])
        y_pred = tf.constant([0.9, 0.1, 0.8, 0.2])

        loss = loss_fn(y_true, y_pred)

        # 损失应该是正数
        assert loss > 0

    def test_perfect_prediction(self):
        """测试完美预测"""
        loss_fn = WeightedBinaryCrossentropy(pos_weight=1.0, from_logits=False)

        y_true = tf.constant([1.0, 0.0, 1.0, 0.0])
        y_pred = tf.constant([1.0, 0.0, 1.0, 0.0])

        loss = loss_fn(y_true, y_pred)

        # 完美预测损失应该接近0
        assert loss < 0.01

    def test_weighted_loss(self):
        """测试加权损失"""
        loss_fn_weighted = WeightedBinaryCrossentropy(pos_weight=2.0, from_logits=False)
        loss_fn_normal = WeightedBinaryCrossentropy(pos_weight=1.0, from_logits=False)

        # 只有正样本
        y_true = tf.constant([1.0, 1.0, 1.0, 1.0])
        y_pred = tf.constant([0.5, 0.5, 0.5, 0.5])

        loss_weighted = loss_fn_weighted(y_true, y_pred)
        loss_normal = loss_fn_normal(y_true, y_pred)

        # 加权损失应该更大
        assert loss_weighted > loss_normal

    def test_from_logits(self):
        """测试logits输入"""
        loss_fn = WeightedBinaryCrossentropy(pos_weight=1.0, from_logits=True)

        y_true = tf.constant([1.0, 0.0, 1.0, 0.0])
        y_pred = tf.constant([2.0, -2.0, 1.5, -1.5])  # logits

        loss = loss_fn(y_true, y_pred)

        assert loss > 0

    def test_label_smoothing(self):
        """测试标签平滑"""
        loss_fn = WeightedBinaryCrossentropy(
            pos_weight=1.0,
            from_logits=False,
            label_smoothing=0.1
        )

        y_true = tf.constant([1.0, 0.0, 1.0, 0.0])
        y_pred = tf.constant([0.9, 0.1, 0.8, 0.2])

        loss = loss_fn(y_true, y_pred)

        assert loss > 0


class TestContrastiveLoss:
    """对比损失测试"""

    def test_loss_creation(self):
        """测试损失函数创建"""
        loss_fn = ContrastiveLoss(margin=1.0)
        assert loss_fn.margin == 1.0

    def test_similar_pair_loss(self):
        """测试相似对损失"""
        loss_fn = ContrastiveLoss(margin=1.0)

        # 相似对（y=1），距离应该小
        y_true = tf.constant([1.0, 1.0])
        y_pred = tf.constant([0.1, 0.2])  # 小距离

        loss = loss_fn(y_true, y_pred)

        # 损失应该很小
        assert loss < 0.1

    def test_dissimilar_pair_loss(self):
        """测试不相似对损失"""
        loss_fn = ContrastiveLoss(margin=1.0)

        # 不相似对（y=0），距离应该大
        y_true = tf.constant([0.0, 0.0])
        y_pred = tf.constant([2.0, 3.0])  # 大距离（超过margin）

        loss = loss_fn(y_true, y_pred)

        # 距离超过margin，损失应该接近0
        assert loss < 0.1

    def test_dissimilar_pair_small_distance(self):
        """测试不相似对但距离小的情况"""
        loss_fn = ContrastiveLoss(margin=1.0)

        # 不相似对（y=0），但距离小于margin
        y_true = tf.constant([0.0, 0.0])
        y_pred = tf.constant([0.3, 0.4])  # 小距离

        loss = loss_fn(y_true, y_pred)

        # 距离小于margin，应该有惩罚
        assert loss > 0.1


class TestFocalLoss:
    """Focal Loss测试"""

    def test_focal_loss_computation(self):
        """测试focal loss计算"""
        y_true = tf.constant([1.0, 0.0, 1.0, 0.0])
        y_pred = tf.constant([0.9, 0.1, 0.8, 0.2])

        loss = focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0, from_logits=False)

        assert loss > 0

    def test_focal_loss_easy_examples(self):
        """测试简单样本的focal loss"""
        # 简单样本（高置信度正确预测）
        y_true = tf.constant([1.0, 0.0])
        y_pred = tf.constant([0.99, 0.01])

        loss = focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0, from_logits=False)

        # 简单样本的损失应该很小
        assert loss < 0.01

    def test_focal_loss_hard_examples(self):
        """测试困难样本的focal loss"""
        # 困难样本（低置信度）
        y_true = tf.constant([1.0, 0.0])
        y_pred = tf.constant([0.6, 0.4])

        loss = focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0, from_logits=False)

        # 困难样本的损失应该较大
        assert loss > 0.1

    def test_focal_loss_from_logits(self):
        """测试logits输入"""
        y_true = tf.constant([1.0, 0.0, 1.0, 0.0])
        y_pred = tf.constant([2.0, -2.0, 1.5, -1.5])  # logits

        loss = focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0, from_logits=True)

        assert loss > 0


class TestGetLossFunction:
    """测试损失函数工厂"""

    def test_get_bce(self):
        """测试获取BCE损失"""
        loss_fn = get_loss_function('bce', from_logits=False)
        assert isinstance(loss_fn, tf.keras.losses.BinaryCrossentropy)

    def test_get_weighted_bce(self):
        """测试获取加权BCE损失"""
        loss_fn = get_loss_function('weighted_bce', pos_weight=2.0)
        assert isinstance(loss_fn, WeightedBinaryCrossentropy)
        assert loss_fn.pos_weight == 2.0

    def test_get_contrastive(self):
        """测试获取对比损失"""
        loss_fn = get_loss_function('contrastive', margin=1.5)
        assert isinstance(loss_fn, ContrastiveLoss)
        assert loss_fn.margin == 1.5

    def test_get_focal(self):
        """测试获取focal loss"""
        loss_fn = get_loss_function('focal', alpha=0.25, gamma=2.0)

        # focal loss返回的是函数
        y_true = tf.constant([1.0, 0.0])
        y_pred = tf.constant([0.9, 0.1])

        loss = loss_fn(y_true, y_pred)
        assert loss > 0

    def test_invalid_loss_type(self):
        """测试无效的损失类型"""
        with pytest.raises(ValueError):
            get_loss_function('invalid_loss')
