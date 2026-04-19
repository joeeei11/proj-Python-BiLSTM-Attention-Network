"""
测试评估指标模块
"""

import pytest
import numpy as np
from utils.metrics import (
    calculate_eer,
    calculate_accuracy,
    calculate_far_frr,
    calculate_metrics_at_threshold,
    find_optimal_threshold
)


def test_calculate_eer():
    """测试EER计算"""
    # 完美分类
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    eer, threshold = calculate_eer(y_true, y_scores)

    assert 0 <= eer <= 1
    assert 0 <= threshold <= 1
    assert eer < 0.5  # 应该很低


def test_calculate_eer_random():
    """测试随机预测的EER"""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_scores = np.random.rand(100)

    eer, threshold = calculate_eer(y_true, y_scores)

    # 随机预测EER应该接近0.5
    assert 0.3 < eer < 0.7


def test_calculate_accuracy():
    """测试准确率计算"""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])

    acc = calculate_accuracy(y_true, y_pred)
    assert acc == 1.0

    y_pred = np.array([1, 1, 0, 0])
    acc = calculate_accuracy(y_true, y_pred)
    assert acc == 0.0

    y_pred = np.array([0, 1, 1, 0])
    acc = calculate_accuracy(y_true, y_pred)
    assert acc == 0.5


def test_calculate_far_frr():
    """测试FAR和FRR计算"""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.8, 0.3, 0.7, 0.9])

    # 阈值0.5
    far, frr = calculate_far_frr(y_true, y_scores, threshold=0.5)

    # FAR: 1/3 (一个伪签名被接受)
    # FRR: 1/3 (一个真签名被拒绝)
    assert abs(far - 1/3) < 0.01
    assert abs(frr - 1/3) < 0.01


def test_calculate_far_frr_extreme():
    """测试极端阈值的FAR和FRR"""
    y_true = np.array([0, 0, 1, 1])
    y_scores = np.array([0.2, 0.3, 0.7, 0.8])

    # 阈值0.0 - 全部接受
    far, frr = calculate_far_frr(y_true, y_scores, threshold=0.0)
    assert far == 1.0  # 所有伪签名被接受
    assert frr == 0.0  # 没有真签名被拒绝

    # 阈值1.0 - 全部拒绝
    far, frr = calculate_far_frr(y_true, y_scores, threshold=1.0)
    assert far == 0.0  # 没有伪签名被接受
    assert frr == 1.0  # 所有真签名被拒绝


def test_calculate_metrics_at_threshold():
    """测试完整指标计算"""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    metrics = calculate_metrics_at_threshold(y_true, y_scores, threshold=0.5)

    assert 'accuracy' in metrics
    assert 'far' in metrics
    assert 'frr' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
    assert 'true_positives' in metrics
    assert 'true_negatives' in metrics
    assert 'false_positives' in metrics
    assert 'false_negatives' in metrics

    # 验证准确率
    assert metrics['accuracy'] == 1.0


def test_find_optimal_threshold_eer():
    """测试寻找EER最优阈值"""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    threshold = find_optimal_threshold(y_true, y_scores, metric='eer')

    assert 0 <= threshold <= 1


def test_find_optimal_threshold_accuracy():
    """测试寻找准确率最优阈值"""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    threshold = find_optimal_threshold(y_true, y_scores, metric='accuracy')

    assert 0 <= threshold <= 1


def test_find_optimal_threshold_f1():
    """测试寻找F1最优阈值"""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    threshold = find_optimal_threshold(y_true, y_scores, metric='f1')

    assert 0 <= threshold <= 1


def test_metrics_consistency():
    """测试指标一致性"""
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_scores = np.random.rand(100)

    # EER阈值
    eer, eer_threshold = calculate_eer(y_true, y_scores)
    far, frr = calculate_far_frr(y_true, y_scores, eer_threshold)

    # 在EER点，FAR应该接近FRR
    assert abs(far - frr) < 0.1


def test_edge_cases():
    """测试边界情况"""
    # 全部正样本
    y_true = np.array([1, 1, 1, 1])
    y_scores = np.array([0.6, 0.7, 0.8, 0.9])

    far, frr = calculate_far_frr(y_true, y_scores, threshold=0.5)
    assert far == 0.0  # 没有负样本

    # 全部负样本
    y_true = np.array([0, 0, 0, 0])
    y_scores = np.array([0.1, 0.2, 0.3, 0.4])

    far, frr = calculate_far_frr(y_true, y_scores, threshold=0.5)
    assert frr == 0.0  # 没有正样本


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
