"""
测试评估器模块
"""

import pytest
import numpy as np
import tensorflow as tf
from pathlib import Path
import tempfile

# 标记需要TensorFlow的测试
pytestmark = pytest.mark.skipif(
    not hasattr(tf, '__version__'),
    reason="TensorFlow not installed"
)


def create_dummy_dataset(num_samples=10, batch_size=2):
    """创建虚拟数据集"""
    def generator():
        for _ in range(num_samples):
            sig1 = tf.random.normal((20, 23))
            sig2 = tf.random.normal((20, 23))
            label = tf.random.uniform((), 0, 2, dtype=tf.int32)
            yield sig1, sig2, label

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 23), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 23), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    return dataset.padded_batch(batch_size, padded_shapes=([None, 23], [None, 23], []))


def create_dummy_model():
    """创建虚拟模型"""
    from models.siamese import SiameseNetwork

    model = SiameseNetwork(
        input_size=23,
        hidden_size=32,
        num_layers=1,
        dropout=0.0,
        bidirectional=False,
        use_attention=False
    )
    return model


def test_evaluator_initialization():
    """测试评估器初始化"""
    from training.evaluator import Evaluator

    model = create_dummy_model()
    test_dataset = create_dummy_dataset()

    evaluator = Evaluator(
        model=model,
        test_dataset=test_dataset
    )

    assert evaluator.model is not None
    assert evaluator.test_dataset is not None


def test_evaluate():
    """测试评估功能"""
    from training.evaluator import Evaluator

    model = create_dummy_model()
    test_dataset = create_dummy_dataset(num_samples=10, batch_size=2)

    evaluator = Evaluator(
        model=model,
        test_dataset=test_dataset
    )

    results = evaluator.evaluate()

    assert 'eer' in results
    assert 'eer_threshold' in results
    assert 'eer_metrics' in results
    assert 'default_metrics' in results
    assert 'num_samples' in results

    # 验证EER在合理范围内
    assert 0 <= results['eer'] <= 1


def test_evaluate_with_save():
    """测试评估结果保存"""
    from training.evaluator import Evaluator

    model = create_dummy_model()
    test_dataset = create_dummy_dataset(num_samples=10, batch_size=2)

    evaluator = Evaluator(
        model=model,
        test_dataset=test_dataset
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        results = evaluator.evaluate(save_dir=tmpdir)

        # 检查文件是否生成
        save_path = Path(tmpdir)
        assert (save_path / 'evaluation_results.json').exists()
        assert (save_path / 'predictions.npz').exists()
        assert (save_path / 'roc_curve.png').exists()
        assert (save_path / 'det_curve.png').exists()


def test_evaluate_metrics_structure():
    """测试评估指标结构"""
    from training.evaluator import Evaluator

    model = create_dummy_model()
    test_dataset = create_dummy_dataset(num_samples=10, batch_size=2)

    evaluator = Evaluator(
        model=model,
        test_dataset=test_dataset
    )

    results = evaluator.evaluate()

    # 检查EER指标
    eer_metrics = results['eer_metrics']
    assert 'accuracy' in eer_metrics
    assert 'far' in eer_metrics
    assert 'frr' in eer_metrics
    assert 'precision' in eer_metrics
    assert 'recall' in eer_metrics
    assert 'f1_score' in eer_metrics

    # 检查默认阈值指标
    default_metrics = results['default_metrics']
    assert 'accuracy' in default_metrics
    assert 'far' in default_metrics
    assert 'frr' in default_metrics


def test_evaluate_sample_counts():
    """测试样本计数"""
    from training.evaluator import Evaluator

    model = create_dummy_model()
    test_dataset = create_dummy_dataset(num_samples=20, batch_size=4)

    evaluator = Evaluator(
        model=model,
        test_dataset=test_dataset
    )

    results = evaluator.evaluate()

    # 验证样本数量
    assert results['num_samples'] == 20
    assert results['num_positives'] + results['num_negatives'] == 20


def test_compute_confusion_matrix():
    """测试混淆矩阵计算"""
    from training.evaluator import Evaluator

    model = create_dummy_model()
    test_dataset = create_dummy_dataset(num_samples=10, batch_size=2)

    evaluator = Evaluator(
        model=model,
        test_dataset=test_dataset
    )

    cm = evaluator.compute_confusion_matrix(threshold=0.5)

    # 混淆矩阵应该是2x2
    assert cm.shape == (2, 2)
    assert cm.sum() == 10  # 总样本数


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
