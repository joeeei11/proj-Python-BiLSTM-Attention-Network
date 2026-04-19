"""
测试回调模块
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


def test_early_stopping_callback_initialization():
    """测试早停回调初始化"""
    from training.callbacks import EarlyStoppingCallback

    callback = EarlyStoppingCallback(
        monitor='val_loss',
        patience=5,
        min_delta=0.001
    )

    assert callback.monitor == 'val_loss'
    assert callback.patience == 5
    assert callback.min_delta == 0.001


def test_early_stopping_improvement():
    """测试早停改善检测"""
    from training.callbacks import EarlyStoppingCallback

    callback = EarlyStoppingCallback(
        monitor='val_loss',
        patience=3,
        mode='min',
        verbose=0
    )

    # 模拟训练开始
    callback.on_train_begin()

    # 模拟改善的情况
    logs = {'val_loss': 1.0, 'model': None}
    callback.on_epoch_end(0, logs)
    assert callback.wait == 0

    logs = {'val_loss': 0.8, 'model': None}
    callback.on_epoch_end(1, logs)
    assert callback.wait == 0  # 改善了

    logs = {'val_loss': 0.9, 'model': None}
    callback.on_epoch_end(2, logs)
    assert callback.wait == 1  # 没有改善


def test_early_stopping_trigger():
    """测试早停触发"""
    from training.callbacks import EarlyStoppingCallback

    callback = EarlyStoppingCallback(
        monitor='val_loss',
        patience=2,
        mode='min',
        verbose=0
    )

    callback.on_train_begin()

    # 模拟没有改善的情况
    logs = {'val_loss': 1.0, 'model': None}
    callback.on_epoch_end(0, logs)

    logs = {'val_loss': 1.1, 'model': None}
    callback.on_epoch_end(1, logs)
    assert 'stop_training' not in logs

    logs = {'val_loss': 1.2, 'model': None}
    callback.on_epoch_end(2, logs)
    assert 'stop_training' not in logs

    logs = {'val_loss': 1.3, 'model': None}
    callback.on_epoch_end(3, logs)
    assert logs.get('stop_training') == True  # 应该触发早停


def test_model_checkpoint_callback():
    """测试模型检查点回调"""
    from training.callbacks import ModelCheckpointCallback

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = str(Path(tmpdir) / 'model.h5')

        callback = ModelCheckpointCallback(
            filepath=filepath,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=0
        )

        assert callback.filepath == filepath
        assert callback.monitor == 'val_loss'


def test_learning_rate_scheduler_reduce_on_plateau():
    """测试学习率调度（ReduceOnPlateau）"""
    from training.callbacks import LearningRateSchedulerCallback

    callback = LearningRateSchedulerCallback(
        schedule='reduce_on_plateau',
        monitor='val_loss',
        factor=0.5,
        patience=2,
        verbose=0
    )

    # 创建模拟优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    callback.on_train_begin({'optimizer': optimizer})

    # 模拟没有改善的情况
    logs = {'val_loss': 1.0, 'optimizer': optimizer}
    callback.on_epoch_end(0, logs)

    logs = {'val_loss': 1.1, 'optimizer': optimizer}
    callback.on_epoch_end(1, logs)

    logs = {'val_loss': 1.2, 'optimizer': optimizer}
    callback.on_epoch_end(2, logs)

    # 学习率应该降低
    current_lr = optimizer.learning_rate.numpy()
    assert current_lr < 0.001


def test_learning_rate_scheduler_step():
    """测试学习率调度（Step）"""
    from training.callbacks import LearningRateSchedulerCallback

    callback = LearningRateSchedulerCallback(
        schedule='step',
        step_size=2,
        gamma=0.1,
        verbose=0
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    callback.on_train_begin({'optimizer': optimizer})

    # Epoch 0, 1: lr = 0.1
    callback.on_epoch_end(0, {'optimizer': optimizer})
    assert abs(optimizer.learning_rate.numpy() - 0.1) < 1e-6

    callback.on_epoch_end(1, {'optimizer': optimizer})
    assert abs(optimizer.learning_rate.numpy() - 0.1) < 1e-6

    # Epoch 2: lr = 0.01
    callback.on_epoch_end(2, {'optimizer': optimizer})
    assert abs(optimizer.learning_rate.numpy() - 0.01) < 1e-6


def test_tensorboard_callback():
    """测试TensorBoard回调"""
    from training.callbacks import TensorBoardCallback

    with tempfile.TemporaryDirectory() as tmpdir:
        callback = TensorBoardCallback(
            log_dir=tmpdir,
            update_freq='epoch'
        )

        assert callback.log_dir == tmpdir


def test_callback_base_class():
    """测试回调基类"""
    from training.callbacks import Callback

    callback = Callback()

    # 所有方法应该可以调用而不报错
    callback.on_train_begin()
    callback.on_train_end()
    callback.on_epoch_begin(0)
    callback.on_epoch_end(0)
    callback.on_batch_begin(0)
    callback.on_batch_end(0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
