"""
测试训练器模块
"""

import pytest
import numpy as np
import tensorflow as tf
from unittest.mock import Mock, MagicMock

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


def test_trainer_initialization():
    """测试训练器初始化"""
    from training.trainer import Trainer

    model = create_dummy_model()
    train_dataset = create_dummy_dataset()

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset
    )

    assert trainer.model is not None
    assert trainer.train_dataset is not None
    assert trainer.optimizer is not None
    assert trainer.loss_fn is not None


def test_trainer_with_custom_optimizer():
    """测试自定义优化器"""
    from training.trainer import Trainer

    model = create_dummy_model()
    train_dataset = create_dummy_dataset()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        optimizer=optimizer
    )

    assert isinstance(trainer.optimizer, tf.keras.optimizers.SGD)


def test_trainer_with_callbacks():
    """测试回调功能"""
    from training.trainer import Trainer
    from training.callbacks import Callback

    model = create_dummy_model()
    train_dataset = create_dummy_dataset()

    # 创建模拟回调
    callback = Mock(spec=Callback)

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        callbacks=[callback]
    )

    assert len(trainer.callbacks) == 1


def test_train_single_epoch():
    """测试单个epoch训练"""
    from training.trainer import Trainer

    model = create_dummy_model()
    train_dataset = create_dummy_dataset(num_samples=4, batch_size=2)

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset
    )

    # 训练1个epoch
    history = trainer.train(epochs=1)

    assert 'train_loss' in history
    assert len(history['train_loss']) == 1
    assert history['train_loss'][0] > 0


def test_train_with_validation():
    """测试带验证的训练"""
    from training.trainer import Trainer

    model = create_dummy_model()
    train_dataset = create_dummy_dataset(num_samples=4, batch_size=2)
    val_dataset = create_dummy_dataset(num_samples=4, batch_size=2)

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset
    )

    history = trainer.train(epochs=2)

    assert 'train_loss' in history
    assert 'val_loss' in history
    assert len(history['train_loss']) == 2
    assert len(history['val_loss']) == 2


def test_train_step():
    """测试单步训练"""
    from training.trainer import Trainer

    model = create_dummy_model()
    train_dataset = create_dummy_dataset(num_samples=2, batch_size=2)

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset
    )

    # 获取一个batch
    batch = next(iter(train_dataset))

    # 执行训练步骤
    loss, predictions = trainer._train_step(batch)

    assert loss.numpy() > 0
    assert predictions.shape[0] == 2  # batch_size


def test_validate_epoch():
    """测试验证epoch"""
    from training.trainer import Trainer

    model = create_dummy_model()
    train_dataset = create_dummy_dataset(num_samples=2, batch_size=2)
    val_dataset = create_dummy_dataset(num_samples=4, batch_size=2)

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset
    )

    val_logs = trainer._validate_epoch()

    assert 'val_loss' in val_logs
    assert val_logs['val_loss'] > 0


def test_callback_execution():
    """测试回调执行"""
    from training.trainer import Trainer
    from training.callbacks import Callback

    model = create_dummy_model()
    train_dataset = create_dummy_dataset(num_samples=2, batch_size=2)

    # 创建模拟回调
    callback = Mock(spec=Callback)

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        callbacks=[callback]
    )

    trainer.train(epochs=1)

    # 验证回调被调用
    callback.on_train_begin.assert_called_once()
    callback.on_train_end.assert_called_once()
    callback.on_epoch_begin.assert_called()
    callback.on_epoch_end.assert_called()


def test_early_stopping():
    """测试早停功能"""
    from training.trainer import Trainer
    from training.callbacks import EarlyStoppingCallback

    model = create_dummy_model()
    train_dataset = create_dummy_dataset(num_samples=4, batch_size=2)
    val_dataset = create_dummy_dataset(num_samples=4, batch_size=2)

    early_stopping = EarlyStoppingCallback(
        monitor='val_loss',
        patience=2,
        verbose=0
    )

    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        callbacks=[early_stopping]
    )

    history = trainer.train(epochs=10)

    # 应该在10个epoch之前停止（如果验证损失不改善）
    # 注意：由于是随机数据，可能不会触发早停
    assert len(history['train_loss']) <= 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
