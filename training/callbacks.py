"""
训练回调模块

实现训练过程中的回调功能：
- 早停 (Early Stopping)
- 模型检查点保存 (Model Checkpoint)
- 学习率调度 (Learning Rate Scheduler)
- TensorBoard日志记录
"""

import os
import numpy as np
from typing import Optional, Dict, Any
import tensorflow as tf
from pathlib import Path


class Callback:
    """回调基类"""

    def on_train_begin(self, logs: Optional[Dict] = None):
        """训练开始时调用"""
        pass

    def on_train_end(self, logs: Optional[Dict] = None):
        """训练结束时调用"""
        pass

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        """每个epoch开始时调用"""
        pass

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """每个epoch结束时调用"""
        pass

    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        """每个batch开始时调用"""
        pass

    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        """每个batch结束时调用"""
        pass


class EarlyStoppingCallback(Callback):
    """早停回调"""

    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = True,
        verbose: int = 1
    ):
        """
        Args:
            monitor: 监控的指标名称
            patience: 容忍的epoch数
            min_delta: 最小改善量
            mode: 'min' 或 'max'
            restore_best_weights: 是否恢复最佳权重
            verbose: 日志级别
        """
        super().__init__()
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.best_value = np.inf if mode == 'min' else -np.inf

    def on_train_begin(self, logs: Optional[Dict] = None):
        """重置状态"""
        self.wait = 0
        self.stopped_epoch = 0
        self.best_value = np.inf if self.mode == 'min' else -np.inf

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """检查是否需要早停"""
        if logs is None:
            return

        current = logs.get(self.monitor)
        if current is None:
            return

        # 判断是否改善
        if self.mode == 'min':
            improved = current < (self.best_value - self.min_delta)
        else:
            improved = current > (self.best_value + self.min_delta)

        if improved:
            self.best_value = current
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = [w.numpy() for w in logs.get('model').trainable_variables]
            if self.verbose > 0:
                print(f"\nEpoch {epoch+1}: {self.monitor} improved to {current:.6f}")
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                logs['stop_training'] = True
                if self.verbose > 0:
                    print(f"\nEpoch {epoch+1}: early stopping triggered")

    def on_train_end(self, logs: Optional[Dict] = None):
        """恢复最佳权重"""
        if self.stopped_epoch > 0 and self.verbose > 0:
            print(f"Early stopping at epoch {self.stopped_epoch + 1}")

        if self.restore_best_weights and self.best_weights is not None:
            if self.verbose > 0:
                print("Restoring best weights")
            model = logs.get('model')
            if model is not None:
                for var, weight in zip(model.trainable_variables, self.best_weights):
                    var.assign(weight)


class ModelCheckpointCallback(Callback):
    """模型检查点保存回调"""

    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        save_best_only: bool = True,
        mode: str = 'min',
        save_weights_only: bool = True,
        verbose: int = 1
    ):
        """
        Args:
            filepath: 保存路径（支持格式化，如 'model_{epoch:02d}.h5'）
            monitor: 监控的指标
            save_best_only: 是否只保存最佳模型
            mode: 'min' 或 'max'
            save_weights_only: 是否只保存权重
            verbose: 日志级别
        """
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.save_weights_only = save_weights_only
        self.verbose = verbose

        self.best_value = np.inf if mode == 'min' else -np.inf

        # 创建保存目录
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """保存模型"""
        if logs is None:
            return

        model = logs.get('model')
        if model is None:
            return

        current = logs.get(self.monitor)
        if current is None:
            return

        # 判断是否需要保存
        should_save = False
        if self.save_best_only:
            if self.mode == 'min':
                improved = current < self.best_value
            else:
                improved = current > self.best_value

            if improved:
                self.best_value = current
                should_save = True
        else:
            should_save = True

        if should_save:
            # 格式化文件名
            filepath = self.filepath.format(epoch=epoch+1, **logs)

            # 保存模型
            if self.save_weights_only:
                model.save_weights(filepath)
            else:
                model.save(filepath)

            if self.verbose > 0:
                print(f"\nEpoch {epoch+1}: saving model to {filepath}")


class LearningRateSchedulerCallback(Callback):
    """学习率调度回调"""

    def __init__(
        self,
        schedule: str = 'reduce_on_plateau',
        monitor: str = 'val_loss',
        factor: float = 0.5,
        patience: int = 5,
        min_lr: float = 1e-7,
        mode: str = 'min',
        verbose: int = 1
    ):
        """
        Args:
            schedule: 调度策略 ('reduce_on_plateau', 'step', 'cosine')
            monitor: 监控的指标
            factor: 学习率衰减因子
            patience: ReduceLROnPlateau的容忍epoch数
            min_lr: 最小学习率
            mode: 'min' 或 'max'
            verbose: 日志级别
        """
        super().__init__()
        self.schedule = schedule
        self.monitor = monitor
        self.factor = float(factor)
        self.patience = int(patience)
        self.min_lr = float(min_lr)
        self.mode = mode
        self.verbose = verbose

        self.wait = 0
        self.best_value = np.inf if mode == 'min' else -np.inf

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """调整学习率"""
        if logs is None:
            return

        optimizer = logs.get('optimizer')
        if optimizer is None:
            return

        current_lr = optimizer.learning_rate.numpy()

        if self.schedule == 'reduce_on_plateau':
            current = logs.get(self.monitor)
            if current is None:
                return

            # 判断是否改善
            if self.mode == 'min':
                improved = current < self.best_value
            else:
                improved = current > self.best_value

            if improved:
                self.best_value = current
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    new_lr = max(current_lr * self.factor, self.min_lr)
                    if new_lr < current_lr:
                        optimizer.learning_rate.assign(new_lr)
                        if self.verbose > 0:
                            print(f"\nEpoch {epoch+1}: reducing learning rate to {new_lr:.2e}")
                        self.wait = 0


class TensorBoardCallback(Callback):
    """TensorBoard日志回调"""

    def __init__(
        self,
        log_dir: str,
        update_freq: str = 'epoch',
        profile_batch: int = 0
    ):
        """
        Args:
            log_dir: 日志目录
            update_freq: 更新频率 ('epoch' 或 'batch')
            profile_batch: 性能分析的batch范围
        """
        super().__init__()
        self.log_dir = log_dir
        self.update_freq = update_freq
        self.profile_batch = profile_batch

        # 创建日志目录
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        # 创建writer
        self.train_writer = tf.summary.create_file_writer(
            os.path.join(log_dir, 'train')
        )
        self.val_writer = tf.summary.create_file_writer(
            os.path.join(log_dir, 'validation')
        )

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        """记录epoch级别的指标"""
        if logs is None:
            return

        # 记录训练指标
        with self.train_writer.as_default():
            for key, value in logs.items():
                if key.startswith('train_') or key in ['loss', 'accuracy']:
                    tf.summary.scalar(key, value, step=epoch)

        # 记录验证指标
        with self.val_writer.as_default():
            for key, value in logs.items():
                if key.startswith('val_'):
                    tf.summary.scalar(key, value, step=epoch)

        # 记录学习率
        optimizer = logs.get('optimizer')
        if optimizer is not None:
            with self.train_writer.as_default():
                tf.summary.scalar(
                    'learning_rate',
                    optimizer.learning_rate.numpy(),
                    step=epoch
                )

    def on_train_end(self, logs: Optional[Dict] = None):
        """关闭writer"""
        self.train_writer.close()
        self.val_writer.close()


class CallbackList:
    """回调列表管理器"""

    def __init__(self, callbacks: list):
        self.callbacks = callbacks or []

    def on_train_begin(self, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_train_end(logs)

    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch: int, logs: Optional[Dict] = None):
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)
