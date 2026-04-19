"""
训练器模块

实现完整的训练循环，包括：
- 前向传播和反向传播
- 梯度计算和参数更新
- 训练/验证循环
- 回调管理
- 日志记录
"""

import os
import sys
import time
from typing import Optional, List, Dict, Any
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm

from training.callbacks import Callback
from utils.logger import get_logger


class Trainer:
    """训练器类"""

    def __init__(
        self,
        model: tf.keras.Model,
        train_dataset: tf.data.Dataset,
        val_dataset: Optional[tf.data.Dataset] = None,
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        loss_fn: Optional[tf.keras.losses.Loss] = None,
        metrics: Optional[List[tf.keras.metrics.Metric]] = None,
        callbacks: Optional[List[Callback]] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            model: 待训练的模型
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            optimizer: 优化器
            loss_fn: 损失函数
            metrics: 评估指标列表
            callbacks: 回调列表
            config: 配置字典
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.callbacks = callbacks or []
        self.config = config or {}

        # 优化器
        if optimizer is None:
            lr = self.config.get('learning_rate', 0.001)
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        else:
            self.optimizer = optimizer

        # 损失函数
        if loss_fn is None:
            self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        else:
            self.loss_fn = loss_fn

        # 评估指标
        self.metrics = metrics or [
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.AUC(name='auc')
        ]

        # 日志
        self.logger = get_logger(__name__)

        # 训练状态
        self.current_epoch = 0
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

    def train(
        self,
        epochs: int,
        initial_epoch: int = 0,
        steps_per_epoch: Optional[int] = None,
        validation_steps: Optional[int] = None
    ) -> Dict[str, List[float]]:
        """
        执行训练

        Args:
            epochs: 训练轮数
            initial_epoch: 起始轮数
            steps_per_epoch: 每轮的步数
            validation_steps: 验证步数

        Returns:
            history: 训练历史
        """
        self.logger.info(f"Starting training for {epochs} epochs")

        # 训练开始回调
        self._call_callbacks('on_train_begin', logs={'model': self.model})

        try:
            for epoch in range(initial_epoch, epochs):
                self.current_epoch = epoch

                # Epoch开始回调
                self._call_callbacks('on_epoch_begin', epoch=epoch)

                # 训练阶段
                train_logs = self._train_epoch(steps_per_epoch)

                # 验证阶段
                val_logs = {}
                if self.val_dataset is not None:
                    val_logs = self._validate_epoch(validation_steps)

                # 合并日志（注意：optimizer 对 LR 调度器 callback 是必需的）
                epoch_logs = {
                    **train_logs,
                    **val_logs,
                    'model': self.model,
                    'optimizer': self.optimizer,
                }

                # 记录历史
                self._update_history(epoch_logs)

                # 打印日志
                self._print_epoch_logs(epoch, epoch_logs)

                # Epoch结束回调
                self._call_callbacks('on_epoch_end', epoch=epoch, logs=epoch_logs)

                # 检查是否需要停止训练
                if epoch_logs.get('stop_training', False):
                    self.logger.info("Training stopped by callback")
                    break

        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")

        finally:
            # 训练结束回调
            self._call_callbacks('on_train_end', logs={'model': self.model})

        return self.history

    def _train_epoch(self, steps_per_epoch: Optional[int] = None) -> Dict[str, float]:
        """训练一个epoch"""
        self.logger.info(f"Epoch {self.current_epoch + 1} - Training")

        # 重置指标
        for metric in self.metrics:
            metric.reset_states()

        # 训练循环
        total_loss = 0.0
        num_batches = 0

        dataset = self.train_dataset
        if steps_per_epoch is not None:
            dataset = dataset.take(steps_per_epoch)

        progress_bar = tqdm(
            dataset,
            desc=f"Epoch {self.current_epoch + 1}",
            file=sys.stdout,
            mininterval=1.0,
            dynamic_ncols=True,
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Batch开始回调
            self._call_callbacks('on_batch_begin', batch=batch_idx)

            # 训练步骤
            loss, predictions = self._train_step(batch)

            # 更新指标
            labels = batch[2] if len(batch) > 2 else batch[1]
            for metric in self.metrics:
                metric.update_state(labels, predictions)

            # 累计损失
            total_loss += loss.numpy()
            num_batches += 1

            # 更新进度条
            metrics_str = ' - '.join([
                f"{metric.name}: {metric.result().numpy():.4f}"
                for metric in self.metrics
            ])
            progress_bar.set_postfix_str(
                f"loss: {loss.numpy():.4f} - {metrics_str}"
            )

            # Batch结束回调
            batch_logs = {
                'loss': loss.numpy(),
                'batch': batch_idx
            }
            self._call_callbacks('on_batch_end', batch=batch_idx, logs=batch_logs)

        # 计算平均损失和指标
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        logs = {'train_loss': avg_loss}

        for metric in self.metrics:
            logs[f'train_{metric.name}'] = metric.result().numpy()

        return logs

    @tf.function(reduce_retracing=True)
    def _train_step(self, batch):
        """单个训练步骤"""
        sig1, sig2, labels = batch

        with tf.GradientTape() as tape:
            # 前向传播
            predictions = self.model((sig1, sig2), training=True)

            # 计算损失
            loss = self.loss_fn(labels, predictions)

        # 反向传播
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss, predictions

    def _validate_epoch(self, validation_steps: Optional[int] = None) -> Dict[str, float]:
        """验证一个epoch"""
        self.logger.info(f"Epoch {self.current_epoch + 1} - Validation starting...")
        val_start = time.time()

        # 重置指标
        for metric in self.metrics:
            metric.reset_states()

        # 验证循环
        total_loss = 0.0
        num_batches = 0

        dataset = self.val_dataset
        if validation_steps is not None:
            dataset = dataset.take(validation_steps)

        # file=sys.stdout 配合 mininterval 保证即使 nohup 重定向也能及时刷新进度
        progress_bar = tqdm(
            dataset,
            desc=f"Epoch {self.current_epoch + 1} Val",
            file=sys.stdout,
            mininterval=1.0,
            dynamic_ncols=True,
        )

        for batch_idx, batch in enumerate(progress_bar):
            # 验证步骤
            loss, predictions = self._val_step(batch)

            # 更新指标
            labels = batch[2] if len(batch) > 2 else batch[1]
            for metric in self.metrics:
                metric.update_state(labels, predictions)

            # 累计损失
            total_loss += loss.numpy()
            num_batches += 1

            if batch_idx % 20 == 0:
                progress_bar.set_postfix_str(f"loss: {loss.numpy():.4f}")

        # 计算平均损失和指标
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        logs = {'val_loss': avg_loss}

        for metric in self.metrics:
            logs[f'val_{metric.name}'] = metric.result().numpy()

        elapsed = time.time() - val_start
        self.logger.info(
            f"Epoch {self.current_epoch + 1} - Validation done in {elapsed:.1f}s "
            f"({num_batches} batches, {elapsed / max(num_batches, 1):.2f}s/batch)"
        )

        return logs

    @tf.function(reduce_retracing=True)
    def _val_step(self, batch):
        """单个验证步骤"""
        sig1, sig2, labels = batch

        # 前向传播（不计算梯度）
        predictions = self.model((sig1, sig2), training=False)

        # 计算损失
        loss = self.loss_fn(labels, predictions)

        return loss, predictions

    def _update_history(self, logs: Dict[str, Any]):
        """更新训练历史"""
        for key, value in logs.items():
            if key in self.history and isinstance(value, (int, float)):
                self.history[key].append(value)

    def _print_epoch_logs(self, epoch: int, logs: Dict[str, Any]):
        """打印epoch日志"""
        log_str = f"Epoch {epoch + 1}/{self.config.get('epochs', '?')}"

        for key, value in logs.items():
            if isinstance(value, (int, float)) and key != 'model':
                log_str += f" - {key}: {value:.4f}"

        self.logger.info(log_str)

    def _call_callbacks(self, method: str, **kwargs):
        """调用回调方法"""
        for callback in self.callbacks:
            getattr(callback, method)(**kwargs)

    def save_checkpoint(self, filepath: str):
        """保存检查点"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_weights': [w.numpy() for w in self.model.trainable_variables],
            'optimizer_weights': self.optimizer.get_weights(),
            'epoch': self.current_epoch,
            'history': self.history
        }

        np.savez(filepath, **checkpoint)
        self.logger.info(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        checkpoint = np.load(filepath, allow_pickle=True)

        # 恢复模型权重
        for var, weight in zip(self.model.trainable_variables, checkpoint['model_weights']):
            var.assign(weight)

        # 恢复优化器权重
        self.optimizer.set_weights(checkpoint['optimizer_weights'])

        # 恢复训练状态
        self.current_epoch = int(checkpoint['epoch'])
        self.history = checkpoint['history'].item()

        self.logger.info(f"Checkpoint loaded from {filepath}")
