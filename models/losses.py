"""
损失函数实现

包含签名验证任务的各种损失函数
"""

import tensorflow as tf
from tensorflow import keras
from typing import Optional


class WeightedBinaryCrossentropy(keras.losses.Loss):
    """加权二元交叉熵损失

    用于处理类别不平衡问题

    Args:
        pos_weight: 正样本权重（相对于负样本）
        from_logits: 输入是否为logits（未经sigmoid）
        label_smoothing: 标签平滑系数
    """

    def __init__(
        self,
        pos_weight: float = 1.0,
        from_logits: bool = False,
        label_smoothing: float = 0.0,
        name: str = 'weighted_binary_crossentropy',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.pos_weight = pos_weight
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """计算损失

        Args:
            y_true: 真实标签 (batch_size, 1) 或 (batch_size,)
            y_pred: 预测值 (batch_size, 1) 或 (batch_size,)

        Returns:
            loss: 标量损失值
        """
        # 确保形状一致
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # 展平
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        # 标签平滑
        if self.label_smoothing > 0:
            y_true = y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing

        # 计算加权BCE
        if self.from_logits:
            # 使用TensorFlow内置的加权BCE（支持logits）
            loss = tf.nn.weighted_cross_entropy_with_logits(
                labels=y_true,
                logits=y_pred,
                pos_weight=self.pos_weight
            )
        else:
            # 手动计算加权BCE
            epsilon = tf.keras.backend.epsilon()
            y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

            # BCE = -[y*log(p) + (1-y)*log(1-p)]
            # 加权: -[pos_weight*y*log(p) + (1-y)*log(1-p)]
            loss = -(
                self.pos_weight * y_true * tf.math.log(y_pred) +
                (1.0 - y_true) * tf.math.log(1.0 - y_pred)
            )

        return tf.reduce_mean(loss)

    def get_config(self):
        """获取配置"""
        config = super().get_config()
        config.update({
            'pos_weight': self.pos_weight,
            'from_logits': self.from_logits,
            'label_smoothing': self.label_smoothing,
        })
        return config


class ContrastiveLoss(keras.losses.Loss):
    """对比损失（Contrastive Loss）

    用于度量学习，拉近相似样本，推远不相似样本

    Args:
        margin: 负样本对的距离边界
    """

    def __init__(
        self,
        margin: float = 1.0,
        name: str = 'contrastive_loss',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.margin = margin

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """计算损失

        Args:
            y_true: 真实标签 (batch_size,) 1表示相似，0表示不相似
            y_pred: 预测距离 (batch_size,)

        Returns:
            loss: 标量损失值
        """
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # 展平
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        # 对比损失公式:
        # L = y * d^2 + (1-y) * max(margin - d, 0)^2
        # y=1: 相似对，最小化距离
        # y=0: 不相似对，距离大于margin时损失为0
        positive_loss = y_true * tf.square(y_pred)
        negative_loss = (1.0 - y_true) * tf.square(tf.maximum(self.margin - y_pred, 0.0))

        loss = positive_loss + negative_loss

        return tf.reduce_mean(loss)

    def get_config(self):
        """获取配置"""
        config = super().get_config()
        config.update({'margin': self.margin})
        return config


def focal_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    from_logits: bool = False
) -> tf.Tensor:
    """Focal Loss

    用于处理类别不平衡和难分样本

    Args:
        y_true: 真实标签 (batch_size,)
        y_pred: 预测值 (batch_size,)
        alpha: 平衡因子
        gamma: 聚焦参数
        from_logits: 输入是否为logits

    Returns:
        loss: 标量损失值
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # 展平
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    if from_logits:
        y_pred = tf.nn.sigmoid(y_pred)

    epsilon = tf.keras.backend.epsilon()
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

    # Focal Loss = -alpha * (1-p)^gamma * log(p)  for y=1
    #              -(1-alpha) * p^gamma * log(1-p) for y=0
    positive_loss = -alpha * tf.pow(1.0 - y_pred, gamma) * tf.math.log(y_pred)
    negative_loss = -(1.0 - alpha) * tf.pow(y_pred, gamma) * tf.math.log(1.0 - y_pred)

    loss = y_true * positive_loss + (1.0 - y_true) * negative_loss

    return tf.reduce_mean(loss)


def get_loss_function(loss_type: str, **kwargs) -> keras.losses.Loss:
    """获取损失函数

    Args:
        loss_type: 损失函数类型
        **kwargs: 损失函数参数

    Returns:
        损失函数实例
    """
    loss_type = loss_type.lower()

    if loss_type == 'bce' or loss_type == 'binary_crossentropy':
        return keras.losses.BinaryCrossentropy(
            from_logits=kwargs.get('from_logits', False),
            label_smoothing=kwargs.get('label_smoothing', 0.0)
        )

    elif loss_type == 'weighted_bce':
        return WeightedBinaryCrossentropy(
            pos_weight=kwargs.get('pos_weight', 1.0),
            from_logits=kwargs.get('from_logits', False),
            label_smoothing=kwargs.get('label_smoothing', 0.0)
        )

    elif loss_type == 'contrastive':
        return ContrastiveLoss(margin=kwargs.get('margin', 1.0))

    elif loss_type == 'focal':
        # Focal loss作为函数返回，需要包装
        alpha = kwargs.get('alpha', 0.25)
        gamma = kwargs.get('gamma', 2.0)
        from_logits = kwargs.get('from_logits', False)

        def focal_loss_wrapper(y_true, y_pred):
            return focal_loss(y_true, y_pred, alpha, gamma, from_logits)

        return focal_loss_wrapper

    else:
        raise ValueError(f"不支持的损失函数类型: {loss_type}")
