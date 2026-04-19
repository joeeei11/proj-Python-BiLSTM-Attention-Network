"""
注意力机制实现

实现加性注意力（Additive Attention / Bahdanau Attention）
用于从RNN输出序列中提取重要特征
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Optional


class Attention(layers.Layer):
    """加性注意力层

    计算注意力权重并生成加权上下文向量

    Args:
        hidden_size: 注意力隐藏层大小
        use_bias: 是否使用偏置
    """

    def __init__(self, hidden_size: int = 128, use_bias: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.use_bias = use_bias

        # 注意力权重计算层
        self.W = layers.Dense(hidden_size, use_bias=use_bias, name='attention_W')
        self.v = layers.Dense(1, use_bias=False, name='attention_v')

    def call(
        self,
        inputs: tf.Tensor,
        mask: Optional[tf.Tensor] = None,
        training: Optional[bool] = None
    ) -> tuple:
        """前向传播

        Args:
            inputs: RNN输出 (batch_size, seq_len, hidden_size)
            mask: 序列mask (batch_size, seq_len)
            training: 是否训练模式

        Returns:
            context: 上下文向量 (batch_size, hidden_size)
            attention_weights: 注意力权重 (batch_size, seq_len)
        """
        # 计算注意力分数
        # score = v^T * tanh(W * h)
        score = self.v(tf.nn.tanh(self.W(inputs)))  # (batch_size, seq_len, 1)
        score = tf.squeeze(score, axis=-1)  # (batch_size, seq_len)

        # 应用mask（将padding位置设为极小值）
        if mask is not None:
            # 确保mask是布尔类型
            if mask.dtype != tf.bool:
                mask = tf.cast(mask, tf.bool)

            # 将mask为False的位置设为-inf
            score = tf.where(mask, score, tf.constant(-1e9, dtype=score.dtype))

        # 计算注意力权重（softmax归一化）
        attention_weights = tf.nn.softmax(score, axis=-1)  # (batch_size, seq_len)

        # 加权求和生成上下文向量
        # context = Σ(attention_weights * hidden_states)
        attention_weights_expanded = tf.expand_dims(attention_weights, axis=1)  # (batch_size, 1, seq_len)
        context = tf.matmul(attention_weights_expanded, inputs)  # (batch_size, 1, hidden_size)
        context = tf.squeeze(context, axis=1)  # (batch_size, hidden_size)

        return context, attention_weights

    def get_config(self):
        """获取配置用于序列化"""
        config = super().get_config()
        config.update({
            'hidden_size': self.hidden_size,
            'use_bias': self.use_bias,
        })
        return config


class AttentionRNN(keras.Model):
    """带注意力机制的RNN模型

    组合StrokeRNN和Attention层

    Args:
        rnn_config: StrokeRNN配置
        attention_hidden_size: 注意力隐藏层大小
    """

    def __init__(
        self,
        rnn_config: dict,
        attention_hidden_size: int = 128,
        **kwargs
    ):
        super().__init__(**kwargs)

        # 导入StrokeRNN
        from models.stroke_rnn import build_stroke_rnn

        # 构建RNN（必须return_sequences=True）
        rnn_config['return_sequences'] = True
        self.rnn = build_stroke_rnn(rnn_config)

        # 注意力层
        self.attention = Attention(hidden_size=attention_hidden_size)

    def call(
        self,
        inputs: tf.Tensor,
        training: Optional[bool] = None
    ) -> tuple:
        """前向传播

        Args:
            inputs: 输入序列 (batch_size, seq_len, input_size)
            training: 是否训练模式

        Returns:
            context: 上下文向量 (batch_size, hidden_size)
            attention_weights: 注意力权重 (batch_size, seq_len)
        """
        # RNN编码
        rnn_output = self.rnn(inputs, training=training)  # (batch_size, seq_len, hidden_size)

        # 获取mask（从Masking层）
        mask = self.rnn.masking.compute_mask(inputs)

        # 注意力机制
        context, attention_weights = self.attention(rnn_output, mask=mask, training=training)

        return context, attention_weights

    def get_config(self):
        """获取配置用于序列化"""
        config = super().get_config()
        config.update({
            'rnn_config': self.rnn.get_config(),
            'attention_hidden_size': self.attention.hidden_size,
        })
        return config
