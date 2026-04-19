"""
Stroke-Based RNN模型实现

基于论文: A Stroke-Based RNN for Writer-Independent Online Signature Verification
使用双向LSTM/GRU处理变长笔画序列
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Optional, Tuple


class StrokeRNN(keras.Model):
    """笔画级RNN模型

    处理变长签名序列，提取时序特征

    Args:
        input_size: 输入特征维度（默认23）
        hidden_size: RNN隐藏层大小
        num_layers: RNN层数
        rnn_type: RNN类型 ('lstm' 或 'gru')
        bidirectional: 是否使用双向RNN
        dropout: Dropout比例
        return_sequences: 是否返回所有时间步（用于注意力机制）
    """

    def __init__(
        self,
        input_size: int = 23,
        hidden_size: int = 128,
        num_layers: int = 2,
        rnn_type: str = 'lstm',
        bidirectional: bool = True,
        dropout: float = 0.3,
        return_sequences: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        self.dropout_rate = dropout
        self.return_sequences = return_sequences

        # Masking层：自动处理padding（mask_value=0.0）
        self.masking = layers.Masking(mask_value=0.0)

        # 构建RNN层
        self.rnn_layers = []
        for i in range(num_layers):
            # 选择RNN类型
            if self.rnn_type == 'lstm':
                rnn_layer = layers.LSTM(
                    hidden_size,
                    return_sequences=True if i < num_layers - 1 or return_sequences else False,
                    dropout=dropout if i < num_layers - 1 else 0.0,
                    recurrent_dropout=0.0,  # 避免与CuDNN冲突
                )
            elif self.rnn_type == 'gru':
                rnn_layer = layers.GRU(
                    hidden_size,
                    return_sequences=True if i < num_layers - 1 or return_sequences else False,
                    dropout=dropout if i < num_layers - 1 else 0.0,
                    recurrent_dropout=0.0,
                )
            else:
                raise ValueError(f"不支持的RNN类型: {self.rnn_type}")

            # 双向包装
            if bidirectional:
                rnn_layer = layers.Bidirectional(rnn_layer, merge_mode='concat')

            self.rnn_layers.append(rnn_layer)

        # 最后的Dropout层
        self.final_dropout = layers.Dropout(dropout)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """前向传播

        Args:
            inputs: 输入张量 (batch_size, seq_len, input_size)
            training: 是否训练模式

        Returns:
            如果return_sequences=True: (batch_size, seq_len, hidden_size*2)
            如果return_sequences=False: (batch_size, hidden_size*2)
        """
        x = inputs

        # Masking
        x = self.masking(x)

        # 通过RNN层
        for rnn_layer in self.rnn_layers:
            x = rnn_layer(x, training=training)

        # 最后的Dropout
        x = self.final_dropout(x, training=training)

        return x

    def get_config(self):
        """获取配置用于序列化"""
        config = super().get_config()
        config.update({
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'rnn_type': self.rnn_type,
            'bidirectional': self.bidirectional,
            'dropout': self.dropout_rate,
            'return_sequences': self.return_sequences,
        })
        return config

    @property
    def output_size(self) -> int:
        """输出特征维度"""
        return self.hidden_size * 2 if self.bidirectional else self.hidden_size


def build_stroke_rnn(config: dict) -> StrokeRNN:
    """从配置字典构建StrokeRNN模型

    Args:
        config: 配置字典，包含模型参数

    Returns:
        StrokeRNN模型实例
    """
    return StrokeRNN(
        input_size=config.get('input_size', 23),
        hidden_size=config.get('hidden_size', 128),
        num_layers=config.get('num_layers', 2),
        rnn_type=config.get('rnn_type', 'lstm'),
        bidirectional=config.get('bidirectional', True),
        dropout=config.get('dropout', 0.3),
        return_sequences=config.get('return_sequences', True),
    )
