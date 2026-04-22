"""
孪生网络实现

用于签名验证的孪生网络架构
两个分支共享权重，提取签名特征并计算相似度
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Optional, Tuple


class SiameseNetwork(keras.Model):
    """孪生网络模型

    使用共享权重的双分支结构提取签名特征
    计算两个签名的相似度分数

    Args:
        rnn_config: StrokeRNN配置字典
        attention_hidden_size: 注意力隐藏层大小
        use_attention: 是否使用注意力机制
        distance_metric: 距离度量方式 ('l2', 'cosine', 'concat')
        output_activation: 输出激活函数 ('sigmoid', 'softmax', None)
    """

    def __init__(
        self,
        rnn_config: dict,
        attention_hidden_size: int = 128,
        use_attention: bool = True,
        distance_metric: str = 'l2',
        output_activation: str = 'sigmoid',
        **kwargs
    ):
        super().__init__(**kwargs)

        self.rnn_config = rnn_config
        self.attention_hidden_size = attention_hidden_size
        self.use_attention = use_attention
        self.distance_metric = distance_metric.lower()
        self.output_activation = output_activation

        # 构建共享的特征提取器
        if use_attention:
            from models.attention import AttentionRNN
            self.feature_extractor = AttentionRNN(
                rnn_config=rnn_config,
                attention_hidden_size=attention_hidden_size
            )
        else:
            from models.stroke_rnn import build_stroke_rnn
            # 不使用注意力时，RNN最后一层不返回序列
            rnn_config['return_sequences'] = False
            self.feature_extractor = build_stroke_rnn(rnn_config)

        # 相似度计算层
        if distance_metric == 'concat':
            # 拼接特征后通过全连接层
            feature_size = self.feature_extractor.rnn.output_size
            self.similarity_fc = keras.Sequential([
                layers.Dense(128, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(1, activation=None)
            ])
        else:
            self.similarity_fc = None

        # 输出激活：仅 concat 分支（输出为原始 logit）才需要 sigmoid；
        # l2/cosine 分支已将距离映射到 (0,1]，再过 sigmoid 会把所有输出压到
        # (0.5, 0.731]，导致 BinaryAccuracy 永远判正类，accuracy 锁死在 0.5。
        if distance_metric == 'concat' and output_activation == 'sigmoid':
            self.output_layer = layers.Activation('sigmoid')
        else:
            self.output_layer = None

    def extract_features(
        self,
        inputs: tf.Tensor,
        training: Optional[bool] = None
    ) -> tf.Tensor:
        """提取单个签名的特征向量

        Args:
            inputs: 签名序列 (batch_size, seq_len, input_size)
            training: 是否训练模式

        Returns:
            features: 特征向量 (batch_size, feature_size)
        """
        if self.use_attention:
            features, _ = self.feature_extractor(inputs, training=training)
        else:
            features = self.feature_extractor(inputs, training=training)

        return features

    def compute_similarity(
        self,
        feat1: tf.Tensor,
        feat2: tf.Tensor
    ) -> tf.Tensor:
        """计算两个特征向量的相似度

        Args:
            feat1: 特征向量1 (batch_size, feature_size)
            feat2: 特征向量2 (batch_size, feature_size)

        Returns:
            similarity: 相似度分数 (batch_size, 1)
        """
        if self.distance_metric == 'l2':
            # L2距离（欧氏距离）
            distance = tf.sqrt(tf.reduce_sum(tf.square(feat1 - feat2), axis=1, keepdims=True))
            # 转换为相似度（距离越小，相似度越高）
            similarity = 1.0 / (1.0 + distance)

        elif self.distance_metric == 'cosine':
            # 余弦相似度
            feat1_norm = tf.nn.l2_normalize(feat1, axis=1)
            feat2_norm = tf.nn.l2_normalize(feat2, axis=1)
            similarity = tf.reduce_sum(feat1_norm * feat2_norm, axis=1, keepdims=True)
            # 将[-1, 1]映射到[0, 1]
            similarity = (similarity + 1.0) / 2.0

        elif self.distance_metric == 'concat':
            # 拼接特征后通过全连接层
            concat_features = tf.concat([feat1, feat2, tf.abs(feat1 - feat2)], axis=1)
            similarity = self.similarity_fc(concat_features)

        else:
            raise ValueError(f"不支持的距离度量: {self.distance_metric}")

        return similarity

    def call(
        self,
        inputs: Tuple[tf.Tensor, tf.Tensor],
        training: Optional[bool] = None
    ) -> tf.Tensor:
        """前向传播

        Args:
            inputs: 签名对 (sig1, sig2)
                sig1: (batch_size, seq_len1, input_size)
                sig2: (batch_size, seq_len2, input_size)
            training: 是否训练模式

        Returns:
            similarity: 相似度分数 (batch_size, 1)
        """
        sig1, sig2 = inputs

        # 提取特征（共享权重）
        feat1 = self.extract_features(sig1, training=training)
        feat2 = self.extract_features(sig2, training=training)

        # 计算相似度
        similarity = self.compute_similarity(feat1, feat2)

        # 输出激活
        if self.output_layer is not None:
            similarity = self.output_layer(similarity)

        return similarity

    def get_config(self):
        """获取配置用于序列化"""
        config = super().get_config()
        config.update({
            'rnn_config': self.rnn_config,
            'attention_hidden_size': self.attention_hidden_size,
            'use_attention': self.use_attention,
            'distance_metric': self.distance_metric,
            'output_activation': self.output_activation,
        })
        return config


def build_siamese_network(config: dict) -> SiameseNetwork:
    """从配置字典构建孪生网络

    Args:
        config: 配置字典

    Returns:
        SiameseNetwork模型实例
    """
    rnn_config = {
        'input_size': config.get('input_size', 23),
        'hidden_size': config.get('hidden_size', 128),
        'num_layers': config.get('num_layers', 2),
        'rnn_type': config.get('rnn_type', 'lstm'),
        'bidirectional': config.get('bidirectional', True),
        'dropout': config.get('dropout', 0.3),
    }

    return SiameseNetwork(
        rnn_config=rnn_config,
        attention_hidden_size=config.get('attention_hidden_size', 128),
        use_attention=config.get('use_attention', True),
        distance_metric=config.get('distance_metric', 'l2'),
        output_activation=config.get('output_activation', 'sigmoid'),
    )
