"""
孪生网络测试
"""

import pytest
import numpy as np

try:
    import tensorflow as tf
    from models.siamese import SiameseNetwork, build_siamese_network
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow未安装")


class TestSiameseNetwork:
    """孪生网络测试"""

    def test_model_creation_with_attention(self):
        """测试创建带注意力的模型"""
        rnn_config = {
            'input_size': 23,
            'hidden_size': 64,
            'num_layers': 2,
            'rnn_type': 'lstm',
            'bidirectional': True,
            'dropout': 0.3
        }

        model = SiameseNetwork(
            rnn_config=rnn_config,
            attention_hidden_size=128,
            use_attention=True,
            distance_metric='l2',
            output_activation='sigmoid'
        )

        assert model.use_attention is True
        assert model.distance_metric == 'l2'
        assert model.output_activation == 'sigmoid'

    def test_model_creation_without_attention(self):
        """测试创建不带注意力的模型"""
        rnn_config = {
            'input_size': 23,
            'hidden_size': 64,
            'num_layers': 2,
            'rnn_type': 'lstm',
            'bidirectional': True,
            'dropout': 0.3
        }

        model = SiameseNetwork(
            rnn_config=rnn_config,
            use_attention=False,
            distance_metric='cosine',
            output_activation='sigmoid'
        )

        assert model.use_attention is False
        assert model.distance_metric == 'cosine'

    def test_forward_pass_l2_distance(self):
        """测试前向传播（L2距离）"""
        rnn_config = {
            'input_size': 23,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.3
        }

        model = SiameseNetwork(
            rnn_config=rnn_config,
            use_attention=True,
            distance_metric='l2',
            output_activation='sigmoid'
        )

        batch_size = 4
        seq_len1 = 50
        seq_len2 = 60

        sig1 = tf.random.normal((batch_size, seq_len1, 23))
        sig2 = tf.random.normal((batch_size, seq_len2, 23))

        # 前向传播
        similarity = model((sig1, sig2), training=False)

        # 检查输出形状
        assert similarity.shape == (batch_size, 1)

        # 检查输出范围[0, 1]（sigmoid激活）
        assert tf.reduce_all(similarity >= 0.0)
        assert tf.reduce_all(similarity <= 1.0)

    def test_forward_pass_cosine_distance(self):
        """测试前向传播（余弦相似度）"""
        rnn_config = {
            'input_size': 23,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.3
        }

        model = SiameseNetwork(
            rnn_config=rnn_config,
            use_attention=True,
            distance_metric='cosine',
            output_activation='sigmoid'
        )

        batch_size = 4
        seq_len = 50

        sig1 = tf.random.normal((batch_size, seq_len, 23))
        sig2 = tf.random.normal((batch_size, seq_len, 23))

        # 前向传播
        similarity = model((sig1, sig2), training=False)

        # 检查输出形状和范围
        assert similarity.shape == (batch_size, 1)
        assert tf.reduce_all(similarity >= 0.0)
        assert tf.reduce_all(similarity <= 1.0)

    def test_forward_pass_concat_distance(self):
        """测试前向传播（拼接特征）"""
        rnn_config = {
            'input_size': 23,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.3
        }

        model = SiameseNetwork(
            rnn_config=rnn_config,
            use_attention=True,
            distance_metric='concat',
            output_activation='sigmoid'
        )

        batch_size = 4
        seq_len = 50

        sig1 = tf.random.normal((batch_size, seq_len, 23))
        sig2 = tf.random.normal((batch_size, seq_len, 23))

        # 前向传播
        similarity = model((sig1, sig2), training=False)

        # 检查输出形状和范围
        assert similarity.shape == (batch_size, 1)
        assert tf.reduce_all(similarity >= 0.0)
        assert tf.reduce_all(similarity <= 1.0)

    def test_extract_features(self):
        """测试特征提取"""
        rnn_config = {
            'input_size': 23,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.3
        }

        model = SiameseNetwork(
            rnn_config=rnn_config,
            use_attention=True
        )

        batch_size = 4
        seq_len = 50
        inputs = tf.random.normal((batch_size, seq_len, 23))

        # 提取特征
        features = model.extract_features(inputs, training=False)

        # 检查输出形状（双向LSTM: 64*2=128）
        assert features.shape == (batch_size, 128)

    def test_same_signature_high_similarity(self):
        """测试相同签名应该有高相似度"""
        rnn_config = {
            'input_size': 23,
            'hidden_size': 64,
            'num_layers': 1,
            'dropout': 0.0
        }

        model = SiameseNetwork(
            rnn_config=rnn_config,
            use_attention=False,
            distance_metric='l2',
            output_activation='sigmoid'
        )

        batch_size = 2
        seq_len = 50

        # 相同的签名
        sig = tf.random.normal((batch_size, seq_len, 23))

        # 计算相似度
        similarity = model((sig, sig), training=False)

        # 相同签名的相似度应该很高（接近1）
        assert tf.reduce_all(similarity > 0.9)

    def test_variable_length_signatures(self):
        """测试不同长度的签名"""
        rnn_config = {
            'input_size': 23,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.3
        }

        model = SiameseNetwork(
            rnn_config=rnn_config,
            use_attention=True,
            distance_metric='l2'
        )

        batch_size = 4

        # 不同长度的签名
        sig1 = tf.random.normal((batch_size, 30, 23))
        sig2 = tf.random.normal((batch_size, 70, 23))

        # 前向传播
        similarity = model((sig1, sig2), training=False)

        # 检查输出形状
        assert similarity.shape == (batch_size, 1)

    def test_build_from_config(self):
        """测试从配置构建模型"""
        config = {
            'input_size': 23,
            'hidden_size': 128,
            'num_layers': 2,
            'rnn_type': 'lstm',
            'bidirectional': True,
            'dropout': 0.3,
            'attention_hidden_size': 128,
            'use_attention': True,
            'distance_metric': 'l2',
            'output_activation': 'sigmoid'
        }

        model = build_siamese_network(config)

        assert model.use_attention is True
        assert model.distance_metric == 'l2'

        # 测试前向传播
        batch_size = 2
        seq_len = 50
        sig1 = tf.random.normal((batch_size, seq_len, 23))
        sig2 = tf.random.normal((batch_size, seq_len, 23))

        similarity = model((sig1, sig2), training=False)
        assert similarity.shape == (batch_size, 1)

    def test_training_mode(self):
        """测试训练模式"""
        rnn_config = {
            'input_size': 23,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.5
        }

        model = SiameseNetwork(
            rnn_config=rnn_config,
            use_attention=True
        )

        batch_size = 4
        seq_len = 50
        sig1 = tf.random.normal((batch_size, seq_len, 23))
        sig2 = tf.random.normal((batch_size, seq_len, 23))

        # 训练模式
        similarity_train = model((sig1, sig2), training=True)
        assert similarity_train.shape == (batch_size, 1)

        # 推理模式
        similarity_eval = model((sig1, sig2), training=False)
        assert similarity_eval.shape == (batch_size, 1)

    def test_get_config(self):
        """测试配置序列化"""
        rnn_config = {
            'input_size': 23,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.3
        }

        model = SiameseNetwork(
            rnn_config=rnn_config,
            attention_hidden_size=128,
            use_attention=True,
            distance_metric='l2',
            output_activation='sigmoid'
        )

        config = model.get_config()

        assert config['use_attention'] is True
        assert config['distance_metric'] == 'l2'
        assert config['output_activation'] == 'sigmoid'
        assert config['attention_hidden_size'] == 128
