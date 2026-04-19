"""
注意力机制测试
"""

import pytest
import numpy as np

try:
    import tensorflow as tf
    from models.attention import Attention, AttentionRNN
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow未安装")


class TestAttention:
    """Attention层测试"""

    def test_attention_creation(self):
        """测试注意力层创建"""
        attention = Attention(hidden_size=128, use_bias=True)
        assert attention.hidden_size == 128
        assert attention.use_bias is True

    def test_attention_forward(self):
        """测试注意力前向传播"""
        attention = Attention(hidden_size=128)

        batch_size = 4
        seq_len = 50
        hidden_size = 256
        inputs = tf.random.normal((batch_size, seq_len, hidden_size))

        # 前向传播
        context, weights = attention(inputs)

        # 检查输出形状
        assert context.shape == (batch_size, hidden_size)
        assert weights.shape == (batch_size, seq_len)

        # 检查注意力权重和为1
        weights_sum = tf.reduce_sum(weights, axis=1)
        assert np.allclose(weights_sum.numpy(), 1.0, atol=1e-5)

    def test_attention_with_mask(self):
        """测试带mask的注意力"""
        attention = Attention(hidden_size=128)

        batch_size = 2
        seq_len = 50
        hidden_size = 256

        # 创建输入
        inputs = tf.random.normal((batch_size, seq_len, hidden_size))

        # 创建mask（前30个为True，后20个为False）
        mask = np.zeros((batch_size, seq_len), dtype=bool)
        mask[:, :30] = True
        mask = tf.constant(mask)

        # 前向传播
        context, weights = attention(inputs, mask=mask)

        # 检查输出形状
        assert context.shape == (batch_size, hidden_size)
        assert weights.shape == (batch_size, seq_len)

        # 检查mask后的权重：后20个应该接近0
        assert np.allclose(weights[:, 30:].numpy(), 0.0, atol=1e-5)

        # 检查前30个权重和为1
        weights_sum = tf.reduce_sum(weights[:, :30], axis=1)
        assert np.allclose(weights_sum.numpy(), 1.0, atol=1e-5)

    def test_attention_get_config(self):
        """测试配置序列化"""
        attention = Attention(hidden_size=128, use_bias=True)
        config = attention.get_config()

        assert config['hidden_size'] == 128
        assert config['use_bias'] is True


class TestAttentionRNN:
    """AttentionRNN模型测试"""

    def test_model_creation(self):
        """测试模型创建"""
        rnn_config = {
            'input_size': 23,
            'hidden_size': 64,
            'num_layers': 2,
            'rnn_type': 'lstm',
            'bidirectional': True,
            'dropout': 0.3
        }

        model = AttentionRNN(
            rnn_config=rnn_config,
            attention_hidden_size=128
        )

        assert model.attention.hidden_size == 128

    def test_forward_pass(self):
        """测试前向传播"""
        rnn_config = {
            'input_size': 23,
            'hidden_size': 64,
            'num_layers': 2,
            'rnn_type': 'lstm',
            'bidirectional': True,
            'dropout': 0.3
        }

        model = AttentionRNN(
            rnn_config=rnn_config,
            attention_hidden_size=128
        )

        batch_size = 4
        seq_len = 50
        inputs = tf.random.normal((batch_size, seq_len, 23))

        # 前向传播
        context, weights = model(inputs, training=False)

        # 检查输出形状
        assert context.shape == (batch_size, 128)  # 64*2 = 128
        assert weights.shape == (batch_size, seq_len)

        # 检查注意力权重和为1
        weights_sum = tf.reduce_sum(weights, axis=1)
        assert np.allclose(weights_sum.numpy(), 1.0, atol=1e-5)

    def test_with_padding(self):
        """测试带padding的输入"""
        rnn_config = {
            'input_size': 23,
            'hidden_size': 64,
            'num_layers': 1,
            'rnn_type': 'lstm',
            'bidirectional': True,
            'dropout': 0.0
        }

        model = AttentionRNN(
            rnn_config=rnn_config,
            attention_hidden_size=128
        )

        batch_size = 2
        seq_len = 50

        # 创建带padding的输入
        inputs = np.zeros((batch_size, seq_len, 23), dtype=np.float32)
        inputs[0, :30, :] = np.random.randn(30, 23)
        inputs[1, :20, :] = np.random.randn(20, 23)
        inputs = tf.constant(inputs)

        # 前向传播
        context, weights = model(inputs, training=False)

        # 检查输出形状
        assert context.shape == (batch_size, 128)
        assert weights.shape == (batch_size, seq_len)

        # padding部分的注意力权重应该接近0
        assert np.allclose(weights[0, 30:].numpy(), 0.0, atol=1e-5)
        assert np.allclose(weights[1, 20:].numpy(), 0.0, atol=1e-5)

    def test_training_mode(self):
        """测试训练模式"""
        rnn_config = {
            'input_size': 23,
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.5
        }

        model = AttentionRNN(
            rnn_config=rnn_config,
            attention_hidden_size=128
        )

        batch_size = 4
        seq_len = 50
        inputs = tf.random.normal((batch_size, seq_len, 23))

        # 训练模式
        context_train, weights_train = model(inputs, training=True)
        assert context_train.shape == (batch_size, 128)

        # 推理模式
        context_eval, weights_eval = model(inputs, training=False)
        assert context_eval.shape == (batch_size, 128)
