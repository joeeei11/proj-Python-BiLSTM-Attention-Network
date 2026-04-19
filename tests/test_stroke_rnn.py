"""
StrokeRNN模型测试
"""

import pytest
import numpy as np

# 尝试导入TensorFlow，如果失败则跳过测试
try:
    import tensorflow as tf
    from models.stroke_rnn import StrokeRNN, build_stroke_rnn
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow未安装")


class TestStrokeRNN:
    """StrokeRNN模型测试"""

    def test_model_creation(self):
        """测试模型创建"""
        model = StrokeRNN(
            input_size=23,
            hidden_size=64,
            num_layers=2,
            rnn_type='lstm',
            bidirectional=True,
            dropout=0.3,
            return_sequences=True
        )
        assert model.input_size == 23
        assert model.hidden_size == 64
        assert model.num_layers == 2
        assert model.bidirectional is True
        assert model.output_size == 128  # 64 * 2

    def test_forward_pass_return_sequences(self):
        """测试前向传播（返回序列）"""
        model = StrokeRNN(
            input_size=23,
            hidden_size=64,
            num_layers=2,
            return_sequences=True
        )

        # 创建测试输入
        batch_size = 4
        seq_len = 50
        inputs = tf.random.normal((batch_size, seq_len, 23))

        # 前向传播
        outputs = model(inputs, training=False)

        # 检查输出形状
        assert outputs.shape == (batch_size, seq_len, 128)

    def test_forward_pass_no_sequences(self):
        """测试前向传播（不返回序列）"""
        model = StrokeRNN(
            input_size=23,
            hidden_size=64,
            num_layers=2,
            return_sequences=False
        )

        # 创建测试输入
        batch_size = 4
        seq_len = 50
        inputs = tf.random.normal((batch_size, seq_len, 23))

        # 前向传播
        outputs = model(inputs, training=False)

        # 检查输出形状
        assert outputs.shape == (batch_size, 128)

    def test_masking(self):
        """测试Masking功能"""
        model = StrokeRNN(
            input_size=23,
            hidden_size=64,
            num_layers=1,
            return_sequences=True
        )

        # 创建带padding的输入
        batch_size = 2
        seq_len = 50
        inputs = np.zeros((batch_size, seq_len, 23), dtype=np.float32)

        # 第一个样本：前30个时间步有值
        inputs[0, :30, :] = np.random.randn(30, 23)

        # 第二个样本：前20个时间步有值
        inputs[1, :20, :] = np.random.randn(20, 23)

        inputs = tf.constant(inputs)

        # 前向传播
        outputs = model(inputs, training=False)

        # 检查输出形状
        assert outputs.shape == (batch_size, seq_len, 128)

        # padding部分的输出应该接近0（被mask掉）
        assert np.allclose(outputs[0, 30:, :].numpy(), 0.0, atol=1e-5)
        assert np.allclose(outputs[1, 20:, :].numpy(), 0.0, atol=1e-5)

    def test_gru_type(self):
        """测试GRU类型"""
        model = StrokeRNN(
            input_size=23,
            hidden_size=64,
            num_layers=2,
            rnn_type='gru',
            return_sequences=True
        )

        batch_size = 4
        seq_len = 50
        inputs = tf.random.normal((batch_size, seq_len, 23))

        outputs = model(inputs, training=False)
        assert outputs.shape == (batch_size, seq_len, 128)

    def test_unidirectional(self):
        """测试单向RNN"""
        model = StrokeRNN(
            input_size=23,
            hidden_size=64,
            num_layers=2,
            bidirectional=False,
            return_sequences=True
        )

        assert model.output_size == 64  # 不是双向，所以是64

        batch_size = 4
        seq_len = 50
        inputs = tf.random.normal((batch_size, seq_len, 23))

        outputs = model(inputs, training=False)
        assert outputs.shape == (batch_size, seq_len, 64)

    def test_build_from_config(self):
        """测试从配置构建模型"""
        config = {
            'input_size': 23,
            'hidden_size': 128,
            'num_layers': 2,
            'rnn_type': 'lstm',
            'bidirectional': True,
            'dropout': 0.3,
            'return_sequences': True
        }

        model = build_stroke_rnn(config)

        assert model.input_size == 23
        assert model.hidden_size == 128
        assert model.num_layers == 2

    def test_training_mode(self):
        """测试训练模式"""
        model = StrokeRNN(
            input_size=23,
            hidden_size=64,
            num_layers=2,
            dropout=0.5,
            return_sequences=True
        )

        batch_size = 4
        seq_len = 50
        inputs = tf.random.normal((batch_size, seq_len, 23))

        # 训练模式
        outputs_train = model(inputs, training=True)
        assert outputs_train.shape == (batch_size, seq_len, 128)

        # 推理模式
        outputs_eval = model(inputs, training=False)
        assert outputs_eval.shape == (batch_size, seq_len, 128)

    def test_invalid_rnn_type(self):
        """测试无效的RNN类型"""
        with pytest.raises(ValueError):
            model = StrokeRNN(rnn_type='invalid')
            inputs = tf.random.normal((2, 10, 23))
            model(inputs)

    def test_get_config(self):
        """测试配置序列化"""
        model = StrokeRNN(
            input_size=23,
            hidden_size=64,
            num_layers=2,
            rnn_type='lstm',
            bidirectional=True,
            dropout=0.3,
            return_sequences=True
        )

        config = model.get_config()

        assert config['input_size'] == 23
        assert config['hidden_size'] == 64
        assert config['num_layers'] == 2
        assert config['rnn_type'] == 'lstm'
        assert config['bidirectional'] is True
        assert config['dropout'] == 0.3
        assert config['return_sequences'] is True
