"""测试模块导入

验证所有核心模块可以正常导入。
"""

import pytest
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_import_utils():
    """测试工具模块导入"""
    try:
        from utils import config, logger
        from utils import Config, load_config, setup_logger, get_logger
        assert True
    except ImportError as e:
        pytest.fail(f"工具模块导入失败: {e}")


def test_import_data():
    """测试数据模块导入"""
    try:
        import data
        assert True
    except ImportError as e:
        pytest.fail(f"数据模块导入失败: {e}")


def test_import_models():
    """测试模型模块导入"""
    try:
        import models
        assert True
    except ImportError as e:
        pytest.fail(f"模型模块导入失败: {e}")


def test_import_training():
    """测试训练模块导入"""
    try:
        import training
        assert True
    except ImportError as e:
        pytest.fail(f"训练模块导入失败: {e}")


def test_config_loading():
    """测试配置加载"""
    try:
        from utils import load_config
        config_path = project_root / "config" / "default.yaml"
        config = load_config(str(config_path))

        # 验证关键配置项存在
        assert config.get('data.root') is not None
        assert config.get('model.hidden_size') is not None
        assert config.get('training.batch_size') is not None

    except Exception as e:
        pytest.fail(f"配置加载失败: {e}")


def test_logger_setup():
    """测试日志设置"""
    try:
        from utils import setup_logger
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = setup_logger(
                name="test_logger",
                log_dir=tmpdir,
                log_level="INFO",
                console=False,
                log_file=True
            )

            logger.info("测试日志消息")
            assert True

    except Exception as e:
        pytest.fail(f"日志设置失败: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
