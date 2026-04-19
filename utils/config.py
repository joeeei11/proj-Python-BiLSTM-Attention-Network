"""配置管理模块

负责加载和合并YAML配置文件与命令行参数。
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class Config:
    """配置管理类"""

    def __init__(self, config_dict: Dict[str, Any]):
        """初始化配置对象

        Args:
            config_dict: 配置字典
        """
        self._config = config_dict
        self._update_nested_dict(self.__dict__, config_dict)

    def _update_nested_dict(self, target: Dict, source: Dict):
        """递归更新嵌套字典

        Args:
            target: 目标字典
            source: 源字典
        """
        for key, value in source.items():
            if isinstance(value, dict):
                nested_config = Config(value)
                target[key] = nested_config
            else:
                target[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值

        Args:
            key: 配置键，支持点号分隔的嵌套键 (如 'model.hidden_size')
            default: 默认值

        Returns:
            配置值
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

            if value is None:
                return default

        return value

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典

        Returns:
            配置字典
        """
        return self._config

    def __repr__(self) -> str:
        return f"Config({self._config})"


def load_config(config_path: str) -> Config:
    """从YAML文件加载配置

    Args:
        config_path: 配置文件路径

    Returns:
        Config对象

    Raises:
        FileNotFoundError: 配置文件不存在
        yaml.YAMLError: YAML解析错误
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)


def merge_config(base_config: Config, override_dict: Dict[str, Any]) -> Config:
    """合并配置

    Args:
        base_config: 基础配置
        override_dict: 覆盖配置字典

    Returns:
        合并后的Config对象
    """
    merged_dict = base_config.to_dict().copy()
    _deep_update(merged_dict, override_dict)
    return Config(merged_dict)


def _deep_update(base_dict: Dict, update_dict: Dict):
    """深度更新字典

    Args:
        base_dict: 基础字典（会被修改）
        update_dict: 更新字典
    """
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value


def parse_args() -> argparse.Namespace:
    """解析命令行参数

    Returns:
        命令行参数
    """
    parser = argparse.ArgumentParser(description='签名验证系统')

    # 配置文件
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='配置文件路径')

    # 数据路径
    parser.add_argument('--data_root', type=str, default=None,
                        help='数据集根目录')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出目录')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率')

    # 硬件配置
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU ID')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='数据加载线程数')

    # 其他
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')

    return parser.parse_args()


def get_config(config_path: Optional[str] = None,
               args: Optional[argparse.Namespace] = None) -> Config:
    """获取最终配置

    优先级: 命令行参数 > 配置文件

    Args:
        config_path: 配置文件路径
        args: 命令行参数

    Returns:
        最终配置对象
    """
    # 加载基础配置
    if config_path is None:
        config_path = 'config/default.yaml'

    base_config = load_config(config_path)

    # 如果没有命令行参数，直接返回基础配置
    if args is None:
        return base_config

    # 构建覆盖字典
    override_dict = {}

    if args.data_root is not None:
        override_dict.setdefault('data', {})['root'] = args.data_root

    if args.output_dir is not None:
        override_dict.setdefault('checkpoint', {})['save_dir'] = args.output_dir

    if args.batch_size is not None:
        override_dict.setdefault('training', {})['batch_size'] = args.batch_size

    if args.epochs is not None:
        override_dict.setdefault('training', {})['num_epochs'] = args.epochs

    if args.lr is not None:
        override_dict.setdefault('training', {})['learning_rate'] = args.lr

    if args.gpu is not None:
        override_dict.setdefault('device', {})['gpu_id'] = args.gpu

    if args.num_workers is not None:
        override_dict.setdefault('device', {})['num_workers'] = args.num_workers

    if args.seed is not None:
        override_dict['seed'] = args.seed

    # 合并配置
    return merge_config(base_config, override_dict)
