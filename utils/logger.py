"""日志管理模块

提供统一的日志记录功能，支持控制台和文件输出。
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logger(
    name: str = "signature_verification",
    log_dir: Optional[str] = None,
    log_level: str = "INFO",
    console: bool = True,
    log_file: bool = True
) -> logging.Logger:
    """设置日志记录器

    Args:
        name: 日志记录器名称
        log_dir: 日志文件目录
        log_level: 日志级别 (DEBUG/INFO/WARNING/ERROR/CRITICAL)
        console: 是否输出到控制台
        log_file: 是否输出到文件

    Returns:
        配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # 清除已有的处理器
    logger.handlers.clear()

    # 日志格式
    formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 控制台处理器
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 文件处理器
    if log_file and log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # 生成日志文件名（带时间戳）
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file_path = log_dir / f"{name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"日志文件: {log_file_path}")

    return logger


def get_logger(name: str = "signature_verification") -> logging.Logger:
    """获取已存在的日志记录器

    Args:
        name: 日志记录器名称

    Returns:
        日志记录器
    """
    return logging.getLogger(name)


class TensorBoardLogger:
    """TensorBoard日志记录器"""

    def __init__(self, log_dir: str):
        """初始化TensorBoard记录器

        Args:
            log_dir: TensorBoard日志目录
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
            self.enabled = True
        except ImportError:
            logging.warning("TensorBoard未安装，跳过TensorBoard日志")
            self.enabled = False

    def log_scalar(self, tag: str, value: float, step: int):
        """记录标量值

        Args:
            tag: 标签名
            value: 值
            step: 步数
        """
        if self.enabled:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        """记录多个标量值

        Args:
            main_tag: 主标签
            tag_scalar_dict: 标签-值字典
            step: 步数
        """
        if self.enabled:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)

    def log_histogram(self, tag: str, values, step: int):
        """记录直方图

        Args:
            tag: 标签名
            values: 值（数组或张量）
            step: 步数
        """
        if self.enabled:
            self.writer.add_histogram(tag, values, step)

    def log_image(self, tag: str, img_tensor, step: int):
        """记录图像

        Args:
            tag: 标签名
            img_tensor: 图像张量
            step: 步数
        """
        if self.enabled:
            self.writer.add_image(tag, img_tensor, step)

    def log_text(self, tag: str, text: str, step: int):
        """记录文本

        Args:
            tag: 标签名
            text: 文本内容
            step: 步数
        """
        if self.enabled:
            self.writer.add_text(tag, text, step)

    def close(self):
        """关闭记录器"""
        if self.enabled:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
