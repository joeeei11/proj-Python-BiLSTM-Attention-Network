"""
SVC2004数据集加载模块
"""
import os
import pickle
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import warnings

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from data.feature_extractor import extract_temporal_features


def parse_txt_file(filepath: str) -> np.ndarray:
    """
    解析SVC2004的.TXT文件

    Args:
        filepath: .TXT文件路径

    Returns:
        形状为 (N, 6) 的数组，列为 [id, x, y, time, pressure, pen_status]
    """
    points = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            try:
                # SVC2004格式: ID x y time pressure pen_status
                point = [float(p) for p in parts[:6]]
                points.append(point)
            except ValueError:
                continue

    if not points:
        warnings.warn(f"No valid points found in {filepath}")
        return np.array([]).reshape(0, 6)

    return np.array(points, dtype=np.float32)


def load_signature(filepath: str, extract_features: bool = True) -> np.ndarray:
    """
    加载单个签名文件

    Args:
        filepath: 签名文件路径
        extract_features: 是否提取特征（True返回23维特征，False返回原始6维数据）

    Returns:
        特征数组，形状为 (N, 23) 或 (N, 6)
    """
    raw_data = parse_txt_file(filepath)

    if raw_data.shape[0] == 0:
        return np.array([]).reshape(0, 23 if extract_features else 6)

    if not extract_features:
        return raw_data

    # 提取特征
    x = raw_data[:, 1]
    y = raw_data[:, 2]
    time = raw_data[:, 3]
    pressure = raw_data[:, 4]

    features = extract_temporal_features(x, y, pressure, time)
    return features


class SVC2004Dataset:
    """
    SVC2004 Task 2 数据集类
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        feature_cache_dir: Optional[str] = None,
        use_cache: bool = True,
        transform=None,
        split_list_file: Optional[str] = None
    ):
        """
        初始化数据集

        Args:
            data_root: 数据集根目录
            split: 数据集划分 ('train', 'val', 'test', 'all')
            feature_cache_dir: 特征缓存目录
            use_cache: 是否使用缓存
            transform: 数据增强函数
            split_list_file: 显式指定 split 文件列表路径（如 train_list.txt），
                优先级最高。若未指定，则在 feature_cache_dir 下按 <split>_list.txt 查找。
        """
        self.data_root = Path(data_root)
        self.split = split
        self.feature_cache_dir = Path(feature_cache_dir) if feature_cache_dir else None
        self.use_cache = use_cache
        self.transform = transform
        self.split_list_file = Path(split_list_file) if split_list_file else None

        # 加载文件列表
        self.file_list = self._load_file_list()

    def _load_file_list(self) -> List[str]:
        """
        加载文件列表

        优先级：
        1. 显式传入的 split_list_file
        2. feature_cache_dir/<split>_list.txt
        3. data_root/*.TXT（仅当 split=='all' 或未提供任何 list 时）

        Returns:
            文件路径列表
        """
        # 优先级 1：显式指定的 split list 文件
        if self.split_list_file is not None and self.split_list_file.exists():
            from data.utils import load_file_list
            files = load_file_list(str(self.split_list_file))
            if len(files) == 0:
                raise ValueError(f"Split list file is empty: {self.split_list_file}")
            return files

        # 优先级 2：feature_cache_dir 下的 <split>_list.txt
        if self.split != 'all' and self.feature_cache_dir is not None:
            list_file = self.feature_cache_dir / f"{self.split}_list.txt"
            if list_file.exists():
                from data.utils import load_file_list
                files = load_file_list(str(list_file))
                if len(files) == 0:
                    raise ValueError(f"Split list file is empty: {list_file}")
                return files
            # 找不到 list 文件时：发出警告（而非报错），允许测试场景直接用目录下所有 .TXT
            warnings.warn(
                f"Split list file not found: {list_file}. "
                f"Falling back to loading all .TXT under {self.data_root}. "
                f"For real training, run `python scripts/preprocess.py` first."
            )

        # 优先级 3：fallback，加载 data_root 下全部 .TXT
        txt_files = list(self.data_root.glob('*.TXT'))

        if len(txt_files) == 0:
            raise ValueError(f"No .TXT files found in {self.data_root}")

        return [str(f) for f in txt_files]

    def preload_features(self, verbose: bool = True) -> Dict[str, np.ndarray]:
        """
        将所有文件的 23 维特征预加载到内存字典中，供训练期 pair 生成器零 I/O 使用。

        优先从 feature_cache_dir 下的 .pkl 缓存读取；缓存缺失则现场提取并回写。

        Args:
            verbose: 是否打印进度条

        Returns:
            {filepath: features (N, 23)} 字典
        """
        try:
            from tqdm import tqdm
            iterator = tqdm(self.file_list, desc=f"Preloading {self.split}") if verbose else self.file_list
        except ImportError:
            iterator = self.file_list

        cache: Dict[str, np.ndarray] = {}
        for filepath in iterator:
            features = self._load_from_cache(filepath)
            if features is None:
                features = load_signature(filepath, extract_features=True)
                self._save_to_cache(filepath, features)
            cache[filepath] = features
        return cache

    def _get_cache_path(self, filepath: str) -> Path:
        """
        获取缓存文件路径

        Args:
            filepath: 原始文件路径

        Returns:
            缓存文件路径
        """
        filename = Path(filepath).stem
        return self.feature_cache_dir / f"{filename}.pkl"

    def _load_from_cache(self, filepath: str) -> Optional[np.ndarray]:
        """
        从缓存加载特征

        Args:
            filepath: 原始文件路径

        Returns:
            特征数组，如果缓存不存在返回None
        """
        if not self.use_cache or self.feature_cache_dir is None:
            return None

        cache_path = self._get_cache_path(filepath)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                warnings.warn(f"Failed to load cache {cache_path}: {e}")
                return None
        return None

    def _save_to_cache(self, filepath: str, features: np.ndarray):
        """
        保存特征到缓存

        Args:
            filepath: 原始文件路径
            features: 特征数组
        """
        if not self.use_cache or self.feature_cache_dir is None:
            return

        self.feature_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._get_cache_path(filepath)

        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(features, f)
        except Exception as e:
            warnings.warn(f"Failed to save cache {cache_path}: {e}")

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Dict]:
        """
        获取单个样本

        Args:
            idx: 样本索引

        Returns:
            (features, metadata) 元组
        """
        filepath = self.file_list[idx]

        # 尝试从缓存加载
        features = self._load_from_cache(filepath)

        # 如果缓存不存在，则加载并提取特征
        if features is None:
            features = load_signature(filepath, extract_features=True)
            self._save_to_cache(filepath, features)

        # 应用数据增强
        if self.transform is not None:
            features = self.transform(features)

        # 元数据
        metadata = {
            'filepath': filepath,
            'filename': Path(filepath).name,
            'length': len(features)
        }

        return features, metadata


def create_tf_dataset(
    data_root: str,
    split: str = 'train',
    batch_size: int = 32,
    feature_cache_dir: Optional[str] = None,
    shuffle: bool = True,
    transform=None
) -> tf.data.Dataset:
    """
    创建TensorFlow数据集

    Args:
        data_root: 数据集根目录
        split: 数据集划分
        batch_size: 批大小
        feature_cache_dir: 特征缓存目录
        shuffle: 是否打乱
        transform: 数据增强函数

    Returns:
        tf.data.Dataset对象
    """
    dataset = SVC2004Dataset(
        data_root=data_root,
        split=split,
        feature_cache_dir=feature_cache_dir,
        transform=transform
    )

    def generator():
        for i in range(len(dataset)):
            features, metadata = dataset[i]
            yield features, metadata['length']

    # 创建TensorFlow数据集
    tf_dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 23), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )

    if shuffle:
        tf_dataset = tf_dataset.shuffle(buffer_size=1000)

    # Padding和批处理
    tf_dataset = tf_dataset.padded_batch(
        batch_size,
        padded_shapes=([None, 23], []),
        padding_values=(0.0, 0)
    )

    tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)

    return tf_dataset
