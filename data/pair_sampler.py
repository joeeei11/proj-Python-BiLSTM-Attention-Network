"""
样本对生成模块
用于生成真签名对和伪造签名对
"""
import numpy as np
from typing import List, Tuple, Dict
from pathlib import Path
import random


def parse_filename(filename: str) -> Tuple[int, int]:
    """
    解析SVC2004文件名，提取用户ID和签名ID

    Args:
        filename: 文件名，格式如 "U1S1.TXT" 或 "U01S01.TXT"

    Returns:
        (user_id, signature_id)
    """
    # 移除扩展名
    name = Path(filename).stem

    # 解析格式: U<user_id>S<signature_id>
    parts = name.split('S')
    if len(parts) != 2:
        raise ValueError(f"Invalid filename format: {filename}")

    user_id = int(parts[0].replace('U', ''))
    signature_id = int(parts[1])

    return user_id, signature_id


def group_by_user(file_list: List[str]) -> Dict[int, List[str]]:
    """
    按用户分组文件

    Args:
        file_list: 文件路径列表

    Returns:
        {user_id: [file_paths]} 字典
    """
    user_groups = {}

    for filepath in file_list:
        try:
            user_id, _ = parse_filename(filepath)
            if user_id not in user_groups:
                user_groups[user_id] = []
            user_groups[user_id].append(filepath)
        except ValueError:
            continue

    return user_groups


def generate_genuine_pairs(
    file_list: List[str],
    num_pairs: int = None
) -> List[Tuple[str, str, int]]:
    """
    生成真签名对（同一用户的不同签名）

    Args:
        file_list: 文件路径列表
        num_pairs: 生成的对数，None表示生成所有可能的对

    Returns:
        [(file1, file2, label)] 列表，label=1表示真签名对
    """
    user_groups = group_by_user(file_list)
    pairs = []

    for user_id, files in user_groups.items():
        if len(files) < 2:
            continue

        # 生成该用户所有可能的签名对
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                pairs.append((files[i], files[j], 1))

    # 如果指定了数量，随机采样
    if num_pairs is not None and num_pairs < len(pairs):
        pairs = random.sample(pairs, num_pairs)

    return pairs


def generate_forgery_pairs(
    file_list: List[str],
    num_pairs: int = None
) -> List[Tuple[str, str, int]]:
    """
    生成伪造签名对（不同用户的签名）

    Args:
        file_list: 文件路径列表
        num_pairs: 生成的对数，None表示生成与真签名对相同数量的伪造对

    Returns:
        [(file1, file2, label)] 列表，label=0表示伪造签名对
    """
    user_groups = group_by_user(file_list)
    user_ids = list(user_groups.keys())

    if len(user_ids) < 2:
        return []

    pairs = []

    # 如果未指定数量，生成与真签名对相同数量
    if num_pairs is None:
        genuine_pairs = generate_genuine_pairs(file_list)
        num_pairs = len(genuine_pairs)

    # 随机生成伪造对
    while len(pairs) < num_pairs:
        # 随机选择两个不同的用户
        user1, user2 = random.sample(user_ids, 2)

        # 随机选择每个用户的一个签名
        file1 = random.choice(user_groups[user1])
        file2 = random.choice(user_groups[user2])

        pairs.append((file1, file2, 0))

    return pairs


def balance_pairs(
    genuine_pairs: List[Tuple[str, str, int]],
    forgery_pairs: List[Tuple[str, str, int]],
    ratio: float = 1.0
) -> List[Tuple[str, str, int]]:
    """
    平衡正负样本对

    Args:
        genuine_pairs: 真签名对列表
        forgery_pairs: 伪造签名对列表
        ratio: 负样本/正样本比例

    Returns:
        平衡后的样本对列表
    """
    num_genuine = len(genuine_pairs)
    num_forgery = int(num_genuine * ratio)

    # 如果伪造对不够，随机重复采样
    if num_forgery > len(forgery_pairs):
        forgery_pairs = random.choices(forgery_pairs, k=num_forgery)
    else:
        forgery_pairs = random.sample(forgery_pairs, num_forgery)

    # 合并并打乱
    all_pairs = genuine_pairs + forgery_pairs
    random.shuffle(all_pairs)

    return all_pairs


class PairSampler:
    """
    样本对采样器
    """

    def __init__(
        self,
        file_list: List[str],
        ratio: float = 1.0,
        seed: int = 42
    ):
        """
        初始化采样器

        Args:
            file_list: 文件路径列表
            ratio: 负样本/正样本比例
            seed: 随机种子
        """
        self.file_list = file_list
        self.ratio = ratio
        random.seed(seed)
        np.random.seed(seed)

        # 生成样本对
        self.genuine_pairs = generate_genuine_pairs(file_list)
        self.forgery_pairs = generate_forgery_pairs(file_list,
                                                     num_pairs=int(len(self.genuine_pairs) * ratio))
        self.pairs = balance_pairs(self.genuine_pairs, self.forgery_pairs, ratio)

    def __len__(self) -> int:
        """返回样本对数量"""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[str, str, int]:
        """
        获取单个样本对

        Args:
            idx: 索引

        Returns:
            (file1, file2, label)
        """
        return self.pairs[idx]

    def get_statistics(self) -> Dict:
        """
        获取统计信息

        Returns:
            统计信息字典
        """
        return {
            'total_pairs': len(self.pairs),
            'genuine_pairs': len(self.genuine_pairs),
            'forgery_pairs': len(self.forgery_pairs),
            'ratio': self.ratio,
            'num_users': len(group_by_user(self.file_list)),
            'num_files': len(self.file_list)
        }
