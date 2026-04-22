"""
样本对生成模块（SVC2004 Task 2）

SVC2004 官方定义（见 svc2004说明.pdf §3.3）：
    文件命名 UxSy.TXT，x=user_id (1..40)，y=signature_id (1..40)
        y ∈ [1, 20]  → 真实签名 (genuine)
        y ∈ [21, 40] → 熟练伪造签名 (skilled forgery)，伪造的正是 user x 的签名

签名验证任务的正负对定义：
    正样本对 (label=1)：同一 user 的两个**真实**签名         (genuine-genuine, 同 user)
    负样本对 (label=0)：
        - 熟练伪造 (skilled forgery)：同一 user 的真签名 vs 熟练伪造该 user 的签名
          (genuine[y∈1..20] vs forgery[y∈21..40], 同 user)  —— 论文主指标
        - 随机伪造 (random forgery)：跨 user 的两个真签名
          (genuineA vs genuineB, A≠B)                       —— 论文次指标

历史 bug：旧版 generate_genuine_pairs 把同 user 所有 40 张文件两两配对都标 1，
         导致 ~76% 的 "正对" 其实含伪造签名；负对只有跨 user 随机对，从未包含
         熟练伪造对。已于 2026-04-22 按官方说明重写。
"""
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import random


# === 官方定义 ============================================================
GENUINE_ID_MIN = 1
GENUINE_ID_MAX = 20
FORGERY_ID_MIN = 21
FORGERY_ID_MAX = 40


def is_genuine(signature_id: int) -> bool:
    """SVC2004: signature_id ∈ [1, 20] 为真实签名"""
    return GENUINE_ID_MIN <= signature_id <= GENUINE_ID_MAX


def is_skilled_forgery(signature_id: int) -> bool:
    """SVC2004: signature_id ∈ [21, 40] 为熟练伪造"""
    return FORGERY_ID_MIN <= signature_id <= FORGERY_ID_MAX


def parse_filename(filename: str) -> Tuple[int, int]:
    """
    解析 SVC2004 文件名，提取 user_id 和 signature_id

    Args:
        filename: 文件名/路径，形如 "U1S1.TXT", "U01S21.TXT", "raw_data/.../U10S5.TXT"

    Returns:
        (user_id, signature_id)
    """
    name = Path(filename).stem
    parts = name.split('S')
    if len(parts) != 2:
        raise ValueError(f"Invalid filename format: {filename}")

    user_id = int(parts[0].replace('U', ''))
    signature_id = int(parts[1])
    return user_id, signature_id


def group_by_user(file_list: List[str]) -> Dict[int, List[str]]:
    """按 user_id 分组"""
    user_groups: Dict[int, List[str]] = {}
    for filepath in file_list:
        try:
            user_id, _ = parse_filename(filepath)
            user_groups.setdefault(user_id, []).append(filepath)
        except ValueError:
            continue
    return user_groups


def split_genuine_forgery(files: List[str]) -> Tuple[List[str], List[str]]:
    """将一组（通常同一 user 的）文件拆为 [真签名列表, 熟练伪造列表]"""
    genuine, forgery = [], []
    for fp in files:
        try:
            _, sid = parse_filename(fp)
        except ValueError:
            continue
        if is_genuine(sid):
            genuine.append(fp)
        elif is_skilled_forgery(sid):
            forgery.append(fp)
    return genuine, forgery


# === 正样本对：同 user 真-真 ============================================
def generate_genuine_pairs(
    file_list: List[str],
    num_pairs: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> List[Tuple[str, str, int]]:
    """
    真签名对：同一 user 的两个真实签名 (signature_id 都 ∈ [1, 20])

    Returns:
        [(file1, file2, 1), ...]
    """
    rng = rng or random
    user_groups = group_by_user(file_list)
    pairs: List[Tuple[str, str, int]] = []

    for _, files in user_groups.items():
        genuine, _ = split_genuine_forgery(files)
        for i in range(len(genuine)):
            for j in range(i + 1, len(genuine)):
                pairs.append((genuine[i], genuine[j], 1))

    if num_pairs is not None and num_pairs < len(pairs):
        pairs = rng.sample(pairs, num_pairs)

    return pairs


# === 负样本对：熟练伪造 (同 user 真-伪) =================================
def generate_skilled_forgery_pairs(
    file_list: List[str],
    num_pairs: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> List[Tuple[str, str, int]]:
    """
    熟练伪造对：同一 user 的真签名 × 熟练伪造签名
    (signature_id 一端 ∈ [1, 20]，另一端 ∈ [21, 40]，同 user)

    Returns:
        [(file_genuine, file_forgery, 0), ...]
    """
    rng = rng or random
    user_groups = group_by_user(file_list)
    pairs: List[Tuple[str, str, int]] = []

    for _, files in user_groups.items():
        genuine, forgery = split_genuine_forgery(files)
        for g in genuine:
            for f in forgery:
                pairs.append((g, f, 0))

    if num_pairs is not None and num_pairs < len(pairs):
        pairs = rng.sample(pairs, num_pairs)

    return pairs


# === 负样本对：随机伪造 (跨 user 真-真) =================================
def generate_random_forgery_pairs(
    file_list: List[str],
    num_pairs: int,
    rng: Optional[random.Random] = None,
) -> List[Tuple[str, str, int]]:
    """
    随机伪造对：跨 user 的两个真签名
    (signature_id 两端都 ∈ [1, 20]，但 user 不同)

    Args:
        file_list: 文件列表
        num_pairs: 需要生成的对数（必填）
        rng: 随机数生成器（用于可复现）

    Returns:
        [(file_userA, file_userB, 0), ...]
    """
    rng = rng or random
    user_groups = group_by_user(file_list)
    # 仅保留真签名，随机伪造概念对伪造签名无意义
    genuine_by_user: Dict[int, List[str]] = {}
    for uid, files in user_groups.items():
        g, _ = split_genuine_forgery(files)
        if g:
            genuine_by_user[uid] = g

    user_ids = list(genuine_by_user.keys())
    if len(user_ids) < 2:
        return []

    pairs: List[Tuple[str, str, int]] = []
    seen = set()  # 去重，避免重复对
    max_attempts = num_pairs * 20
    attempts = 0
    while len(pairs) < num_pairs and attempts < max_attempts:
        u1, u2 = rng.sample(user_ids, 2)
        f1 = rng.choice(genuine_by_user[u1])
        f2 = rng.choice(genuine_by_user[u2])
        key = tuple(sorted([f1, f2]))
        if key in seen:
            attempts += 1
            continue
        seen.add(key)
        pairs.append((f1, f2, 0))
        attempts += 1

    return pairs


# === 背向兼容别名 ======================================================
def generate_forgery_pairs(
    file_list: List[str],
    num_pairs: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> List[Tuple[str, str, int]]:
    """
    【弃用】已被 generate_skilled_forgery_pairs / generate_random_forgery_pairs 取代。
    保留仅为接口兼容：默认按"熟练伪造 80% + 随机伪造 20%"组合返回。
    """
    rng = rng or random
    skilled = generate_skilled_forgery_pairs(file_list, rng=rng)
    if num_pairs is None:
        num_pairs = len(skilled)
    num_skilled = min(int(num_pairs * 0.8), len(skilled))
    num_random = num_pairs - num_skilled
    if num_skilled < len(skilled):
        skilled = rng.sample(skilled, num_skilled)
    random_pairs = generate_random_forgery_pairs(file_list, num_random, rng=rng)
    out = skilled + random_pairs
    rng.shuffle(out)
    return out


# === 样本对采样器 ======================================================
class PairSampler:
    """
    样本对采样器（按官方真伪定义生成）

    生成规则：
        - 正对：同 user 真-真
        - 负对 = skilled_ratio 熟练伪造 + random_ratio 随机伪造
        - |负对| = |正对| * pair_ratio

    参数说明：
        pair_ratio:   负正比（默认 1.0）
        skilled_ratio: 负对里熟练伪造的占比（默认 0.8，论文主指标）
        random_ratio:  = 1 - skilled_ratio，自动推算
    """

    def __init__(
        self,
        file_list: List[str],
        ratio: float = 1.0,
        skilled_ratio: float = 0.8,
        seed: int = 42,
    ):
        if not 0.0 <= skilled_ratio <= 1.0:
            raise ValueError("skilled_ratio must be in [0, 1]")

        self.file_list = file_list
        self.ratio = ratio
        self.skilled_ratio = skilled_ratio
        self.random_ratio = 1.0 - skilled_ratio

        # 独立 RNG，不污染全局 random
        self._rng = random.Random(seed)
        try:
            import numpy as _np  # 延迟导入：保持下游 np.random 的复现性
            _np.random.seed(seed)
        except ImportError:
            pass  # 非训练环境（仅用 sampler 做 smoke test）允许不装 numpy

        # 1) 正对
        self.genuine_pairs = generate_genuine_pairs(file_list, rng=self._rng)

        # 2) 负对
        total_neg = int(len(self.genuine_pairs) * ratio)
        n_skilled = int(total_neg * self.skilled_ratio)
        n_random = total_neg - n_skilled

        skilled_all = generate_skilled_forgery_pairs(file_list, rng=self._rng)
        if n_skilled <= len(skilled_all):
            self.skilled_pairs = self._rng.sample(skilled_all, n_skilled) if n_skilled > 0 else []
        else:
            # 熟练伪造不够时，不循环采样（容易造成过度重复），改为全用 + 补随机
            self.skilled_pairs = skilled_all
            n_random += (n_skilled - len(skilled_all))

        self.random_pairs = generate_random_forgery_pairs(file_list, n_random, rng=self._rng)
        self.forgery_pairs = self.skilled_pairs + self.random_pairs

        # 3) 合并 + shuffle
        self.pairs: List[Tuple[str, str, int]] = list(self.genuine_pairs) + list(self.forgery_pairs)
        self._rng.shuffle(self.pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[str, str, int]:
        return self.pairs[idx]

    def get_statistics(self) -> Dict:
        return {
            'total_pairs': len(self.pairs),
            'genuine_pairs': len(self.genuine_pairs),
            'forgery_pairs': len(self.forgery_pairs),
            'skilled_forgery_pairs': len(self.skilled_pairs),
            'random_forgery_pairs': len(self.random_pairs),
            'pair_ratio': self.ratio,
            'skilled_ratio': self.skilled_ratio,
            'num_users': len(group_by_user(self.file_list)),
            'num_files': len(self.file_list),
        }
