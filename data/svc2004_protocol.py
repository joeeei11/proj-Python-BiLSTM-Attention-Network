"""
SVC2004 官方 10-trial 评测协议（依照 svc2004说明.pdf §4.1）

核心流程（每 test user 重复 10 trials）：
    1. Enrollment: 从 S1-S10 的真签名里随机抽 5 张作模板
    2. Test samples:
        - 10 positives      : S11-S20 的真签名
        - 20 skilled neg    : S21-S40 的熟练伪造
        - 20 random neg     : 其他 20 个 user 的真签名（每 user 抽 1 张），
                              与 test user 不同且彼此不同
    3. 对每个测试样本 s：
        raw_scores = [ siamese(template_i, s) for i in 5 templates ]
        final_score = aggregate(raw_scores)   # 默认 mean
    4. 分别在 {10 pos, 20 skilled} 与 {10 pos, 20 random} 上算 EER

汇总：
    对所有 (user × trial) 的 EER 取 mean / SD / max，分别对 skilled 和 random
    报告 —— 对齐论文 Table 3 的 "Average / SD / Maximum" 列。

本模块只实现"协议骨架"，不直接依赖 TensorFlow：
    通过 `score_fn: Callable[[np.ndarray, np.ndarray], float]` 或
    `batch_score_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]` 注入打分器，
    测试时用 mock，生产时注入真实 Siamese 模型。
"""
from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from data.pair_sampler import (
    parse_filename,
    group_by_user,
    is_genuine,
    is_skilled_forgery,
    GENUINE_ID_MIN,
    GENUINE_ID_MAX,
    FORGERY_ID_MIN,
    FORGERY_ID_MAX,
)


# === 官方协议常量 ========================================================
ENROLLMENT_SIG_ID_MIN = 1
ENROLLMENT_SIG_ID_MAX = 10
TEST_GENUINE_ID_MIN = 11
TEST_GENUINE_ID_MAX = 20
ENROLLMENT_K = 5
N_TRIALS = 10
N_POS_PER_TRIAL = 10
N_SKILLED_PER_TRIAL = 20
N_RANDOM_PER_TRIAL = 20


# === 类型别名 =============================================================
# 打分器：接收两组同长度 feature 矩阵列表，返回分数数组
# (templates: List[np.ndarray], test: np.ndarray) -> np.ndarray of shape (K,)
ScoreFn = Callable[[Sequence, object], Sequence[float]]


@dataclass
class TrialResult:
    user_id: int
    trial: int
    skilled_eer: Optional[float]   # None 若样本 label 单一化
    random_eer: Optional[float]
    skilled_threshold: Optional[float]
    random_threshold: Optional[float]


@dataclass
class ProtocolReport:
    aggregation: str
    n_trials: int
    enrollment_k: int
    per_trial: List[TrialResult] = field(default_factory=list)

    def _collect(self, attr: str) -> List[float]:
        return [getattr(t, attr) for t in self.per_trial if getattr(t, attr) is not None]

    def summary(self) -> Dict[str, Dict[str, float]]:
        """返回 {skilled: {avg, sd, max}, random: {avg, sd, max}}"""
        out: Dict[str, Dict[str, float]] = {}
        for kind in ('skilled', 'random'):
            values = self._collect(f'{kind}_eer')
            if not values:
                out[kind] = {'avg': float('nan'), 'sd': float('nan'), 'max': float('nan'), 'n': 0}
                continue
            out[kind] = {
                'avg': sum(values) / len(values),
                'sd': statistics.stdev(values) if len(values) > 1 else 0.0,
                'max': max(values),
                'n': len(values),
            }
        return out


# === 协议采样 =============================================================
def select_enrollment(
    user_id: int,
    genuine_files: Sequence[str],
    trial_rng: random.Random,
    k: int = ENROLLMENT_K,
) -> List[str]:
    """从 S1-S10 的真签名里随机抽 k 张作模板"""
    pool = [
        f for f in genuine_files
        if ENROLLMENT_SIG_ID_MIN <= parse_filename(f)[1] <= ENROLLMENT_SIG_ID_MAX
    ]
    if len(pool) < k:
        raise ValueError(
            f"user {user_id}: enrollment pool S{ENROLLMENT_SIG_ID_MIN}-"
            f"S{ENROLLMENT_SIG_ID_MAX} only has {len(pool)} files, cannot sample {k}"
        )
    return trial_rng.sample(pool, k)


def select_test_genuines(user_id: int, genuine_files: Sequence[str]) -> List[str]:
    """S11-S20 的 10 张真签名（固定，不随机）"""
    return sorted(
        f for f in genuine_files
        if TEST_GENUINE_ID_MIN <= parse_filename(f)[1] <= TEST_GENUINE_ID_MAX
    )


def select_test_skilled(user_id: int, forgery_files: Sequence[str]) -> List[str]:
    """S21-S40 的 20 张熟练伪造（固定，不随机）"""
    return sorted(
        f for f in forgery_files
        if FORGERY_ID_MIN <= parse_filename(f)[1] <= FORGERY_ID_MAX
    )


def select_test_random_forgeries(
    user_id: int,
    all_genuine_by_user: Dict[int, List[str]],
    trial_rng: random.Random,
    n: int = N_RANDOM_PER_TRIAL,
) -> List[str]:
    """从其他 n 个 user 的 S1-S20 真签名里每 user 抽 1 张（共 n 张），作为 random forgery"""
    other_users = [uid for uid in all_genuine_by_user.keys() if uid != user_id]
    if len(other_users) < n:
        raise ValueError(
            f"need {n} other users for random forgery, but only {len(other_users)} available"
        )
    chosen_users = trial_rng.sample(other_users, n)
    out = []
    for uid in chosen_users:
        pool = all_genuine_by_user[uid]
        if not pool:
            raise ValueError(f"user {uid} has no genuine signatures in pool")
        out.append(trial_rng.choice(pool))
    return out


# === Aggregation ==========================================================
_AGGREGATORS = {
    'mean': lambda scores: sum(scores) / len(scores),
    'max': lambda scores: max(scores),
    'min': lambda scores: min(scores),
}


def aggregate_template_scores(scores: Sequence[float], how: str = 'mean') -> float:
    if how not in _AGGREGATORS:
        raise ValueError(f"unknown aggregation '{how}', choose from {list(_AGGREGATORS)}")
    return _AGGREGATORS[how](scores)


# === EER helper（stdlib 实现，便于无 sklearn 环境下测试） ================
def compute_eer(y_true: Sequence[int], y_scores: Sequence[float]) -> Tuple[float, float]:
    """
    返回 (eer, threshold)。扫过所有可能阈值，取 |FAR-FRR| 最小处。
    输入需至少各含一个正负样本，否则抛 ValueError。
    """
    y_true = list(y_true)
    y_scores = list(y_scores)
    n_pos = sum(1 for v in y_true if v == 1)
    n_neg = sum(1 for v in y_true if v == 0)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("EER requires both positive and negative samples")

    # 候选阈值：所有独立分数值 + 两端微调
    thresholds = sorted(set(y_scores))
    best = None
    best_thr = thresholds[0]
    for thr in thresholds:
        far = sum(1 for t, s in zip(y_true, y_scores) if t == 0 and s >= thr) / n_neg
        frr = sum(1 for t, s in zip(y_true, y_scores) if t == 1 and s < thr) / n_pos
        diff = abs(far - frr)
        eer = (far + frr) / 2
        if best is None or diff < best[0] or (diff == best[0] and eer < best[1]):
            best = (diff, eer)
            best_thr = thr
    return best[1], best_thr


# === 协议主循环 ===========================================================
def run_protocol(
    test_users: Sequence[int],
    all_files: Sequence[str],
    batch_score_fn: ScoreFn,
    aggregation: str = 'mean',
    n_trials: int = N_TRIALS,
    enrollment_k: int = ENROLLMENT_K,
    seed: int = 42,
    other_users_pool: Optional[Sequence[int]] = None,
) -> ProtocolReport:
    """
    执行 SVC2004 官方 10-trial 协议。

    Args:
        test_users: 要评测的 user id 列表（e.g. [35..40]）
        all_files: 所有可用文件（必须包含 test_users 的全部签名，以及可作
                   random forgery 源的 other_users_pool 的 S1-S20）
        batch_score_fn: (templates: List[feat], test: feat) -> List[float]，
                        返回测试签名对每个模板的原始分数（长度 = len(templates)）
        aggregation: 'mean' / 'max' / 'min'
        n_trials: 默认 10
        enrollment_k: 默认 5
        seed: 主种子，每个 user-trial 派生独立子 seed
        other_users_pool: random forgery 的候选 user 池。默认：all_files 里除
                          test_users 之外的所有 user

    Returns:
        ProtocolReport（含 per_trial 和 summary()）
    """
    # 防御性去重：若调用方传入了重复路径（例如 random pool 与 test split 重叠），
    # group_by_user 后会让每个 user 的文件翻倍，触发 "expected 10 test positives,
    # got 20" 报错。保持插入顺序去重。
    all_files = list(dict.fromkeys(list(all_files)))
    user_groups = group_by_user(all_files)

    # 构造 genuine-only / forgery-only map
    genuine_by_user: Dict[int, List[str]] = {}
    forgery_by_user: Dict[int, List[str]] = {}
    for uid, files in user_groups.items():
        g = [f for f in files if is_genuine(parse_filename(f)[1])]
        k = [f for f in files if is_skilled_forgery(parse_filename(f)[1])]
        if g:
            genuine_by_user[uid] = g
        if k:
            forgery_by_user[uid] = k

    if other_users_pool is None:
        other_users_pool = [uid for uid in genuine_by_user if uid not in set(test_users)]
    other_genuine_map = {uid: genuine_by_user[uid] for uid in other_users_pool if uid in genuine_by_user}

    report = ProtocolReport(aggregation=aggregation, n_trials=n_trials, enrollment_k=enrollment_k)

    for user_id in test_users:
        if user_id not in genuine_by_user or user_id not in forgery_by_user:
            raise ValueError(f"test user {user_id} missing genuine or forgery signatures")
        u_genuine = genuine_by_user[user_id]
        u_forgery = forgery_by_user[user_id]

        test_pos = select_test_genuines(user_id, u_genuine)       # 10
        test_skilled = select_test_skilled(user_id, u_forgery)    # 20
        if len(test_pos) != N_POS_PER_TRIAL:
            raise ValueError(f"user {user_id}: expected {N_POS_PER_TRIAL} test positives, got {len(test_pos)}")
        if len(test_skilled) != N_SKILLED_PER_TRIAL:
            raise ValueError(f"user {user_id}: expected {N_SKILLED_PER_TRIAL} skilled, got {len(test_skilled)}")

        for trial in range(n_trials):
            # 派生独立 trial RNG（便于复现）
            trial_rng = random.Random((seed, user_id, trial).__hash__())

            enrollment = select_enrollment(user_id, u_genuine, trial_rng, enrollment_k)
            test_random = select_test_random_forgeries(
                user_id, other_genuine_map, trial_rng, N_RANDOM_PER_TRIAL
            )

            # 对每个测试样本得到 aggregated score
            def _score_list(test_items: Sequence[str]) -> List[float]:
                out: List[float] = []
                for t in test_items:
                    raw = batch_score_fn(enrollment, t)
                    if len(raw) != len(enrollment):
                        raise ValueError(
                            f"batch_score_fn returned {len(raw)} scores, expected {len(enrollment)}"
                        )
                    out.append(aggregate_template_scores(raw, aggregation))
                return out

            pos_scores = _score_list(test_pos)
            skilled_scores = _score_list(test_skilled)
            random_scores = _score_list(test_random)

            # skilled subset
            y_t_sk = [1] * len(pos_scores) + [0] * len(skilled_scores)
            y_s_sk = pos_scores + skilled_scores
            try:
                eer_sk, thr_sk = compute_eer(y_t_sk, y_s_sk)
            except ValueError:
                eer_sk, thr_sk = None, None

            # random subset
            y_t_rd = [1] * len(pos_scores) + [0] * len(random_scores)
            y_s_rd = pos_scores + random_scores
            try:
                eer_rd, thr_rd = compute_eer(y_t_rd, y_s_rd)
            except ValueError:
                eer_rd, thr_rd = None, None

            report.per_trial.append(TrialResult(
                user_id=user_id, trial=trial,
                skilled_eer=eer_sk, random_eer=eer_rd,
                skilled_threshold=thr_sk, random_threshold=thr_rd,
            ))

    return report
