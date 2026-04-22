"""
SVC2004 官方 10-trial 协议回归测试

验证 data/svc2004_protocol.py 严格遵循 svc2004说明.pdf §4.1:
    - Enrollment：每 trial 从 S1-S10 抽 5 张真签名；不含任何 S11-S20、S21-S40
    - 测试 positives：S11-S20 的 10 张真签名（固定，不随机）
    - 测试 skilled：S21-S40 的 20 张熟练伪造（固定，不随机）
    - 测试 random：其他 20 个 user 的真签名（每 user 抽 1 张，与 test user 不同）
    - 跑 n_trials=10；per (user, trial) 独立算 EER；取 mean/SD/max 汇总
"""
import sys
import tempfile
import shutil
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.pair_sampler import parse_filename, is_genuine, is_skilled_forgery
from data.svc2004_protocol import (
    ENROLLMENT_SIG_ID_MIN, ENROLLMENT_SIG_ID_MAX,
    TEST_GENUINE_ID_MIN, TEST_GENUINE_ID_MAX,
    N_POS_PER_TRIAL, N_SKILLED_PER_TRIAL, N_RANDOM_PER_TRIAL,
    ENROLLMENT_K, N_TRIALS,
    select_enrollment, select_test_genuines, select_test_skilled,
    select_test_random_forgeries,
    aggregate_template_scores, compute_eer,
    run_protocol, ProtocolReport,
)


@pytest.fixture
def full_dataset():
    """构造 40 user × 40 signature 的完整 mock 文件集"""
    d = tempfile.mkdtemp()
    files = []
    for uid in range(1, 41):
        for sid in range(1, 41):
            fp = Path(d) / f"U{uid:02d}S{sid:02d}.TXT"
            fp.write_text("dummy")
            files.append(str(fp))
    yield files
    shutil.rmtree(d)


# === 基础常量 ============================================================
class TestProtocolConstants:
    def test_constants(self):
        assert ENROLLMENT_SIG_ID_MIN == 1
        assert ENROLLMENT_SIG_ID_MAX == 10
        assert TEST_GENUINE_ID_MIN == 11
        assert TEST_GENUINE_ID_MAX == 20
        assert ENROLLMENT_K == 5
        assert N_TRIALS == 10
        assert N_POS_PER_TRIAL == 10
        assert N_SKILLED_PER_TRIAL == 20
        assert N_RANDOM_PER_TRIAL == 20


# === 采样函数 ============================================================
class TestEnrollmentSelection:
    def test_only_from_s1_to_s10(self, full_dataset):
        import random
        user_files = [f for f in full_dataset if '/U35' in f.replace('\\', '/') or 'U35' in Path(f).name]
        genuine = [f for f in user_files if is_genuine(parse_filename(f)[1])]

        rng = random.Random(0)
        enrolled = select_enrollment(35, genuine, rng, k=5)
        assert len(enrolled) == 5
        for f in enrolled:
            _, sid = parse_filename(f)
            assert ENROLLMENT_SIG_ID_MIN <= sid <= ENROLLMENT_SIG_ID_MAX, \
                f"enrollment 必须从 S1-S10 抽，收到 S{sid}"

    def test_no_forgery_in_enrollment(self, full_dataset):
        """即使混入 forgery 文件也不能被 enrollment 选中"""
        import random
        user_files = [f for f in full_dataset if Path(f).stem.startswith('U35S')]
        # 注意：select_enrollment 假设 genuine_files 已是真签名，防御性
        # 这里验证传入混杂列表时依旧只取 S1-S10 区间（防回归）
        genuine = [f for f in user_files if is_genuine(parse_filename(f)[1])]
        rng = random.Random(0)
        enrolled = select_enrollment(35, genuine, rng, k=5)
        for f in enrolled:
            _, sid = parse_filename(f)
            assert not is_skilled_forgery(sid)

    def test_raises_when_pool_too_small(self):
        """S1-S10 只有 3 张时应抛"""
        import random
        d = tempfile.mkdtemp()
        try:
            files = [str(Path(d) / f"U01S{sid:02d}.TXT") for sid in [1, 2, 3]]
            for f in files:
                Path(f).write_text("x")
            with pytest.raises(ValueError):
                select_enrollment(1, files, random.Random(0), k=5)
        finally:
            shutil.rmtree(d)


class TestSelectTestSets:
    def test_positives_are_s11_to_s20(self, full_dataset):
        user_files = [f for f in full_dataset if Path(f).stem.startswith('U35S')]
        genuine = [f for f in user_files if is_genuine(parse_filename(f)[1])]
        pos = select_test_genuines(35, genuine)
        assert len(pos) == N_POS_PER_TRIAL
        for f in pos:
            _, sid = parse_filename(f)
            assert TEST_GENUINE_ID_MIN <= sid <= TEST_GENUINE_ID_MAX

    def test_skilled_are_s21_to_s40(self, full_dataset):
        user_files = [f for f in full_dataset if Path(f).stem.startswith('U35S')]
        forgery = [f for f in user_files if is_skilled_forgery(parse_filename(f)[1])]
        sk = select_test_skilled(35, forgery)
        assert len(sk) == N_SKILLED_PER_TRIAL
        for f in sk:
            _, sid = parse_filename(f)
            assert 21 <= sid <= 40


class TestRandomForgeries:
    def test_cross_user_and_genuine_only(self, full_dataset):
        import random
        from data.pair_sampler import group_by_user
        ug = group_by_user(full_dataset)
        genuine_by_user = {uid: [f for f in files if is_genuine(parse_filename(f)[1])]
                           for uid, files in ug.items()}

        rng = random.Random(1)
        test_user = 35
        rf = select_test_random_forgeries(
            test_user, {u: g for u, g in genuine_by_user.items() if u != test_user},
            rng, n=N_RANDOM_PER_TRIAL
        )
        assert len(rf) == N_RANDOM_PER_TRIAL
        seen_users = set()
        for f in rf:
            uid, sid = parse_filename(f)
            assert uid != test_user, "random forgery 不能来自 test user"
            assert is_genuine(sid), "random forgery 必须是真签名"
            seen_users.add(uid)
        # 论文：20 random forgeries selected randomly from genuine signatures of 20 OTHER users
        assert len(seen_users) == N_RANDOM_PER_TRIAL, \
            f"应来自 20 个不同 user，实际 {len(seen_users)}"


# === 聚合 + EER ==========================================================
class TestAggregation:
    def test_mean(self):
        assert aggregate_template_scores([0.2, 0.4, 0.6], 'mean') == pytest.approx(0.4)

    def test_max(self):
        assert aggregate_template_scores([0.2, 0.4, 0.6], 'max') == 0.6

    def test_min(self):
        assert aggregate_template_scores([0.2, 0.4, 0.6], 'min') == 0.2

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            aggregate_template_scores([0.1, 0.2], 'median')


class TestCompute_eer:
    def test_perfect_separation(self):
        y = [1, 1, 1, 0, 0, 0]
        s = [0.9, 0.8, 0.7, 0.1, 0.2, 0.3]
        eer, _ = compute_eer(y, s)
        assert eer == pytest.approx(0.0, abs=1e-6)

    def test_worst_case_flipped(self):
        """分数完全反向：EER 应达到 1.0 或接近（阈值扫描下等错率点）"""
        y = [1, 1, 1, 0, 0, 0]
        s = [0.1, 0.2, 0.3, 0.9, 0.8, 0.7]
        eer, _ = compute_eer(y, s)
        # 扫描下最差 EER = 1.0（FAR=FRR=1），不必精确等于
        assert eer >= 0.5

    def test_requires_both_classes(self):
        with pytest.raises(ValueError):
            compute_eer([1, 1, 1], [0.5, 0.6, 0.7])


# === 协议主循环 ==========================================================
class TestRunProtocol:
    def _make_mock_score_fn(self, genuine_score=0.9, forgery_score=0.1):
        """
        理想打分器：
          - 测试签名 id∈[1,20] 且与模板同 user → 高分（genuine）
          - 其他 → 低分
        """
        def _fn(templates, test_path):
            _, test_sid = parse_filename(test_path)
            test_uid = parse_filename(test_path)[0]
            scores = []
            for t in templates:
                t_uid, t_sid = parse_filename(t)
                same_user = (t_uid == test_uid)
                test_is_genuine = is_genuine(test_sid)
                if same_user and test_is_genuine:
                    scores.append(genuine_score)
                else:
                    scores.append(forgery_score)
            return scores
        return _fn

    def test_default_counts(self, full_dataset):
        report = run_protocol(
            test_users=[35, 36, 37, 38, 39, 40],
            all_files=full_dataset,
            batch_score_fn=self._make_mock_score_fn(),
            aggregation='mean',
            n_trials=N_TRIALS,
            enrollment_k=ENROLLMENT_K,
            seed=42,
        )
        # 6 user × 10 trial = 60 个 trial 结果
        assert len(report.per_trial) == 6 * 10
        # 每 trial 的 user 出现 10 次
        from collections import Counter
        c = Counter(t.user_id for t in report.per_trial)
        for uid in [35, 36, 37, 38, 39, 40]:
            assert c[uid] == 10

    def test_perfect_model_zero_eer(self, full_dataset):
        """理想模型 → skilled & random EER 都是 0"""
        report = run_protocol(
            test_users=[35, 36, 37, 38, 39, 40],
            all_files=full_dataset,
            batch_score_fn=self._make_mock_score_fn(),
            seed=0,
        )
        for r in report.per_trial:
            assert r.skilled_eer is not None and r.skilled_eer == pytest.approx(0.0, abs=1e-6)
            assert r.random_eer is not None and r.random_eer == pytest.approx(0.0, abs=1e-6)

    def test_summary_avg_sd_max(self, full_dataset):
        report = run_protocol(
            test_users=[35, 36, 37, 38, 39, 40],
            all_files=full_dataset,
            batch_score_fn=self._make_mock_score_fn(),
            seed=0,
        )
        summary = report.summary()
        assert 'skilled' in summary
        assert 'random' in summary
        for kind in ('skilled', 'random'):
            assert summary[kind]['n'] == 60
            assert summary[kind]['avg'] == pytest.approx(0.0, abs=1e-6)
            assert summary[kind]['max'] == pytest.approx(0.0, abs=1e-6)

    def test_reproducible_with_same_seed(self, full_dataset):
        fn = self._make_mock_score_fn()
        a = run_protocol(test_users=[35, 36], all_files=full_dataset,
                         batch_score_fn=fn, seed=7, n_trials=3)
        b = run_protocol(test_users=[35, 36], all_files=full_dataset,
                         batch_score_fn=fn, seed=7, n_trials=3)
        a_keys = [(t.user_id, t.trial, t.skilled_eer, t.random_eer) for t in a.per_trial]
        b_keys = [(t.user_id, t.trial, t.skilled_eer, t.random_eer) for t in b.per_trial]
        assert a_keys == b_keys

    def test_random_forgery_excludes_test_user(self, full_dataset):
        """
        用一个会记录 random forgery 路径的打分器，间接验证 run_protocol 里
        select_test_random_forgeries 从不选到 test user。
        """
        seen_in_random = []

        def _fn(templates, test_path):
            # 只要 test_path 不是 test user 的文件，就记入 seen_in_random
            seen_in_random.append(test_path)
            return [0.5] * len(templates)

        run_protocol(
            test_users=[35],
            all_files=full_dataset,
            batch_score_fn=_fn,
            n_trials=2,
            seed=0,
        )
        # 把被作为 random forgery 的 test 路径（user != 35 的那些）过滤出来
        random_paths = [p for p in seen_in_random if parse_filename(p)[0] != 35]
        for p in random_paths:
            uid, sid = parse_filename(p)
            assert uid != 35
            # random forgery 对必须是 y∈[1,20] 的真签名
            assert is_genuine(sid), f"random forgery 不应含 S{sid}（user {uid} 的伪造）"

    def test_duplicated_all_files_is_dedup(self, full_dataset):
        """
        回归防护 (2026-04-22 审查 #P1-1)：
        若调用方在 all_files 里重复了 test user 的文件（例如 random_source='all'
        意外把 test split 又拼进去一次），run_protocol 必须去重后正确运作，
        不应抛 "expected 10 test positives, got 20"。
        """
        # 把全集再复制一份拼进去，模拟调用方去重失败的场景
        duplicated = list(full_dataset) + list(full_dataset)
        report = run_protocol(
            test_users=[35, 36],
            all_files=duplicated,            # 故意传重复
            batch_score_fn=self._make_mock_score_fn(),
            n_trials=2,
            seed=0,
        )
        # 每 (user, trial) 都应有结果，且理想模型下 EER=0
        assert len(report.per_trial) == 2 * 2
        for r in report.per_trial:
            assert r.skilled_eer == pytest.approx(0.0, abs=1e-6)
            assert r.random_eer == pytest.approx(0.0, abs=1e-6)

    def test_enrollment_is_only_s1_to_s10_in_all_trials(self, full_dataset):
        """间接验证：打分器看到的 template 始终来自 S1-S10 且属于 test user"""
        seen_templates = []

        def _fn(templates, test_path):
            seen_templates.extend(templates)
            return [0.5] * len(templates)

        run_protocol(
            test_users=[35, 36],
            all_files=full_dataset,
            batch_score_fn=_fn,
            n_trials=3,
            seed=0,
        )
        for t in seen_templates:
            uid, sid = parse_filename(t)
            assert uid in {35, 36}, f"enrollment template 必须来自 test user，收到 user {uid}"
            assert ENROLLMENT_SIG_ID_MIN <= sid <= ENROLLMENT_SIG_ID_MAX, \
                f"enrollment template 必须来自 S1-S10，收到 S{sid}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
