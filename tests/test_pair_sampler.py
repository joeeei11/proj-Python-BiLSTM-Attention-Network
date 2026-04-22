"""
样本对生成测试（按 SVC2004 官方真伪定义）

SVC2004 约定：UxSy.TXT
    y ∈ [1, 20]  → 真实签名 (genuine)
    y ∈ [21, 40] → 熟练伪造 (skilled forgery)
"""
import pytest
import sys
from pathlib import Path
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.pair_sampler import (
    parse_filename,
    is_genuine,
    is_skilled_forgery,
    split_genuine_forgery,
    group_by_user,
    generate_genuine_pairs,
    generate_skilled_forgery_pairs,
    generate_random_forgery_pairs,
    PairSampler,
)


@pytest.fixture
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


@pytest.fixture
def sample_files(temp_dir):
    """
    构造两个 user 的完整 20 真 + 20 伪造样本：
        user 1: S1..S20 真,  S21..S40 伪造
        user 2: 同样
    """
    files = []
    for uid in [1, 2]:
        for sid in range(1, 41):
            fp = Path(temp_dir) / f"U{uid:02d}S{sid:02d}.TXT"
            fp.write_text("dummy")
            files.append(str(fp))
    return files


class TestIdDefinition:
    def test_is_genuine_boundary(self):
        assert is_genuine(1) is True
        assert is_genuine(20) is True
        assert is_genuine(21) is False
        assert is_genuine(0) is False

    def test_is_skilled_forgery_boundary(self):
        assert is_skilled_forgery(21) is True
        assert is_skilled_forgery(40) is True
        assert is_skilled_forgery(20) is False
        assert is_skilled_forgery(41) is False


class TestParseFilename:
    def test_basic(self):
        assert parse_filename("U01S01.TXT") == (1, 1)
        assert parse_filename("U10S05.TXT") == (10, 5)
        assert parse_filename("U40S40.TXT") == (40, 40)

    def test_no_zero_pad(self):
        assert parse_filename("U1S1.TXT") == (1, 1)

    def test_full_path(self):
        assert parse_filename("/data/raw/U03S25.TXT") == (3, 25)

    def test_invalid(self):
        with pytest.raises(ValueError):
            parse_filename("invalid.TXT")


class TestSplitGenuineForgery:
    def test_split(self, sample_files):
        user1_files = [f for f in sample_files if 'U01' in f]
        genuine, forgery = split_genuine_forgery(user1_files)
        assert len(genuine) == 20
        assert len(forgery) == 20
        # 所有 genuine 的 y 都 ∈ [1, 20]
        for fp in genuine:
            _, sid = parse_filename(fp)
            assert 1 <= sid <= 20
        # 所有 forgery 的 y 都 ∈ [21, 40]
        for fp in forgery:
            _, sid = parse_filename(fp)
            assert 21 <= sid <= 40


class TestGenuinePairs:
    def test_only_genuine_same_user(self, sample_files):
        pairs = generate_genuine_pairs(sample_files)
        # 每个 user 20 张真签名 → C(20,2)=190 对，两个 user 共 380
        assert len(pairs) == 2 * 190
        for f1, f2, label in pairs:
            assert label == 1
            u1, s1 = parse_filename(f1)
            u2, s2 = parse_filename(f2)
            # 同 user
            assert u1 == u2, "Genuine pair must be from same user"
            # 两端都是真实签名
            assert 1 <= s1 <= 20, f"{f1} is not genuine"
            assert 1 <= s2 <= 20, f"{f2} is not genuine"

    def test_no_forgery_leaks_into_genuine(self, sample_files):
        pairs = generate_genuine_pairs(sample_files)
        forgery_leak = [
            (f1, f2) for f1, f2, _ in pairs
            if parse_filename(f1)[1] > 20 or parse_filename(f2)[1] > 20
        ]
        assert forgery_leak == [], f"伪造签名被错误放入正对：{forgery_leak}"


class TestSkilledForgeryPairs:
    def test_same_user_cross_region(self, sample_files):
        pairs = generate_skilled_forgery_pairs(sample_files)
        # 每个 user: 20 真 × 20 伪 = 400 对；两个 user 共 800
        assert len(pairs) == 2 * 400
        for f1, f2, label in pairs:
            assert label == 0
            u1, s1 = parse_filename(f1)
            u2, s2 = parse_filename(f2)
            assert u1 == u2, "Skilled forgery 必须同 user"
            # 一端 genuine 一端 forgery
            assert (is_genuine(s1) and is_skilled_forgery(s2)) or \
                   (is_skilled_forgery(s1) and is_genuine(s2))


class TestRandomForgeryPairs:
    def test_cross_user_both_genuine(self, sample_files):
        pairs = generate_random_forgery_pairs(sample_files, num_pairs=50)
        assert len(pairs) == 50
        for f1, f2, label in pairs:
            assert label == 0
            u1, s1 = parse_filename(f1)
            u2, s2 = parse_filename(f2)
            assert u1 != u2, "Random forgery 必须跨 user"
            # 两端都是真签名（random forgery 对伪造签名无意义）
            assert is_genuine(s1)
            assert is_genuine(s2)


class TestPairSampler:
    def test_default(self, sample_files):
        sampler = PairSampler(sample_files, ratio=1.0, skilled_ratio=0.8, seed=42)
        stats = sampler.get_statistics()

        # 正对 = 2 user × C(20,2) = 380
        assert stats['genuine_pairs'] == 380
        # 负对总数 = 正对数 × ratio = 380
        assert stats['forgery_pairs'] == 380
        # skilled ≈ 80%, random ≈ 20%
        assert stats['skilled_forgery_pairs'] == int(380 * 0.8)
        assert stats['skilled_forgery_pairs'] + stats['random_forgery_pairs'] == 380
        assert stats['num_users'] == 2

    def test_label_semantics(self, sample_files):
        """综合断言：每种对的标签和签名 id 区间都一致"""
        sampler = PairSampler(sample_files, ratio=1.0, skilled_ratio=0.5, seed=0)
        for f1, f2, label in sampler.pairs:
            u1, s1 = parse_filename(f1)
            u2, s2 = parse_filename(f2)
            if label == 1:
                # 正对：同 user + 两端都真
                assert u1 == u2
                assert is_genuine(s1) and is_genuine(s2)
            else:
                # 负对：skilled（同 user，真+伪）或 random（跨 user，真+真）
                if u1 == u2:
                    kinds = (is_genuine(s1), is_genuine(s2))
                    assert kinds in [(True, False), (False, True)], \
                        "同 user 的负对只能是 skilled forgery（真 vs 熟练伪造）"
                else:
                    assert is_genuine(s1) and is_genuine(s2), \
                        "跨 user 的负对只能是 random forgery（真 vs 真）"

    def test_reproducible(self, sample_files):
        a = PairSampler(sample_files, ratio=1.0, seed=123).pairs
        b = PairSampler(sample_files, ratio=1.0, seed=123).pairs
        assert a == b

    def test_sampler_does_not_mix_forgery_into_positive(self, sample_files):
        sampler = PairSampler(sample_files, ratio=2.0, skilled_ratio=1.0, seed=7)
        for f1, f2, label in sampler.pairs:
            if label == 1:
                _, s1 = parse_filename(f1)
                _, s2 = parse_filename(f2)
                assert is_genuine(s1) and is_genuine(s2), \
                    "历史 bug 回归：伪造签名不可出现在正对"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
