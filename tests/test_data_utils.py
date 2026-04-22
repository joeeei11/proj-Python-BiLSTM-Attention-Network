"""
data/utils.py 测试 —— 重点验证 writer-independent 划分
"""
import sys
import tempfile
import shutil
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.utils import split_train_val_test
from data.pair_sampler import group_by_user


@pytest.fixture
def svc2004_mock_files():
    """模拟 40 user × 40 signature 的目录"""
    d = tempfile.mkdtemp()
    files = []
    for uid in range(1, 41):
        for sid in range(1, 41):
            fp = Path(d) / f"U{uid:02d}S{sid:02d}.TXT"
            fp.write_text("dummy")
            files.append(str(fp))
    yield files
    shutil.rmtree(d)


class TestWriterIndependentSplit:
    def test_official_default_ratio(self, svc2004_mock_files):
        """默认比例 70/15/15 对应 28/6/6 user"""
        tr, va, te = split_train_val_test(svc2004_mock_files, seed=42)
        tr_u = set(group_by_user(tr).keys())
        va_u = set(group_by_user(va).keys())
        te_u = set(group_by_user(te).keys())

        # writer-independent
        assert tr_u.isdisjoint(va_u)
        assert tr_u.isdisjoint(te_u)
        assert va_u.isdisjoint(te_u)

        # user 数量
        assert len(tr_u) == 28
        assert len(va_u) == 6
        assert len(te_u) == 6

        # 每个 user 的 40 个文件都必须完整落入同一 split
        assert len(tr) == 28 * 40
        assert len(va) == 6 * 40
        assert len(te) == 6 * 40

    def test_explicit_user_list(self, svc2004_mock_files):
        """显式指定 user 1-28/29-34/35-40（官方推荐）"""
        tr, va, te = split_train_val_test(
            svc2004_mock_files,
            train_users=list(range(1, 29)),
            val_users=list(range(29, 35)),
            test_users=list(range(35, 41)),
        )
        assert set(group_by_user(tr).keys()) == set(range(1, 29))
        assert set(group_by_user(va).keys()) == set(range(29, 35))
        assert set(group_by_user(te).keys()) == set(range(35, 41))

    def test_overlap_rejected(self, svc2004_mock_files):
        with pytest.raises(ValueError):
            split_train_val_test(
                svc2004_mock_files,
                train_users=[1, 2, 3],
                val_users=[3, 4],
                test_users=[5],
            )

    def test_reproducibility(self, svc2004_mock_files):
        a = split_train_val_test(svc2004_mock_files, seed=7)
        b = split_train_val_test(svc2004_mock_files, seed=7)
        assert a == b


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
