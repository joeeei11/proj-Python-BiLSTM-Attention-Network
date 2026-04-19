"""
样本对生成测试
"""
import pytest
import sys
from pathlib import Path
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.pair_sampler import (
    parse_filename,
    group_by_user,
    generate_genuine_pairs,
    generate_forgery_pairs,
    balance_pairs,
    PairSampler
)


class TestPairSampler:
    """样本对生成测试"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_files(self, temp_dir):
        """创建示例文件"""
        files = []
        # 用户1: 3个签名
        for i in range(1, 4):
            filepath = Path(temp_dir) / f"U01S0{i}.TXT"
            filepath.write_text("dummy content")
            files.append(str(filepath))

        # 用户2: 2个签名
        for i in range(1, 3):
            filepath = Path(temp_dir) / f"U02S0{i}.TXT"
            filepath.write_text("dummy content")
            files.append(str(filepath))

        return files

    def test_parse_filename(self):
        """测试文件名解析"""
        user_id, sig_id = parse_filename("U01S01.TXT")
        assert user_id == 1
        assert sig_id == 1

        user_id, sig_id = parse_filename("U10S05.TXT")
        assert user_id == 10
        assert sig_id == 5

    def test_parse_filename_invalid(self):
        """测试无效文件名"""
        with pytest.raises(ValueError):
            parse_filename("invalid.TXT")

    def test_group_by_user(self, sample_files):
        """测试按用户分组"""
        groups = group_by_user(sample_files)

        assert len(groups) == 2
        assert 1 in groups
        assert 2 in groups
        assert len(groups[1]) == 3
        assert len(groups[2]) == 2

    def test_generate_genuine_pairs(self, sample_files):
        """测试生成真签名对"""
        pairs = generate_genuine_pairs(sample_files)

        # 用户1: C(3,2)=3对, 用户2: C(2,2)=1对, 总共4对
        assert len(pairs) == 4

        # 检查标签
        for file1, file2, label in pairs:
            assert label == 1
            # 检查是同一用户
            user1, _ = parse_filename(file1)
            user2, _ = parse_filename(file2)
            assert user1 == user2

    def test_generate_forgery_pairs(self, sample_files):
        """测试生成伪造签名对"""
        pairs = generate_forgery_pairs(sample_files, num_pairs=10)

        assert len(pairs) == 10

        # 检查标签
        for file1, file2, label in pairs:
            assert label == 0
            # 检查是不同用户
            user1, _ = parse_filename(file1)
            user2, _ = parse_filename(file2)
            assert user1 != user2

    def test_balance_pairs(self, sample_files):
        """测试平衡样本对"""
        genuine_pairs = generate_genuine_pairs(sample_files)
        forgery_pairs = generate_forgery_pairs(sample_files, num_pairs=10)

        balanced = balance_pairs(genuine_pairs, forgery_pairs, ratio=1.0)

        # 应该有相同数量的真签名对和伪造对
        num_genuine = sum(1 for _, _, label in balanced if label == 1)
        num_forgery = sum(1 for _, _, label in balanced if label == 0)

        assert num_genuine == len(genuine_pairs)
        assert num_forgery == len(genuine_pairs)

    def test_pair_sampler(self, sample_files):
        """测试样本对采样器"""
        sampler = PairSampler(sample_files, ratio=1.0, seed=42)

        assert len(sampler) > 0

        # 测试获取样本对
        file1, file2, label = sampler[0]
        assert isinstance(file1, str)
        assert isinstance(file2, str)
        assert label in [0, 1]

        # 测试统计信息
        stats = sampler.get_statistics()
        assert 'total_pairs' in stats
        assert 'genuine_pairs' in stats
        assert 'forgery_pairs' in stats
        assert 'num_users' in stats
        assert stats['num_users'] == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
