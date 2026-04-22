"""
数据集测试
"""
import pytest
import numpy as np
import sys
from pathlib import Path
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import parse_txt_file, load_signature, SVC2004Dataset


class TestDataset:
    """数据集测试"""

    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_txt_file(self, temp_dir):
        """创建示例TXT文件"""
        filepath = Path(temp_dir) / "U01S01.TXT"
        content = """1 100.0 200.0 0.0 0.5 1
2 101.0 201.0 10.0 0.6 1
3 102.0 202.0 20.0 0.7 1
4 103.0 203.0 30.0 0.8 1
5 104.0 204.0 40.0 0.9 1
"""
        filepath.write_text(content)
        return str(filepath)

    def test_parse_txt_file(self, sample_txt_file):
        """测试TXT文件解析"""
        data = parse_txt_file(sample_txt_file)

        assert data.shape == (5, 6)
        assert data.dtype == np.float32
        assert data[0, 0] == 1.0  # ID
        assert data[0, 1] == 100.0  # X
        assert data[0, 2] == 200.0  # Y

    def test_parse_txt_file_invalid(self, temp_dir):
        """测试无效文件"""
        filepath = Path(temp_dir) / "invalid.TXT"
        filepath.write_text("invalid content\n")

        data = parse_txt_file(str(filepath))
        assert data.shape == (0, 6)

    def test_load_signature_raw(self, sample_txt_file):
        """测试加载原始数据"""
        data = load_signature(sample_txt_file, extract_features=False)

        assert data.shape == (5, 6)
        assert data.dtype == np.float32

    def test_load_signature_features(self, sample_txt_file):
        """测试加载特征"""
        features = load_signature(sample_txt_file, extract_features=True)

        assert features.shape == (5, 23)
        assert features.dtype == np.float32
        assert not np.any(np.isnan(features))

    def test_svc2004_dataset(self, temp_dir):
        """测试SVC2004数据集类（split='all' 可直接扫目录，无需 split list）"""
        # 创建多个示例文件
        for i in range(3):
            filepath = Path(temp_dir) / f"U01S0{i+1}.TXT"
            content = f"""1 {100+i}.0 {200+i}.0 0.0 0.5 1
2 {101+i}.0 {201+i}.0 10.0 0.6 1
3 {102+i}.0 {202+i}.0 20.0 0.7 1
"""
            filepath.write_text(content)

        # split='all' 是唯一允许直接扫目录的模式
        dataset = SVC2004Dataset(
            data_root=temp_dir,
            split='all',
            feature_cache_dir=None,
            use_cache=False
        )

        assert len(dataset) == 3

        # 测试获取样本
        features, metadata = dataset[0]
        assert features.shape[1] == 23
        assert 'filepath' in metadata
        assert 'filename' in metadata
        assert 'length' in metadata

    def test_svc2004_dataset_with_cache(self, temp_dir):
        """测试带缓存的数据集（split='all' + feature_cache_dir 仅做缓存，不是 split 来源）"""
        filepath = Path(temp_dir) / "U01S01.TXT"
        content = """1 100.0 200.0 0.0 0.5 1
2 101.0 201.0 10.0 0.6 1
3 102.0 202.0 20.0 0.7 1
"""
        filepath.write_text(content)

        cache_dir = Path(temp_dir) / "cache"

        dataset1 = SVC2004Dataset(
            data_root=temp_dir,
            split='all',
            feature_cache_dir=str(cache_dir),
            use_cache=True
        )
        features1, _ = dataset1[0]

        cache_file = cache_dir / "U01S01.pkl"
        assert cache_file.exists()

        dataset2 = SVC2004Dataset(
            data_root=temp_dir,
            split='all',
            feature_cache_dir=str(cache_dir),
            use_cache=True
        )
        features2, _ = dataset2[0]

        np.testing.assert_array_equal(features1, features2)

    # --- 回归防护：train/val/test 缺 split list 时必须 hard fail --------
    def test_train_split_without_list_hard_fails(self, temp_dir):
        """split='train' 且未提供 split list → 必须抛异常，不得 silent fallback"""
        # 造几个文件仅为 data_root 存在
        (Path(temp_dir) / "U01S01.TXT").write_text("1 1 1 0 0 1\n")
        import pytest as _pytest
        with _pytest.raises((FileNotFoundError, ValueError)):
            SVC2004Dataset(
                data_root=temp_dir,
                split='train',
                feature_cache_dir=None,   # 未提供 cache dir
                use_cache=False,
            )

    def test_val_split_with_missing_list_file_hard_fails(self, temp_dir):
        """split='val' + feature_cache_dir 下没有 val_list.txt → 必须抛异常"""
        (Path(temp_dir) / "U01S01.TXT").write_text("1 1 1 0 0 1\n")
        cache_dir = Path(temp_dir) / "cache"
        cache_dir.mkdir()
        import pytest as _pytest
        with _pytest.raises(FileNotFoundError):
            SVC2004Dataset(
                data_root=temp_dir,
                split='val',
                feature_cache_dir=str(cache_dir),
                use_cache=True,
            )

    def test_explicit_split_list_file_loaded(self, temp_dir):
        """显式传 split_list_file 时应直接读取"""
        f1 = Path(temp_dir) / "U01S01.TXT"
        f1.write_text("1 100 200 0 0.5 1\n2 101 201 10 0.6 1\n")
        list_file = Path(temp_dir) / "mylist.txt"
        list_file.write_text(str(f1) + "\n")
        ds = SVC2004Dataset(
            data_root=temp_dir,
            split='train',
            feature_cache_dir=None,
            split_list_file=str(list_file),
            use_cache=False,
        )
        assert len(ds) == 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
