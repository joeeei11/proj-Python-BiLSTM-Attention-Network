"""
数据预处理脚本
提取特征并生成训练/验证/测试列表
"""
import argparse
import sys
from pathlib import Path
from tqdm import tqdm
import json

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import load_signature
from data.utils import (
    split_train_val_test,
    compute_dataset_statistics,
    save_file_list,
    save_statistics,
    visualize_signature
)
from data.pair_sampler import PairSampler
from utils.logger import setup_logger


# SVC2004 Task 2 官方推荐 writer-independent 划分（28/6/6 = 70%/15%/15%）
DEFAULT_TRAIN_USERS = list(range(1, 29))     # user 1-28
DEFAULT_VAL_USERS = list(range(29, 35))      # user 29-34
DEFAULT_TEST_USERS = list(range(35, 41))     # user 35-40


def _parse_user_list(s: str) -> list:
    """解析 '1-28' 或 '1,3,5' 为 int 列表"""
    if not s:
        return []
    s = s.strip()
    if '-' in s and ',' not in s:
        a, b = s.split('-')
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(',') if x.strip()]


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='预处理SVC2004数据集')

    parser.add_argument(
        '--data_root',
        type=str,
        default='./raw_data/SVC2004_Task2',
        help='数据集根目录'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs/features',
        help='输出目录'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.7,
        help='训练集比例'
    )
    parser.add_argument(
        '--val_ratio',
        type=float,
        default=0.15,
        help='验证集比例'
    )
    parser.add_argument(
        '--test_ratio',
        type=float,
        default=0.15,
        help='测试集比例'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子'
    )
    parser.add_argument(
        '--split_mode',
        type=str,
        default='official',
        choices=['official', 'ratio', 'custom'],
        help=('划分模式：'
              'official=官方推荐 user 1-28/29-34/35-40 (28/6/6)； '
              'ratio=按 --train/val/test_ratio 按 user 随机分组 (seed 固定可复现)； '
              'custom=手动指定 --train_users/--val_users/--test_users')
    )
    parser.add_argument('--train_users', type=str, default='',
                        help='custom 模式下的训练 user，格式 "1-28" 或 "1,2,3"')
    parser.add_argument('--val_users', type=str, default='',
                        help='custom 模式下的验证 user')
    parser.add_argument('--test_users', type=str, default='',
                        help='custom 模式下的测试 user')
    parser.add_argument(
        '--pair_ratio',
        type=float,
        default=1.0,
        help='样本对采样时 负/正 比（仅 dry-run 打印统计用）'
    )
    parser.add_argument(
        '--skilled_ratio',
        type=float,
        default=0.8,
        help='负样本对中熟练伪造的比例（其余为随机伪造）'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='可视化部分样本'
    )
    parser.add_argument(
        '--num_visualize',
        type=int,
        default=5,
        help='可视化样本数量'
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 设置日志
    logger = setup_logger('preprocess', log_file='./outputs/logs/preprocess.log')
    logger.info('开始数据预处理')
    logger.info(f'参数: {vars(args)}')

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 查找所有.TXT文件
    data_root = Path(args.data_root)
    txt_files = list(data_root.glob('*.TXT'))

    if len(txt_files) == 0:
        logger.error(f'在 {data_root} 中未找到.TXT文件')
        return

    logger.info(f'找到 {len(txt_files)} 个签名文件')

    # 提取特征并缓存
    logger.info('提取特征...')
    features_list = []
    valid_files = []

    for txt_file in tqdm(txt_files, desc='提取特征'):
        try:
            features = load_signature(str(txt_file), extract_features=True)

            if features.shape[0] < 2:
                logger.warning(f'跳过文件 {txt_file.name}（点数 < 2）')
                continue

            features_list.append(features)
            valid_files.append(str(txt_file))

        except Exception as e:
            logger.error(f'处理文件 {txt_file.name} 时出错: {e}')
            continue

    logger.info(f'成功处理 {len(valid_files)} 个文件')

    # 划分数据集 (writer-independent: 同一 user 的所有文件进同一 split)
    logger.info(f'划分数据集 (split_mode={args.split_mode})...')
    if args.split_mode == 'official':
        train_users, val_users, test_users = (
            DEFAULT_TRAIN_USERS, DEFAULT_VAL_USERS, DEFAULT_TEST_USERS
        )
    elif args.split_mode == 'custom':
        train_users = _parse_user_list(args.train_users)
        val_users = _parse_user_list(args.val_users)
        test_users = _parse_user_list(args.test_users)
        if not (train_users and val_users and test_users):
            raise SystemExit('custom 模式下 --train_users/--val_users/--test_users 必填')
    else:  # ratio
        train_users = val_users = test_users = None  # 交给 split_train_val_test 按比例随机分配 user

    train_files, val_files, test_files = split_train_val_test(
        valid_files,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        train_users=train_users,
        val_users=val_users,
        test_users=test_users,
    )

    # 日志：每 split 的 user 集合
    from data.pair_sampler import group_by_user
    train_uids = sorted(group_by_user(train_files).keys())
    val_uids = sorted(group_by_user(val_files).keys())
    test_uids = sorted(group_by_user(test_files).keys())
    logger.info(f'训练集: {len(train_files)} 文件, {len(train_uids)} user: {train_uids}')
    logger.info(f'验证集: {len(val_files)} 文件, {len(val_uids)} user: {val_uids}')
    logger.info(f'测试集: {len(test_files)} 文件, {len(test_uids)} user: {test_uids}')

    # 核验 writer-independent
    assert not (set(train_uids) & set(val_uids)), 'train 与 val user 有交集'
    assert not (set(train_uids) & set(test_uids)), 'train 与 test user 有交集'
    assert not (set(val_uids) & set(test_uids)), 'val 与 test user 有交集'
    logger.info('✓ writer-independent 检查通过: train/val/test user 集合严格不相交')

    # 保存文件列表
    save_file_list(train_files, output_dir / 'train_list.txt')
    save_file_list(val_files, output_dir / 'val_list.txt')
    save_file_list(test_files, output_dir / 'test_list.txt')
    logger.info('文件列表已保存')

    # Dry-run: 生成样本对并打印统计，校验标签正确性
    logger.info('样本对 dry-run 统计...')
    for name, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        if not files:
            continue
        sampler = PairSampler(
            file_list=files,
            ratio=args.pair_ratio,
            skilled_ratio=args.skilled_ratio,
            seed=args.seed,
        )
        stats = sampler.get_statistics()
        logger.info(
            f'  [{name}] total={stats["total_pairs"]} '
            f'genuine={stats["genuine_pairs"]} '
            f'skilled_forgery={stats["skilled_forgery_pairs"]} '
            f'random_forgery={stats["random_forgery_pairs"]}'
        )

    # 计算统计信息
    logger.info('计算统计信息...')
    stats = compute_dataset_statistics(features_list)
    save_statistics(stats, output_dir / 'statistics.json')
    logger.info(f'数据集统计信息:')
    logger.info(f'  样本数: {stats["num_samples"]}')
    logger.info(f'  总点数: {stats["total_points"]}')
    logger.info(f'  平均长度: {stats["length_mean"]:.2f} ± {stats["length_std"]:.2f}')
    logger.info(f'  长度范围: [{stats["length_min"]}, {stats["length_max"]}]')

    # 可视化部分样本
    if args.visualize:
        logger.info(f'可视化 {args.num_visualize} 个样本...')
        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)

        for i in range(min(args.num_visualize, len(features_list))):
            save_path = vis_dir / f'sample_{i:03d}.png'
            visualize_signature(
                features_list[i],
                save_path=str(save_path),
                show=False
            )

        logger.info(f'可视化结果已保存到 {vis_dir}')

    logger.info('数据预处理完成！')


if __name__ == '__main__':
    main()
