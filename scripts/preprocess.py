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
from utils.logger import setup_logger


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

    # 划分数据集
    logger.info('划分数据集...')
    train_files, val_files, test_files = split_train_val_test(
        valid_files,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )

    logger.info(f'训练集: {len(train_files)} 个文件')
    logger.info(f'验证集: {len(val_files)} 个文件')
    logger.info(f'测试集: {len(test_files)} 个文件')

    # 保存文件列表
    save_file_list(train_files, output_dir / 'train_list.txt')
    save_file_list(val_files, output_dir / 'val_list.txt')
    save_file_list(test_files, output_dir / 'test_list.txt')
    logger.info('文件列表已保存')

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
