"""
评估脚本

执行模型评估的主脚本
"""

import os
import sys
import argparse
from pathlib import Path
import tensorflow as tf

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import SVC2004Dataset
from data.pair_sampler import PairSampler
from models.siamese import SiameseNetwork
from training.evaluator import Evaluator
from utils.config import load_config
from utils.logger import setup_logger, get_logger


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Evaluate Stroke-Based RNN for Signature Verification')

    # 模型参数
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to config file')

    # 数据参数
    parser.add_argument('--data_root', type=str, default='./raw_data/SVC2004_Task2',
                        help='Path to SVC2004 dataset')
    parser.add_argument('--feature_cache', type=str, default='./outputs/features',
                        help='Path to feature cache directory')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate')

    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./outputs/results',
                        help='Output directory for results')

    # 其他参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')

    return parser.parse_args()


def setup_gpu(gpu_id: int):
    """配置GPU"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
            print(f"Using GPU {gpu_id}: {gpus[gpu_id]}")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPU available, using CPU")


def create_test_dataset(config: dict, args):
    """创建测试数据集"""
    logger = get_logger(__name__)
    logger.info(f"Creating {args.split} dataset...")

    # 加载数据集
    dataset = SVC2004Dataset(
        data_root=args.data_root,
        split=args.split,
        feature_cache=args.feature_cache
    )

    logger.info(f"{args.split.capitalize()} dataset: {len(dataset)} signatures")

    # 创建样本对生成器
    sampler = PairSampler(
        dataset=dataset,
        num_genuine_per_user=config['data']['num_genuine_per_user'],
        num_forgery_per_user=config['data']['num_forgery_per_user']
    )

    # 生成样本对
    pairs = sampler.generate_pairs()
    logger.info(f"Test pairs: {len(pairs)}")

    # 创建TensorFlow数据集
    def pair_generator():
        for sig1_idx, sig2_idx, label in pairs:
            sig1 = dataset[sig1_idx]['features']
            sig2 = dataset[sig2_idx]['features']
            yield sig1, sig2, label

    test_tf_dataset = tf.data.Dataset.from_generator(
        pair_generator,
        output_signature=(
            tf.TensorSpec(shape=(None, 23), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 23), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    test_tf_dataset = test_tf_dataset.padded_batch(
        args.batch_size,
        padded_shapes=([None, 23], [None, 23], [])
    ).prefetch(tf.data.AUTOTUNE)

    return test_tf_dataset


def load_model(checkpoint_path: str, config: dict):
    """加载模型"""
    logger = get_logger(__name__)
    logger.info(f"Loading model from {checkpoint_path}")

    # 创建模型
    model = SiameseNetwork(
        input_size=23,
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout=config['model']['dropout'],
        bidirectional=config['model']['bidirectional'],
        use_attention=config['model']['use_attention'],
        distance_metric=config['model']['distance_metric']
    )

    # 加载权重
    try:
        model.load_weights(checkpoint_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    return model


def main():
    """主函数"""
    args = parse_args()

    # 设置日志
    log_dir = Path(args.output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(log_dir / 'evaluate.log')
    logger = get_logger(__name__)

    logger.info("="*60)
    logger.info("Starting Evaluation")
    logger.info("="*60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Data root: {args.data_root}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Output dir: {args.output_dir}")

    # 设置GPU
    setup_gpu(args.gpu)

    # 加载配置
    config = load_config(args.config)

    # 创建测试数据集
    test_dataset = create_test_dataset(config, args)

    # 加载模型
    model = load_model(args.checkpoint, config)

    # 创建评估器
    evaluator = Evaluator(
        model=model,
        test_dataset=test_dataset,
        config=config
    )

    # 执行评估
    results = evaluator.evaluate(save_dir=args.output_dir)

    # 打印最终结果
    logger.info("="*60)
    logger.info("Evaluation Completed")
    logger.info("="*60)
    logger.info(f"EER: {results['eer']:.4f}")
    logger.info(f"Accuracy (at EER threshold): {results['eer_metrics']['accuracy']:.4f}")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
