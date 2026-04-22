"""
训练脚本

执行模型训练的主脚本
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
from models.losses import WeightedBinaryCrossentropy
from training.trainer import Trainer
from training.callbacks import (
    EarlyStoppingCallback,
    ModelCheckpointCallback,
    LearningRateSchedulerCallback,
    TensorBoardCallback
)
from utils.config import load_config
from utils.logger import setup_logger, get_logger
from utils.visualization import plot_training_curves


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Train Stroke-Based RNN for Signature Verification')

    # 配置文件
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to config file')

    # 数据参数
    parser.add_argument('--data_root', type=str, default='./raw_data/SVC2004_Task2',
                        help='Path to SVC2004 dataset')
    parser.add_argument('--feature_cache', type=str, default='./outputs/features',
                        help='Path to feature cache directory')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')

    # 模型参数
    parser.add_argument('--hidden_size', type=int, default=None,
                        help='Hidden size of RNN')
    parser.add_argument('--num_layers', type=int, default=None,
                        help='Number of RNN layers')
    parser.add_argument('--dropout', type=float, default=None,
                        help='Dropout rate')

    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Checkpoint directory')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Log directory')

    # 其他参数
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    return parser.parse_args()


def setup_gpu(gpu_id: int):
    """配置GPU"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # 设置可见GPU
            tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
            # 设置内存增长
            tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
            print(f"Using GPU {gpu_id}: {gpus[gpu_id]}")
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPU available, using CPU")


def create_datasets(config: dict, args):
    """创建数据集

    关键优化：
    - 按 split_list 正确加载 train/val/test（修复之前三者共用全部文件的 bug）
    - 训练启动时一次性把所有特征预加载到内存 dict，pair_generator 零 I/O
    - train 集启用 shuffle，避免每个 epoch 固定顺序
    """
    logger = get_logger(__name__)
    logger.info("Creating datasets...")

    # 训练集
    train_dataset = SVC2004Dataset(
        data_root=args.data_root,
        split='train',
        feature_cache_dir=args.feature_cache
    )

    # 验证集
    val_dataset = SVC2004Dataset(
        data_root=args.data_root,
        split='val',
        feature_cache_dir=args.feature_cache
    )

    logger.info(f"Train dataset: {len(train_dataset)} signatures")
    logger.info(f"Val dataset: {len(val_dataset)} signatures")

    # 预加载特征到内存（零 I/O 训练）
    logger.info("Preloading features into memory cache...")
    train_feature_cache = train_dataset.preload_features(verbose=True)
    val_feature_cache = val_dataset.preload_features(verbose=True)
    logger.info(f"Preloaded {len(train_feature_cache)} train + {len(val_feature_cache)} val signatures")

    # 创建样本对生成器
    # pair_ratio: train 侧用 2.0（Plan C：更多负样本对抗过拟合）；val 保持 1.0 用于公平评估
    train_ratio = float(getattr(config.training, 'pair_ratio', 1.0))
    train_sampler = PairSampler(
        file_list=train_dataset.file_list,
        ratio=train_ratio
    )

    val_sampler = PairSampler(
        file_list=val_dataset.file_list,
        ratio=1.0
    )
    logger.info(f"Pair ratio: train={train_ratio}, val=1.0")

    # 生成样本对
    train_pairs = train_sampler.pairs
    val_pairs = val_sampler.pairs

    logger.info(f"Train pairs: {len(train_pairs)}")
    logger.info(f"Val pairs: {len(val_pairs)}")

    # 固定序列长度：数据统计 min=80 max=713 mean=208 p95=384 p99=563
    # 选择 MAX_LEN=400 覆盖 95%+ 样本，消除 tf.function retracing，大幅提速
    MAX_LEN = int(getattr(config.training, 'max_seq_len', 400))
    FEAT_DIM = 23
    logger.info(f"Fixed sequence length: MAX_LEN={MAX_LEN} (truncate longer, pad shorter)")

    import numpy as np

    def _fix_length(arr, max_len=MAX_LEN, feat_dim=FEAT_DIM):
        """截断或零填充到固定长度 (max_len, feat_dim)"""
        n = arr.shape[0]
        if n >= max_len:
            return arr[:max_len].astype(np.float32)
        out = np.zeros((max_len, feat_dim), dtype=np.float32)
        out[:n] = arr
        return out

    # 基于内存特征缓存的 pair 生成器（零 I/O + 固定长度）
    def make_pair_generator(pairs, feature_cache):
        def gen():
            for file1, file2, label in pairs:
                yield (
                    _fix_length(feature_cache[file1]),
                    _fix_length(feature_cache[file2]),
                    label
                )
        return gen

    batch_size = args.batch_size or config.training.batch_size

    # 固定 shape → tf.function 只 trace 一次，消除 retracing 警告
    train_tf_dataset = tf.data.Dataset.from_generator(
        make_pair_generator(train_pairs, train_feature_cache),
        output_signature=(
            tf.TensorSpec(shape=(MAX_LEN, FEAT_DIM), dtype=tf.float32),
            tf.TensorSpec(shape=(MAX_LEN, FEAT_DIM), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    train_tf_dataset = train_tf_dataset.shuffle(
        buffer_size=min(len(train_pairs), 4096), reshuffle_each_iteration=True
    ).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    val_tf_dataset = tf.data.Dataset.from_generator(
        make_pair_generator(val_pairs, val_feature_cache),
        output_signature=(
            tf.TensorSpec(shape=(MAX_LEN, FEAT_DIM), dtype=tf.float32),
            tf.TensorSpec(shape=(MAX_LEN, FEAT_DIM), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    val_tf_dataset = val_tf_dataset.batch(
        batch_size, drop_remainder=True
    ).prefetch(tf.data.AUTOTUNE)

    return train_tf_dataset, val_tf_dataset


def create_model(config: dict, args):
    """创建模型"""
    logger = get_logger(__name__)
    logger.info("Creating model...")

    # 构建RNN配置
    rnn_config = {
        'input_size': 23,
        'hidden_size': args.hidden_size or config.model.hidden_size,
        'num_layers': args.num_layers or config.model.num_layers,
        'dropout': args.dropout or config.model.dropout,
        'bidirectional': config.model.bidirectional,
        'return_sequences': True
    }

    model = SiameseNetwork(
        rnn_config=rnn_config,
        use_attention=config.model.use_attention,
        distance_metric=config.model.distance_metric
    )

    logger.info(f"Model created: {model.__class__.__name__}")
    return model


def create_callbacks(config: dict, args):
    """创建回调"""
    checkpoint_dir = args.checkpoint_dir or os.path.join(args.output_dir, 'checkpoints')
    log_dir = args.log_dir or os.path.join(args.output_dir, 'logs')

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    callbacks = [
        # 早停
        EarlyStoppingCallback(
            monitor='val_loss',
            patience=config.training.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),

        # 模型检查点
        ModelCheckpointCallback(
            filepath=os.path.join(checkpoint_dir, 'best_model.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),

        # 学习率调度
        LearningRateSchedulerCallback(
            schedule='reduce_on_plateau',
            monitor='val_loss',
            factor=getattr(config.training, 'scheduler_factor', 0.5),
            patience=getattr(config.training, 'scheduler_patience', 10),
            min_lr=getattr(config.training, 'scheduler_min_lr', 1e-7),
            verbose=1
        ),

        # TensorBoard
        TensorBoardCallback(
            log_dir=log_dir,
            update_freq='epoch'
        )
    ]

    return callbacks


def main():
    """主函数"""
    # 解析参数
    args = parse_args()

    # 加载配置
    config = load_config(args.config)

    # 设置日志
    log_dir = args.log_dir or os.path.join(args.output_dir, 'logs')
    setup_logger(log_dir=log_dir, log_level='INFO')
    logger = get_logger(__name__)

    logger.info("="*60)
    logger.info("Starting Training")
    logger.info("="*60)

    # 设置随机种子
    tf.random.set_seed(args.seed)

    # 配置GPU
    setup_gpu(args.gpu)

    # 创建数据集
    train_dataset, val_dataset = create_datasets(config, args)

    # 创建模型
    model = create_model(config, args)

    # 加载预训练权重（fine-tune 用）
    resume_path = args.resume or getattr(config.training, 'resume', None)
    if resume_path and resume_path != 'null':
        import numpy as np
        dummy = np.zeros((1, int(getattr(config.training, 'max_seq_len', 400)), 23), dtype='float32')
        _ = model([dummy, dummy])
        model.load_weights(resume_path)
        logger.info(f"Loaded pretrained weights from: {resume_path}")

    # 创建损失函数
    loss_fn = WeightedBinaryCrossentropy(pos_weight=getattr(config.training, 'pos_weight', 1.0))

    # 创建优化器
    lr = args.lr or config.training.learning_rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    # 创建回调
    callbacks = create_callbacks(config, args)

    # 创建训练器
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        optimizer=optimizer,
        loss_fn=loss_fn,
        callbacks=callbacks,
        config=config.training.to_dict() if hasattr(config.training, 'to_dict') else config.training._config
    )

    # 开始训练
    epochs = args.epochs or config.training.epochs
    history = trainer.train(epochs=epochs)

    # 保存训练曲线
    plot_path = os.path.join(args.output_dir, 'training_curves.png')
    plot_training_curves(history, save_path=plot_path, show=False)
    logger.info(f"Training curves saved to {plot_path}")

    logger.info("="*60)
    logger.info("Training Completed")
    logger.info("="*60)


if __name__ == '__main__':
    main()
