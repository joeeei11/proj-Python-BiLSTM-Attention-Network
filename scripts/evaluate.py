"""
Pair-level 快速评估脚本 (sanity check / 训练期监控)

⚠️  本脚本 *不是* SVC2004 官方协议实现 —— 官方协议定义于 svc2004说明.pdf §4.1：
        5-shot enrollment（每 user 从 S1-S10 随机抽 5 张真签名作模板）
        测试集：10 pos (S11-S20) + 20 skilled (S21-S40) + 20 random (其他 user)
        跑 10 trials，对每个 (user, trial) 独立算 EER，汇报 mean/SD/max
    想得到与论文 Table 3 可对齐的指标，请使用：
        scripts/evaluate_svc2004_protocol.py

本脚本做什么：
    - 对测试集所有 genuine-genuine 组合产生正对
    - 同等数量的 skilled forgery 对和 random forgery 对
    - 对整个 test split 单次扫阈值算 EER
    这是"pair 分类"视角下的快速指标，适合训练期快速验证标签正确性；
    绝不可声称"复现论文 Table 3"。
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import SVC2004Dataset
from data.pair_sampler import (
    generate_genuine_pairs,
    generate_skilled_forgery_pairs,
    generate_random_forgery_pairs,
)
from models.siamese import SiameseNetwork
from utils.config import load_config
from utils.logger import setup_logger, get_logger
from utils.metrics import (
    calculate_eer,
    calculate_metrics_at_threshold,
    plot_roc_curve,
    plot_det_curve,
)
from sklearn.metrics import roc_curve, auc


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Stroke-Based RNN for Signature Verification')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to config file')
    parser.add_argument('--data_root', type=str, default='./raw_data/SVC2004_Task2')
    parser.add_argument('--feature_cache', type=str, default='./outputs/features',
                        help='Path to feature cache dir (必须含 <split>_list.txt)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--output_dir', type=str, default='./outputs/results')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--fixed_threshold', type=float, default=0.5,
                        help='与 EER 阈值并列给出的固定阈值')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def setup_gpu(gpu_id: int):
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


def _fix_length(arr: np.ndarray, max_len: int, feat_dim: int = 23) -> np.ndarray:
    n = arr.shape[0]
    if n >= max_len:
        return arr[:max_len].astype(np.float32)
    out = np.zeros((max_len, feat_dim), dtype=np.float32)
    out[:n] = arr
    return out


def build_pairwise_eval_set(file_list, seed: int):
    """
    Pair-level 评估集（*非* SVC2004 官方协议，仅作训练期 sanity check）：
        genuine  : 同 user 真-真，全部可能对（不下采样）
        skilled  : 同 user 真-伪造，|skilled| = |genuine|
        random   : 跨 user 真-真，|random|  = |genuine|
    Returns:
        pairs: [(f1, f2, label)]
        kinds: list[str] "genuine" / "skilled" / "random"
    """
    import random
    rng = random.Random(seed)

    genuine = generate_genuine_pairs(file_list, rng=rng)
    n_pos = len(genuine)
    if n_pos == 0:
        raise ValueError("No genuine pairs generated; file_list contains no genuine signatures.")

    skilled = generate_skilled_forgery_pairs(file_list, num_pairs=n_pos, rng=rng)
    random_f = generate_random_forgery_pairs(file_list, num_pairs=n_pos, rng=rng)

    pairs, kinds = [], []
    for p in genuine: pairs.append(p); kinds.append('genuine')
    for p in skilled: pairs.append(p); kinds.append('skilled')
    for p in random_f: pairs.append(p); kinds.append('random')

    # 同步 shuffle
    order = list(range(len(pairs)))
    rng.shuffle(order)
    pairs = [pairs[i] for i in order]
    kinds = [kinds[i] for i in order]
    return pairs, kinds


def load_model(checkpoint_path: str, config, max_len: int):
    logger = get_logger(__name__)
    logger.info(f"Loading model from {checkpoint_path}")

    rnn_config = {
        'input_size': 23,
        'hidden_size': config.model.hidden_size,
        'num_layers': config.model.num_layers,
        'dropout': config.model.dropout,
        'bidirectional': config.model.bidirectional,
        'return_sequences': True,
    }
    model = SiameseNetwork(
        rnn_config=rnn_config,
        use_attention=getattr(config.model, 'use_attention',
                              getattr(config.model, 'attention', True)),
        distance_metric=getattr(config.model, 'distance_metric', 'l2'),
    )
    dummy = tf.zeros((1, max_len, 23), dtype=tf.float32)
    model((dummy, dummy), training=False)
    model.load_weights(checkpoint_path)
    logger.info("Model loaded successfully")
    return model


def _eval_subset(y_true, y_scores, fixed_threshold: float):
    if len(np.unique(y_true)) < 2:
        return None
    eer, eer_thr = calculate_eer(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return {
        'n': int(len(y_true)),
        'n_pos': int(y_true.sum()),
        'n_neg': int((1 - y_true).sum()),
        'eer': float(eer),
        'eer_threshold': float(eer_thr),
        'auc': float(auc(fpr, tpr)),
        'metrics_at_eer': calculate_metrics_at_threshold(y_true, y_scores, eer_thr),
        'metrics_at_fixed': calculate_metrics_at_threshold(y_true, y_scores, fixed_threshold),
        'fixed_threshold': float(fixed_threshold),
    }


def main():
    args = parse_args()

    log_dir = Path(args.output_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(log_dir=str(log_dir), log_file=str(log_dir / 'evaluate.log'))
    logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("Pair-level evaluation (sanity check, NOT SVC2004 10-trial protocol)")
    logger.info("For paper-comparable metrics, run scripts/evaluate_svc2004_protocol.py")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Split:      {args.split}")
    logger.info(f"Output dir: {args.output_dir}")

    setup_gpu(args.gpu)
    config = load_config(args.config)
    MAX_LEN = int(getattr(config.training, 'max_seq_len', 400))

    # --- 数据集（hard-fail 若 split list 缺失） ---------------
    dataset = SVC2004Dataset(
        data_root=args.data_root,
        split=args.split,
        feature_cache_dir=args.feature_cache,
    )
    logger.info(f"{args.split} dataset: {len(dataset)} signatures")
    feat_cache = dataset.preload_features(verbose=True)

    # --- pair 级评估对：genuine + skilled + random 各一份（非官方协议） ---
    pairs, kinds = build_pairwise_eval_set(dataset.file_list, seed=args.seed)
    n_gen = kinds.count('genuine')
    n_sk = kinds.count('skilled')
    n_rd = kinds.count('random')
    logger.info(f"Pair-level eval set: genuine={n_gen}, skilled={n_sk}, random={n_rd}, total={len(pairs)}")

    # --- 模型 ---
    model = load_model(args.checkpoint, config, MAX_LEN)

    # --- 批量推理 ---
    y_true = np.empty(len(pairs), dtype=np.int32)
    y_scores = np.empty(len(pairs), dtype=np.float32)
    y_kind = np.array(kinds)

    batch_size = args.batch_size
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        b1 = np.stack([_fix_length(feat_cache[p[0]], MAX_LEN) for p in batch])
        b2 = np.stack([_fix_length(feat_cache[p[1]], MAX_LEN) for p in batch])
        scores = model((tf.constant(b1), tf.constant(b2)), training=False).numpy().flatten()
        for j, (_, _, label) in enumerate(batch):
            y_true[i + j] = int(label)
            y_scores[i + j] = float(scores[j])

    # --- 协议指标 ---------------------------------------------
    genuine_mask = (y_kind == 'genuine')
    skilled_subset = genuine_mask | (y_kind == 'skilled')
    random_subset = genuine_mask | (y_kind == 'random')

    reports = {
        'skilled_forgery': _eval_subset(y_true[skilled_subset], y_scores[skilled_subset], args.fixed_threshold),
        'random_forgery': _eval_subset(y_true[random_subset], y_scores[random_subset], args.fixed_threshold),
        'overall': _eval_subset(y_true, y_scores, args.fixed_threshold),
    }

    # --- 持久化输出 -------------------------------------------
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    np.savez(out / 'predictions.npz', y_true=y_true, y_scores=y_scores, y_kind=y_kind)

    with open(out / 'evaluation_results.json', 'w') as f:
        json.dump(reports, f, indent=2)

    with open(out / 'metrics.txt', 'w') as f:
        f.write("Pair-level evaluation (NOT SVC2004 10-trial protocol; sanity check only)\n")
        f.write("For Table 3-comparable numbers, run scripts/evaluate_svc2004_protocol.py\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"checkpoint: {args.checkpoint}\n")
        f.write(f"split: {args.split}\n")
        f.write(f"fixed_threshold: {args.fixed_threshold}\n\n")
        f.write(f"Pair set: genuine={n_gen}, skilled={n_sk}, random={n_rd}\n\n")
        for tag in ('skilled_forgery', 'random_forgery', 'overall'):
            r = reports[tag]
            if r is None:
                f.write(f"[{tag}] skipped (label imbalance)\n\n"); continue
            f.write(f"--- {tag} ---\n")
            f.write(f"  n={r['n']}  pos={r['n_pos']}  neg={r['n_neg']}\n")
            f.write(f"  AUC: {r['auc']:.4f}\n")
            f.write(f"  EER: {r['eer']:.4f} ({r['eer']:.2%})  thr={r['eer_threshold']:.4f}\n")
            m = r['metrics_at_eer']
            f.write(f"  @EER    Acc={m['accuracy']:.4f}  FAR={m['far']:.4f}  FRR={m['frr']:.4f}  F1={m['f1_score']:.4f}\n")
            m = r['metrics_at_fixed']
            f.write(f"  @{r['fixed_threshold']:.3f}  Acc={m['accuracy']:.4f}  FAR={m['far']:.4f}  FRR={m['frr']:.4f}  F1={m['f1_score']:.4f}\n\n")

    # ROC / DET：分别出图（用于论文）
    for tag, subset_mask in [('skilled', skilled_subset), ('random', random_subset), ('overall', slice(None))]:
        yt = y_true[subset_mask] if not isinstance(subset_mask, slice) else y_true
        ys = y_scores[subset_mask] if not isinstance(subset_mask, slice) else y_scores
        if len(np.unique(yt)) < 2:
            continue
        plot_roc_curve(yt, ys, save_path=str(out / f'roc_{tag}.png'), show=False)
        plot_det_curve(yt, ys, save_path=str(out / f'det_{tag}.png'), show=False)

    # 打印
    print("\n" + "=" * 60)
    for tag in ('skilled_forgery', 'random_forgery', 'overall'):
        r = reports[tag]
        if r is None:
            print(f"[{tag}] skipped"); continue
        print(f"[{tag}]  AUC={r['auc']:.4f}  EER={r['eer']:.2%} (thr={r['eer_threshold']:.3f})  "
              f"Acc@{r['fixed_threshold']:.2f}={r['metrics_at_fixed']['accuracy']:.2%}")
    print("=" * 60)

    logger.info("Evaluation Completed")
    logger.info(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
