"""
Pair-level 批量推理脚本 (sanity check, 训练期快速监控用)

⚠️  *不是* SVC2004 官方 10-trial 协议实现。官方协议 (svc2004说明.pdf §4.1):
        每 user 从 S1-S10 抽 5 张做 enrollment；测试 S11-S20/S21-S40/20 其他 user 真签；
        跑 10 trials，按 (user, trial) 独立算 EER，汇报 mean/SD/max。
    对应实现见 scripts/evaluate_svc2004_protocol.py。本脚本输出不应作为
    与论文 Table 3 的对比依据。

本脚本做什么：
    - 同 user 真-真全部两两组合当正对
    - 同等数量 skilled forgery 与 random forgery 负对
    - 单次阈值扫描算 EER；分别报告 skilled / random / overall 三套

用法:
    python scripts/batch_inference.py \
        --checkpoint outputs/checkpoints/<your_new_model>.h5 \
        --test_list outputs/features/test_list.txt \
        --data_root raw_data/SVC2004_Task2 \
        --output_dir outputs/results/batch_eval

输出:
    - results.csv         每对签名的分数、标签、类型(kind)
    - metrics.txt         三类分别的 EER/AUC/FAR/FRR（非 Table 3 口径）
    - roc_curve.png       ROC曲线（pair-level）
    - det_curve.png       DET曲线（pair-level）
"""

import argparse
import os
import sys
import random
import csv
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf

from data.feature_extractor import load_signature_txt, extract_temporal_features
from data.pair_sampler import (
    group_by_user,
    generate_genuine_pairs,
    generate_skilled_forgery_pairs,
    generate_random_forgery_pairs,
)
from utils.metrics import (
    calculate_eer, calculate_metrics_at_threshold,
    plot_roc_curve, plot_det_curve
)
from sklearn.metrics import roc_curve, auc

MAX_LEN = 400

MODEL_CONFIG = {
    'input_size': 23,
    'hidden_size': 256,
    'num_layers': 2,
    'rnn_type': 'lstm',
    'bidirectional': True,
    'dropout': 0.2,
    'use_attention': True,
    'attention_hidden_size': 128,
    'distance_metric': 'concat',
    'output_activation': 'sigmoid',
}


def fix_length(features: np.ndarray) -> np.ndarray:
    n = features.shape[0]
    if n >= MAX_LEN:
        return features[:MAX_LEN]
    return np.vstack([features, np.zeros((MAX_LEN - n, features.shape[1]), dtype=np.float32)])


def load_and_extract(filepath: str) -> np.ndarray:
    raw = load_signature_txt(filepath)
    x, y, t, p = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
    return fix_length(extract_temporal_features(x, y, p, time=t))


def build_model(checkpoint: str) -> tf.keras.Model:
    from models.siamese import build_siamese_network
    model = build_siamese_network(MODEL_CONFIG)
    dummy = tf.zeros((1, MAX_LEN, 23))
    model((dummy, dummy), training=False)
    model.load_weights(checkpoint)
    return model


def generate_test_pairs(file_list: list, seed: int = 42):
    """
    Pair-level 评估对（*非* SVC2004 官方 10-trial 协议），仅作快速 sanity check：
        - 正对  (label=1): 同 user 真-真            (genuine pairs)
        - 负对1 (label=0): 同 user 真-熟练伪造       (skilled forgery)
        - 负对2 (label=0): 跨 user 真-真             (random forgery)

    |负对1| = |正对|，|负对2| = |正对|。

    官方协议是 5-shot enrollment + 10 trials 模式，见
    scripts/evaluate_svc2004_protocol.py。

    Returns:
        (all_pairs, meta): meta 标注每对的类型 'genuine' / 'skilled' / 'random'
    """
    rng = random.Random(seed)
    genuine = generate_genuine_pairs(file_list, rng=rng)
    n_pos = len(genuine)
    skilled = generate_skilled_forgery_pairs(file_list, num_pairs=n_pos, rng=rng)
    random_forg = generate_random_forgery_pairs(file_list, num_pairs=n_pos, rng=rng)

    pairs = []
    meta = []
    for p in genuine:
        pairs.append(p); meta.append('genuine')
    for p in skilled:
        pairs.append(p); meta.append('skilled')
    for p in random_forg:
        pairs.append(p); meta.append('random')

    # 同步 shuffle
    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    pairs = [pairs[i] for i in idx]
    meta = [meta[i] for i in idx]
    return pairs, meta


def run_batch(checkpoint: str, test_list_path: str, data_root: str, output_dir: str):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # 读取测试集文件列表（路径已含 raw_data/SVC2004_Task2/ 前缀）
    with open(test_list_path, 'r') as f:
        file_names = [l.strip() for l in f if l.strip()]
    # 若路径已含 data_root 前缀则直接用，否则拼接
    if file_names and file_names[0].startswith(data_root):
        file_list = file_names
    else:
        file_list = [os.path.join(data_root, fn) for fn in file_names]
    print(f"测试集文件数: {len(file_list)}")

    # 生成签名对
    pairs, meta = generate_test_pairs(file_list)
    n_gen = meta.count('genuine')
    n_skilled = meta.count('skilled')
    n_random = meta.count('random')
    print(f"生成签名对: {len(pairs)} 对  "
          f"(genuine: {n_gen}, skilled_forgery: {n_skilled}, random_forgery: {n_random})")

    # 预加载特征
    print("预加载特征...")
    feat_cache = {}
    for fp in file_list:
        feat_cache[fp] = load_and_extract(fp)

    # 构建模型
    print("加载模型...")
    model = build_model(checkpoint)

    # 批量推理
    print("推理中...")
    y_true, y_scores, y_kind = [], [], []
    batch_size = 32
    all_results = []

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        batch_meta = meta[i:i + batch_size]
        b1 = np.stack([feat_cache[p[0]] for p in batch])
        b2 = np.stack([feat_cache[p[1]] for p in batch])
        scores = model((tf.constant(b1), tf.constant(b2)), training=False).numpy().flatten()

        for (f1, f2, label), kind, score in zip(batch, batch_meta, scores):
            y_true.append(label)
            y_scores.append(float(score))
            y_kind.append(kind)
            all_results.append({
                'sig1': os.path.basename(f1),
                'sig2': os.path.basename(f2),
                'label': label,
                'kind': kind,
                'score': round(float(score), 4),
            })

        if (i // batch_size + 1) % 10 == 0:
            print(f"  {i + len(batch)}/{len(pairs)} 对完成")

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_kind = np.array(y_kind)

    # 保存输出
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # CSV结果
    csv_path = out / 'results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['sig1', 'sig2', 'label', 'kind', 'score'])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n结果已保存: {csv_path}")

    # 分别对齐论文 Table 3 两套指标 -----------------------------------
    def _eval_subset(mask, tag):
        yt = y_true[mask]
        ys = y_scores[mask]
        if len(np.unique(yt)) < 2:
            return None
        e, th = calculate_eer(yt, ys)
        fpr_, tpr_, _ = roc_curve(yt, ys)
        return {
            'tag': tag,
            'n': int(mask.sum()),
            'n_pos': int(yt.sum()),
            'n_neg': int((1 - yt).sum()),
            'eer': float(e),
            'eer_threshold': float(th),
            'auc': float(auc(fpr_, tpr_)),
            'metrics_at_eer': calculate_metrics_at_threshold(yt, ys, th),
        }

    genuine_mask = (y_kind == 'genuine')
    skilled_mask = genuine_mask | (y_kind == 'skilled')   # genuine + skilled → "skilled forgery EER"
    random_mask = genuine_mask | (y_kind == 'random')     # genuine + random  → "random forgery EER"
    overall_mask = np.ones(len(y_true), dtype=bool)

    reports = [
        ('skilled_forgery', skilled_mask, _eval_subset(skilled_mask, 'skilled_forgery')),
        ('random_forgery', random_mask, _eval_subset(random_mask, 'random_forgery')),
        ('overall', overall_mask, _eval_subset(overall_mask, 'overall')),
    ]

    metrics_path = out / 'metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Pair-level evaluation (NOT the SVC2004 10-trial protocol)\n")
        f.write("For paper-comparable Table 3 metrics, use scripts/evaluate_svc2004_protocol.py\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"测试对总数: {len(y_true)}  (genuine={genuine_mask.sum()}, "
                f"skilled={int((y_kind == 'skilled').sum())}, "
                f"random={int((y_kind == 'random').sum())})\n\n")
        for tag, _, r in reports:
            if r is None:
                f.write(f"[{tag}] 样本不足，跳过\n\n"); continue
            f.write(f"--- {tag} ---\n")
            f.write(f"  n={r['n']}  pos={r['n_pos']}  neg={r['n_neg']}\n")
            f.write(f"  AUC: {r['auc']:.4f}\n")
            f.write(f"  EER: {r['eer']:.4f} ({r['eer']:.2%})  阈值 {r['eer_threshold']:.4f}\n")
            m = r['metrics_at_eer']
            f.write(f"  @EER  Acc={m['accuracy']:.4f}  FAR={m['far']:.4f}  FRR={m['frr']:.4f}  F1={m['f1_score']:.4f}\n\n")
    print(f"指标已保存: {metrics_path}")

    # 终端打印
    print("\n" + "=" * 60)
    for tag, _, r in reports:
        if r is None:
            print(f"[{tag}] 样本不足，跳过"); continue
        print(f"[{tag}]  AUC={r['auc']:.4f}  EER={r['eer']:.2%} (thr={r['eer_threshold']:.3f})  "
              f"Acc@EER={r['metrics_at_eer']['accuracy']:.2%}")
    print("=" * 60)

    # 用于后续图表（用 overall 或 skilled 视情而定，这里保留 overall 与论文对比参考）
    eer, eer_thresh = reports[2][2]['eer'], reports[2][2]['eer_threshold']  # overall

    # 生成图表
    roc_path = str(out / 'roc_curve.png')
    plot_roc_curve(y_true, y_scores, save_path=roc_path, show=False)
    print(f"ROC曲线: {roc_path}")

    det_path = str(out / 'det_curve.png')
    plot_det_curve(y_true, y_scores, save_path=det_path, show=False)
    print(f"DET曲线: {det_path}")


def main():
    parser = argparse.ArgumentParser(description="批量签名验证推理")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best_model.h5",
                        help="模型权重路径（重训后的新模型）")
    parser.add_argument("--test_list", default="outputs/features/test_list.txt",
                        help="测试集文件路径列表（由 preprocess.py --split_mode official 生成）")
    parser.add_argument("--data_root", default="raw_data/SVC2004_Task2")
    parser.add_argument("--output_dir", default="outputs/results/batch_eval")
    args = parser.parse_args()

    run_batch(args.checkpoint, args.test_list, args.data_root, args.output_dir)


if __name__ == "__main__":
    main()
