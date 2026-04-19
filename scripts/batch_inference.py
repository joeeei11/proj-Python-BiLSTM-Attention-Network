"""
批量推理脚本

对测试集所有签名对进行推理，输出评估指标和论文图表。

用法:
    python scripts/batch_inference.py \
        --checkpoint outputs/checkpoints/best_model_planB_epoch11.h5 \
        --test_list outputs/features/test_list.txt \
        --data_root raw_data/SVC2004_Task2 \
        --output_dir outputs/results/batch_eval

输出:
    - results.csv         每对签名的分数和标签
    - metrics.txt         EER、准确率、AUC、FAR/FRR
    - roc_curve.png       ROC曲线（论文用）
    - det_curve.png       DET曲线（论文用）
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
from data.pair_sampler import group_by_user
from utils.metrics import (
    calculate_eer, calculate_metrics_at_threshold,
    plot_roc_curve, plot_det_curve
)
from sklearn.metrics import roc_curve, auc

MAX_LEN = 400
THRESHOLD = 0.776

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


def generate_test_pairs(file_list: list, num_forgery_per_genuine: int = 1, seed: int = 42):
    """生成测试对：同用户=真(1)，跨用户=假(0)"""
    random.seed(seed)
    user_groups = group_by_user(file_list)
    user_ids = list(user_groups.keys())

    pairs = []

    # 真签名对：同用户任意两签名
    for uid, files in user_groups.items():
        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                pairs.append((files[i], files[j], 1))

    num_genuine = len(pairs)

    # 伪造对：跨用户，数量与真签名对相同
    forgery_count = num_genuine * num_forgery_per_genuine
    forgery_pairs = []
    while len(forgery_pairs) < forgery_count:
        u1, u2 = random.sample(user_ids, 2)
        f1 = random.choice(user_groups[u1])
        f2 = random.choice(user_groups[u2])
        forgery_pairs.append((f1, f2, 0))

    pairs += forgery_pairs
    random.shuffle(pairs)
    return pairs


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
    pairs = generate_test_pairs(file_list)
    print(f"生成签名对: {len(pairs)} 对  (真: {sum(l for _,_,l in pairs)}, 假: {sum(1-l for _,_,l in pairs)})")

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
    y_true, y_scores = [], []
    batch_size = 32
    all_results = []

    for i in range(0, len(pairs), batch_size):
        batch = pairs[i:i + batch_size]
        b1 = np.stack([feat_cache[p[0]] for p in batch])
        b2 = np.stack([feat_cache[p[1]] for p in batch])
        scores = model((tf.constant(b1), tf.constant(b2)), training=False).numpy().flatten()

        for (f1, f2, label), score in zip(batch, scores):
            y_true.append(label)
            y_scores.append(float(score))
            all_results.append({
                'sig1': os.path.basename(f1),
                'sig2': os.path.basename(f2),
                'label': label,
                'score': round(float(score), 4),
                'prediction': int(float(score) >= THRESHOLD)
            })

        if (i // batch_size + 1) % 10 == 0:
            print(f"  {i + len(batch)}/{len(pairs)} 对完成")

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    # 保存输出
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # CSV结果
    csv_path = out / 'results.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['sig1', 'sig2', 'label', 'score', 'prediction'])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\n结果已保存: {csv_path}")

    # 计算指标
    eer, eer_thresh = calculate_eer(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc_score = auc(fpr, tpr)

    best_thresh_metrics = calculate_metrics_at_threshold(y_true, y_scores, THRESHOLD)
    eer_metrics = calculate_metrics_at_threshold(y_true, y_scores, eer_thresh)

    # 保存指标文本
    metrics_path = out / 'metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write("=" * 50 + "\n")
        f.write("Plan B 测试集评估结果\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"测试对总数:    {len(y_true)}\n")
        f.write(f"真签名对:      {y_true.sum()}\n")
        f.write(f"伪造对:        {(1 - y_true).sum()}\n\n")
        f.write(f"AUC:           {auc_score:.4f}\n")
        f.write(f"EER:           {eer:.4f} ({eer:.2%})\n")
        f.write(f"EER阈值:       {eer_thresh:.4f}\n\n")
        f.write(f"最优阈值 {THRESHOLD} 下:\n")
        f.write(f"  准确率:      {best_thresh_metrics['accuracy']:.4f} ({best_thresh_metrics['accuracy']:.2%})\n")
        f.write(f"  FAR:         {best_thresh_metrics['far']:.4f}\n")
        f.write(f"  FRR:         {best_thresh_metrics['frr']:.4f}\n")
        f.write(f"  F1:          {best_thresh_metrics['f1_score']:.4f}\n")
    print(f"指标已保存: {metrics_path}")

    # 打印到终端
    print("\n" + "=" * 50)
    print(f"AUC:      {auc_score:.4f}")
    print(f"EER:      {eer:.2%}  (阈值 {eer_thresh:.3f})")
    print(f"准确率:   {best_thresh_metrics['accuracy']:.2%}  (阈值 {THRESHOLD})")
    print(f"FAR:      {best_thresh_metrics['far']:.4f}")
    print(f"FRR:      {best_thresh_metrics['frr']:.4f}")
    print("=" * 50)

    # 生成图表
    roc_path = str(out / 'roc_curve.png')
    plot_roc_curve(y_true, y_scores, save_path=roc_path, show=False)
    print(f"ROC曲线: {roc_path}")

    det_path = str(out / 'det_curve.png')
    plot_det_curve(y_true, y_scores, save_path=det_path, show=False)
    print(f"DET曲线: {det_path}")


def main():
    parser = argparse.ArgumentParser(description="批量签名验证推理")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best_model_planB_epoch11.h5")
    parser.add_argument("--test_list", default="outputs/features/test_list.txt")
    parser.add_argument("--data_root", default="raw_data/SVC2004_Task2")
    parser.add_argument("--output_dir", default="outputs/results/batch_eval")
    args = parser.parse_args()

    run_batch(args.checkpoint, args.test_list, args.data_root, args.output_dir)


if __name__ == "__main__":
    main()
