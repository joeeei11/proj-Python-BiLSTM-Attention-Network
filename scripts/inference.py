"""
签名验证推理脚本

用法:
    python scripts/inference.py \
        --sig1 path/to/sig1.txt \
        --sig2 path/to/sig2.txt \
        --checkpoint outputs/checkpoints/best_model_planB_epoch11.h5

输出: 相似度分数 + 真/假判断
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tensorflow as tf

from data.feature_extractor import load_signature_txt, extract_temporal_features

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


def fix_length(features: np.ndarray, max_len: int = MAX_LEN) -> np.ndarray:
    n = features.shape[0]
    if n >= max_len:
        return features[:max_len]
    pad = np.zeros((max_len - n, features.shape[1]), dtype=np.float32)
    return np.vstack([features, pad])


def load_and_extract(filepath: str) -> np.ndarray:
    raw = load_signature_txt(filepath)
    if raw.shape[0] == 0:
        raise ValueError(f"无法读取签名文件: {filepath}")
    x, y, t, p = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
    features = extract_temporal_features(x, y, p, time=t)
    return fix_length(features)


def build_model() -> tf.keras.Model:
    from models.siamese import build_siamese_network
    model = build_siamese_network(MODEL_CONFIG)
    dummy = tf.zeros((1, MAX_LEN, 23))
    model((dummy, dummy), training=False)
    return model


def run_inference(sig1_path: str, sig2_path: str, checkpoint: str, threshold: float = THRESHOLD):
    print(f"加载签名1: {sig1_path}")
    feat1 = load_and_extract(sig1_path)

    print(f"加载签名2: {sig2_path}")
    feat2 = load_and_extract(sig2_path)

    print("构建模型并加载权重...")
    model = build_model()
    model.load_weights(checkpoint)

    inp1 = tf.expand_dims(feat1, 0)
    inp2 = tf.expand_dims(feat2, 0)
    score = model((inp1, inp2), training=False).numpy()[0, 0]

    result = "真签名 (Genuine)" if score >= threshold else "伪签名 (Forgery)"
    print(f"\n相似度分数: {score:.4f}")
    print(f"判断阈值:   {threshold:.4f}")
    print(f"验证结果:   {result}")
    return score, result


def main():
    parser = argparse.ArgumentParser(description="在线签名验证推理")
    parser.add_argument("--sig1", required=True, help="签名1的.txt文件路径")
    parser.add_argument("--sig2", required=True, help="签名2的.txt文件路径")
    parser.add_argument(
        "--checkpoint",
        default="outputs/checkpoints/best_model_planB_epoch11.h5",
        help="模型权重路径"
    )
    parser.add_argument("--threshold", type=float, default=THRESHOLD, help="判断阈值")
    args = parser.parse_args()

    run_inference(args.sig1, args.sig2, args.checkpoint, args.threshold)


if __name__ == "__main__":
    main()
