"""
SVC2004 10-trial 评测协议实现（strict / adapted 两种模式）

⚠️  方法学说明（命名边界）：
    - `strict` = same-split 10-trial protocol：random forgery 与被评估 user
      来自**同一个 split** 且该 split 其他 user ≥ 20。"strict" 仅表示"随机
      伪造源与评估集同域"，**不等于 paper-aligned**。例如 `--split train
      --random_source test` 仍会满足 strict 条件（如果 test split 够大），
      但那显然不是论文的评估设定（论文在 evaluation database 上评估所有
      test user）。是否作为"与论文口径一致"的结果，由团队在论文/报告层面
      人工认定，脚本不会自动打 "paper-aligned" 标签。
    - `adapted` = 为 writer-independent split 做的适配：random forgery 取自
      其他 split（例如默认的 `random_source='train_val'`）。我们 28/6/6 的
      test split 只有 6 个 user，不足 20 个其他 user，无法走 strict，只能
      走 adapted。报告中应明确标 "adapted 10-trial protocol (random forgery
      from <source>)"，**不要写 Table 3 aligned**。

    论文 SVC2004 §4.1 原始协议：test user 的 random forgery 来自**同一
    evaluation database** 内的其他 user（每 user 1 张真签名 × 20 user）。
    要真正复现该口径，至少需要测试集 ≥ 21 个 user。

协议主流程（两种模式共用）：
    每 test user × 10 trials：
        enrollment = 5 张随机真签名（从 S1-S10 抽）
        test = 10 pos(S11-S20) + 20 skilled(S21-S40) + 20 random(其他 user 真签)
        对每个测试签名，用 Siamese 对 5 个模板分别打分，按 --aggregation 聚合
        分别在 {pos, skilled} / {pos, random} 上算 EER
    汇总：所有 (user, trial) EER 的 mean / SD / max

用法:
    python scripts/evaluate_svc2004_protocol.py \
        --checkpoint outputs/checkpoints/<new_model>.h5 \
        --feature_cache outputs/features \
        --split test \
        --aggregation mean \
        --random_source train_val \
        --output_dir outputs/results/protocol

输出：
    - per_trial.csv          每 (user, trial) 的 skilled/random EER
    - summary.json           avg/SD/max + meta + protocol_mode
    - summary.txt            人类可读（含 adapted/strict 明确标注）
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path
from typing import List, Sequence

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import SVC2004Dataset
from data.pair_sampler import group_by_user, parse_filename, is_genuine, is_skilled_forgery
from data.svc2004_protocol import run_protocol, ProtocolReport
from models.siamese import SiameseNetwork
from utils.config import load_config
from utils.logger import setup_logger, get_logger


def parse_args():
    p = argparse.ArgumentParser(
        description='SVC2004 10-trial protocol evaluator. '
                    'STRICT mode: random_source=test AND other_users>=20. '
                    'ADAPTED mode: otherwise (not paper-aligned, see summary.txt for details).'
    )
    p.add_argument('--checkpoint', type=str, required=True)
    p.add_argument('--config', type=str, default='config/default.yaml')
    p.add_argument('--data_root', type=str, default='./raw_data/SVC2004_Task2')
    p.add_argument('--feature_cache', type=str, default='./outputs/features')
    p.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    p.add_argument('--output_dir', type=str, default='./outputs/results/protocol')
    p.add_argument('--aggregation', type=str, default='mean', choices=['mean', 'max', 'min'])
    p.add_argument('--n_trials', type=int, default=10)
    p.add_argument('--enrollment_k', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--gpu', type=int, default=0)
    # 随机伪造来源：默认从训练/验证集 user 里抽（避免跨 split 信息泄漏有待讨论）
    p.add_argument('--random_source', type=str, default='train_val',
                   choices=['train_val', 'test', 'all'],
                   help='其他 user 池来源。train_val：从训练和验证集 user 抽（不含 test user）'
                        '；test：仅从 test split 其他 user 抽；all：全集')
    return p.parse_args()


def setup_gpu(gpu_id: int):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[gpu_id], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[gpu_id], True)
        except RuntimeError as e:
            print(f"GPU setup error: {e}")


def _fix_length(arr: np.ndarray, max_len: int, feat_dim: int = 23) -> np.ndarray:
    n = arr.shape[0]
    if n >= max_len:
        return arr[:max_len].astype(np.float32)
    out = np.zeros((max_len, feat_dim), dtype=np.float32)
    out[:n] = arr
    return out


def build_score_fn(model, feature_cache, max_len: int):
    """
    返回一个 batch_score_fn(templates, test_path) -> list[float]，
    对 5 个模板分别打分。
    """
    def _score(templates: Sequence[str], test_path: str) -> List[float]:
        test_feat = _fix_length(feature_cache[test_path], max_len)
        K = len(templates)
        t_batch = np.stack([_fix_length(feature_cache[t], max_len) for t in templates])
        s_batch = np.stack([test_feat] * K)
        scores = model((tf.constant(t_batch), tf.constant(s_batch)), training=False).numpy().flatten()
        return scores.tolist()
    return _score


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
        use_attention=config.model.use_attention,
        distance_metric=config.model.distance_metric,
    )
    dummy = tf.zeros((1, max_len, 23), dtype=tf.float32)
    model((dummy, dummy), training=False)
    model.load_weights(checkpoint_path)
    return model


def _collect_random_pool(args) -> List[str]:
    """
    收集 random forgery 候选 user 的所有签名文件。
    关键防御：**永远排除 args.split 对应的 split**，避免重复拼接到 all_files 后
    触发 run_protocol 的计数校验失败（expected 10 pos, got 20）。

    - random_source='test'       → 返回 []（由 run_protocol 用 test split 内 other user）
    - random_source='train_val'  → train + val，但排除 args.split
    - random_source='all'        → train + val + test，但排除 args.split
    """
    if args.random_source == 'test':
        return []

    if args.random_source == 'train_val':
        candidate_splits = [s for s in ('train', 'val') if s != args.split]
    elif args.random_source == 'all':
        candidate_splits = [s for s in ('train', 'val', 'test') if s != args.split]
    else:
        return []

    pool_files: List[str] = []
    for s in candidate_splits:
        ds = SVC2004Dataset(data_root=args.data_root, split=s, feature_cache_dir=args.feature_cache)
        pool_files.extend(ds.file_list)
    return pool_files


def main():
    args = parse_args()
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / 'logs'; log_dir.mkdir(exist_ok=True)
    setup_logger(log_dir=str(log_dir), log_file=str(log_dir / 'protocol_eval.log'))
    logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("SVC2004 10-trial Protocol Evaluation")
    logger.info("(protocol_mode will be printed below once test users are loaded)")
    logger.info("=" * 60)
    logger.info(f"checkpoint      : {args.checkpoint}")
    logger.info(f"split           : {args.split}")
    logger.info(f"aggregation     : {args.aggregation}")
    logger.info(f"n_trials        : {args.n_trials}")
    logger.info(f"enrollment_k    : {args.enrollment_k}")
    logger.info(f"random source   : {args.random_source}")
    logger.info(f"seed            : {args.seed}")

    setup_gpu(args.gpu)
    config = load_config(args.config)
    MAX_LEN = int(getattr(config.training, 'max_seq_len', 400))

    # 测试 split（test_users 由这里决定）
    test_ds = SVC2004Dataset(data_root=args.data_root, split=args.split, feature_cache_dir=args.feature_cache)
    test_users = sorted(group_by_user(test_ds.file_list).keys())
    logger.info(f"test users: {test_users}")

    # random forgery 池
    extra_files = _collect_random_pool(args)
    if extra_files:
        # dict.fromkeys 保持顺序去重（path 相同视为同文件；_collect_random_pool 已
        # 排除 args.split，但这里再做一次以防御任何调用方的重复）
        all_files = list(dict.fromkeys(list(test_ds.file_list) + extra_files))
        pool_by_user = group_by_user(all_files)
        other_users_pool = [u for u in pool_by_user if u not in set(test_users)]
        logger.info(f"random forgery pool: {len(other_users_pool)} other users "
                    f"(from {args.random_source}, excluding split '{args.split}')")
    else:
        all_files = list(dict.fromkeys(list(test_ds.file_list)))
        other_users_pool = None  # 让 run_protocol 用默认（test split 内其他 user）
        n_other_in_test = len(set(test_users)) - 1
        logger.info(f"random forgery pool: within '{args.split}' split "
                    f"({n_other_in_test} other users)")

    # 判定 protocol_mode: strict 需 random_source='test' 且 test split 其他
    # user 数 >= 20（论文原始要求）
    n_other_users_available = (
        len(other_users_pool) if other_users_pool is not None
        else (len(test_users) - 1)
    )
    if args.random_source == 'test' and n_other_users_available >= 20:
        protocol_mode = 'strict'
    else:
        protocol_mode = 'adapted'

    logger.info(f"protocol_mode   : {protocol_mode}")
    if protocol_mode == 'strict':
        logger.info(
            "protocol_mode='strict' (same-split 10-trial protocol). "
            "注意：strict 不等于 paper-aligned —— 论文在 evaluation database 上"
            "评估全部 test user；是否作为'与论文口径一致'的结果由报告层面人工认定。"
        )
    else:
        logger.warning(
            "protocol_mode='adapted': random forgery 源与论文 SVC2004 原始协议"
            "不完全一致（论文要求从 evaluation database 内其他 user 抽）。"
            "在论文/报告中应写 'adapted 10-trial protocol (random forgery from "
            f"{args.random_source})'，不要写成 'Table 3 aligned' 或 'paper-aligned'。"
        )
    if n_other_users_available < 20:
        logger.error(
            f"random forgery pool only has {n_other_users_available} other users "
            f"(< 20 required per trial). run_protocol will fail."
        )

    # 预加载所有需要的特征到内存
    logger.info("Preloading features...")
    feat_cache = test_ds.preload_features(verbose=True)
    if extra_files:
        # 把 train/val 集的特征也加载进来（用于 random forgery）
        for s in (['train', 'val'] if args.random_source == 'train_val'
                  else ['train', 'val', 'test']):
            ds_aux = SVC2004Dataset(data_root=args.data_root, split=s, feature_cache_dir=args.feature_cache)
            feat_cache.update(ds_aux.preload_features(verbose=False))

    # 模型 & 打分器
    model = load_model(args.checkpoint, config, MAX_LEN)
    score_fn = build_score_fn(model, feat_cache, MAX_LEN)

    # 跑协议
    report = run_protocol(
        test_users=test_users,
        all_files=all_files,
        batch_score_fn=score_fn,
        aggregation=args.aggregation,
        n_trials=args.n_trials,
        enrollment_k=args.enrollment_k,
        seed=args.seed,
        other_users_pool=other_users_pool,
    )

    # 输出 per-trial CSV
    with open(out_dir / 'per_trial.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['user_id', 'trial', 'skilled_eer', 'skilled_threshold',
                         'random_eer', 'random_threshold'])
        for r in report.per_trial:
            writer.writerow([r.user_id, r.trial,
                             r.skilled_eer, r.skilled_threshold,
                             r.random_eer, r.random_threshold])

    summary = report.summary()
    meta = {
        'aggregation': report.aggregation,
        'n_trials': report.n_trials,
        'enrollment_k': report.enrollment_k,
        'n_test_users': len(test_users),
        'test_users': test_users,
        'total_eer_points_per_kind': len(report.per_trial),
        'random_source': args.random_source,
        'protocol_mode': protocol_mode,
        'split': args.split,
        'notes': (
            "strict: same-split 10-trial protocol —— random forgery 与 test "
            "user 同属 split，且 other user ≥ 20。"
            "*不等于 paper-aligned*：论文在 evaluation database 上评估全部 "
            "test user，是否对齐论文口径需由报告层面人工认定。"
            " | adapted: random forgery 源自其他 split，为 writer-"
            "independent 28/6/6 划分做的适配；不可直接声称复现论文 Table 3。"
        ),
    }
    with open(out_dir / 'summary.json', 'w') as f:
        json.dump({'meta': meta, 'summary': summary}, f, indent=2)

    # 人类可读
    with open(out_dir / 'summary.txt', 'w') as f:
        mode_line = {
            'strict': "SVC2004 10-trial Protocol — STRICT (same-split 10-trial protocol)",
            'adapted': "SVC2004 10-trial Protocol — ADAPTED (for writer-independent split)",
        }[protocol_mode]
        f.write(mode_line + "\n")
        f.write("=" * 60 + "\n")
        # 不论 strict 还是 adapted，都要明确说明与论文口径的关系
        f.write("⚠ 方法学说明：\n")
        if protocol_mode == 'strict':
            f.write("    STRICT 仅表示 random forgery 与被评估 user 同属 split "
                    f"('{args.split}')。这 *不等于 paper-aligned*：论文原始协议\n"
                    "    在 evaluation database 上评估全部 test user；是否作为\n"
                    "    \"与论文口径一致\"的结果，需由团队在论文/报告层面人工认定。\n")
        else:
            f.write(f"    ADAPTED: random forgery drawn from a different split "
                    f"('{args.random_source}') than the evaluated split ('{args.split}').\n")
            f.write("    论文 SVC2004 原始协议要求 random forgery 与 test user "
                    "同属 evaluation database；\n")
            f.write("    本实现为 28/6/6 writer-independent 划分做的适配。\n")
            f.write("    在论文/报告中应写 \"adapted 10-trial protocol (random "
                    f"forgery from {args.random_source})\"，\n")
            f.write("    而非 \"Table 3 aligned\" / \"paper-aligned\"。\n")
        f.write("\n")
        f.write(f"checkpoint   : {args.checkpoint}\n")
        f.write(f"aggregation  : {args.aggregation}\n")
        f.write(f"random source: {args.random_source} (split excluded)\n")
        f.write(f"test users ({len(test_users)}): {test_users}\n")
        f.write(f"total (user × trial) points: {len(report.per_trial)} per kind\n\n")
        for kind in ('skilled', 'random'):
            s = summary[kind]
            f.write(f"--- {kind} forgery ---\n")
            if s['n'] == 0:
                f.write("  no valid EER points\n\n"); continue
            f.write(f"  Average EER : {s['avg']:.4f} ({s['avg']:.2%})\n")
            f.write(f"  SD          : {s['sd']:.4f}\n")
            f.write(f"  Maximum EER : {s['max']:.4f} ({s['max']:.2%})\n")
            f.write(f"  n points    : {s['n']}\n\n")

    # 终端输出
    print("\n" + "=" * 60)
    print(f"protocol_mode = {protocol_mode}  "
          f"(random forgery from '{args.random_source}', split '{args.split}' excluded)")
    if protocol_mode == 'strict':
        print("  ℹ strict = same-split 10-trial protocol.")
        print("    strict ≠ paper-aligned. Paper evaluates on the full evaluation")
        print("    database; whether your setup matches the paper is a team call,")
        print("    not a property this script can verify.")
    else:
        print("  ⚠ Not identical to the paper's original SVC2004 protocol.")
        print("    Report as 'adapted 10-trial protocol', NOT 'Table 3 aligned'.")
    print("-" * 60)
    for kind in ('skilled', 'random'):
        s = summary[kind]
        if s['n'] == 0:
            print(f"[{kind}] no data"); continue
        print(f"[{kind}]  Avg EER = {s['avg']:.2%}   SD = {s['sd']:.4f}   Max = {s['max']:.2%}   (n={s['n']})")
    print("=" * 60)
    logger.info(f"Report saved to {out_dir}")


if __name__ == '__main__':
    main()
