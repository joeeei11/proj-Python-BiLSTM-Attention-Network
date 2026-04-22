![Language](https://img.shields.io/badge/language-Python-blue) ![License](https://img.shields.io/badge/license-MIT-green)

# BiLSTM-Attention Online Signature Verification

Implementation of [A Stroke-Based RNN for Writer-Independent Online Signature Verification](https://ieeexplore.ieee.org/document/xxx) on SVC2004 Task2.

## Patch Notes (2026-04-22)

本次补丁修复了初始实现中的两处根本性错误，历史 checkpoint 全部作废，需重新训练。

**Bug 1 — 正/负对标签定义错误**

原实现将 genuine-genuine 对标记为负样本（标签 0），genuine-skilled_forgery 对标记为正样本（标签 1），语义完全相反。修复后：正对（同一人真签名）= 1，负对（真签名 vs 伪签名）= 0，与论文一致。

**Bug 2 — Writer 泄漏（writer-independent 划分失效）**

原实现按签名文件随机划分 train/val/test，导致同一用户的签名同时出现在训练集和测试集，违反 writer-independent 评估要求。修复后严格按用户划分：user 1–28 训练 / 29–34 验证 / 35–40 测试，与 SVC2004 官方协议一致。

**新增 — SVC2004 官方 10-trial 协议评估脚本**

新增 `scripts/evaluate_svc2004_protocol.py`，实现论文 Table 3 的可比指标：每用户随机抽取 5 条真签名作为参考，分别对 skilled forgery 和 random forgery 计算 EER，重复 10 次取均值。输出 `per_trial.csv`、`summary.json`、`summary.txt`。

---

## Results

> **注意**：下表结果来自历史 checkpoint（已作废）。  
> 历史训练存在正负对定义错误和 writer 泄漏问题，数据不可信，待重新训练后更新。

| | Val | Test |
|---|---|---|
| Accuracy | — | — |
| EER | — | — |
| AUC | — | — |

重新训练完成后，可用以下两种方式评估：

- **Pair-level sanity check**（训练期监控）：`scripts/evaluate.py`
- **SVC2004 10-trial 协议**（论文对标指标）：`scripts/evaluate_svc2004_protocol.py`

## Requirements

Python 3.8, TensorFlow 2.9

```bash
conda create -n sig38 python=3.8 -y
conda activate sig38
pip install tensorflow==2.9.0
pip install -r requirements.txt
```

## Quick Start

> 历史预训练权重已作废（标签定义有误），需重新训练后使用。训练完成后权重路径为 `outputs/checkpoints/best_model.h5`。

```bash
python scripts/inference.py \
    --sig1 path/to/sig1.TXT \
    --sig2 path/to/sig2.TXT \
    --checkpoint outputs/checkpoints/best_model.h5 \
    --threshold 0.XXX   # 从 evaluate_svc2004_protocol.py 生成的 summary.json 中取 eer_threshold
```

输出：
```
相似度分数: 0.9999
验证结果:   真签名 (Genuine)
```

## Training from Scratch

**1. 准备数据**

SVC2004 Task2 数据集需向官方申请下载：https://www.cse.ust.hk/svc2004/

下载后将数据解压到 `raw_data/SVC2004_Task2/`，然后：

```bash
# 使用官方 writer-independent 划分（user 1-28 训练 / 29-34 验证 / 35-40 测试）
python scripts/preprocess.py \
    --data_root ./raw_data/SVC2004_Task2 \
    --split_mode official
```

**2. 训练**

```bash
python scripts/train.py --config config/train.yaml
```

**3. Pair-level 快速评估（训练期 sanity check，非论文指标）**

```bash
python scripts/batch_inference.py \
    --checkpoint outputs/checkpoints/best_model.h5 \
    --test_list outputs/features/test_list.txt \
    --output_dir outputs/results
```

**4. SVC2004 10-trial 协议评估**

```bash
python scripts/evaluate_svc2004_protocol.py \
    --checkpoint outputs/checkpoints/best_model.h5 \
    --split test \
    --random_source train_val
```

输出 `per_trial.csv`、`summary.json`、`summary.txt`，分别报告 skilled forgery EER 和 random forgery EER。

> **注意**：`--random_source train_val`（默认值）时进入 **adapted** 模式——random forgery 来自训练/验证集用户，与论文要求的"从评估库内其他用户抽取"不完全一致。`summary.txt` 会明确标注 `ADAPTED`，**不应将此结果直接写成"与 Table 3 一致"**。若需 strict 模式，需要 `--random_source test` 且 test split 有 ≥ 20 个 other user。

## API

```bash
python api/app.py
```

```bash
curl -X POST http://localhost:5000/verify \
  -F "sig1=@sig1.TXT" \
  -F "sig2=@sig2.TXT" \
  -F "threshold=0.XXX"   # 必填：从 evaluate_svc2004_protocol.py 的 summary.json 取 eer_threshold
# {"score": 0.9999, "threshold": 0.XXX, "result": "genuine", "result_cn": "真签名"}
```

## Model

孪生网络结构，两路共享 BiLSTM（2层，hidden=256）+ Attention 提取特征，Concat 拼接后全连接层输出相似度。输入为 23 维时序特征，序列统一截断/补零至长度 400。
