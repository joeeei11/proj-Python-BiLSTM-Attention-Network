# 代码说明文档

> BiLSTM-Attention Online Signature Verification  
> 基于论文《A Stroke-Based RNN for Writer-Independent Online Signature Verification》

---

## 目录

1. [项目结构总览](#1-项目结构总览)
2. [数据处理模块 `data/`](#2-数据处理模块-data)
3. [模型模块 `models/`](#3-模型模块-models)
4. [训练模块 `training/`](#4-训练模块-training)
5. [工具模块 `utils/`](#5-工具模块-utils)
6. [脚本入口 `scripts/`](#6-脚本入口-scripts)
7. [API 服务 `api/`](#7-api-服务-api)
8. [数据流向图](#8-数据流向图)
9. [核心算法说明](#9-核心算法说明)
10. [配置参数说明](#10-配置参数说明)

---

## 1. 项目结构总览

```
ZZU/
├── data/                   # 数据处理：加载、特征提取、增强、采样
│   ├── dataset.py          # SVC2004 数据集类
│   ├── feature_extractor.py# 23维特征提取（核心）
│   ├── augmentation.py     # 数据增强
│   ├── pair_sampler.py     # 正负样本对生成
│   └── utils.py            # 数据工具函数
├── models/                 # 神经网络模型定义
│   ├── stroke_rnn.py       # 基础 BiLSTM/BiGRU 编码器
│   ├── attention.py        # Bahdanau 加性注意力
│   ├── siamese.py          # 孪生网络（整体架构）
│   └── losses.py           # 损失函数（二元交叉熵等）
├── training/               # 训练与评估逻辑
│   ├── trainer.py          # 训练器（训练循环、梯度更新）
│   ├── evaluator.py        # 评估器（EER/ACC/AUC计算）
│   └── callbacks.py        # 训练回调（早停、学习率调整等）
├── utils/                  # 通用工具
│   ├── config.py           # YAML配置加载
│   ├── logger.py           # 日志管理
│   ├── metrics.py          # EER/FAR/FRR指标计算
│   └── visualization.py    # ROC/DET曲线绘制
├── scripts/                # 命令行入口脚本
│   ├── preprocess.py       # 数据预处理
│   ├── train.py            # 训练启动
│   ├── evaluate.py         # 评估
│   ├── inference.py        # 单对推理
│   └── batch_inference.py  # 批量推理
├── api/
│   └── app.py              # Flask REST API
├── config/
│   ├── default.yaml        # 默认配置
│   └── train.yaml          # 训练配置
├── checkpoints/
│   └── best_model.h5       # 训练产出的最优权重（需重训后生成，旧权重已移至 deprecated/）
└── assets/
    ├── roc_curve.png        # ROC曲线图
    └── det_curve.png        # DET曲线图
```

---

## 2. 数据处理模块 `data/`

### 2.1 特征提取 `data/feature_extractor.py`

这是整个系统的数据基础，将原始签名点序列转换为 **23 维时序特征**。

#### `load_signature_txt(filepath) -> np.ndarray`

从 SVC2004 `.TXT` 格式文件读取原始数据。

**SVC2004 文件格式（每行）：**
```
ID  x  y  time  pressure  pen_status
```

**返回：** `(N, 4)` 数组，列为 `[x, y, time, pressure]`

---

#### `extract_temporal_features(x, y, p, time) -> np.ndarray`

核心特征提取函数，输出 `(N, 23)` 特征矩阵。

| 维度 | 特征名 | 说明 |
|------|--------|------|
| 0 | `stroke_mark` | 笔画标记（1=笔按下，0=笔抬起） |
| 1 | `time_norm` | 归一化时间戳 [0,1] |
| 2 | `x_norm` | 归一化 X 坐标 [0,1] |
| 3 | `y_norm` | 归一化 Y 坐标 [0,1] |
| 4 | `p_norm` | 归一化压力 [0,1] |
| 5 | `v_x` | X 方向速度（一阶差分） |
| 6 | `v_y` | Y 方向速度 |
| 7 | `v_p` | 压力变化率 |
| 8 | `a_x` | X 方向加速度（二阶差分） |
| 9 | `a_y` | Y 方向加速度 |
| 10 | `a_p` | 压力加速度 |
| 11 | `v_abs` | 笔尖合速度 `√(vx²+vy²)` |
| 12 | `a_abs` | 笔尖合加速度 `√(ax²+ay²)` |
| 13 | `curvature` | 曲率 `(ax·vy - ay·vx) / |v|³` |
| 14 | `angle` | 运动方向角 `arctan2(vy, vx)` |
| 15 | `cumulative_arc_length` | 累积弧长 |
| 16 | `arc_length_norm` | 归一化累积弧长 [0,1] |
| 17 | `v_t` | 时间速度 |
| 18 | `a_t` | 时间加速度 |
| 19-22 | — | 零填充（保留扩展） |

---

### 2.2 数据集类 `data/dataset.py`

#### `SVC2004Dataset`

| 参数 | 类型 | 说明 |
|------|------|------|
| `data_root` | `str` | 数据集根目录 |
| `split` | `str` | `'train'` / `'test'` |
| `feature_cache` | `str` | 预提取特征缓存目录 |

---

### 2.3 样本对采样 `data/pair_sampler.py`

孪生网络需要成对输入，`pair_sampler` 按 SVC2004 官方定义生成三类样本对：

| 类型 | 定义 | 标签 |
|------|------|------|
| **genuine pair**（真对） | 同一用户的两个**真签名**（`y∈[1,20]`） | `1` |
| **skilled forgery pair**（熟练伪造对） | 同一用户的真签名 vs 熟练伪造（`y∈[21,40]`） | `0` |
| **random forgery pair**（随机伪造对） | 不同用户的两个真签名 | `0` |

> **注意**：`y∈[1,20]` 为真签名，`y∈[21,40]` 为熟练伪造，这是 SVC2004 官方定义。  
> 负样本需同时包含两种伪造类型，两者 EER 分别报告（skilled 通常更难识别）。

`PairSampler(ratio, skilled_ratio)` 中：
- `ratio`：负对与正对的数量比（训练期可设 2.0，评估期固定 1.0）
- `skilled_ratio`：负对中 skilled forgery 占比（其余为 random forgery）

---

### 2.4 数据增强 `data/augmentation.py`

对训练集签名施加随机扰动提升泛化性：

- 坐标随机缩放、平移
- 时间轴随机抖动
- 随机裁剪部分点

---

## 3. 模型模块 `models/`

### 3.1 整体架构

```
输入签名对 (sig1, sig2)
       │
       ▼
┌─────────────────────────────────┐
│        共享权重特征提取器          │
│                                 │
│  输入 (batch, 400, 23)           │
│       │                         │
│  Masking Layer                  │   ← 自动忽略零填充
│       │                         │
│  BiLSTM Layer 1 (256*2=512)     │
│       │                         │
│  BiLSTM Layer 2 (256*2=512)     │   ← return_sequences=True
│       │                         │
│  Attention Layer                │   ← Bahdanau 加性注意力
│       │                         │
│  Context Vector (512 维)        │
└─────────────────────────────────┘
       │                   │
     feat1               feat2
       │                   │
       └────────┬──────────┘
                ▼
     Concat([feat1, feat2, |feat1-feat2|])
                │
         Dense(128, relu)
                │
          Dropout(0.3)
                │
          Dense(64, relu)
                │
          Dropout(0.3)
                │
           Dense(1, sigmoid)
                │
          相似度分数 [0, 1]
```

---

### 3.2 基础编码器 `models/stroke_rnn.py`

#### `StrokeRNN`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `input_size` | `23` | 输入特征维度 |
| `hidden_size` | `256` | 每个方向的隐藏单元数 |
| `num_layers` | `2` | LSTM 层数 |
| `rnn_type` | `'lstm'` | 可选 `'lstm'` / `'gru'` |
| `bidirectional` | `True` | 双向，输出维度翻倍 |
| `dropout` | `0.2` | 层间 Dropout |
| `return_sequences` | `True` | 返回所有时间步（供注意力使用） |

**输出维度：** `hidden_size * 2`（双向），即 `512`

---

### 3.3 注意力机制 `models/attention.py`

#### `Attention`（Bahdanau 加性注意力）

计算公式：

```
score_t = v^T · tanh(W · h_t)
α_t     = softmax(score_t)          # 注意力权重
context = Σ α_t · h_t               # 加权上下文向量
```

**关键特性：** 自动处理 Masking（对零填充位置输出 `-1e9` 后 softmax，使其权重趋近于 0）

#### `AttentionRNN`

`StrokeRNN` + `Attention` 的组合封装，对外统一输出：
- `context`：`(batch, 512)` 上下文向量
- `attention_weights`：`(batch, 400)` 各时间步权重

---

### 3.4 孪生网络 `models/siamese.py`

#### `SiameseNetwork`

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `distance_metric` | `'concat'` | 距离度量：`'l2'` / `'cosine'` / `'concat'` |
| `use_attention` | `True` | 是否使用注意力 |
| `output_activation` | `'sigmoid'` | 输出激活函数 |

**三种距离度量：**

| 方式 | 公式 | 特点 |
|------|------|------|
| `l2` | `1/(1+‖f1-f2‖₂)` | 简单，无额外参数 |
| `cosine` | `(f1·f2/(‖f1‖‖f2‖)+1)/2` | 对幅度不敏感 |
| `concat` | `FC([f1, f2, |f1-f2|])` | **当前使用**，可学习，性能最好 |

#### `build_siamese_network(config) -> SiameseNetwork`

从字典构建网络的工厂函数，`inference.py` 中使用的配置：

```python
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
```

---

## 4. 训练模块 `training/`

### 4.1 训练器 `training/trainer.py`

#### `Trainer`

| 参数 | 说明 |
|------|------|
| `model` | 孪生网络模型 |
| `train_dataset` | `tf.data.Dataset` 训练集 |
| `val_dataset` | `tf.data.Dataset` 验证集 |
| `optimizer` | 默认 `Adam(lr=0.001)` |
| `loss_fn` | 默认 `BinaryCrossentropy` |
| `callbacks` | 早停、模型保存等回调 |

**训练循环流程：**

```
for epoch in epochs:
    for batch in train_dataset:
        with GradientTape():
            similarity = model([sig1, sig2], training=True)
            loss = loss_fn(labels, similarity)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(...)
    
    if val_dataset:
        val_metrics = validate()
        callbacks.on_epoch_end(epoch, val_metrics)
```

---

### 4.2 回调 `training/callbacks.py`

| 回调类 | 功能 |
|--------|------|
| `ModelCheckpoint` | 保存验证集最优权重 |
| `EarlyStopping` | 指标不再改善时提前终止 |
| `LearningRateScheduler` | 学习率衰减 |
| `TensorBoardCallback` | 写入 TensorBoard 日志 |

---

### 4.3 评估器 `training/evaluator.py`

调用 `utils/metrics.py` 中的函数，输出：

```python
{
    'accuracy': ...,       # 重训后更新
    'eer': ...,
    'auc': ...,
    'eer_threshold': ...,  # 重训后更新（旧值 0.776 来自已作废的 Plan B checkpoint）
    'far': ...,
    'frr': ...
}
```

---

## 5. 工具模块 `utils/`

### 5.1 评估指标 `utils/metrics.py`

#### `calculate_eer(y_true, y_scores) -> (eer, threshold)`

等错误率（EER）：FAR = FRR 时的错误率，是签名验证的核心指标。

计算方法：
1. 计算 ROC 曲线的 FPR、TPR、阈值序列
2. 求 `|FNR - FPR|` 最小处对应的 FPR 值即为 EER

#### `calculate_far_frr(y_true, y_scores, threshold) -> (far, frr)`

- **FAR（错误接受率）**：伪签名被判为真的比例
- **FRR（错误拒绝率）**：真签名被判为伪的比例

#### `find_optimal_threshold(y_true, y_scores, metric='eer') -> float`

自动搜索最优判决阈值（FAR = FRR 时对应的阈值）。重训完成后需重新标定，将结果通过 `scripts/inference.py --threshold <值>` 传入（该常量已移除，必须显式指定）。旧值 `0.776` 来自已作废的 Plan B checkpoint，不可继续使用。

---

### 5.2 配置管理 `utils/config.py`

加载 `config/train.yaml`，支持命令行参数覆盖：

```python
config = load_config('config/train.yaml')
config.update(cmd_args)  # 命令行参数优先级更高
```

---

### 5.3 日志 `utils/logger.py`

统一使用 Python `logging` 模块，禁止 `print` 调试。

```python
from utils.logger import get_logger
logger = get_logger(__name__)
logger.info("Training started")
```

---

## 6. 脚本入口 `scripts/`

### 6.1 数据预处理 `scripts/preprocess.py`

```bash
# 使用官方 writer-independent 划分（必须指定 --split_mode official）
python scripts/preprocess.py \
    --data_root ./raw_data/SVC2004_Task2 \
    --split_mode official
```

**执行流程：**
1. 按 `--split_mode` 划分用户：
   - `official`（默认推荐）：user 1-28 → train，29-34 → val，35-40 → test（writer-independent，不同 user 严格隔离）
   - `ratio`：按比例随机划分 user
   - `custom`：指定 user 列表
2. 遍历所有 `.TXT` 签名文件，调用 `load_signature_txt()` 读取原始点
3. 调用 `extract_temporal_features()` 提取 23 维特征
4. 截断/补零至长度 400，保存为 `.npy` 缓存文件
5. 生成 `train_list.txt` / `val_list.txt` / `test_list.txt`（文件路径列表）
6. 输出样本对 dry-run 统计（genuine / skilled / random 各类数量）

> **重要**：`split ∈ {train, val, test}` 时若缺少对应的 `*_list.txt`，`SVC2004Dataset` 将直接抛出 `FileNotFoundError`（不会静默回退到全量加载，以防止 writer 泄漏）。

---

### 6.2 训练脚本 `scripts/train.py`

```bash
python scripts/train.py --config config/train.yaml
```

**执行流程：**
1. 加载配置
2. 构建 `SVC2004Dataset` + `tf.data.Dataset`
3. `build_siamese_network(config)` 构建模型
4. `Trainer.train()` 启动训练
5. 保存最优权重到 `checkpoints/best_model.h5`

---

### 6.3 推理脚本 `scripts/inference.py`

```bash
python scripts/inference.py \
    --sig1 path/to/sig1.TXT \
    --sig2 path/to/sig2.TXT \
    --checkpoint outputs/checkpoints/best_model.h5 \
    --threshold 0.XXX   # 从 evaluate_svc2004_protocol.py 的 summary.json 中取 eer_threshold
```

**执行流程：**

```python
raw = load_signature_txt(filepath)          # 读取原始点
features = extract_temporal_features(...)   # 23维特征
features = fix_length(features, 400)        # 截断/补零至400
model = build_siamese_network(MODEL_CONFIG) # 构建模型
model.load_weights(checkpoint)              # 加载权重
score = model([feat1, feat2])               # 前向推理
result = "真签名" if score >= THRESHOLD else "伪签名"
```

**关键参数：**
- `MAX_LEN = 400`：序列固定长度
- `--threshold`（必填）：判决阈值（EER 点），从 `evaluate_svc2004_protocol.py` 的 `summary.json` 取 `eer_threshold` 字段；旧值 `0.776` 已作废，脚本不再有内置默认值

---

### 6.4 批量推理 `scripts/batch_inference.py`

```bash
python scripts/batch_inference.py \
    --checkpoint outputs/checkpoints/best_model.h5 \
    --test_list outputs/features/test_list.txt \
    --output_dir outputs/results
```

批量处理测试集样本对，输出 EER/ACC/AUC 汇总报告。分别报告：
- skilled forgery EER
- random forgery EER
- overall EER

> **注意**：此脚本是 **pair-level sanity check**，适合训练期快速监控。其结果**不等同于** SVC2004 官方 10-trial 协议的 Table 3 指标，不可直接用于论文对比。

### 6.5 官方 10-trial 协议评估 `scripts/evaluate_svc2004_protocol.py`

```bash
python scripts/evaluate_svc2004_protocol.py \
    --checkpoint outputs/checkpoints/best_model.h5 \
    --split test \
    --random_source train_val
```

按 SVC2004 §4.1 协议：每位 test user 取 S1-S10（5 次）作为 enrollment 模板，S11-S20 作为正例 trial，skilled/random forgery 各 10 个作为负例 trial，通过模板聚合分数计算 EER。

**输出文件**：
- `per_trial.csv`：逐 trial 分数与标签
- `summary.json`：skilled EER / random EER / overall EER + 协议模式标注
- `summary.txt`：人类可读报告（含方法学说明）

**协议模式**：
- `STRICT`：`random_source='test'` 且 test split 其他 user ≥ 20
- `ADAPTED`：其他情况（结果不作为 Table 3 对比依据，报告中会明确标注）

---

## 7. API 服务 `api/`

### `api/app.py`

基于 Flask 的 REST API，提供 HTTP 接口供外部调用。

**启动：**
```bash
python api/app.py
```

**接口：`POST /verify`**

```bash
curl -X POST http://localhost:5000/verify \
  -F "sig1=@path/to/sig1.TXT" \
  -F "sig2=@path/to/sig2.TXT" \
  -F "threshold=0.XXX"   # 必填：从 evaluate_svc2004_protocol.py 的 summary.json 取 eer_threshold
```

**响应：**
```json
{
  "score": 0.9999,
  "threshold": 0.XXX,
  "result": "genuine",
  "result_cn": "真签名"
}
```

| 字段 | 说明 |
|------|------|
| `score` | 相似度分数 `[0, 1]`，越高越可能是真签名 |
| `threshold` | 请求中传入的判决阈值（原样返回，便于确认） |
| `result` | `"genuine"`（真签名）或 `"forgery"`（伪签名） |
| `result_cn` | 中文结果：`"真签名"` 或 `"伪签名"` |

---

## 8. 数据流向图

```
SVC2004 .TXT 文件
        │
        ▼
load_signature_txt()
→ (N, 4): [x, y, time, pressure]
        │
        ▼
extract_temporal_features()
→ (N, 23): 23维时序特征
        │
        ▼
fix_length(max_len=400)
→ (400, 23): 统一长度（截断或零填充）
        │
   ┌────┴────┐
   ▼         ▼
 sig1      sig2       ← 组成样本对
   │         │
   └────┬────┘
        ▼
SiameseNetwork.call([sig1, sig2])
   │
   ├─ feature_extractor(sig1) → feat1 (512,)
   ├─ feature_extractor(sig2) → feat2 (512,)
   │
   └─ Concat([feat1, feat2, |feat1-feat2|])
              │
          FC Layers
              │
        similarity ∈ [0, 1]
              │
       ≥ THRESHOLD → 真签名
       < THRESHOLD → 伪签名
       (THRESHOLD 待重训后从 evaluate_svc2004_protocol 结果标定)
```

---

## 9. 核心算法说明

### 9.1 为什么用孪生网络？

签名验证是**验证任务**而非分类任务。系统需要判断"两个签名是否来自同一人"，而非识别"这是谁的签名"。孪生网络天然适合这种相似度学习范式：

- 两路**共享权重**，保证对两个签名使用相同的特征提取标准
- 训练时直接优化"相同/不同"的二分类损失
- 测试时对从未见过的用户也能泛化

### 9.2 为什么用 Attention？

BiLSTM 最后一个时间步的隐藏状态可能遗忘早期信息（长序列问题）。注意力机制让模型**自适应地**对所有时间步加权求和，关注签名中最具判别性的局部笔画段，而非依赖最后一个时间步。

### 9.3 为什么序列长度固定为 400？

SVC2004 中签名点数分布在 100-800 之间，取 400 覆盖约 80% 的样本：
- 超过 400 点的序列：截断末尾（签名起始段信息更完整）
- 不足 400 点的序列：末尾零填充 + Masking 层忽略补零

### 9.4 判决阈值的标定方法

重训完成后，通过以下步骤确定判决阈值：

1. 运行 `scripts/evaluate_svc2004_protocol.py`，得到 `summary.json`
2. 取 `eer_threshold` 字段值（即 FAR = FRR 时对应的分数阈值）
3. 将该值写入 `scripts/inference.py` 的 `--threshold` 参数

旧阈值 `0.776` 来自已作废的 Plan B checkpoint（标签定义错误），不代表当前系统口径，禁止沿用。

---

## 10. 配置参数说明

`config/train.yaml` 关键参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model.hidden_size` | `256` | BiLSTM 单向隐藏单元数 |
| `model.num_layers` | `2` | LSTM 层数 |
| `model.dropout` | `0.2` | Dropout 比例 |
| `model.attention_hidden_size` | `128` | 注意力隐藏层大小 |
| `model.distance_metric` | `concat` | 相似度度量方式 |
| `training.batch_size` | `32` | 批大小 |
| `training.epochs` | `100` | 最大训练轮数 |
| `training.learning_rate` | `0.001` | 初始学习率 |
| `training.early_stopping_patience` | `15` | 早停耐心轮数 |
| `data.max_len` | `400` | 序列最大长度 |
| `data.threshold` | TBD | 推理判决阈值（重训后从 evaluate_svc2004_protocol 结果标定，旧值 0.776 已作废） |

---

## 附：性能指标

> **以下数据已作废**，来自历史 checkpoint（正负对定义错误 + writer 泄漏），不可用于交付或论文。  
> 详见 `outputs/checkpoints/deprecated/DEPRECATED.md`。

| 指标 | 验证集 | 测试集 |
|------|--------|--------|
| Accuracy | ~~90.39%~~ | ~~90.07%~~ |
| EER | ~~9.83%~~ | ~~9.55%~~ |
| AUC | ~~0.9654~~ | ~~0.9695~~ |

**目标**（重新训练后衡量）：Accuracy ≥ 95%，EER < 7%（skilled forgery EER 和 random forgery EER 分别报告，与论文 Table 3 口径对齐）。
