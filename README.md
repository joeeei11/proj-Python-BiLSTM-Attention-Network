![Language](https://img.shields.io/badge/language-Python-blue) ![License](https://img.shields.io/badge/license-MIT-green)

# proj-Python-BiLSTM-Attention-Network

**基于 BiLSTM 与注意力机制的孪生网络，用于序列相似度计算与脑电/手势信号识别。**

## 功能特性

- 孪生网络架构，支持序列对相似度比较
- 多头注意力机制强化时序特征提取
- 提供 FastAPI REST 接口，支持在线推理
- 完整训练 / 评估 / 批量推理流水线
- YAML 配置化训练参数

## 快速开始

### 环境要求

- Python >= 3.8
- TensorFlow / Keras >= 2.10
- FastAPI, uvicorn

### 安装步骤

1. 克隆仓库

```bash
git clone https://github.com/joeeei11/proj-Python-BiLSTM-Attention-Network.git
cd proj-Python-BiLSTM-Attention-Network
```

2. 安装依赖

```bash
pip install -r requirements.txt
```

3. 启动 API 服务

```bash
cd api && uvicorn app:app --reload
```

### 基础用法

```bash
python scripts/train.py --config config/train.yaml
python scripts/batch_inference.py
python scripts/evaluate.py
```
