# 在线手写签名验证系统 (Stroke-Based RNN)

## 项目概述
基于论文《A Stroke-Based RNN for Writer-Independent Online Signature Verification》实现的在线手写签名验证系统。使用笔画级RNN和注意力机制，在SVC2004 Task 2数据集上实现准确率≥95%、等错误率<7%的签名真伪判别。

## 技术栈
- **Python**: 3.8
- **TensorFlow**: 2.9.0 (CUDA 11.2)
- **NumPy**: 1.21-1.23
- **Pandas**: 1.3+
- **scikit-learn**: 1.0+
- **Matplotlib**: 3.5+
- **tqdm**: 4.65+
- **PyYAML**: 6.0+
- **pytest**: 7.4+

## 硬件环境
- **本地开发**: Windows 11, CPU调试
- **服务器训练**: NVIDIA RTX 5090 (24GB VRAM), CUDA 11.2
- **服务器系统**: Linux (Ubuntu 20.04+)

## 目录结构
```
ZZU/
├── AGENTS.md                    # 项目说明文档
├── README.md                    # 用户使用文档
├── requirements.txt             # 依赖列表
├── setup.py                     # 安装配置
├── config/
│   ├── default.yaml            # 默认配置
│   ├── train.yaml              # 训练配置
│   └── eval.yaml               # 评估配置
├── data/
│   ├── __init__.py
│   ├── dataset.py              # SVC2004数据集类
│   ├── feature_extractor.py    # 23维特征提取
│   ├── augmentation.py         # 数据增强
│   ├── pair_sampler.py         # 正负样本对生成
│   └── utils.py                # 数据工具函数
├── models/
│   ├── __init__.py
│   ├── stroke_rnn.py           # 笔画级RNN模型
│   ├── attention.py            # 注意力机制
│   ├── siamese.py              # 孪生网络
│   └── losses.py               # 损失函数
├── training/
│   ├── __init__.py
│   ├── trainer.py              # 训练器
│   ├── evaluator.py            # 评估器
│   └── callbacks.py            # 训练回调
├── utils/
│   ├── __init__.py
│   ├── config.py               # 配置管理
│   ├── logger.py               # 日志管理
│   ├── metrics.py              # 评估指标(EER/ACC)
│   └── visualization.py        # 可视化工具
├── scripts/
│   ├── train.py                # 训练脚本
│   ├── evaluate.py             # 评估脚本
│   ├── preprocess.py           # 数据预处理
│   └── inference.py            # 推理脚本
├── tests/
│   ├── test_dataset.py         # 数据集测试
│   ├── test_model.py           # 模型测试
│   └── test_features.py        # 特征提取测试
├── notebooks/
│   ├── data_exploration.ipynb  # 数据探索
│   └── result_analysis.ipynb   # 结果分析
├── tasks/
│   ├── current.md              # 当前任务
│   ├── progress.md             # 进度快照
│   └── decisions.md            # 技术决策记录
├── outputs/
│   ├── checkpoints/            # 模型检查点
│   ├── logs/                   # 训练日志
│   ├── results/                # 评估结果
│   └── visualizations/         # 可视化图表
└── raw_data/
    └── SVC2004_Task2/          # 原始数据集
        ├── train/
        └── test/
```

## 环境变量
```bash
# 数据路径
export DATA_ROOT="./raw_data/SVC2004_Task2"
export OUTPUT_DIR="./outputs"

# 训练配置
export CUDA_VISIBLE_DEVICES="0"
export BATCH_SIZE="32"
export NUM_EPOCHS="100"
export LEARNING_RATE="0.001"

# 日志配置
export LOG_LEVEL="INFO"
export TENSORBOARD_DIR="./outputs/logs"

# 模型配置
export HIDDEN_SIZE="128"
export NUM_LAYERS="2"
export DROPOUT="0.3"
```

## 启动方式

### 1. 环境安装
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据预处理
```bash
# 提取特征并生成训练/验证列表
python scripts/preprocess.py \
    --data_root ./raw_data/SVC2004_Task2 \
    --output_dir ./outputs/features \
    --num_workers 4
```

### 3. 模型训练
```bash
# 使用默认配置训练
python scripts/train.py --config config/train.yaml

# 自定义参数训练
python scripts/train.py \
    --batch_size 64 \
    --epochs 150 \
    --lr 0.0005 \
    --gpu 0
```

### 4. 模型评估
```bash
# 评估最佳模型
python scripts/evaluate.py \
    --checkpoint ./outputs/checkpoints/best_model.pth \
    --data_root ./raw_data/SVC2004_Task2
```

### 5. 推理验证
```bash
# 验证两个签名是否为同一人
python scripts/inference.py \
    --signature1 ./test_data/U01_S1_N01.TXT \
    --signature2 ./test_data/U01_S1_N02.TXT \
    --checkpoint ./outputs/checkpoints/best_model.pth
```

### 6. TensorBoard监控
```bash
tensorboard --logdir ./outputs/logs --port 6006
```

---

## 服务器部署指南 (RTX 5090)

### 🚨 何时需要使用服务器
以下情况需要在服务器上运行：
- **Phase 4**: 首次完整模型训练（预计6-12小时）
- **Phase 5**: 超参数调优和性能优化（预计24-48小时）
- **Phase 8**: 对比实验和消融实验（预计12-24小时）

### 📋 服务器环境信息
- **GPU**: NVIDIA RTX 5090 (24GB VRAM)
- **系统**: Linux (Ubuntu 20.04+)
- **CUDA**: 12.1+
- **Python**: 3.8+

### 🔧 服务器部署步骤

#### 1. 上传代码到服务器
```bash
# 在本地打包项目（排除大文件）
tar -czf zzu_project.tar.gz \
    --exclude='outputs/*' \
    --exclude='raw_data/*' \
    --exclude='venv/*' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    .

# 上传到服务器（替换为实际服务器地址）
scp zzu_project.tar.gz username@server_ip:/path/to/project/

# SSH登录服务器
ssh username@server_ip

# 解压项目
cd /path/to/project/
tar -xzf zzu_project.tar.gz
```

#### 2. 上传数据集到服务器
```bash
# 在本地打包数据集
tar -czf svc2004_data.tar.gz raw_data/SVC2004_Task2/

# 上传到服务器
scp svc2004_data.tar.gz username@server_ip:/path/to/project/

# 在服务器上解压
ssh username@server_ip
cd /path/to/project/
tar -xzf svc2004_data.tar.gz
```

#### 3. 服务器环境配置
```bash
# 创建虚拟环境
python3.8 -m venv venv
source venv/bin/activate

# 安装TensorFlow (CUDA 11.2)
pip install tensorflow==2.9.0

# 安装其他依赖
pip install -r requirements.txt

# 验证GPU可用
python -c "import tensorflow as tf; print(f'TF version: {tf.__version__}'); print(f'GPU available: {tf.config.list_physical_devices(\"GPU\")}')"
```

#### 4. 数据预处理（服务器）
```bash
# 激活环境
source venv/bin/activate

# 运行预处理（使用多核加速）
python scripts/preprocess.py \
    --data_root ./raw_data/SVC2004_Task2 \
    --output_dir ./outputs/features \
    --num_workers 8
```

#### 5. 启动训练（后台运行）
```bash
# 使用nohup后台运行，输出重定向到日志
nohup python scripts/train.py \
    --config config/train.yaml \
    --gpu 0 \
    --batch_size 64 \
    --epochs 100 \
    > train.log 2>&1 &

# 查看进程
ps aux | grep train.py

# 实时查看日志
tail -f train.log

# 或使用tmux/screen保持会话
tmux new -s training
python scripts/train.py --config config/train.yaml
# Ctrl+B, D 分离会话
# tmux attach -t training 重新连接
```

#### 6. 监控训练进度
```bash
# 方法1: 查看日志文件
tail -f outputs/logs/train_*.log

# 方法2: TensorBoard（需要端口转发）
# 在服务器上启动TensorBoard
tensorboard --logdir ./outputs/logs --port 6006 --bind_all

# 在本地浏览器访问（需要SSH端口转发）
# ssh -L 6006:localhost:6006 username@server_ip
# 然后访问 http://localhost:6006
```

#### 7. 下载训练结果
```bash
# 在本地执行，下载模型和结果
scp -r username@server_ip:/path/to/project/outputs/checkpoints ./outputs/
scp -r username@server_ip:/path/to/project/outputs/results ./outputs/
scp -r username@server_ip:/path/to/project/outputs/logs ./outputs/
```

### 🔍 常见问题排查

#### GPU内存不足
```bash
# 减小batch size
python scripts/train.py --batch_size 32  # 或16

# 启用梯度累积
python scripts/train.py --gradient_accumulation_steps 2
```

#### 训练中断恢复
```bash
# 从检查点恢复训练
python scripts/train.py \
    --resume ./outputs/checkpoints/last_checkpoint.pth \
    --config config/train.yaml
```

#### 查看GPU使用情况
```bash
# 实时监控GPU
watch -n 1 nvidia-smi

# 或使用gpustat
pip install gpustat
gpustat -i 1
```

### 📝 服务器使用检查清单

**训练前检查**:
- [ ] 代码已上传到服务器
- [ ] 数据集已上传并解压
- [ ] 虚拟环境已创建并激活
- [ ] PyTorch CUDA版本已安装
- [ ] GPU可用性已验证 (`torch.cuda.is_available()`)
- [ ] 特征预处理已完成
- [ ] 配置文件参数已确认

**训练中监控**:
- [ ] 训练进程正常运行 (`ps aux | grep train`)
- [ ] GPU利用率正常 (`nvidia-smi`)
- [ ] 日志文件正常输出 (`tail -f train.log`)
- [ ] TensorBoard曲线正常
- [ ] 磁盘空间充足 (`df -h`)

**训练后操作**:
- [ ] 下载最佳模型权重
- [ ] 下载训练日志和结果
- [ ] 下载TensorBoard日志
- [ ] 清理临时文件释放空间
- [ ] 更新 `tasks/progress.md`

---

## API 规范

### 核心接口

#### 1. 数据加载
```python
from data.dataset import SVC2004Dataset

dataset = SVC2004Dataset(
    data_root='./raw_data/SVC2004_Task2',
    split='train',
    feature_cache='./outputs/features'
)
```

#### 2. 模型初始化
```python
from models.siamese import SiameseStrokeRNN

model = SiameseStrokeRNN(
    input_size=23,
    hidden_size=128,
    num_layers=2,
    dropout=0.3
)
```

#### 3. 训练
```python
from training.trainer import Trainer

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config
)
trainer.train()
```

#### 4. 评估
```python
from training.evaluator import Evaluator

evaluator = Evaluator(model, test_loader)
metrics = evaluator.evaluate()
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"EER: {metrics['eer']:.4f}")
```

### 特征提取函数
```python
from data.feature_extractor import extract_temporal_features

# 输入: (N, 6) 原始点 [id, x, y, t, p, pen_status]
# 输出: (N, 23) 时序特征
features = extract_temporal_features(points)
```

### 评估指标
```python
from utils.metrics import calculate_eer, calculate_accuracy

eer = calculate_eer(y_true, y_scores)
acc = calculate_accuracy(y_true, y_pred)
```

## 开发规范

### 命名规范
- **文件名**: 小写+下划线 `feature_extractor.py`
- **类名**: 大驼峰 `StrokeRNN`
- **函数名**: 小写+下划线 `extract_features()`
- **常量**: 大写+下划线 `MAX_STROKE_LENGTH`
- **私有方法**: 前缀下划线 `_compute_velocity()`

### 代码风格
- 遵循 PEP 8 规范
- 使用 black 格式化 (行长120)
- 使用 flake8 检查
- 类型注解: `def func(x: np.ndarray) -> float:`
- 文档字符串: Google风格

### 提交规范
```
<type>(<scope>): <subject>

<body>

<footer>
```

**Type类型**:
- `feat`: 新功能
- `fix`: 修复bug
- `refactor`: 重构
- `test`: 测试
- `docs`: 文档
- `style`: 格式
- `perf`: 性能优化

**示例**:
```
feat(model): 添加多头注意力机制

- 实现MultiheadAttention层
- 在StrokeRNN中集成注意力
- 添加注意力权重可视化

Closes #12
```

### 测试规范
- 单元测试覆盖率 > 80%
- 每个模块必须有对应测试文件
- 使用 pytest 运行: `pytest tests/ -v`
- 关键函数必须有边界测试

## 任务管理

### 工作流程
每次新 session 开头：
1. 读取 `AGENTS.md` 了解项目全貌
2. 读取 `tasks/progress.md` 了解当前进度
3. 读取 `tasks/decisions.md` 了解技术决策
4. 读取 `tasks/current.md` 执行当前阶段任务

每次结束前：
1. 更新 `tasks/progress.md` 记录完成内容
2. 如有新技术决策，更新 `tasks/decisions.md`
3. 如当前阶段完成，更新 `tasks/current.md` 到下一阶段

### 文件说明
- **tasks/current.md**: 当前阶段的详细任务清单
- **tasks/progress.md**: 各阶段完成情况快照
- **tasks/decisions.md**: 重要技术决策及理由

## 禁止事项
1. **不允许删除原始数据**: `raw_data/` 目录下的所有文件
2. **不允许修改环境变量文件**: `.env` 文件需手动编辑
3. **不允许直接修改配置文件**: 使用命令行参数覆盖
4. **不允许提交大文件**: 模型权重、数据集不进入版本控制
5. **不允许硬编码路径**: 所有路径使用配置文件或环境变量
6. **不允许跳过测试**: 修改代码后必须运行相关测试
7. **不允许使用全局变量**: 除常量外禁止全局状态
8. **不允许忽略异常**: 必须正确处理或向上抛出
9. **不允许使用 print 调试**: 使用 logging 模块
10. **不允许提交未格式化代码**: 提交前运行 black 和 flake8

## 性能目标
- **准确率 (Accuracy)**: ≥ 95%
- **等错误率 (EER)**: < 7%
- **训练时间**: < 24小时 (单GPU)
- **推理速度**: < 100ms/对 (CPU)
- **内存占用**: < 4GB (训练时)

## 参考资料
- 论文: A Stroke-Based RNN for Writer-Independent Online Signature Verification
- 数据集: SVC2004 Task 2
- 原始代码: StrokeBasedRNN-master/
