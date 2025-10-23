# 化神境题库详解

化神阶段要将深度学习炼至元神出窍：熟悉 PyTorch 张量与自动求导、构建与训练模型、拓展视觉/序列任务，并掌握迁移学习和部署。以下详解每个试炼的关键步骤与注意事项。

## 难度阶梯导航

| 难度 | 核心关键词 | 代表练习 |
| --- | --- | --- |
| 🌱 入门 | 张量基础 | 创建张量、切换设备、掌握自动求导 |
| 🌿 进阶 | 模型构建 | 自定义 `nn.Module`、搭建训练/验证循环 |
| 🔥 突破 | 视觉与序列 | 数据增强、LSTM/CNN、Grad-CAM 可视化 |
| 🌟 圆满 | 迁移学习 | 冻结/解冻策略、性能对比表、实验记录 |
| 🛡️ 化神 | 部署守护 | 导出 TorchScript/ONNX、FastAPI 推理、监控回滚 |

## 实验管理提示

- **小步验证**：先在迷你数据集上跑通流程，确认损失下降、指标计算无误，再扩大规模。
- **记录配置**：把超参数、硬件信息、训练时长写入 YAML/JSON 或实验日志，方便复现实验。
- **观察梯度**：定期检查梯度范数与学习率调度，防止梯度爆炸/消失；必要时使用梯度裁剪。

## 元神凝练 · PyTorch 入门

### 题目回顾
1. 🌱 入门：创建张量、查看形状、数据类型以及所在设备。
2. 🌿 进阶：将 NumPy 数组转换为张量并验证共享内存。
3. 🔥 突破：设置随机种子，保证 CPU/GPU 的结果可复现。
4. 🌟 圆满：理解 `requires_grad`、`backward` 与梯度累积。
5. 🛡️ 化神：比较 `torch.no_grad()` 与 `detach()` 的应用场景。

### 逐题拆解
- **题 1**：`tensor = torch.rand((2, 3), dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')`；输出 `tensor.shape`, `tensor.dtype`, `tensor.device`，并演示 `tensor.cpu()`/`tensor.cuda()` 切换。
- **题 2**：`arr = np.ones((2, 2)); tensor = torch.from_numpy(arr)`；修改 `arr[0, 0] = 5` 后，`tensor` 同步变化。提醒若需独立副本使用 `torch.tensor(arr.copy())`。
- **题 3**：调用 `torch.manual_seed(42)`、`np.random.seed(42)`、`random.seed(42)`；若使用 GPU，再调用 `torch.cuda.manual_seed_all(42)`。在 DataLoader 中设置 `worker_init_fn` 以防并行 worker 打乱随机性。
- **题 4**：`x = torch.tensor([2.0], requires_grad=True); y = x**2; y.backward()`；查看 `x.grad`。演示多次 `backward()` 会累积梯度，需 `x.grad.zero_()` 清零。
- **题 5**：`with torch.no_grad():` 可在推理阶段禁用梯度计算，节省显存；`tensor.detach()` 返回共享内存的张量用于日志或可视化，但仍保留在计算图之外。提醒在训练循环中输出 `loss.detach().cpu().item()`。

## 法身塑造 · 模型构建

### 题目回顾
1. 🌱 入门：自定义 `Net(nn.Module)` 并实现 `forward` 前向逻辑。
2. 🌿 进阶：使用 `torchsummary` 或 `torchinfo` 查看参数量。
3. 🔥 突破：编写标准训练/验证循环，包含梯度清零与反向传播。
4. 🌟 圆满：引入学习率调度器，如 `StepLR` 或 `ReduceLROnPlateau`。
5. 🛡️ 化神：将训练指标写入 TensorBoard 便于监控。

### 逐题拆解
- **题 1**：定义类 `class Net(nn.Module): def __init__(self): super().__init__(); self.fc = nn.Linear(784, 10)`；`def forward(self, x): x = x.view(x.size(0), -1); return self.fc(x)`。
- **题 2**：`from torchinfo import summary; summary(model, input_size=(64, 1, 28, 28))`；检查每层输出形状与参数量，帮助定位维度不匹配问题。
- **题 3**：典型循环：```
for epoch in range(epochs):
    model.train()
    for x, y in train_loader:
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        val_loss = ...
```
强调 `optimizer.zero_grad()` 要放在每次迭代开始。
- **题 4**：`scheduler = StepLR(optimizer, step_size=5, gamma=0.1)`；在每个 epoch 末执行 `scheduler.step()`。说明 `ReduceLROnPlateau` 需传入验证集指标。
- **题 5**：`from torch.utils.tensorboard import SummaryWriter; writer = SummaryWriter(); writer.add_scalar('loss/train', loss.item(), step)`；训练结束 `writer.close()`，使用 `tensorboard --logdir runs` 查看。

## 灵识扩展 · 计算机视觉与序列

### 题目回顾
1. 🌱 入门：使用 `torchvision.transforms` 对图像进行数据增强。
2. 🌿 进阶：构建简单的 LSTM 文本分类模型。
3. 🔥 突破：实现自定义 `Dataset`/`DataLoader` 用于非标准数据。
4. 🌟 圆满：运用 Grad-CAM 可视化模型关注区域。
5. 🛡️ 化神：对比不同批大小对训练速度与稳定性的影响。

### 逐题拆解
- **题 1**：`transform = transforms.Compose([transforms.Resize(224), transforms.RandomHorizontalFlip(), transforms.ColorJitter(brightness=0.2)])`；解释训练/验证集应使用不同增强策略。
- **题 2**：构造 `Embedding` + `LSTM` + `Linear`，需要 `pack_padded_sequence` 处理变长序列。设置 `batch_first=True` 方便使用。
- **题 3**：重写 `__len__` 返回样本数量，`__getitem__` 返回 `(image, label)` 或 `(sequence, target)`。在 `DataLoader` 中设置 `num_workers`、`collate_fn` 处理变长数据。
- **题 4**：使用 `pytorch-grad-cam`：`cam = GradCAM(model=model, target_layers=[model.layer4[-1]])`；传入输入张量，得到热力图叠加。注意模型需处于 `eval()` 状态。
- **题 5**：实验不同 `batch_size`（如 32/64/128），记录 `epoch` 时间和验证损失；观察较大批次可能导致显存不足或收敛缓慢，推荐配合学习率调整。

## 元神游历 · 迁移学习与部署

### 题目回顾
1. 🌱 入门：加载预训练 ResNet 并冻结大部分层。
2. 🌿 进阶：替换全连接层以适配新的类别数。
3. 🔥 突破：使用 `torch.jit.trace` 导出 TorchScript 模型。
4. 🌟 圆满：将 TorchScript 模型接入 FastAPI 推理服务。
5. 🛡️ 化神：设计健康检查、版本控制与性能监控。

### 逐题拆解
- **题 1**：`model = torchvision.models.resnet18(weights='DEFAULT')`；遍历 `model.parameters()` 设置 `requires_grad=False`，仅微调新头部。
- **题 2**：`model.fc = nn.Linear(model.fc.in_features, num_classes)`；若是 `EfficientNet` 等模型需修改 `classifier`。确保新层权重随机初始化。
- **题 3**：`example = torch.randn(1, 3, 224, 224); traced = torch.jit.trace(model, example); traced.save('model.ts')`；说明 `trace` 适用于结构固定模型，动态控制流需用 `script`。
- **题 4**：在 FastAPI 中加载 TorchScript：`model = torch.jit.load('model.ts'); model.eval()`；在接口中执行张量转换、归一化、推理，再将结果 `tolist()` 返回。
- **题 5**：实现 `/healthz` 返回 `{'status': 'ok', 'model_version': 'v1.0'}`；记录推理耗时 `time.perf_counter()` 并上报至日志或监控系统；为异常返回 503。

## 道果巩固 · 持续精进

### 题目回顾
1. 🌱 入门：选择一篇近期论文或官方教程进行复现。
2. 🌿 进阶：使用 PyTorch Lightning / Accelerate 重构训练流程。
3. 🔥 突破：尝试混合精度训练提升效率。
4. 🌟 圆满：编写自动化部署脚本，包含打包与上线步骤。
5. 🛡️ 化神：规划下一阶段进阶主题（NLP、CV、RL 等）。

### 逐题拆解
- **题 1**：挑选小型论文，例如《CutMix》或 `PyTorch tutorials`，逐段阅读实现，记录差异点。建议写博客总结复现经验。
- **题 2**：安装 `pytorch-lightning`，将训练循环重构为 `LightningModule`；重点理解 `training_step`、`validation_step`、`configure_optimizers`。
- **题 3**：`scaler = torch.cuda.amp.GradScaler()`；在训练循环中 `with torch.cuda.amp.autocast(): loss = criterion(model(x), y)`；使用 `scaler.scale(loss).backward()` 再 `scaler.step(optimizer); scaler.update()`。
- **题 4**：编写 `Dockerfile`、`docker-compose`、CI 配置脚本；实现自动化测试、镜像构建、推送仓库、部署到服务器/云平台。
- **题 5**：制作路线图，列出要深挖的专题、推荐课程或比赛，设置里程碑（如“完成 NLP Transformer 项目”），并安排复盘周期。
