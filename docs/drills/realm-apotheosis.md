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

#### 韩立演练：题 1 全解

```python
import torch  # ① 引入 PyTorch，以便韩立在灵兽山洞中调度张量灵气。
device = "cuda" if torch.cuda.is_available() else "cpu"  # ② 判断可用设备，呼应韩立在乱星海时随时切换飞剑与遁光。
tensor = torch.rand((2, 3), dtype=torch.float32, device=device)  # ③ 生成 2×3 随机张量，模拟韩立吸纳的六缕天地灵气。
print(tensor.shape)  # ④ 输出张量形状，让韩立确认灵气阵列尺寸。
print(tensor.dtype)  # ⑤ 输出数据类型，对应灵气纯度。
print(tensor.device)  # ⑥ 输出张量所在设备，确保与韩立的法器契合。
tensor_cpu = tensor.cpu()  # ⑦ 若当前在 GPU，则转回凡间（CPU），仿佛韩立回归黄枫谷闭关。
print(tensor_cpu.device)  # ⑧ 验证转换生效。
if torch.cuda.is_available():  # ⑨ 若韩立手握噬灵火（GPU），再演示返回飞升界（CUDA）。
    tensor_cuda = tensor_cpu.cuda()
    print(tensor_cuda.device)  # ⑩ 输出 CUDA 设备位置。
```

- **①** 导入 `torch` 后，韩立便能操控张量之力。
- **②** 判断 `cuda` 可用性时，若有 GPU 会返回 `'cuda'`，否则 `'cpu'`，如同韩立随身携带的青竹蜂云剑。
- **③** `torch.rand((2, 3), ...)` 生成一个 2 行 3 列的随机张量，数值均匀分布在 `[0, 1)`，象征六道灵气柱。
- **④-⑥** 依次打印形状 `torch.Size([2, 3])`、数据类型 `torch.float32`、设备如 `cuda:0` 或 `cpu`。
- **⑦-⑧** `tensor.cpu()` 返回一个位于 CPU 的张量，输出 `cpu`，对应韩立短暂回到七玄门。
- **⑨-⑩** 若存在 GPU，再次 `.cuda()` 会输出 `cuda:0`，象征韩立驾驭紫纹雷竹飞升高空。

示例输出（假设存在 GPU）：

```
torch.Size([2, 3])
torch.float32
cuda:0
cpu
cuda:0
```

#### 韩立演练：题 2 全解

```python
import numpy as np  # ① 先引入凡俗学堂的 NumPy。
import torch  # ② 再次引入 PyTorch，让韩立比较两种灵气容器。
arr = np.ones((2, 2), dtype=np.float32)  # ③ 构造 2×2 的灵石矩阵，初始每颗灵石含量为 1。
tensor = torch.from_numpy(arr)  # ④ 将 NumPy 阵列转化为张量，保持共用底层灵力脉络。
print("初始 NumPy:")
print(arr)  # ⑤ 输出 NumPy 数组。
print("初始 Tensor:")
print(tensor)  # ⑥ 输出 PyTorch 张量。
arr[0, 0] = 5.0  # ⑦ 韩立以玄天果汁灌注第一颗灵石。
print("修改后 NumPy:")
print(arr)  # ⑧ NumPy 视图更新。
print("同步的 Tensor:")
print(tensor)  # ⑨ PyTorch 张量同步变化，体现同源。
detached = torch.tensor(arr.copy())  # ⑩ 若韩立想封存副本，可显式复制后再建张量。
```

- **①-②** 分别导入 NumPy 与 PyTorch，准备灵石（数组）与法阵（张量）。
- **③** `np.ones((2, 2))` 创建全 1 阵列，表示四颗灵石初始灵气相等。
- **④** `torch.from_numpy` 会共享内存，犹如韩立以《青元剑诀》连接灵石脉络。
- **⑤-⑥** 打印可见两者均为 `[[1. 1.]
 [1. 1.]]`。
- **⑦-⑨** 当 NumPy 阵列中某元素更新为 `5.0` 时，张量同步显示 `5.0`，仿佛韩立在紫灵岛将灵气注入同一法阵。
- **⑩** 若使用 `torch.tensor(arr.copy())`，则张量与原数组互不影响，相当于韩立在天南另起一炉丹药。

示例输出：

```
初始 NumPy:
[[1. 1.]
 [1. 1.]]
初始 Tensor:
tensor([[1., 1.],
        [1., 1.]])
修改后 NumPy:
[[5. 1.]
 [1. 1.]]
同步的 Tensor:
tensor([[5., 1.],
        [1., 1.]])
```

#### 韩立演练：题 3 全解

```python
import random  # ① 引入凡俗骰子，用于控制韩立闭关时的机缘。
import numpy as np  # ② 导入 NumPy，确保洞府内阵法一致。
import torch  # ③ 导入 PyTorch，统一随机种子。
seed = 1640  # ④ 设定韩立在太南小世界的纪年作为随机种子。
random.seed(seed)  # ⑤ 固定 Python 标准库随机性。
np.random.seed(seed)  # ⑥ 固定 NumPy。
torch.manual_seed(seed)  # ⑦ 固定 CPU 上的张量随机数。
if torch.cuda.is_available():  # ⑧ 若韩立携带噬金虫（GPU），需进一步固定 CUDA。
    torch.cuda.manual_seed_all(seed)
print(random.random())  # ⑨ 输出一个伪随机数，韩立预测灵兽出世概率。
print(np.random.rand(2))  # ⑩ 输出 NumPy 随机数组。
print(torch.rand(2))  # ⑪ 输出 PyTorch 随机张量。
```

- **①-③** 分别导入 `random`、`numpy`、`torch`，对应韩立调度的三类法阵。
- **④** 选定固定值 `1640`，可替换为任意整数。
- **⑤-⑦** 依次设种子，确保多次运行得到相同结果，犹如韩立在化神战场布下可重复的幻阵。
- **⑧** 对 GPU 调用 `manual_seed_all`，防止多张显卡结果不一致。
- **⑨-⑪** 打印出的随机值在每次运行中保持稳定，示例：

```
0.7208888851403657
[0.93871914 0.81585354]
tensor([0.0250, 0.3015])
```

韩立可据此验证闭关前后的灵气波动始终一致。

#### 韩立演练：题 4 全解

```python
import torch  # ① 再次召唤 PyTorch 灵阵。
x = torch.tensor([2.0], requires_grad=True)  # ② 定义标量张量 x，开启梯度追踪，如韩立注视金阙玉书。
y = x ** 2  # ③ 令 y = x²，模拟功法运转带来的灵力提升。
print(y)  # ④ 输出当前的 y。
y.backward()  # ⑤ 反向传播，相当于韩立逆推功法脉络。
print(x.grad)  # ⑥ 查看梯度，即 d(x²)/dx = 2x。
x.grad.zero_()  # ⑦ 将梯度清零，准备下一轮演练。
z = x ** 3  # ⑧ 更换功法为 x³。
z.backward()  # ⑨ 再次回传。
print(x.grad)  # ⑩ 查看新的梯度 3x²。
```

- **①-②** 创建 `x` 时设置 `requires_grad=True`，表示此灵力节点需记录梯度。
- **③-④** 计算 `y = x ** 2`，输出 `tensor([4.], grad_fn=<PowBackward0>)`。
- **⑤-⑥** `y.backward()` 后，`x.grad` 为 `tensor([4.])`，因为 `2 * 2 = 4`。
- **⑦** `zero_()` 将梯度原地清零，防止累积，如同韩立在虚天鼎前重置灵力。
- **⑧-⑩** 再次计算 `z = x ** 3`，回传后梯度为 `tensor([12.])`，对应 `3 * 2² = 12`。

示例输出：

```
tensor([4.], grad_fn=<PowBackward0>)
tensor([4.])
tensor([12.])
```

#### 韩立演练：题 5 全解

```python
import torch  # ① 引入 PyTorch。
x = torch.tensor([3.0], requires_grad=True)  # ② 准备含梯度的灵力节点。
with torch.no_grad():  # ③ 在无梯度语境中，韩立仿佛施展《大庚剑诀》中的隐匿篇，屏蔽灵气波动。
    y = x * 2  # ④ 计算 y = 6，此时不会记录梯度。
print(y)  # ⑤ 输出张量，显示无 grad_fn。
z = x.detach()  # ⑥ 使用 detach 复制视图，仍共享内存但不追踪梯度。
print(z)  # ⑦ 输出张量，device 与值与 x 相同。
loss = (x ** 2).sum()  # ⑧ 定义可回传的损失。
loss.backward()  # ⑨ 回传后 x.grad 得到 6。
print(x.grad.detach().cpu().item())  # ⑩ 通过 detach()+cpu()+item() 提取纯数值供韩立记录。
```

- **①-②** 创建含梯度的变量。
- **③-④** 进入 `torch.no_grad()` 后的运算不会被计算图记录，打印 `tensor([6.])`，无 `grad_fn`。
- **⑥-⑦** `detach()` 返回与 `x` 共享存储的张量；若后续修改 `x`，`z` 同步。
- **⑧-⑨** 计算 `loss = x ** 2` 并回传，`x.grad` 变为 `tensor([6.])`。
- **⑩** 使用 `detach()` 防止梯度继续参与图计算，`cpu()` 便于在无 GPU 环境输出，`item()` 转换为 Python 浮点数。

示例输出：

```
tensor([6.])
tensor([3.])
6.0
```

## 法身塑造 · 模型构建

### 题目回顾
1. 🌱 入门：自定义 `Net(nn.Module)` 并实现 `forward` 前向逻辑。
2. 🌿 进阶：使用 `torchsummary` 或 `torchinfo` 查看参数量。
3. 🔥 突破：编写标准训练/验证循环，包含梯度清零与反向传播。
4. 🌟 圆满：引入学习率调度器，如 `StepLR` 或 `ReduceLROnPlateau`。
5. 🛡️ 化神：将训练指标写入 TensorBoard 便于监控。

### 逐题拆解

#### 韩立演练：题 1 全解

```python
import torch
import torch.nn as nn  # ① 引入神识构筑模块。

class Net(nn.Module):  # ② 定义模型，如韩立炼制的青元剑阵。
    def __init__(self):
        super().__init__()  # ③ 初始化父类，确保内功心法继承。
        self.flatten = nn.Flatten()  # ④ 将 28×28 灵纹摊平成 784 长度。
        self.fc = nn.Linear(784, 10)  # ⑤ 全连接层，输出十种灵器类别。

    def forward(self, x):
        x = self.flatten(x)  # ⑥ 展平输入，仿佛韩立将银月的幻化拆解为单线条。
        logits = self.fc(x)  # ⑦ 线性映射，得到各类别的灵气值。
        return logits  # ⑧ 返回 logits，供后续损失计算。

model = Net()  # ⑨ 实例化模型，如同韩立在虚天鼎内炼成法器。
sample = torch.randn(16, 1, 28, 28)  # ⑩ 生成 16 张手写符文。
output = model(sample)  # ⑪ 前向传播。
print(output.shape)  # ⑫ 输出形状，验证为 [16, 10]。
```

- 运行后输出 `torch.Size([16, 10])`，确认批量大小与类别维度正确。

#### 韩立演练：题 2 全解

```python
from torchinfo import summary  # ① 引入 torchinfo，仿佛韩立手持天机符卷洞察法器结构。
model = Net()  # ② 沿用上题模型。
report = summary(model, input_size=(64, 1, 28, 28))  # ③ 生成摘要，批次 64。
print(report)  # ④ 打印详细参数。
```

- 输出会列出每层形状与参数数量，例如 `Linear` 层参数为 7,850，帮助韩立检视功法结构。

#### 韩立演练：题 3 全解

```python
for epoch in range(1, epochs + 1):  # ① 外层循环遍历轮次，相当于韩立闭关多次冲击瓶颈。
    model.train()  # ② 切换训练模式，激活诸如 Dropout 的阵法。
    for batch_idx, (features, labels) in enumerate(train_loader, start=1):  # ③ 遍历批次，记录索引。
        optimizer.zero_grad()  # ④ 清理旧梯度，犹如韩立每次运功前先归元。
        preds = model(features)  # ⑤ 前向传播，获取预测灵压。
        loss = criterion(preds, labels)  # ⑥ 计算损失，评估偏差。
        loss.backward()  # ⑦ 反向传播，追踪灵力流向。
        optimizer.step()  # ⑧ 更新参数，韩立调整功法路线。
        if batch_idx % 50 == 0:
            print(f"第{epoch}轮-第{batch_idx}批，训练损失={loss.item():.4f}")  # ⑨ 输出阶段性战报。
    model.eval()  # ⑩ 切换为评估模式。
    with torch.no_grad():  # ⑪ 禁用梯度，避免灵力波动干扰验证。
        val_loss = 0.0
        for features, labels in val_loader:  # ⑫ 遍历验证集。
            preds = model(features)
            val_loss += criterion(preds, labels).item()
    val_loss /= len(val_loader)  # ⑬ 平均化验证损失。
    print(f"第{epoch}轮验证损失={val_loss:.4f}")  # ⑭ 报告成果，如同韩立与南宫婉互通密信。
```

- 示例输出：`第1轮-第50批，训练损失=0.9451`、`第1轮验证损失=0.8123`。

#### 韩立演练：题 4 全解

```python
from torch.optim.lr_scheduler import StepLR  # ① 引入学习率调度器。
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # ② 每 5 轮衰减一次，如韩立逐步收束灵力。
for epoch in range(1, epochs + 1):
    train_one_epoch()
    validate()
    scheduler.step()  # ③ 在每轮结束后调整学习率。
    print(f"第{epoch}轮学习率={scheduler.get_last_lr()[0]:.6f}")  # ④ 输出当前学习率。
```

- 输出示例：`第1轮学习率=0.010000`、`第5轮学习率=0.001000`，帮助韩立掌握灵力节奏。

#### 韩立演练：题 5 全解

```python
from torch.utils.tensorboard import SummaryWriter  # ① 引入日志法阵。
writer = SummaryWriter(log_dir="runs/hanli-apotheosis")  # ② 指定日志目录。
global_step = 0
for epoch in range(1, epochs + 1):
    for features, labels in train_loader:
        optimizer.zero_grad()
        preds = model(features)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        writer.add_scalar("loss/train", loss.item(), global_step)  # ③ 记录训练损失。
        global_step += 1
    writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)  # ④ 每轮记录学习率。
writer.close()  # ⑤ 关闭写入器，如同韩立合上炼丹日志。
print("运行 tensorboard --logdir runs/hanli-apotheosis 以查看修炼曲线")  # ⑥ 提示查看方式。
```

- 打开 TensorBoard 后能观察损失与学习率曲线，韩立即可洞察修炼走势。

## 灵识扩展 · 计算机视觉与序列

### 题目回顾
1. 🌱 入门：使用 `torchvision.transforms` 对图像进行数据增强。
2. 🌿 进阶：构建简单的 LSTM 文本分类模型。
3. 🔥 突破：实现自定义 `Dataset`/`DataLoader` 用于非标准数据。
4. 🌟 圆满：运用 Grad-CAM 可视化模型关注区域。
5. 🛡️ 化神：对比不同批大小对训练速度与稳定性的影响。

### 逐题拆解

#### 韩立演练：题 1 全解

```python
from torchvision import transforms  # ① 引入图像炼丹秘术。
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ② 统一尺寸，犹如韩立将灵兽缩放入灵兽袋。
    transforms.RandomHorizontalFlip(p=0.5),  # ③ 随机水平翻转，模拟幻阵镜像。
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # ④ 微调亮度与对比度，如同洒下青竹蜂云剑光。
    transforms.ToTensor(),  # ⑤ 转为张量。
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ⑥ 验证集仅做尺度统一。
    transforms.ToTensor(),
])
```

- 训练集通过随机扰动提升模型鲁棒性，验证集保持稳定，方便韩立评估真实战力。

#### 韩立演练：题 2 全解

```python
import torch
import torch.nn as nn

class SpellClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # ① 将法诀符文映射为向量。
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)  # ② 设定 batch_first 方便处理。
        self.classifier = nn.Linear(hidden_dim, num_classes)  # ③ 输出法诀类别。

    def forward(self, inputs, lengths):
        embedded = self.embedding(inputs)  # ④ 嵌入层输出。
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)  # ⑤ 打包变长序列。
        _, (hidden, _) = self.lstm(packed)  # ⑥ 获取最后一层隐藏状态。
        logits = self.classifier(hidden[-1])  # ⑦ 使用末层隐藏态分类。
        return logits
```

- 前向结果形状为 `[batch_size, num_classes]`，韩立即可判断弟子适合哪门功法。

#### 韩立演练：题 3 全解

```python
from torch.utils.data import Dataset

class BeastDataset(Dataset):
    def __init__(self, records):
        self.records = records  # ① 保存灵兽档案。

    def __len__(self):
        return len(self.records)  # ② 返回样本数量。

    def __getitem__(self, idx):
        entry = self.records[idx]
        image = load_image(entry["image_path"])  # ③ 自定义读取灵兽画像。
        label = entry["label"]  # ④ 对应品阶。
        return image, label
```

- 搭配 `DataLoader(..., num_workers=4, collate_fn=custom_collate)` 可稳定处理异形数据，如韩立驭使众多傀儡。

#### 韩立演练：题 4 全解

```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

model.eval()  # ① 切换评估模式。
target_layers = [model.layer4[-1]]  # ② 指定注意力层。
cam = GradCAM(model=model, target_layers=target_layers)  # ③ 构建 Grad-CAM。
grayscale_cam = cam(input_tensor=inputs)  # ④ 生成热力图。
overlay = show_cam_on_image(rgb_image, grayscale_cam[0], use_rgb=True)  # ⑤ 将注意力叠加到原图。
```

- 韩立可借此洞察模型关注的灵兽部位，判断是否聚焦于关键法纹，如他曾辨识七绝门护山灵兽弱点。

#### 韩立演练：题 5 全解

```python
import time

for batch_size in [32, 64, 128]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    start = time.perf_counter()  # ① 记录起始时间。
    train_one_epoch(loader)
    duration = time.perf_counter() - start  # ② 计算耗时。
    val_loss = evaluate(val_loader)
    print(f"批次 {batch_size}: 耗时 {duration:.2f}s, 验证损失 {val_loss:.4f}")  # ③ 输出比较结果。
```

- 输出类似 `批次 64: 耗时 12.45s, 验证损失 0.8213`，韩立即可选择最合适的批次规模。

## 元神游历 · 迁移学习与部署

### 题目回顾
1. 🌱 入门：加载预训练 ResNet 并冻结大部分层。
2. 🌿 进阶：替换全连接层以适配新的类别数。
3. 🔥 突破：使用 `torch.jit.trace` 导出 TorchScript 模型。
4. 🌟 圆满：将 TorchScript 模型接入 FastAPI 推理服务。
5. 🛡️ 化神：设计健康检查、版本控制与性能监控。

### 逐题拆解

#### 韩立演练：题 1 全解

```python
import torchvision.models as models
import torch.nn as nn

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # ① 加载预训练模型，如韩立借用灵界上古阵基。
for param in model.parameters():
    param.requires_grad = False  # ② 冻结主干参数，保留原有法力底蕴。
model.fc = nn.Linear(model.fc.in_features, num_classes)  # ③ 替换顶层以适配新任务。
```

- 仅训练新层即可快速适应灵兽分类，节省韩立的灵石消耗。

#### 韩立演练：题 2 全解

```python
nn.init.kaiming_normal_(model.fc.weight)  # ① 以开明初始化激活新层，如韩立点燃玄天火。 
model.fc.bias.data.zero_()  # ② 偏置清零，保持平衡。
```

- 通过显式初始化确保训练初期稳定，避免像落云宗大战时法阵失衡。

#### 韩立演练：题 3 全解

```python
import torch

example = torch.randn(1, 3, 224, 224)  # ① 构造示例输入。
traced = torch.jit.trace(model, example)  # ② 追踪模型计算图。
traced.save("model.ts")  # ③ 保存成 TorchScript 文件。
```

- 若模型含条件分支需改用 `torch.jit.script`，否则会遗漏动态逻辑。

#### 韩立演练：题 4 全解

```python
from fastapi import FastAPI
from fastapi.responses import Response
import torch
import pandas as pd

app = FastAPI(title="韩立灵兽鉴定台")
ts_model = torch.jit.load("model.ts")  # ① 加载脚本模型。
ts_model.eval()  # ② 切换推理模式。

@app.post("/predict")
def predict(payload: dict):
    df = pd.DataFrame([payload])  # ③ 将请求转为 DataFrame。
    tensor = preprocess(df)  # ④ 复用训练期预处理。
    with torch.no_grad():
        logits = ts_model(tensor)
        proba = logits.softmax(dim=1)  # ⑤ 计算概率分布。
    return {"probability": proba[0].tolist()}  # ⑥ 返回可 JSON 化的列表。
```

- 韩立可在坊市摊位上实时鉴定灵兽潜力。

#### 韩立演练：题 5 全解

```python
import time

@app.get("/healthz")
def healthz():
    start = time.perf_counter()
    try:
        _ = ts_model(torch.zeros(1, 3, 224, 224))  # ① 进行空推理，确保模型存活。
    except Exception as exc:
        return Response(status_code=503, content=f"异常: {exc}")
    latency = (time.perf_counter() - start) * 1000
    return {"status": "ok", "model_version": "v1.0", "latency_ms": round(latency, 2)}  # ② 返回健康状态。
```

- 该守护接口帮助韩立随时监控服务延迟，防止像逆星盟突袭时措手不及。

## 道果巩固 · 持续精进

### 题目回顾
1. 🌱 入门：选择一篇近期论文或官方教程进行复现。
2. 🌿 进阶：使用 PyTorch Lightning / Accelerate 重构训练流程。
3. 🔥 突破：尝试混合精度训练提升效率。
4. 🌟 圆满：编写自动化部署脚本，包含打包与上线步骤。
5. 🛡️ 化神：规划下一阶段进阶主题（NLP、CV、RL 等）。

### 逐题拆解

#### 韩立演练：题 1 全解

```python
from pathlib import Path
import json

plan = {
    "paper": "CutMix: Regularization Strategy to Train Strong Classifiers",  # ① 选定复现目标，韩立如同挑选上古功法。
    "repo": "https://github.com/clovaai/CutMix-PyTorch",  # ② 标记官方代码。
    "checkpoints": [],
}

notes_dir = Path("hanli_research/notes")  # ③ 创建记录目录。
notes_dir.mkdir(parents=True, exist_ok=True)

for section in ["数据预处理", "模型结构", "训练超参", "实验结果"]:  # ④ 逐段拆解论文。
    file = notes_dir / f"{section}.md"
    file.write_text(f"# {section}\n韩立复盘：\n")  # ⑤ 写入初始笔记。
    plan["checkpoints"].append(file.name)  # ⑥ 记录生成的文件。

Path("hanli_research/plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2))  # ⑦ 输出整体计划。
```

- 执行后会生成笔记文件与 `plan.json`，韩立即可随时复盘与分享。

#### 韩立演练：题 2 全解

```python
import pytorch_lightning as pl
import torch.nn.functional as F

class HanLiClassifier(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone  # ① 注入模型骨架，仿佛韩立请傀儡坐镇。

    def forward(self, x):
        return self.backbone(x)  # ② Lightning 自动处理分布式同步。

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)  # ③ 前向传播。
        loss = F.cross_entropy(logits, y)  # ④ 计算损失。
        self.log("train_loss", loss, on_step=True, prog_bar=True)  # ⑤ 实时记录。
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("val_loss", loss, prog_bar=True)  # ⑥ 验证损失。

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)  # ⑦ 指定优化器。
```

- 结合 `Trainer(max_epochs=10)` 即可启动训练，Lightning 负责日志与多卡同步，宛如韩立坐镇天机堂调度诸峰弟子。

#### 韩立演练：题 3 全解

```python
scaler = torch.cuda.amp.GradScaler()  # ① 准备混合精度缩放器，如韩立驾驭梵圣真火。
for epoch in range(epochs):
    for features, labels in train_loader:
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():  # ② 在自动混合精度环境运算。
            logits = model(features)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()  # ③ 缩放后回传。
        scaler.step(optimizer)  # ④ 更新参数。
        scaler.update()  # ⑤ 调整缩放因子，防止溢出。
```

- 借此韩立即可在灵界大战中兼顾速度与稳定性，减少显存消耗。

#### 韩立演练：题 4 全解

```python
from subprocess import run

scripts = [
    ("lint", ["ruff", "check", "app"]),  # ① 代码规约，如韩立检查法阵纹路。
    ("tests", ["pytest", "-q"]),  # ② 单元试炼。
    ("build", ["docker", "build", "-t", "hanli/vision:latest", "."]),  # ③ 构建镜像。
    ("push", ["docker", "push", "hanli/vision:latest"]),  # ④ 推送到灵界仓库。
]

for name, command in scripts:
    result = run(command, check=True)
    print(f"步骤 {name} 完成，返回码 {result.returncode}")  # ⑤ 输出每步结果。
```

- 可结合 CI（如 GitHub Actions）执行此脚本，实现从测试到部署的一条龙流程。

#### 韩立演练：题 5 全解

```python
roadmap = [
    {"quarter": "Q1", "focus": "NLP Transformer", "milestone": "完成小型翻译模型", "ally": "小天", "story": "重现乱星海拍卖会"},  # ①
    {"quarter": "Q2", "focus": "CV 医疗影像", "milestone": "提交 Kaggle 竞赛", "ally": "南宫婉", "story": "联手炼制辟毒丹"},
    {"quarter": "Q3", "focus": "RL", "milestone": "复现 DQN 在修仙塔", "ally": "婴童金阙", "story": "操控傀儡守卫"},
]

for item in roadmap:
    print(f"季度 {item['quarter']} · 主题 {item['focus']} · 目标 {item['milestone']} · 搭档 {item['ally']} · 剧情 {item['story']}")  # ② 输出规划。
```

- 运行后将打印三阶段计划，帮助韩立像布局天渊城大战那样未雨绸缪，并将学习目标与《凡人修仙传》剧情相呼应。
