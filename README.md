# 零基础 Python + 机器学习 + PyTorch 自学手册

这份仓库是一套针对零基础学习者的完整自学路线，覆盖从 Python 基础、数据分析、经典机器学习到 PyTorch 深度学习的全部关键知识。按照顺序学习，每一步都有明确目标、推荐资源、实践项目以及配套的“训练题 + 解析”，帮助你真正掌握技能并构建个人作品集。

## 目录

1. [学习总览](#学习总览)
2. [学习路径速览](#学习路径速览)
3. [阶段 0 · 零基础入门（第 1 周）](#阶段-0--零基础入门第-1-周)
4. [阶段 1 · Python 基础语法（第 2-3 周）](#阶段-1--python-基础语法第-2-3-周)
5. [阶段 2 · 数据分析与可视化（第 4-5 周）](#阶段-2--数据分析与可视化第-4-5-周)
6. [阶段 3 · 经典机器学习算法（第 6-8 周）](#阶段-3--经典机器学习算法第-6-8-周)
7. [阶段 4 · 项目化与部署实战（第 9-10 周）](#阶段-4--项目化与部署实战第-9-10-周)
8. [阶段 5 · 深度学习与 PyTorch（第 11-12 周）](#阶段-5--深度学习与-pytorch第-11-12-周)
9. [实战工具箱](#实战工具箱)
10. [项目灵感库](#项目灵感库)
11. [学习方法与复盘建议](#学习方法与复盘建议)
12. [附录 · 常见问题 & 资源推荐](#附录--常见问题--资源推荐)

---

## 学习总览

- **学习时长**：建议每周投入 10 ~ 12 小时，约 12 周完成全部内容。
- **准备工具**：Python 3.10+、Anaconda 或 Miniconda、VS Code、Git、Jupyter Notebook。
- **学习原则**：先理解概念 → 立即动手实践 → 复盘整理笔记 → 分享输出。
- **成果输出**：每一阶段完成至少 1 个小项目或报告，所有代码托管到 GitHub，构建学习档案。

> ⭐️ 建议按照“打基础 → 做项目 → 总结复盘”的节奏学习。每个阶段的实践任务和训练题都是检验成果的依据。

---

## 学习路径速览

| 周数 | 阶段 | 主要目标 | 推荐产出 |
| --- | --- | --- | --- |
| 第 1 周 | 阶段 0 | 完成环境搭建，熟悉命令行与 VS Code | 学习日志、环境搭建笔记 |
| 第 2-3 周 | 阶段 1 | 掌握 Python 语法与基础编程能力 | 猜数字游戏、通讯录脚本 |
| 第 4-5 周 | 阶段 2 | 熟悉 NumPy/Pandas 与可视化 | Titanic 数据探索报告 |
| 第 6-8 周 | 阶段 3 | 完成传统机器学习项目并调参 | 随机森林分类器、模型评估报告 |
| 第 9-10 周 | 阶段 4 | 学会模型服务化与部署 | FastAPI 推理接口、部署说明 |
| 第 11-12 周 | 阶段 5 | 入门 PyTorch 并完成深度学习项目 | MNIST/CIFAR-10 模型、复盘博客 |

**达成标准**：能独立完成一个从数据处理、建模到部署的端到端项目，并具备继续深入深度学习的基础。

---

## 阶段 0 · 零基础入门（第 1 周）

### 学习目标
- 熟悉操作系统基础、命令行与文件管理。
- 完成 Python、VS Code、Git、虚拟环境等开发环境搭建。

### 学习内容
1. 电脑基础：路径概念、压缩/解压、常用快捷键。
2. 命令行：`cd`、`ls/dir`、`mkdir`、`rm/rmdir`、`python --version` 等。
3. Python 环境：安装 Anaconda/Miniconda、创建虚拟环境、配置镜像源。
4. VS Code：安装 Python 插件、配置格式化工具（Black）、集成终端。

### 实践任务
- 输出第一行 Python 代码：`print("Hello, Machine Learning!")`。
- 使用 `pip` 安装 `numpy`、`pandas` 并验证。
- 在 GitHub 创建仓库，记录环境搭建步骤与踩坑笔记。

### 训练题与解析

| 训练题 | 操作步骤提示 | 参考解析 |
| --- | --- | --- |
| 在命令行中创建一个名为 `ml-study` 的文件夹并进入 | 打开终端 → `mkdir ml-study` → `cd ml-study` | `mkdir` 创建文件夹，`cd` 进入；若提示权限不足，先执行 `pwd` 查看当前路径再调整 |
| 创建一个虚拟环境并在其中安装 `numpy` | `conda create -n ml-base python=3.10` → `conda activate ml-base` → `pip install numpy` | 首次安装速度慢可执行 `pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple` 切换镜像 |
| 使用 VS Code 运行 `hello.py` 并解决编码错误 | 在文件中写 `print("Hello, Machine Learning!")` → 选择右下角 Python 解释器 → 终端运行 | 若报 `ModuleNotFoundError`，说明解释器未切换到虚拟环境；按 `Ctrl+Shift+P` 搜索 “Python: Select Interpreter” 重新选择 |

### 能力自检
- 能在命令行中完成文件创建、删除与移动。
- 清楚解释虚拟环境的作用，并能在不同环境之间切换。
- 会在 GitHub 上创建仓库并提交第一份学习记录。

---

## 阶段 1 · Python 基础语法（第 2-3 周）

### 核心知识点
- 数据类型：数字、字符串、布尔、列表、元组、集合、字典。
- 运算符与表达式：算术、比较、逻辑、成员、切片。
- 控制语句：`if/elif/else`、`for`、`while`、列表推导式、异常处理。
- 函数与模块：参数、返回值、作用域、lambda、内置模块（`random`、`datetime`、`os`）。
- 文件与类：读写文本、JSON、CSV；类与对象、继承与多态、特殊方法（`__str__`、`__len__`）。

### 推荐资源
- Python 官方教程，廖雪峰 Python 教程。
- B 站“Python 入门”系列或 CS50P 课程。
- LeetCode/牛客网简单题巩固算法思维。

### 实战练习
- 猜数字小游戏（运用循环、条件、随机数）。
- 通讯录管理脚本：新增/删除/搜索联系人并持久化为 JSON。
- 汇总常用内置函数，输出自己的“Python 速查表”。

### 训练题与解析

| 训练题 | 操作步骤提示 | 参考解析 |
| --- | --- | --- |
| 编写程序判断用户输入的年份是否为闰年 | 使用 `input()` 获取字符串 → 转为整数 → 套用闰年公式 | ```python
year = int(input("请输入年份:"))
if (year % 4 == 0 and year % 100 != 0) or year % 400 == 0:
    print("闰年")
else:
    print("平年")
``` |
| 统计一句话中每个单词出现的次数 | 调用 `split()` → 遍历列表 → 使用字典计数 | ```python
sentence = input("请输入一句话:")
words = sentence.lower().split()
count = {}
for word in words:
    count[word] = count.get(word, 0) + 1
print(count)
``` |
| 设计 `Student` 类并支持打印“姓名-平均分” | 定义 `__init__`、`add_score`、`average`、`__str__` 方法 | ```python
class Student:
    def __init__(self, name):
        self.name = name
        self.scores = []

    def add_score(self, score):
        self.scores.append(score)

    def average(self):
        return sum(self.scores) / len(self.scores)

    def __str__(self):
        return f"{self.name}-{self.average():.1f}"
``` |
| 使用文件读写实现“今日待办”小工具 | `open()` → 写入任务 → 读取并输出 | ```python
with open("todo.txt", "a", encoding="utf-8") as f:
    f.write(input("写下今日待办：") + "\n")

with open("todo.txt", encoding="utf-8") as f:
    print("今日清单：")
    for line in f:
        print("-", line.strip())
``` |

### 能力自检
- 能解释可变与不可变数据类型的差异。
- 知道如何封装函数并在其他模块中导入。
- 能用类封装业务逻辑，并在文件间共享代码。

---

## 阶段 2 · 数据分析与可视化（第 4-5 周）

### 技能点拆解
1. **NumPy**：数组创建、切片、广播、矩阵运算、随机数生成。
2. **Pandas**：DataFrame 操作、缺失值处理、数据清洗、分组聚合、透视表。
3. **可视化**：Matplotlib、Seaborn、Plotly 基本图表与美化技巧，中文显示、双坐标轴等。
4. **数据预处理**：数据类型转换、标准化与归一化、异常值检测、特征构造。

### 练习项目
- Kaggle Titanic 数据探索：幸存率分析、年龄/票价分布、特征之间的关系图。
- 城市房价分析报告：至少 5 种可视化图表，输出数据洞察与结论。
- 数据清洗流水线：将原始数据清洗成训练集（缺失值、重复值、异常值处理）。

### 训练题与解析

| 训练题 | 操作步骤提示 | 参考解析 |
| --- | --- | --- |
| 使用 NumPy 生成 0~1 之间 100 个随机数并计算平均值 | `np.random.rand(100)` → `np.mean()` | ```python
import numpy as np
numbers = np.random.rand(100)
print(f"平均值：{numbers.mean():.3f}")
``` |
| 读取 CSV 后统计每个城市的平均房价 | `pd.read_csv()` → `groupby('city')['price'].mean()` → `reset_index()` | ```python
import pandas as pd
df = pd.read_csv('house.csv')
city_price = df.groupby('city')['price'].mean().reset_index()
print(city_price)
``` |
| 绘制乘客年龄的直方图并标注平均年龄 | `plt.hist()` → `plt.axvline()` → 设置中文字体 | ```python
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
ages = df['Age'].dropna()
plt.hist(ages, bins=20, color='#4C72B0')
plt.axvline(ages.mean(), color='red', linestyle='--', label='平均年龄')
plt.legend()
plt.xlabel('年龄')
plt.ylabel('人数')
plt.show()
``` |
| 使用 `Pipeline` 串联标准化与随机森林回归 | `ColumnTransformer` → `Pipeline` → `fit` | ```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor

numeric_features = ['age', 'fare']
categorical_features = ['sex', 'embarked']

preprocess = ColumnTransformer([
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

pipe = Pipeline([
    ('preprocess', preprocess),
    ('model', RandomForestRegressor(random_state=42))
])
pipe.fit(train_df[numeric_features + categorical_features], train_df['survived'])
``` |

### 能力自检
- 知道如何清洗缺失值、重复值与异常值。
- 能使用 Pandas 进行数据透视与聚合分析。
- 能用可视化图表讲述数据洞察故事并撰写报告。

---

## 阶段 3 · 经典机器学习算法（第 6-8 周）

### 学习重点
- 机器学习流程：问题定义 → 数据准备 → 特征工程 → 建模 → 评估 → 调参 → 部署。
- 监督学习：训练/验证/测试集划分、交叉验证、过拟合与欠拟合。
- 分类算法：逻辑回归、kNN、朴素贝叶斯、决策树、随机森林、梯度提升、XGBoost。
- 回归算法：线性回归、岭回归、Lasso、随机森林回归、XGBoost 回归。
- 评估指标：准确率、召回率、F1、AUC、混淆矩阵、均方误差、R²、MAE。
- 特征工程：特征缩放、独热编码、目标编码、特征选择、`Pipeline` 与 `ColumnTransformer`。

### 实操路径
1. 使用 `train_test_split` 完成数据集划分，理解随机种子与分层抽样。
2. 熟悉 Scikit-Learn API：`fit`、`predict`、`predict_proba`、`score`、`Pipeline`。
3. 学习调参：`GridSearchCV`、`RandomizedSearchCV`、`cross_val_score`。
4. 模型解释：SHAP、Permutation Importance、部分依赖图（PDP）。

### 阶段项目
- **随机森林贷款违约预测**：绘制 ROC、PR 曲线与混淆矩阵，撰写评估报告。
- **XGBoost 房价预测**：生成特征重要性图、学习曲线与误差分析。
- **自动化训练脚本**：封装参数配置、日志记录、模型保存/加载、指标输出。

### 训练题与解析

| 训练题 | 操作步骤提示 | 参考解析 |
| --- | --- | --- |
| 将数据集按 8:2 划分训练集和测试集 | `from sklearn.model_selection import train_test_split` → 指定 `test_size=0.2` → 使用 `stratify` | ```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
``` `stratify` 能保持类别比例稳定 |
| 训练随机森林分类器并输出准确率 | 初始化 `RandomForestClassifier` → `fit` → `score` | ```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)
print(f"准确率：{clf.score(X_test, y_test):.3f}")
``` |
| 使用 `GridSearchCV` 调参并查看最优参数 | 定义 `param_grid` → 构造 `GridSearchCV` → `fit` → 打印 `best_params_` | ```python
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20]
}
grid = GridSearchCV(clf, param_grid, cv=5, scoring='f1')
grid.fit(X_train, y_train)
print(grid.best_params_)
``` |
| 绘制混淆矩阵并解释分类错误 | `confusion_matrix` → `ConfusionMatrixDisplay` | ```python
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test, cmap='Blues')
plt.title('随机森林混淆矩阵')
plt.show()
```
重点观察假阳性/假阴性占比，分析改进方向 |

### 能力自检
- 能解释不同评估指标的适用场景。
- 知道如何利用交叉验证和调参提升模型表现。
- 会对模型结果进行可视化并撰写技术报告。

---

## 阶段 4 · 项目化与部署实战（第 9-10 周）

### 关键技能
- 数据版本管理：DVC、Git LFS 或 MLflow Tracking。
- API 服务化：使用 FastAPI/Fastify 构建预测接口，了解 Uvicorn/Gunicorn、Docker 部署流程。
- 自动化测试与监控：编写单元测试、数据漂移监控、日志与告警。
- 报告与可视化：自动生成模型评估报告、可解释性分析图表。

### 综合项目建议
选择真实业务主题，如“用户流失预测”或“二手车定价”，完成以下交付：
- 标准化数据处理 Pipeline 与 Notebook。
- 多模型对比与调参记录，形成技术报告。
- 基于 FastAPI 的推理服务，并提供简单的前端界面（Streamlit/Gradio/自定义网页）。
- 部署说明书与 README，上传至 GitHub 展示。

### 训练题与解析

| 训练题 | 操作步骤提示 | 参考解析 |
| --- | --- | --- |
| 使用 FastAPI 构建一个 `/ping` 接口返回 `{"status": "ok"}` | 新建 `main.py` → 创建 `FastAPI()` 应用 → 定义 `@app.get` 函数 → `uvicorn main:app --reload` | ```python
from fastapi import FastAPI
app = FastAPI()

@app.get("/ping")
def ping():
    return {"status": "ok"}
``` |
| 为模型预测函数编写单元测试 | 安装 `pytest` → 构造虚拟输入 → 断言输出形状与数值范围 | ```python
def test_predict_shape():
    sample = pd.DataFrame([features])
    result = model.predict(sample)
    assert result.shape == (1,)
    assert 0 <= result[0] <= 1
``` |
| 设计部署说明模板 | 拆分“环境依赖 → 启动命令 → API 文档” | ```markdown
## 部署步骤
1. `pip install -r requirements.txt`
2. `uvicorn app.main:app --host 0.0.0.0 --port 8000`
3. 访问 `http://localhost:8000/docs` 查看接口
``` |
| 使用 Docker 打包推理服务 | 编写 `Dockerfile` → `docker build` → `docker run -p` | ```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
通过 `docker run -p 8000:8000` 启动并验证接口 |

### 能力自检
- 知道如何将 Notebook 中的模型封装为 API。
- 能编写基本的单元测试与日志记录。
- 会打包镜像并编写部署文档。

---

## 阶段 5 · 深度学习与 PyTorch（第 11-12 周）

### 基础铺垫
- 数学复习：线性代数（矩阵乘法、特征向量）、微积分（梯度、链式法则）、概率论（交叉熵、KL 散度）。
- PyTorch 核心：张量与自动求导、GPU 加速、`Dataset` / `DataLoader`、模型保存与加载。
- 神经网络基础：激活函数、损失函数、优化器、BatchNorm、Dropout。

### 进阶主题
1. 多层感知机（MLP）：在 MNIST 上实现分类，绘制训练/验证曲线。
2. 卷积神经网络（CNN）：理解卷积、池化、Padding，对 CIFAR-10 进行图像分类。
3. 迁移学习：加载 ResNet/VGG 预训练模型，完成小样本分类或特征提取。
4. 自然语言处理（可选）：RNN/LSTM 文本分类，或使用 Transformers（BERT、GPT）。
5. 模型部署：TorchScript、ONNX、FastAPI/Gradio 在线推理服务。

### 最终大作业
- 自主选题（图像/文本/表格均可），完成数据准备 → 模型设计 → 训练调参 → 结果分析 → Demo 部署。
- 输出成果：技术博客或演示视频 + GitHub 仓库（包含 README、代码、模型权重、部署指南）。
- 形成复盘总结：记录遇到的问题、解决方案、下一步学习计划。

### 训练题与解析

| 训练题 | 操作步骤提示 | 参考解析 |
| --- | --- | --- |
| 使用 PyTorch 创建一个 3×3 的随机张量并计算求和 | `torch.rand((3, 3))` → `tensor.sum()` | ```python
import torch
x = torch.rand((3, 3))
print(x)
print(x.sum())
``` |
| 构建两层全连接网络并完成前向传播 | 使用 `nn.Sequential` → 输入随机张量 | ```python
import torch.nn as nn
model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)
sample = torch.randn(2, 4)
print(model(sample))
``` |
| 编写训练循环并记录每轮损失 | `for epoch in range()` → `zero_grad()` → 前向 → 反向 → `step()` | ```python
for epoch in range(5):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}: loss={loss.item():.4f}")
``` |
| 计算模型准确率与保存权重 | `with torch.no_grad()` → `torch.save()` | ```python
with torch.no_grad():
    preds = model(val_inputs).argmax(dim=1)
    acc = (preds == val_labels).float().mean().item()
    print(f"验证准确率：{acc:.3f}")

torch.save(model.state_dict(), 'model.pt')
``` |

### 能力自检
- 能阅读并修改官方 PyTorch 教程示例。
- 清楚区分训练集、验证集与测试集的作用。
- 能把训练好的模型保存并在推理脚本中加载使用。

---

## 实战工具箱

### 环境与效率工具
- Conda/Poetry：管理虚拟环境和依赖，确保项目可复现。
- VS Code 插件：Python、Pylance、Black、isort、GitLens、Jupyter。
- 版本管理：Git 基本操作、GitHub Issues/Projects、`.gitignore` 规范。
- 日志与调试：`logging`、`pdb`、VS Code Debugger。

### 数据资源
- Kaggle、天池、Google Dataset Search：获取公开数据集。
- OpenML、UCI Machine Learning Repository：经典教学数据集。
- Hugging Face Datasets：海量 NLP/CV 数据集，支持流式加载。

### 模型与部署工具
- Scikit-Learn、LightGBM、XGBoost：传统机器学习核心库。
- PyTorch Lightning、FastAI：加速深度学习实验。
- MLflow、Weights & Biases：实验追踪与可视化。
- Docker、Docker Compose、Render、Railway：部署与在线托管方案。

---

## 项目灵感库

| 主题 | 项目点子 | 关键能力 |
| --- | --- | --- |
| 数据分析 | 城市空气质量监测、零售销售趋势分析、电影评分洞察 | 数据清洗、可视化、叙事报告 |
| 经典机器学习 | 客户流失预测、信用卡欺诈检测、二手车价格预测 | 特征工程、模型对比、指标评估 |
| 深度学习 | 手写数字识别、花卉图像分类、商品评论情感分析 | PyTorch 建模、迁移学习、推理部署 |
| 综合应用 | 智能问答机器人、个性化推荐系统、语音情绪识别 | 数据管道、模型服务化、系统设计 |

> ✅ 建议从公开数据集开始，逐步尝试自建或爬取数据，形成可展示的项目组合。

---

## 学习方法与复盘建议

1. **制定周计划**：写下本周要完成的课程、练习和项目。每天坚持打卡，积累“可见进度”。
2. **费曼学习法**：尝试用自己的话向他人解释所学知识，发现盲点后及时补强。
3. **记录错题与 Bug**：将踩坑记录在 Issue 或博客中，复习时快速回顾。
4. **定期复盘**：每两周回答三个问题——“我学会了什么？”、“我做成了什么？”、“下一步打算是什么？”。
5. **参与社区**：在 Kaggle、知乎、GitHub 讨论区提问或解答问题，保持输入与输出的平衡。

---

## 附录 · 常见问题 & 资源推荐

### 常见问题
- **数学基础薄弱怎么办？** 结合 3Blue1Brown、Essence of Linear Algebra、吴恩达《机器学习》数学附录复习。
- **英文文档看不懂？** 借助翻译工具，但核心 API 与官方文档建议阅读英文原版以保持准确理解。
- **缺少练习数据集？** 多使用 Kaggle、天池、Google Dataset Search 等开放数据平台。
- **学习动力不足？** 与同伴互相监督、写博客分享、参与开源项目保持反馈。

### 工具与资源清单
- 文档：Python Docs、Pandas Cookbook、Scikit-Learn User Guide、PyTorch Tutorials。
- 课程：CS50P、吴恩达《机器学习》《深度学习专项课程》、FastAI Practical Deep Learning。
- 社区：Kaggle Discuss、Stack Overflow、知乎「机器学习」、PyTorch Forums。
- 书籍：《利用 Python 进行数据分析》《Hands-On Machine Learning》《Deep Learning with PyTorch》。

### 术语速查
- **EDA (Exploratory Data Analysis)**：探索性数据分析，用于理解数据特征与分布。
- **Overfitting / Underfitting**：过拟合/欠拟合，指模型在训练集/测试集表现差异的状态。
- **Batch、Epoch、Iteration**：批次/轮次/迭代次数，描述深度学习训练过程。
- **Inference**：推理，将训练好的模型应用在新数据上得到预测结果。

### 下一步发展方向
- 强化学习、推荐系统、时间序列、图神经网络。
- MLOps：CI/CD、模型监控、特征存储、自动化部署。
- AIGC：大语言模型、Stable Diffusion、LoRA 微调等前沿方向。

保持持续学习与项目实战，你将具备胜任初级/中级数据科学或算法工程岗位的能力。祝学习顺利！
