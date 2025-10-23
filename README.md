# Python 修仙志：从练气到化身的全门派功法

欢迎来到专为 **零基础修仙者** 打造的 Python × 机器学习门派。此卷以修仙五大境界贯穿全程：

1. **练气境**：点亮灵根，熟悉电脑法器、Python 语法与基础心法。
2. **筑基境**：打磨内功，掌握控制流、数据结构、函数、面向对象等核心灵技。
3. **结丹境**：凝聚丹道之力，学会 NumPy、Pandas、可视化与数据清洗，炼成数据炼丹术。
4. **元婴境**：唤醒内婴，精通经典机器学习算法、特征工程与模型评估。
5. **化身境**：元神出窍，驾驭 PyTorch 深度学习与模型部署，完成真正的飞升。

每一境均提供：

- 📘 **修行目标**：明确阶段要点与能力清单。
- 🧭 **修炼路径**：按步骤修习，知道先后顺序与推荐节奏。
- 🧪 **试炼任务**：结合实操练功巩固所学。
- 🧩 **练气题库**：覆盖常见知识点，每项不少于 5 题，并附详细点拨与解析。
- 🧰 **心法锦囊**：工具、资源、加速技巧与常见疑问解答。

> 🌟 **阅读建议**：将 README 当作随身心法。同步打开网站版本（见下方“道场指引”）获得更沉浸的修仙体验。

---

## 目录

1. [道场指引：如何使用本仓库](#道场指引如何使用本仓库)
2. [修行路线速览（12 周时间表）](#修行路线速览12-周时间表)
3. [练气境 · 灵根初醒（第 1-2 周）](#练气境--灵根初醒第-1-2-周)
4. [筑基境 · 内功淬炼（第 3-5 周）](#筑基境--内功淬炼第-3-5-周)
5. [结丹境 · 数据炼丹（第 6-7 周）](#结丹境--数据炼丹第-6-7-周)
6. [元婴境 · 御阵驭兽（第 8-10 周）](#元婴境--御阵驭兽第-8-10-周)
7. [化身境 · 元神出窍（第 11-12 周）](#化身境--元神出窍第-11-12-周)
8. [外功密卷 · 工具与资源](#外功密卷--工具与资源)
9. [灵感宝库 · 项目创意](#灵感宝库--项目创意)
10. [心法总纲 · 复盘与精进](#心法总纲--复盘与精进)
11. [答疑密信 · 常见问题](#答疑密信--常见问题)

---

## 道场指引：如何使用本仓库

### 1. 领取法器（克隆与环境搭建）

```bash
# 克隆仓库
git clone https://github.com/<your-name>/Machine-Learning.git
cd Machine-Learning

# 创建与激活虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows 请运行 .venv\Scripts\activate

# 装备常用法术
pip install -r requirements.txt
```

> ⚙️ **镜像加速**：若下载缓慢，可执行 `pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple`。

### 2. 启动修仙道场（本地教学网站）

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- 在浏览器访问 <http://127.0.0.1:8000>。
- 固定目录左栏呈现境界导航，可快速跳转练气题与解析。
- 若要停止，返回终端使用 <kbd>Ctrl</kbd>+<kbd>C</kbd>。

### 3. 修炼建议

1. **双线修行**：先在 README 熟悉整体路线，再在网站逐章练习。
2. **亲手过招**：每道练气题都动手完成，再比对解析体悟差距。
3. **结丹成册**：将解题代码、心得写入 GitHub 日志，形成成长档案。
4. **同门交流**：积极在社区或学习群讨论心得，互为道友。

---

## 修行路线速览（12 周时间表）

| 周数 | 境界 | 核心修行 | 推荐产出 |
| --- | --- | --- | --- |
| 第 1-2 周 | 练气境 | 环境搭建、命令行、Python 基础语法与常用工具 | 环境搭建手册、基础脚本、练气题解析日志 |
| 第 3-5 周 | 筑基境 | 控制流、数据结构、函数、面向对象与模块化 | 猜数字/通讯录项目、OOP 实战、自动化脚本 |
| 第 6-7 周 | 结丹境 | NumPy、Pandas、数据清洗、可视化、探索性分析 | 数据炼丹报告、可视化仪表盘、清洗脚本 |
| 第 8-10 周 | 元婴境 | 经典机器学习、特征工程、模型评估与调参 | 随机森林/梯度提升项目、模型对比报告 |
| 第 11-12 周 | 化身境 | PyTorch、深度学习项目、部署与持续学习 | 图像/文本模型、推理服务、飞升复盘博客 |

> ✅ **飞升标准**：能独立完成从数据收集、建模、部署到复盘的完整闭环，并对后续深度修炼有清晰规划。

---

## 练气境 · 灵根初醒（第 1-2 周）

### 修行目标

- 认识命令行、路径与文件操作，搭建 Python/VS Code/Git 等法器。
- 熟悉 Python 解释器、变量、基础数据类型与交互式环境。
- 明白如何运行脚本、安装包、编写第一个程序。

### 修炼路径

1. 走访灵材商铺：完成 Python 官方安装或创建虚拟环境。
2. 练习终端步法：熟悉 `cd`、`ls/dir`、`mkdir`、`rm` 等指令。
3. 打开灵识之眼：在 VS Code 中运行 `hello_cultivator.py`。
4. 记账练气：使用 Git 记录第一次 commit 与推送。

### 试炼任务

1. 在桌面创建 `dao-field` 文件夹，初始化 Git 仓库并写入 `README`。
2. 创建虚拟环境并安装 `numpy`、`pandas`，用脚本打印版本信息。
3. 在命令行与 VS Code 各运行一次“修仙宣言”脚本。
4. 编写一个交互式脚本，请求用户姓名与目标境界并打印祝福。
5. 在 GitHub 发布“练气周志”，总结环境搭建与指令体验。

### 练气题库

#### 灵材备置 · 环境与命令行

| # | 练习题 | 点拨 | 解析 |
| --- | --- | --- | --- |
| 1 | 使用命令行创建多层目录 `sect/training/day1` | 组合 `mkdir -p` | Linux/macOS 可 `mkdir -p sect/training/day1`，Windows 用 `mkdir sect\training\day1` |
| 2 | 查看当前路径并写入 `path.txt` | `pwd` 与重定向 | 运行 `pwd > path.txt`，若在 PowerShell 使用 `Get-Location` |
| 3 | 激活虚拟环境并列出已安装包 | `source` 与 `pip list` | 激活 `.venv` 后 `pip list`；若失败检查路径是否正确 |
| 4 | 设置 pip 镜像并验证是否生效 | `pip config set` | 配置镜像后执行 `pip install requests -i` 查看下载速度变化 |
| 5 | 使用 VS Code 调试 `hello.py` | 断点 + 运行 | 在编辑器中设置断点，按 F5 选择 Python 调试配置，观察变量面板 |

#### 灵根觉醒 · 变量与数据类型

| # | 练习题 | 点拨 | 解析 |
| --- | --- | --- | --- |
| 1 | 声明 `spirit_name`、`age`、`is_novice` 并打印类型 | 使用 `type()` | `print(type(spirit_name))` 等，辨识 `str`/`int`/`bool` |
| 2 | 判断变量 `mana = None` 是否等同空字符串 | `is` 与 `==` 对比 | `mana is None` 为真，但 `mana == ""` 为假，说明含义不同 |
| 3 | 将字符串 `"108"` 与整数 `12` 相加 | 类型转换 | `int("108") + 12`，理解 `ValueError` 的产生原因 |
| 4 | 解释 `id()` 函数查看对象内存地址 | 引导内存概念 | `id(a)` 显示引用，强调同值不同对象与缓存机制 |
| 5 | 比较两个变量是否指向同一对象 | `is` 运算符 | `a = []; b = a` → `a is b` 为真；`a = []; b = []` 则为假 |

#### 灵气入体 · 数值与字符串

| # | 练习题 | 点拨 | 解析 |
| --- | --- | --- | --- |
| 1 | 计算 `(5 ** 2) // 3` 与 `5 ** (2 // 3)` | 运算优先级 | 指出指数先算，整除影响结果，演示两者差异 |
| 2 | 判断 `0.1 + 0.2 == 0.3` | 浮点误差 | 使用 `math.isclose(0.1 + 0.2, 0.3)`，解释 IEEE 754 |
| 3 | 将 `"剑"*3` 与 `"仙"*2` 拼接 | 字符串乘法 | 结果为 `"剑剑剑仙仙"`，演示重复拼接技巧 |
| 4 | 使用 f-string 输出修士信息 | f-string 占位 | `f"{name} 当前境界 {realm}"`，可加格式化 `:.2f` |
| 5 | 统计咒语字符串中元音数量 | 字符串方法 | `sum(1 for ch in mantra.lower() if ch in 'aeiou')` |

#### 灵识初开 · 输入输出

| # | 练习题 | 点拨 | 解析 |
| --- | --- | --- | --- |
| 1 | 使用 `input` 获取名字并打印欢迎语 | 字符串拼接 | `name = input("道友名号：")` 后输出格式化句 |
| 2 | 读取多个数字并求平均 | `split` 与 `map` | `nums = list(map(float, input().split()))`，再求和除以长度 |
| 3 | 将结果写入文件 `blessing.txt` | `with open` | `with open("blessing.txt", "w", encoding="utf-8") as f:` |
| 4 | 从文件读取并输出到终端 | `readlines` | `for line in f: print(line.strip())`，注意编码 |
| 5 | 使用 `print(..., file=sys.stderr)` | `sys` 模块 | 了解错误输出通道，需先 `import sys` |

#### 气脉调息 · 调试与笔记

| # | 练习题 | 点拨 | 解析 |
| --- | --- | --- | --- |
| 1 | 在脚本加入 `__name__ == "__main__"` 判断 | 程序入口 | 确保脚本被导入时不执行主流程 |
| 2 | 使用 `dir()` 探索模块提供的成员 | 交互式探索 | `dir(math)` 返回可用属性，配合 `help()` 查文档 |
| 3 | 在 REPL 中使用 `_` 变量复用上次结果 | 交互技巧 | CPython 默认 `_` 保存上一次表达式值 |
| 4 | 使用 `logging` 输出调试信息 | `basicConfig` | 设置 `logging.basicConfig(level=logging.INFO)`，替换 print |
| 5 | 记录学习日志模板 | Markdown | 包含“遇到的问题 / 解决方案 / 下一步计划”三段 |

---

## 筑基境 · 内功淬炼（第 3-5 周）

### 修行目标

- 熟练掌握控制流、容器、推导式与函数式思想。
- 能够设计模块化脚本、面向对象模型与异常处理流程。
- 学会撰写单元测试、使用第三方库，迈向实战级代码。

### 修炼路径

1. 复习基础语法并尝试将多段脚本封装为函数。
2. 实现命令行菜单程序（如通讯录、图书管理）。
3. 引入 `unittest` 或 `pytest` 为核心函数编写测试。
4. 练习使用 `venv`/`pip` 打包并分享自己的小工具。

### 试炼任务

1. 完成“灵宠属性管理器”：支持增删改查、模糊搜索、CSV 持久化。
2. 编写“灵药交易账本”：使用字典与列表统计总价、折扣、税额。
3. 用面向对象实现“飞剑竞速”模拟，支持继承与多态。
4. 创建 `package` 目录，封装常用函数并发布至 TestPyPI。
5. 为以上项目编写至少 5 条自动化测试并接入 CI（可用 GitHub Actions）。

### 练气题库

#### 阵法初成 · 控制流与推导式

| # | 练习题 | 点拨 | 解析 |
| --- | --- | --- | --- |
| 1 | 使用 `for` 与 `range` 打印 1-9 九九乘法表 | 双层循环 | 内层 `for j in range(1, i+1)`，格式化输出 |
| 2 | 使用 `while` 计算 100 内能被 3 整除但不被 5 整除的和 | 条件判断 | 逐一累加或用推导式 `sum(i for i in range(1, 101) if i % 3 == 0 and i % 5 != 0)` |
| 3 | 比较 `break` 与 `continue` 在循环中的效果 | 小实验 | 通过记录日志展示跳出循环与跳过本次迭代的差别 |
| 4 | 利用列表推导式生成平方表并过滤偶数 | 推导式条件 | `[i**2 for i in range(10) if i % 2 == 0]` |
| 5 | 将嵌套循环转化为生成器表达式 | 节省内存 | 使用 `sum(i*j for i in range(1, 10) for j in range(1, 10))` |

#### 灵兽驯养 · 列表、元组、集合

| # | 练习题 | 点拨 | 解析 |
| --- | --- | --- | --- |
| 1 | 用列表保存灵药并删除重复 | `set` 去重 | `unique = list(dict.fromkeys(elixirs))` 保留顺序 |
| 2 | 将坐标 `(x, y)` 以元组存储并解包 | 多变量赋值 | `x, y = position`，强调不可变特性 |
| 3 | 使用集合求两名修士掌握法术的交集与差集 | 集合运算 | `skills_a & skills_b`、`skills_a - skills_b` |
| 4 | 嵌套列表表示灵宠信息并排序 | `sorted` + `key` | `sorted(pets, key=lambda p: p[1])` |
| 5 | 使用 `enumerate` 与 `zip` 同时遍历索引与元素 | 内置函数 | `for idx, (name, power) in enumerate(zip(names, powers), 1)` |

#### 宝库开启 · 字典与映射

| # | 练习题 | 点拨 | 解析 |
| --- | --- | --- | --- |
| 1 | 创建字典记录灵器库存并更新数量 | `dict` 方法 | `inventory["flying-sword"] = inventory.get("flying-sword", 0) + 1` |
| 2 | 使用字典推导式从列表生成映射 | 推导式 | `{item: len(item) for item in items}` |
| 3 | 合并两个字典并保留后者优先 | 解包 | `{**base, **override}` 或 `base | override` (3.9+) |
| 4 | 使用 `collections.Counter` 统计灵石颜色频率 | 标准库 | `Counter(stones).most_common()` |
| 5 | 实现嵌套字典的安全访问 | `get` 默认值 | `spellbook.get("fire", {}).get("level", 0)` |

#### 灵术传承 · 函数与参数

| # | 练习题 | 点拨 | 解析 |
| --- | --- | --- | --- |
| 1 | 定义函数计算灵力恢复 `recover(mana, rate=0.1)` | 默认参数 | 理解默认值只计算一次的特性 |
| 2 | 使用关键字参数调用函数 | 调用方式 | `recover(rate=0.2, mana=100)` |
| 3 | 演示可变参数 `*args`、`**kwargs` | 解包 | 在函数中打印两种参数形态 |
| 4 | 编写递归函数计算灵兽族谱深度 | 递归思路 | 终止条件 + 对子节点递归取最大值 |
| 5 | 使用闭包记忆法术冷却时间 | 闭包 | 返回内部函数保存状态，演示 `nonlocal` |

#### 气海调息 · 匿名函数与迭代器

| # | 练习题 | 点拨 | 解析 |
| --- | --- | --- | --- |
| 1 | 使用 `map`+`lambda` 将灵药价格折半 | 函数式思维 | `map(lambda price: price * 0.5, prices)` |
| 2 | 使用 `filter` 筛选修为≥80的弟子 | 条件过滤 | `list(filter(lambda disciple: disciple['power'] >= 80, disciples))` |
| 3 | 创建自定义迭代器按周输出修炼日志 | `__iter__`/`__next__` | 类实现迭代协议并在 `StopIteration` 结束 |
| 4 | 用 `itertools.cycle` 创建无限法术循环器 | 标准库 | `cycle(['fireball', 'ice'])` |
| 5 | 比较生成器函数与列表的内存占用 | `sys.getsizeof` | 显示同等元素数目的差异 |

#### 真经铭刻 · 面向对象

| # | 练习题 | 点拨 | 解析 |
| --- | --- | --- | --- |
| 1 | 实现 `Cultivator` 类，含 `level_up` 方法 | 基本类定义 | `self.realm += 1`，强调 `__init__` |
| 2 | 设计继承结构 `Sword` → `FlyingSword` | 继承 | 子类调用 `super().__init__`，覆写方法 |
| 3 | 使用 `@property` 管理灵力值上下限 | 属性装饰器 | 在 setter 中限制范围 |
| 4 | 定义类方法创建预设角色 | `@classmethod` | `@classmethod def novice(cls): return cls("新弟子")` |
| 5 | 使用数据类简化属性声明 | `dataclasses` | `@dataclass` 自动生成 `__init__` 与 `__repr__` |

#### 火候把握 · 文件与上下文

| # | 练习题 | 点拨 | 解析 |
| --- | --- | --- | --- |
| 1 | 使用 `with open` 写入灵药日志 | 上下文 | 自动关闭文件避免资源泄露 |
| 2 | 将 CSV 读取为字典列表 | `csv.DictReader` | 注意编码与换行参数 |
| 3 | 使用 `pathlib` 处理路径与扩展名 | 面向对象路径 | `Path('logs').glob('*.txt')` |
| 4 | 压缩文件夹备份灵材 | `shutil.make_archive` | 指定根目录与格式，如 `zip` |
| 5 | 利用 `tempfile` 创建临时文件进行测试 | 临时资源 | `NamedTemporaryFile` 自动清理 |

#### 天罚护身 · 异常处理与调试

| # | 练习题 | 点拨 | 解析 |
| --- | --- | --- | --- |
| 1 | 捕获 `ValueError` 提示输入非法 | `try`/`except` | `except ValueError as exc: print(...)` |
| 2 | 自定义异常 `InsufficientManaError` | 自定义类 | 继承 `Exception`，用于业务校验 |
| 3 | 使用 `else` 与 `finally` | 完整结构 | `else` 在无异常时执行，`finally` 保证善后 |
| 4 | 利用 `pdb` 单步调试循环 | 断点调试 | `import pdb; pdb.set_trace()` |
| 5 | 在日志中记录异常堆栈 | `logging.exception` | 自动附带 stack trace，便于排查 |

---

## 结丹境 · 数据炼丹（第 6-7 周）

### 修行目标

- 熟练掌握 NumPy、Pandas 的数组与表格操作。
- 能够进行数据清洗、缺失值处理、特征工程与统计描述。
- 具备绘制 Matplotlib/Seaborn 可视化的能力，完成探索性数据分析。

### 修炼路径

1. 通过 Kaggle 或天池获取练习数据集，熟悉 CSV/JSON 读写。
2. 使用 NumPy 完成向量化运算，理解广播与矩阵乘法。
3. 借助 Pandas 清洗缺失值、异常值并生成统计报表。
4. 结合 Seaborn/Matplotlib 绘制趋势图、箱线图、相关性热力图。

### 试炼任务

1. 对“灵草药性”数据集完成缺失值填补、异常检测与可视化。
2. 编写 `eda.py`，输出各字段均值、中位数、标准差与分位数。
3. 使用 `plotly` 或 `altair` 构建交互式可视化仪表盘。
4. 设计数据清洗 pipeline，封装为函数供后续机器学习使用。
5. 撰写《结丹日志》总结采样、清洗、探索的关键心得。

### 练气题库

#### 灵石阵列 · NumPy 基础

| # | 练习题 | 点拨 | 解析 |
| --- | --- | --- | --- |
| 1 | 创建 3×3 单位矩阵并计算行列式 | `np.eye`、`np.linalg.det` | 结果为 1，强调线性代数函数 |
| 2 | 将一维数组重塑为 2×6 并切片取前两列 | `reshape`、切片 | `arr.reshape(2, 6)[:, :2]` |
| 3 | 解释广播规则计算 `arr + np.array([1, 2, 3])` | 维度匹配 | 行向量会扩展到每一行 |
| 4 | 使用布尔索引筛选能量值 > 50 的元素 | 条件筛选 | `arr[arr > 50]` |
| 5 | 计算数组均值、方差、标准差 | 聚合函数 | `arr.mean()`、`arr.var()`、`arr.std()` |

#### 丹炉调配 · Pandas 数据帧

| # | 练习题 | 点拨 | 解析 |
| --- | --- | --- | --- |
| 1 | 从字典创建 DataFrame 并查看前几行 | `pd.DataFrame`、`head` | `df.head()` 显示前 5 行 |
| 2 | 设置 `id` 为索引并排序 | `set_index`、`sort_index` | `df.set_index('id').sort_index()` |
| 3 | 选择多列并添加新特征列 | 列操作 | `df[['mana', 'power']]; df['ratio'] = df['power'] / df['mana']` |
| 4 | 使用 `groupby` 统计各类灵草平均效力 | 分组聚合 | `df.groupby('type')['power'].mean()` |
| 5 | 合并两个 DataFrame | `merge` | `pd.merge(df1, df2, on='id', how='left')` |

#### 杂质祛除 · 数据清洗

| # | 练习题 | 点拨 | 解析 |
| --- | --- | --- | --- |
| 1 | 检查缺失值并输出比例 | `isna` 与 `mean` | `df.isna().mean()` |
| 2 | 使用 `fillna` 与插值填补缺失 | 多策略 | 数值列用均值，时间序列用 `interpolate()` |
| 3 | 检测离群值并以 IQR 过滤 | 四分位数 | 计算 Q1、Q3 与 IQR，过滤 `(value < Q1 - 1.5*IQR)` |
| 4 | 统一字符串大小写并去除首尾空格 | 清洗 | `df['name'].str.strip().str.title()` |
| 5 | 编写函数自动记录清洗步骤日志 | 函数封装 | 使用装饰器打印前后数据形状 |

#### 灵火绘阵 · 可视化

| # | 练习题 | 点拨 | 解析 |
| --- | --- | --- | --- |
| 1 | 使用 Matplotlib 绘制修为随时间变化折线图 | 基础绘图 | `plt.plot(days, power)` + `plt.xlabel` |
| 2 | 绘制多类别灵草功效箱线图 | Seaborn | `sns.boxplot(x='type', y='power', data=df)` |
| 3 | 创建相关性热力图并标注数值 | `sns.heatmap` | `annot=True, cmap='YlGnBu'` |
| 4 | 使用双轴图展示成本与收益 | `twinx` | 左轴成本、右轴收益，注意颜色区分 |
| 5 | 输出 PNG、SVG 两种格式 | `plt.savefig` | 传入 `dpi=300`、`format='svg'` |

#### 丹方评估 · 数据报告

| # | 练习题 | 点拨 | 解析 |
| --- | --- | --- | --- |
| 1 | 计算描述统计并转为 Markdown 表格 | `describe` | `df.describe().to_markdown()` |
| 2 | 编写 `profile_data(df)` 返回缺失、唯一值、类型信息 | 自定义函数 | 利用 `df.dtypes`、`nunique` |
| 3 | 生成 `pandas-profiling` 报告 | 第三方库 | `ProfileReport(df, minimal=True)` |
| 4 | 将清洗前后数据差异写入日志 | `compare` | `df.compare(df_clean)` |
| 5 | 总结数据风险点并给出下一步建议 | 文档化 | 列出数据质量指标、潜在偏差 |

---

## 元婴境 · 御阵驭兽（第 8-10 周）

### 修行目标

- 掌握经典监督/无监督算法的原理与实现。
- 熟悉特征工程、数据集拆分、交叉验证与模型评估。
- 能够构建完整的机器学习 pipeline，并进行调参与模型解释。

### 修炼路径

1. 选择结构化数据集（如 Kaggle 泰坦尼克、信贷评分）。
2. 编写数据预处理模块，完成标准化、编码与特征选择。
3. 实现多种模型（线性回归、逻辑回归、决策树、随机森林、XGBoost）。
4. 使用交叉验证、学习曲线、特征重要性评估模型表现。

### 试炼任务

1. 构建“灵兽资质评估”模型，比较逻辑回归与随机森林效果。
2. 对比三种特征缩放方式（MinMax、Standard、Robust）对模型影响。
3. 使用 `GridSearchCV` 或 `Optuna` 进行超参数调优，并记录搜索过程。
4. 通过 SHAP/LIME 对模型结果进行可解释性分析。
5. 撰写《元婴悟道》报告，汇总实验流程、评估指标与改进方向。

### 练气题库

#### 玄阵布局 · 数据预处理

| # | 练习题 | 点拨 | 解析 |
| --- | --- | --- | --- |
| 1 | 将数据划分为训练集与测试集 | `train_test_split` | `test_size=0.2`，设置 `random_state` 确保可复现 |
| 2 | 使用 `ColumnTransformer` 组合数值与分类特征处理 | 预处理管道 | 数值列加 `StandardScaler`，分类列加 `OneHotEncoder` |
| 3 | 检测并处理类别不平衡 | 采样方法 | 尝试 `SMOTE` 或 class_weight |
| 4 | 构建自定义特征如“灵石密度” | 自定义函数 | `df['density'] = df['mana'] / df['weight']` |
| 5 | 保存预处理器以便部署 | `joblib` | `joblib.dump(preprocessor, 'preprocess.pkl')` |

#### 御兽要诀 · 监督学习

| # | 练习题 | 点拨 | 解析 |
| --- | --- | --- | --- |
| 1 | 训练线性回归预测灵药价格 | `LinearRegression` | 比较训练集与测试集 R² |
| 2 | 实现逻辑回归分类修士是否飞升 | `LogisticRegression` | 使用 `C` 调整正则，观察精度与召回 |
| 3 | 训练决策树并可视化 | `export_graphviz` | 控制 `max_depth`、`min_samples_split` |
| 4 | 使用随机森林并查看特征重要性 | `feature_importances_` | 排序可视化条形图 |
| 5 | 尝试梯度提升或 XGBoost | 高级模型 | 对比训练时间与性能 |

#### 幻阵迷踪 · 无监督学习

| # | 练习题 | 点拨 | 解析 |
| --- | --- | --- | --- |
| 1 | 对灵石属性使用 KMeans 聚类 | `KMeans` | 选择 `n_clusters`，查看簇中心 |
| 2 | 使用 PCA 降维并绘制散点图 | `PCA` | 解释方差比例，观察聚类分布 |
| 3 | 实现层次聚类并绘制树状图 | `scipy.cluster.hierarchy` | `linkage` + `dendrogram` |
| 4 | 使用 DBSCAN 发现异常灵石 | 密度聚类 | 调整 `eps` 与 `min_samples` |
| 5 | 评估聚类结果的轮廓系数 | `silhouette_score` | 指导如何选择最佳簇数 |

#### 阵法验收 · 模型评估

| # | 练习题 | 点拨 | 解析 |
| --- | --- | --- | --- |
| 1 | 计算分类模型的准确率、精确率、召回率、F1 | `classification_report` | 理解各指标意义 |
| 2 | 绘制混淆矩阵并解释每个象限 | `ConfusionMatrixDisplay` | 识别 FP/FN 影响 |
| 3 | 绘制 ROC 曲线与 AUC | `roc_curve` | 讨论阈值与曲线下面积 |
| 4 | 绘制学习曲线观察是否过拟合 | `LearningCurveDisplay` | 解读训练/验证分数差距 |
| 5 | 使用交叉验证估计模型稳定性 | `cross_val_score` | 设置 `cv=5`，输出均值与方差 |

#### 灵阵调优 · Pipeline 与部署

| # | 练习题 | 点拨 | 解析 |
| --- | --- | --- | --- |
| 1 | 构建包含预处理与模型的 `Pipeline` | 一体化流程 | `Pipeline(steps=[('prep', preprocessor), ('model', clf)])` |
| 2 | 使用 `GridSearchCV` 对 Pipeline 调参 | 网格搜索 | 将参数命名为 `model__max_depth` 等 |
| 3 | 保存最佳模型并加载复现 | `joblib.dump` | `clf = joblib.load('best_model.joblib')` |
| 4 | 使用 FastAPI 暴露预测接口 | 简易部署 | `@app.post('/predict')` 接收 JSON 并返回概率 |
| 5 | 编写 `make_predictions.py` 脚本批量推理 | 脚本化 | 读取 CSV → 调用模型 → 写回结果 |

---

## 化身境 · 元神出窍（第 11-12 周）

### 修行目标

- 掌握 PyTorch 张量运算、自动求导、模型构建与训练循环。
- 了解卷积神经网络、循环/注意力模型与迁移学习思路。
- 能够将训练好的模型导出、部署为推理服务并持续监控。

### 修炼路径

1. 完成 PyTorch 官方 60 分钟教程，熟悉 `Tensor` 与 `nn.Module`。
2. 实现手写数字或猫狗分类模型，掌握数据加载、训练、验证流程。
3. 尝试使用预训练模型（ResNet、BERT）进行迁移学习。
4. 将模型导出为 TorchScript 或 ONNX，并用 FastAPI 提供推理接口。

### 试炼任务

1. 编写 `pytorch_basics.ipynb` 演练张量运算、自动微分与优化器。
2. 构建 CNN 模型训练灵兽识别数据集，记录损失与准确率曲线。
3. 使用 `torchvision.models` 进行迁移学习并对比冻结/解冻策略。
4. 将模型导出为 ONNX，并用 `onnxruntime` 编写推理脚本。
5. 撰写《化身笔记》，总结训练技巧、部署经验与下一步方向。

### 练气题库

#### 元神凝练 · PyTorch 入门

| # | 练习题 | 点拨 | 解析 |
| --- | --- | --- | --- |
| 1 | 创建张量并查看形状、类型、设备 | `tensor.shape` | 了解 CPU/GPU 切换 `to('cuda')` |
| 2 | 将 NumPy 数组转换为张量并保持共享内存 | `from_numpy` | 修改其中一方另一方同步变化 |
| 3 | 使用随机种子保证实验可复现 | `torch.manual_seed` | 同时设置 `torch.cuda.manual_seed_all` |
| 4 | 探索 `requires_grad` 的作用 | 自动求导 | 对张量调用 `backward()`，查看梯度 |
| 5 | 比较 `torch.no_grad()` 与 `detach()` | 推理模式 | 两者都会阻断梯度，但应用场景不同 |

#### 法身塑造 · 模型构建

| # | 练习题 | 点拨 | 解析 |
| --- | --- | --- | --- |
| 1 | 自定义 `Net(nn.Module)` 并实现 `forward` | 模型结构 | 在 `__init__` 定义层，`forward` 中串联 |
| 2 | 使用 `torchsummary` 查看模型参数量 | 工具使用 | `summary(model, input_size=(1, 28, 28))` |
| 3 | 编写训练循环与验证循环 | 标准流程 | 包含梯度清零、反向传播、优化器更新 |
| 4 | 使用 `torch.optim.lr_scheduler` 调整学习率 | 学习率策略 | 如 `StepLR(optimizer, step_size=5, gamma=0.1)` |
| 5 | 记录训练指标到 TensorBoard | 可视化 | `SummaryWriter` 写入 loss/accuracy |

#### 灵识扩展 · 计算机视觉与序列

| # | 练习题 | 点拨 | 解析 |
| --- | --- | --- | --- |
| 1 | 使用 `torchvision.transforms` 进行数据增强 | 数据预处理 | `RandomHorizontalFlip`、`ColorJitter` |
| 2 | 实现简单的 RNN/LSTM 文本分类 | 序列模型 | 使用 `nn.Embedding` + `nn.LSTM` |
| 3 | 读取自定义数据集并实现 `Dataset`/`DataLoader` | 自定义类 | 重写 `__len__`、`__getitem__` |
| 4 | 使用 Grad-CAM 可视化关注区域 | 模型解释 | 调用 `pytorch-grad-cam` 或手写实现 |
| 5 | 对比不同批大小对训练稳定性的影响 | 实验设计 | 记录损失曲线与 GPU 占用 |

#### 元神游历 · 迁移学习与部署

| # | 练习题 | 点拨 | 解析 |
| --- | --- | --- | --- |
| 1 | 加载预训练 ResNet 并冻结前几层 | 迁移学习 | `for param in model.parameters(): param.requires_grad = False` |
| 2 | 替换最后全连接层适配自定义类别数 | 模型修改 | `model.fc = nn.Linear(model.fc.in_features, num_classes)` |
| 3 | 导出模型为 TorchScript | `torch.jit.trace` | 保存为 `model.ts` 供 C++/部署使用 |
| 4 | 将 TorchScript 模型接入 FastAPI 推理服务 | 部署 | 加载模型 → 预处理 → 推理 → 返回结果 |
| 5 | 设计推理服务的健康检查与监控指标 | 生产思维 | 返回版本号、耗时日志、错误处理 |

#### 道果巩固 · 持续精进

| # | 练习题 | 点拨 | 解析 |
| --- | --- | --- | --- |
| 1 | 复现论文中的模型或实验 | 研读论文 | 选择小型论文，阅读实现细节 |
| 2 | 使用 `Lightning` 或 `Accelerate` 重构训练流程 | 框架封装 | 简化训练循环，提高可复用性 |
| 3 | 尝试混合精度训练提升速度 | `torch.cuda.amp` | `with autocast():` 包裹前向传播 |
| 4 | 编写部署流水线脚本（打包、推送、上线） | DevOps | 包含 Dockerfile、CI/CD 配置 |
| 5 | 设计学习路线图（下一步深度学习专题） | 规划能力 | 列出 NLP、CV、强化学习等方向 |

---

## 外功密卷 · 工具与资源

- **书籍**：《Python 编程：从入门到实践》《流畅的 Python》《Hands-On Machine Learning》《Deep Learning with PyTorch》。
- **官方文档**：Python 官方文档、NumPy/Pandas/Matplotlib/PyTorch 文档。
- **课程**：MIT 6.0001、CS231n、fast.ai Practical Deep Learning for Coders。
- **练习平台**：LeetCode、Kaggle、天池、Datawhale 打卡营。
- **效率法器**：VS Code、JupyterLab、Black、isort、Poetry、DVC、Weights & Biases。

---

## 灵感宝库 · 项目创意

| 项目 | 境界 | 灵感说明 |
| --- | --- | --- |
| 灵药交易分析仪 | 结丹境 | 清洗交易数据、分析供需、构建交互式仪表盘 |
| 灵兽资质评定系统 | 元婴境 | 结合分类算法预测灵兽等级并输出解释 |
| 门派秘术推荐引擎 | 元婴境 | 基于用户喜好推荐修炼法门，应用协同过滤 |
| 天机占卜卷轴 | 化身境 | 使用时间序列或 Transformer 预测灵气波动 |
| 云端法阵守卫 | 化身境 | 将模型部署在云端并实现监控与自动回滚 |

---

## 心法总纲 · 复盘与精进

1. **周度复盘**：每周回答“学到了什么 / 卡在哪 / 下一步计划”。
2. **月度回顾**：整理阶段项目，寻找可复用模块与最佳实践。
3. **对外分享**：在博客、公众号或社区分享心得，深化理解。
4. **师徒互助**：主动指导新入门的道友，以教促学。

---

## 答疑密信 · 常见问题

**Q1：我完全没基础，可以直接练气吗？** 当然可以，按照练气境步骤依次完成即可，遇到术语不懂就搜索官方文档或附录资源。

**Q2：如何选择练气题？** 建议全部动手做，若时间有限，至少完成每个知识点的前三题并理解解析。

**Q3：需要掌握多少数学？** 至少复习高中函数、微积分入门与概率统计基础，在元婴境前逐渐补齐。

**Q4：没有 GPU 如何修炼化身境？** 可以使用 Kaggle Notebook、Google Colab 或租用云 GPU，亦可先在 CPU 上跑小型数据集。

**Q5：我完成了所有境界，下一步呢？** 深入研究专攻方向（CV/NLP/强化学习）、参与开源项目，或挑战更高难度的比赛。

祝你从练气启程，终有一日化身万千，纵横代码仙界！
