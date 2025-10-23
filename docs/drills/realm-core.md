# 结丹境题库详解

结丹阶段聚焦数据炼丹术：NumPy 阵列、Pandas 炼丹炉、清洗祛杂、可视化绘阵与数据报告。本卷按难度梯度拆解每道题的核心动作与常见陷阱。

## 难度阶梯导航

| 难度 | 核心关键词 | 代表练习 |
| --- | --- | --- |
| 🌱 入门 | NumPy | 创建矩阵、变形数组、使用布尔索引 |
| 🌿 进阶 | Pandas | 构造 DataFrame、设置索引、特征构造 |
| 🔥 突破 | 数据清洗 | 统计缺失、填补空值、处理离群点 |
| 🌟 圆满 | 可视化 | 绘制折线、箱线、热力与双轴图 |
| 🛡️ 化神 | 报告输出 | 生成 Markdown/HTML 报告，总结洞见 |

## 实战建议

- **一份数据两份副本**：保留原始数据，所有清洗步骤在副本上进行，必要时可对照还原。
- **记录假设**：每次清洗或可视化前写下“猜测”与“验证方式”，结果与预期不符时更易定位原因。
- **善用 Notebook**：在 Notebook 中穿插 Markdown 单元记录结论，最终导出即可作为报告初稿。

## 灵石阵列 · NumPy 基础

### 题目回顾
1. 🌱 入门：创建 3×3 单位矩阵并计算行列式。
2. 🌿 进阶：将一维数组重塑为 2×6，截取前两列。
3. 🔥 突破：演示广播规则，计算 `arr + np.array([1, 2, 3])`。
4. 🌟 圆满：使用布尔索引筛选能量值 > 50 的元素。
5. 🛡️ 化神：同时计算均值、方差、标准差并解释差异。

### 逐题拆解
- **题 1**：`matrix = np.eye(3); det = np.linalg.det(matrix)`。说明单位矩阵行列式恒为 1，同时使用 `np.linalg.inv` 验证逆矩阵仍为单位。
- **题 2**：`reshaped = arr.reshape(2, 6)`；切片 `reshaped[:, :2]` 取前两列。提醒 `reshape` 需保证元素数量匹配，否则会触发 `ValueError`。
- **题 3**：打印 `arr.shape` 与 `(1, 3)` 的数组形状，解释行向量如何沿着行广播。建议尝试列向量 `np.array([[1], [2], [3]])` 感受差异。
- **题 4**：`mask = arr > 50; filtered = arr[mask]`。展示 `mask.sum()` 统计满足条件的数量，强调布尔数组与原数组等长。
- **题 5**：`mean = arr.mean(); var = arr.var(); std = arr.std()`；解释 `var` 是方差、`std` 是标准差。可指定 `ddof=1` 计算无偏估计。

## 丹炉调配 · Pandas 数据帧

### 题目回顾
1. 🌱 入门：从字典创建 DataFrame，查看前几行与基本信息。
2. 🌿 进阶：设置 `id` 为索引并按索引排序。
3. 🔥 突破：选择多列并新增特征列，例如功效比值。
4. 🌟 圆满：使用 `groupby` 统计各类灵草平均效力。
5. 🛡️ 化神：合并两个 DataFrame，对齐缺失并解释连接方式。

### 逐题拆解
- **题 1**：`df = pd.DataFrame(data); print(df.head()); print(df.info())`。说明 `info()` 能同时查看列类型与缺失情况。
- **题 2**：`df = df.set_index('id').sort_index()`；提醒索引不再是普通列，如需恢复可 `df.reset_index()`。
- **题 3**：`subset = df[['mana', 'power']]; df['ratio'] = df['power'] / df['mana']`。注意除零风险，可使用 `df['mana'].replace(0, np.nan)`。
- **题 4**：`avg = df.groupby('type')['power'].mean().sort_values(ascending=False)`；可链式调用 `.reset_index()` 方便可视化。
- **题 5**：`merged = pd.merge(df1, df2, on='id', how='left')`；解释 `how` 参数（inner/outer/left/right），观察合并后缺失值，用 `fillna` 处理。

## 杂质祛除 · 数据清洗

### 题目回顾
1. 🌱 入门：统计每列缺失比例并输出格式化结果。
2. 🌿 进阶：使用 `fillna` 或 `interpolate` 填补不同类型的缺失。
3. 🔥 突破：基于 IQR 规则过滤异常值，并统计剩余数量。
4. 🌟 圆满：统一字符串列大小写、去除空白与特殊字符。
5. 🛡️ 化神：封装数据清洗函数，自动记录操作日志。

### 逐题拆解
- **题 1**：`missing = df.isna().mean().mul(100).round(2)`；使用 `to_frame('missing_ratio')` 转成表格并输出 `to_markdown()`。
- **题 2**：对数值列 `df['mana'].fillna(df['mana'].mean(), inplace=True)`；对时间序列 `df['power'].interpolate(method='time')`。提醒在填补前复制数据以免污染原始数据。
- **题 3**：计算 `Q1 = df['mana'].quantile(0.25)`、`Q3 = df['mana'].quantile(0.75)`，IQR = `Q3 - Q1`，过滤 `df[(df['mana'] >= Q1 - 1.5*IQR) & (df['mana'] <= Q3 + 1.5*IQR)]`。
- **题 4**：`df['herb'] = df['herb'].str.strip().str.title().str.replace(r'[^\w\s]', '', regex=True)`；说明 `.str` 方法可链式调用，正则用于清理符号。
- **题 5**：封装 `def clean_dataset(df): steps = []; ...; steps.append("填补 mana 均值"); return cleaned_df, steps`。将步骤写入日志文件或返回列表。

## 灵火绘阵 · 可视化

### 题目回顾
1. 🌱 入门：使用 Matplotlib 绘制修为随时间的折线图。
2. 🌿 进阶：绘制多类别灵草功效箱线图，比较分布。
3. 🔥 突破：创建相关性热力图，标注数值并调整色系。
4. 🌟 圆满：制作双轴图展示成本与收益随时间变化。
5. 🛡️ 化神：分别导出 PNG 与 SVG 图像，控制尺寸与 DPI。

### 逐题拆解
- **题 1**：`plt.plot(days, power, marker='o'); plt.xlabel('修炼日'); plt.ylabel('灵力'); plt.title('修为走势')`。添加 `plt.grid(True)` 提升可读性。
- **题 2**：`sns.boxplot(x='type', y='power', data=df)`；添加 `sns.stripplot` 展示数据点。说明箱线图四分位含义。
- **题 3**：`sns.heatmap(df.corr(), annot=True, cmap='YlGnBu', fmt='.2f')`；建议设置 `mask=np.triu(np.ones_like(corr, dtype=bool))` 仅展示下三角。
- **题 4**：`fig, ax1 = plt.subplots(); ax2 = ax1.twinx(); ax1.plot(days, cost, color='tab:red'); ax2.plot(days, revenue, color='tab:blue')`。注意添加图例和轴标签。
- **题 5**：`plt.savefig('trend.png', dpi=300); plt.savefig('trend.svg', format='svg')`。解释矢量图（SVG）与位图（PNG）的适用场景。

## 丹方评估 · 数据报告

### 题目回顾
1. 🌱 入门：调用 `describe` 输出描述统计，并转为 Markdown 表格。
2. 🌿 进阶：编写 `profile_data(df)` 返回缺失率、唯一值数量与列类型。
3. 🔥 突破：生成 `pandas-profiling` 报告，保存为 HTML。
4. 🌟 圆满：比较清洗前后 DataFrame 的差异，写入日志。
5. 🛡️ 化神：撰写数据风险综述，提出下一步采集与建模建议。

### 逐题拆解
- **题 1**：`df.describe().transpose().to_markdown()`；说明 `transpose` 可以让指标作为列，更易阅读。
- **题 2**：`def profile_data(df): return pd.DataFrame({'dtype': df.dtypes, 'missing': df.isna().sum(), 'unique': df.nunique()})`；可计算占比字段。
- **题 3**：`from ydata_profiling import ProfileReport; report = ProfileReport(df, minimal=True); report.to_file('profile.html')`；提醒大数据集需设置 `minimal=True`。
- **题 4**：`diff = df_raw.compare(df_clean)`；将 `diff` 输出到 `diff.to_csv('clean_diff.csv')`。说明 `compare` 会列出两个 DataFrame 不同的单元格。
- **题 5**：从数据缺失、异常值、类别不平衡、时间漂移等角度撰写总结。提供推荐行动，例如“补充采集 2020 年之前的数据以减少漂移”。
