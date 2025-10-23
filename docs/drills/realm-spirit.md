# 元婴境题库详解

元婴阶段重在御阵驭兽：构建数据预处理流水线、驾驭监督/无监督算法、评估模型表现并准备部署。以下逐题详解帮助修士理解每项试炼的目标、关键代码与常见问题。

## 玄阵布局 · 数据预处理

### 题目回顾
1. 🌱 入门：使用 `train_test_split` 划分训练集与测试集，设定随机种子。
2. 🌿 进阶：构建 `ColumnTransformer`，为数值列做标准化，为类别列做独热编码。
3. 🔥 突破：识别类别不平衡并应用采样或 class_weight 解决。
4. 🌟 圆满：基于原始字段构造自定义特征（如灵石密度）。
5. 🛡️ 化神：将训练好的预处理器序列化保存，以便部署重用。

### 逐题拆解
- **题 1**：`X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`；解释 `stratify` 可保持类别比例。
- **题 2**：```
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ]
)
```
说明 `handle_unknown='ignore'` 可避免预测阶段出现未知类别时报错。
- **题 3**：计算类别分布 `y_train.value_counts(normalize=True)`；若极度不平衡，可尝试 `SMOTE`、`RandomOverSampler` 或模型参数 `class_weight='balanced'`。
- **题 4**：在 `ColumnTransformer` 外部先添加新列 `df['density'] = df['mana'] / df['weight'].replace(0, np.nan)`。注意除零处理，并考虑对数变换。
- **题 5**：`joblib.dump(preprocessor, 'preprocess.pkl')`；加载时 `preprocess = joblib.load('preprocess.pkl')`。提醒与模型一并存储版本号，确保兼容。

## 御兽要诀 · 监督学习

### 题目回顾
1. 🌱 入门：训练线性回归预测灵药价格，比较训练/测试 R²。
2. 🌿 进阶：使用逻辑回归预测修士是否飞升，调节正则化强度。
3. 🔥 突破：训练决策树并导出图形化结构。
4. 🌟 圆满：训练随机森林并分析特征重要性。
5. 🛡️ 化神：尝试梯度提升或 XGBoost，对比性能与训练时间。

### 逐题拆解
- **题 1**：`model = LinearRegression(); model.fit(X_train, y_train)`；输出 `r2_score(y_test, model.predict(X_test))`，并对比欠拟合/过拟合迹象。
- **题 2**：`clf = LogisticRegression(max_iter=1000, C=1.0)`；通过网格搜索 `C`、`penalty`（`l1`/`l2`）观察精确率与召回率变化。
- **题 3**：`tree = DecisionTreeClassifier(max_depth=5, random_state=42)`；使用 `plot_tree` 或 `export_graphviz` 可视化。强调限制深度避免过拟合。
- **题 4**：`rf = RandomForestClassifier(n_estimators=200, random_state=42)`；`importance = pd.Series(rf.feature_importances_, index=feature_names).sort_values()` 绘制条形图解释关键特征。
- **题 5**：安装 `xgboost` 或使用 `GradientBoostingClassifier`；记录训练时间 `time.perf_counter()` 与指标差异，提醒合理设置 `n_estimators`。

## 幻阵迷踪 · 无监督学习

### 题目回顾
1. 🌱 入门：使用 KMeans 对灵石属性聚类，观察簇中心。
2. 🌿 进阶：采用 PCA 降维至 2 维，并绘制聚类散点图。
3. 🔥 突破：进行层次聚类并绘制树状图。
4. 🌟 圆满：使用 DBSCAN 检测异常灵石，调整超参数。
5. 🛡️ 化神：计算轮廓系数，评估聚类效果。

### 逐题拆解
- **题 1**：`kmeans = KMeans(n_clusters=3, random_state=42)`；训练后 `kmeans.cluster_centers_` 给出簇中心。绘制散点图标注簇标签。
- **题 2**：`pca = PCA(n_components=2)`；`X_pca = pca.fit_transform(scaled_X)`；使用 `plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)`，说明方差贡献率。
- **题 3**：`Z = linkage(scaled_X, method='ward'); dendrogram(Z, truncate_mode='level', p=5)`；强调样本数量多时需裁剪图像。
- **题 4**：`db = DBSCAN(eps=0.3, min_samples=5)`；`labels = db.fit_predict(scaled_X)`；`labels == -1` 表示噪声。可用 `sklearn.neighbors.NearestNeighbors` 估计 `eps`。
- **题 5**：`from sklearn.metrics import silhouette_score; score = silhouette_score(scaled_X, labels)`；指导 0~1 的评分如何解读，低于 0 表示簇重叠。

## 阵法验收 · 模型评估

### 题目回顾
1. 🌱 入门：输出 `classification_report`，理解准确率、精确率、召回率与 F1。
2. 🌿 进阶：绘制混淆矩阵，解释四个象限的含义。
3. 🔥 突破：绘制 ROC 曲线并计算 AUC。
4. 🌟 圆满：绘制学习曲线，判断是否需要更多数据或正则化。
5. 🛡️ 化神：使用 `cross_val_score` 做 K 折交叉验证，统计均值与方差。

### 逐题拆解
- **题 1**：`from sklearn.metrics import classification_report; print(classification_report(y_test, y_pred))`；逐项解释 Precision/Recall/F1 含义。
- **题 2**：`from sklearn.metrics import ConfusionMatrixDisplay; ConfusionMatrixDisplay.from_predictions(y_test, y_pred)`；解读 TP/TN/FP/FN 对业务的影响。
- **题 3**：`fpr, tpr, thresholds = roc_curve(y_test, y_proba)`；`auc = roc_auc_score(y_test, y_proba)`；说明阈值调整如何影响召回率。
- **题 4**：`LearningCurveDisplay.from_estimator(model, X, y, cv=5)`；观察训练/验证曲线是否趋于收敛，评估是否过拟合。
- **题 5**：`scores = cross_val_score(model, X, y, cv=5)`；输出 `scores.mean()` 与 `scores.std()`。提醒交叉验证需保持数据泄漏可控。

## 灵阵调优 · Pipeline 与部署

### 题目回顾
1. 🌱 入门：构建包含预处理与模型的 `Pipeline`。
2. 🌿 进阶：使用 `GridSearchCV` 对 Pipeline 参数调优。
3. 🔥 突破：保存最佳模型并重新加载预测。
4. 🌟 圆满：使用 FastAPI 暴露预测接口。
5. 🛡️ 化神：编写批量推理脚本 `make_predictions.py`。

### 逐题拆解
- **题 1**：`pipeline = Pipeline([('prep', preprocessor), ('model', RandomForestClassifier())])`；强调 `fit` 会先拟合预处理再训练模型。
- **题 2**：`param_grid = {'model__n_estimators': [100, 200], 'model__max_depth': [None, 10, 20]}`；`GridSearchCV(pipeline, param_grid, cv=5)` 自动搜索。`model__` 前缀指向 Pipeline 子组件。
- **题 3**：`joblib.dump(grid.best_estimator_, 'best_model.joblib')`；加载后 `estimator = joblib.load('best_model.joblib')` 直接调用 `predict`。
- **题 4**：编写 FastAPI：```
from fastapi import FastAPI
import joblib
app = FastAPI()
model = joblib.load('best_model.joblib')

@app.post('/predict')
def predict(payload: dict):
    df = pd.DataFrame([payload])
    proba = model.predict_proba(df)[0, 1]
    return {'probability': float(proba)}
```
提醒启动 `uvicorn` 并测试。
- **题 5**：`python make_predictions.py --input data.csv --output predictions.csv`。脚本步骤：读取 CSV → `model.predict_proba` → 合并原始 ID 与预测结果 → 保存。加入日志说明处理数量。
