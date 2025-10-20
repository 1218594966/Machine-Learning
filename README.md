# 机器学习模型管理平台

该项目提供了一个基于 FastAPI 的端到端机器学习管理平台，支持在网页端上传数据集、训练模型、预测结果并生成 SHAP 可视化分析，适合部署在宝塔面板（BT Panel）环境中。

## 功能特性

- 支持 CSV 数据集上传与管理。
- 集成随机森林（Random Forest）和 XGBoost 分类模型，可扩展其他算法。
- 完整的训练、预测、评估流程，提供常见指标与分类报告。
- 提供 SHAP 可视化解释能力，帮助理解模型决策。
- 前后端模块化设计，方便后续维护和扩展。

## 目录结构

```
app/
├── core/
│   ├── data_manager.py      # 数据集管理
│   ├── evaluation.py        # 模型评估指标
│   ├── model_manager.py     # 模型训练、持久化与解释
│   └── preprocess.py        # 数据预处理流程
├── static/
│   ├── css/style.css        # 页面样式
│   └── js/app.js            # 前端交互逻辑
├── templates/index.html     # 主页面模版
└── main.py                  # FastAPI 应用入口
models/                      # 已训练模型与元数据
uploads/                     # 上传的数据集
requirements.txt             # 项目依赖
```

## 环境准备

```bash
python -m venv venv
source venv/bin/activate  # Windows 使用 venv\\Scripts\\activate
pip install -r requirements.txt
```

## 开发环境运行

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

访问 `http://localhost:8000` 即可使用网页界面。

## 宝塔面板部署建议

1. 在宝塔面板中创建 Python 项目或使用 `Python 项目管理器`。
2. 上传/拉取代码，并安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 使用以下启动命令配置守护进程（Supervisor）：
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
4. 在宝塔面板中设置反向代理或使用 `Python 项目` 功能绑定域名，将外部请求转发到 8000 端口。
5. 确保 `uploads/` 与 `models/` 目录具有写权限，用于保存数据集与模型。

## 常见问题

- **XGBoost 安装**：若服务器缺乏编译环境，可通过 `pip install xgboost` 直接获取预编译 wheel（推荐 Python 3.8+）。
- **SHAP 图像生成**：首次生成 SHAP 图像可能耗时较长，确保服务器具备足够的内存。

欢迎根据业务需求继续拓展模型、可视化与鉴权等能力。
