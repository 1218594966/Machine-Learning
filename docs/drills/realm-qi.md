# 练气境题库详解

本卷收录灵根初醒阶段的全部试炼题。每题都给出**任务目标 → 点拨思路 → 详细拆解**，并附带常见报错与修正建议，帮助初学者按梯度巩固基础。

## 灵材备置 · 环境与命令行

### 题目回顾
1. 🌱 入门：使用命令行创建多层目录 `sect/training/day1`，并切换到最深层后列出内容。
2. 🌿 进阶：查看当前路径并写入 `path.txt`，确认文件记录的路径与终端输出一致。
3. 🔥 突破：激活虚拟环境 `.venv`，执行 `pip list` 对比全局环境与虚拟环境的差别。
4. 🌟 圆满：配置 pip 镜像，加速包安装，并记录修改前后的配置。
5. 🛡️ 化神：在 VS Code 设置断点调试 `hello.py`，逐行观察变量变化。

### 逐题拆解
- **题 1**：先用 `pwd`（Windows 用 `cd`）确认当前目录，再执行 `mkdir -p sect/training/day1`。如果系统不支持 `-p`，可以逐级 `mkdir`。完成后 `cd sect/training/day1`，执行 `ls -a` 或 `dir` 验证空目录结构。
- **题 2**：`pwd > path.txt` 会把路径写入文件；PowerShell 则使用 `Get-Location | Out-File path.txt`。打开 `path.txt` 检查尾部是否有多余空行，并比较是否与终端一致。
- **题 3**：激活命令在不同平台不同：macOS/Linux 使用 `source .venv/bin/activate`，PowerShell 使用 `.venv\Scripts\Activate.ps1`。激活后运行 `which python`/`where python` 确认指向 `.venv` 目录，再比较两次 `pip list` 输出差异。
- **题 4**：运行 `pip config list` 备份原设置，再执行 `pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple`。安装一个小包（如 `pip install rich -U`）观察下载速度，最后记录如何通过 `pip config unset` 恢复。
- **题 5**：VS Code 调试流程为：打开文件 → 左侧单击行号设置断点 → `F5` 选择 Python 配置 → 在断点处使用调试面板查看变量值。结束时按 `Shift+F5`，以免调试会话占用终端。

## 灵根觉醒 · 变量与数据类型

### 题目回顾
1. 🌱 入门：声明 `spirit_name`、`age`、`is_novice` 并打印值与类型。
2. 🌿 进阶：判断 `mana = None` 是否等同空字符串，同时比较布尔值。
3. 🔥 突破：将字符串 `"108"` 与整数 `12` 相加，展示错误并修复。
4. 🌟 圆满：使用 `id()` 比较相同值不同对象的内存地址，解释差异。
5. 🛡️ 化神：通过 `is` 判断两个变量是否指向同一对象，结合浅拷贝实验。

### 逐题拆解
- **题 1**：声明后使用 `print(type(spirit_name), spirit_name)` 同时输出类型和值，并补充 `type(type(spirit_name))` 观察 `type` 自身的类型。
- **题 2**：打印 `id(mana)` 与 `id(None)` 证明 `None` 是单例；再比较 `mana == ""` 与 `bool(mana)` 的结果，帮助理解身份与布尔语义。
- **题 3**：直接 `"108" + 12` 会抛出 `TypeError`。引导先 `value = "108".strip()` 再 `int(value)`。若输入含中文，可提醒先过滤非数字字符。
- **题 4**：创建 `a = [1, 2]; b = [1, 2]`，对比 `id(a)`、`id(b)`。再对小整数（`id(10)`）与大整数（`id(1000)`）进行测试，引出整数驻留机制。
- **题 5**：构造 `a = []; b = a; c = a.copy()`，比较 `a is b` 与 `a is c`。进一步引入 `copy.deepcopy`，说明嵌套结构需要深拷贝。

## 灵气入体 · 数值与字符串

### 题目回顾
1. 🌱 入门：计算 `(5 ** 2) // 3` 与 `5 ** (2 // 3)`，写出运算顺序解释。
2. 🌿 进阶：验证 `0.1 + 0.2 == 0.3`，并打印误差来源。
3. 🔥 突破：构造 `"剑"*3` 与 `"仙"*2` 的组合字符串，输出带分隔符的版本。
4. 🌟 圆满：使用 f-string 格式化输出修士信息，包含对齐与千分位。
5. 🛡️ 化神：统计咒语字符串中元音个数与占比，忽略大小写和空格。

### 逐题拆解
- **题 1**：手绘运算顺序树或使用 Python Tutor 可视化，帮助理解指数比整除优先。可要求写出 `5 ** 2 // 3` 做比较。
- **题 2**：打印 `0.1 + 0.2 - 0.3` 观察 `5.551115123125783e-17`，再使用 `math.isclose` 与 `Decimal('0.1')` 解释浮点误差。
- **题 3**：建议先写出 `"剑" * 3 + "-" + "仙" * 2`，再尝试 `"剑".join(['剑', '剑', '剑'])` 以及 `"·".join(list('剑剑剑仙仙'))`，体会性能差异。
- **题 4**：演示 `f"{name:^10}"`、`f"{realm:>4}"`、`f"{mana:,.2f}"` 等格式化；要求写注释说明各部分用途。
- **题 5**：将字符串转小写后迭代统计。遇到标点符号可用 `str.isalpha()` 过滤，并最终输出百分比格式 `f"{ratio:.1%}"`。

## 灵识初开 · 输入输出

### 题目回顾
1. 🌱 入门：通过 `input` 获取姓名，首字母大写后打印欢迎语。
2. 🌿 进阶：读取多组数字求平均，需处理空输入或非法字符。
3. 🔥 突破：把姓名与境界写入 `blessing.txt`，采用追加模式并换行。
4. 🌟 圆满：按行读取 `blessing.txt`，编号输出且跳过空行。
5. 🛡️ 化神：将错误信息输出到 `stderr`，并演示如何分离日志与正常输出。

### 逐题拆解
- **题 1**：`input().strip().title()` 可移除首尾空格并美化名称。提醒中文不一定适用 `.title()`，可改用自定义函数。
- **题 2**：使用 `while True` 与 `try/except` 捕获 `ValueError`。若输入为空列表，提示“至少输入一个数字”。
- **题 3**：`with open("blessing.txt", "a", encoding="utf-8") as f:` 写入 `f"{name},{realm}\n"`。强调 `a` 追加不会清空旧记录。
- **题 4**：`with open(...) as f: for idx, line in enumerate(f, 1):` 输出 `f"{idx:02d} | {line.strip()}"`。`if not line.strip(): continue` 可跳过空行。
- **题 5**：导入 `sys`，使用 `print("格式错误", file=sys.stderr)`。在命令行演示 `python script.py 1>out.log 2>err.log` 分离输出。

## 气脉调息 · 调试与笔记

### 题目回顾
1. 🌱 入门：将主要逻辑包裹在 `main()` 函数，并以 `if __name__ == "__main__"` 作为入口。
2. 🌿 进阶：在 REPL 中使用 `dir()` 与 `help()` 查阅模块说明。
3. 🔥 突破：理解交互式解释器的 `_` 变量，观察何时会更新。
4. 🌟 圆满：使用 `logging` 替换 `print`，自定义格式和级别。
5. 🛡️ 化神：设计 Markdown 学习日志模板，包含“问题-尝试-成果”。

### 逐题拆解
- **题 1**：编写 `def main():` 并在末尾调用。导入该模块到另一个脚本，观察无入口保护时会立即执行的差异。
- **题 2**：在 Python 中运行 `dir(math)`、`help(math.sqrt)`，截屏记录。补充 `pydoc` 命令行用法：`python -m pydoc math`。
- **题 3**：在 REPL 依次执行表达式和赋值，观察 `_` 是否更新。强调 `_` 仅在交互式环境存在，脚本中不会自动生成。
- **题 4**：通过 `logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')` 配置输出，比较 `print` 与 `logging` 的差异。
- **题 5**：推荐日志模板：`### 今日问题` → `### 调试过程` → `### 收获与遗留`。建议同步至 GitHub Wiki、Notion 或 Obsidian 形成知识库。
