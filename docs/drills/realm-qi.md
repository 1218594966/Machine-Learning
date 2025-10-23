# 练气境题库详解

本卷收录灵根初醒阶段的全部试炼题。每题都给出**任务目标 → 点拨思路 → 详细拆解**，并附带常见报错与修正建议，帮助初学者按梯度巩固基础。

## 难度阶梯导航

| 难度 | 核心关键词 | 代表练习 |
| --- | --- | --- |
| 🌱 入门 | 命令行、环境检测 | 创建多层目录、确认 Python/VS Code 是否就绪 |
| 🌿 进阶 | 变量、类型转换 | 使用 `type()`、`id()` 比较对象与数值差异 |
| 🔥 突破 | 运算、字符串 | 解决浮点误差，利用 f-string 制作信息卡 |
| 🌟 圆满 | 输入输出、文件 | 将 `input` 数据写入文件，再编号读取与处理空行 |
| 🛡️ 化神 | 调试、日志 | 使用 `logging`、`pdb`，并记录 Markdown 学习日志 |

## 零基础护身符

- **循序渐进**：每题都可先照抄示例运行，再尝试替换变量名或输出格式。
- **报错不慌**：把错误原文复制进日志，记录“触发动作 → 报错内容 → 尝试解决方案”。
- **多设备同步**：练气阶段建议使用 Git 提交或云盘同步，避免环境重装时丢失代码。

## 灵材备置 · 环境与命令行

### 题目回顾
1. 🌱 入门：使用命令行创建多层目录 `sect/training/day1`，并切换到最深层后列出内容。
2. 🌿 进阶：查看当前路径并写入 `path.txt`，确认文件记录的路径与终端输出一致。
3. 🔥 突破：激活虚拟环境 `.venv`，执行 `pip list` 对比全局环境与虚拟环境的差别。
4. 🌟 圆满：配置 pip 镜像，加速包安装，并记录修改前后的配置。
5. 🛡️ 化神：在 VS Code 设置断点调试 `hello.py`，逐行观察变量变化。

### 逐题拆解
- **题 1**：先用 `pwd`（Windows 用 `cd`）确认当前目录，再执行 `mkdir -p sect/training/day1`。如果系统不支持 `-p`，可以逐级 `mkdir`。完成后 `cd sect/training/day1`，执行 `ls -a` 或 `dir` 验证空目录结构。
  **示例命令**：

  ```bash
  pwd
  mkdir -p sect/training/day1
  cd sect/training/day1
  ls -a
  ```

  **预期输出**：

  ```text
  /Users/adept/projects/Machine-Learning
  .
  ..
  ```
- **题 2**：`pwd > path.txt` 会把路径写入文件；PowerShell 则使用 `Get-Location | Out-File path.txt`。打开 `path.txt` 检查尾部是否有多余空行，并比较是否与终端一致。
  **示例命令**：

  ```bash
  pwd > path.txt
  cat path.txt
  ```

  **预期输出**：

  ```text
  /Users/adept/projects/Machine-Learning
  ```
- **题 3**：激活命令在不同平台不同：macOS/Linux 使用 `source .venv/bin/activate`，PowerShell 使用 `.venv\Scripts\Activate.ps1`。激活后运行 `which python`/`where python` 确认指向 `.venv` 目录，再比较两次 `pip list` 输出差异。
  **示例命令**：

  ```bash
  python -m venv .venv
  source .venv/bin/activate
  which python
  pip list | head -n 5
  deactivate
  ```

  **预期输出**：

  ```text
  /Users/adept/projects/Machine-Learning/.venv/bin/python
  Package    Version
  ---------- -------
  pip        23.2.1
  setuptools 68.2.2
  ```
- **题 4**：运行 `pip config list` 备份原设置，再执行 `pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple`。安装一个小包（如 `pip install rich -U`）观察下载速度，最后记录如何通过 `pip config unset` 恢复。
  **示例命令**：

  ```bash
  pip config list
  pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
  pip install rich -U
  pip config unset global.index-url
  ```

  **预期输出**：

  ```text
  Writing to /Users/adept/Library/Application Support/pip/pip.ini
  Collecting rich
    Downloading rich-13.6.0-py3-none-any.whl (238 kB)
  ```
- **题 5**：VS Code 调试流程为：打开文件 → 左侧单击行号设置断点 → `F5` 选择 Python 配置 → 在断点处使用调试面板查看变量值。结束时按 `Shift+F5`，以免调试会话占用终端。
  **示例脚本**（`hello.py`）：

  ```python
  def greet(name: str) -> str:
      message = f"你好，{name}!"
      return message


  if __name__ == "__main__":
      print(greet("修士"))
  ```

  命中断点后，**调试控制台**会输出：

  ```text
  你好，修士!
  ```

## 灵根觉醒 · 变量与数据类型

### 题目回顾
1. 🌱 入门：声明 `spirit_name`、`age`、`is_novice` 并打印值与类型。
2. 🌿 进阶：判断 `mana = None` 是否等同空字符串，同时比较布尔值。
3. 🔥 突破：将字符串 `"108"` 与整数 `12` 相加，展示错误并修复。
4. 🌟 圆满：使用 `id()` 比较相同值不同对象的内存地址，解释差异。
5. 🛡️ 化神：通过 `is` 判断两个变量是否指向同一对象，结合浅拷贝实验。

### 逐题拆解
- **题 1**：声明后使用 `print(type(spirit_name), spirit_name)` 同时输出类型和值，并补充 `type(type(spirit_name))` 观察 `type` 自身的类型。
  **示例代码**：

  ```python
  spirit_name = "余音"
  age = 18
  is_novice = True

  print(spirit_name, type(spirit_name))
  print(age, type(age))
  print(is_novice, type(is_novice))
  print(type(type(spirit_name)))
  ```

  **运行结果**：

  ```text
  余音 <class 'str'>
  18 <class 'int'>
  True <class 'bool'>
  <class 'type'>
  ```
- **题 2**：打印 `id(mana)` 与 `id(None)` 证明 `None` 是单例；再比较 `mana == ""` 与 `bool(mana)` 的结果，帮助理解身份与布尔语义。
  **示例代码**：

  ```python
  mana = None

  print(id(mana), id(None))
  print("mana == ''?", mana == "")
  print("bool(mana)", bool(mana))
  ```

  **运行结果**：

  ```text
  9790944 9790944
  mana == ''? False
  bool(mana) False
  ```
- **题 3**：直接 `"108" + 12` 会抛出 `TypeError`。引导先 `value = "108".strip()` 再 `int(value)`。若输入含中文，可提醒先过滤非数字字符。
  **示例代码**：

  ```python
  value = "108"
  try:
      print(value + 12)
  except TypeError as exc:
      print("报错信息:", exc)

  total = int(value) + 12
  print("修复后:", total)
  ```

  **运行结果**：

  ```text
  报错信息: can only concatenate str (not "int") to str
  修复后: 120
  ```
- **题 4**：创建 `a = [1, 2]; b = [1, 2]`，对比 `id(a)`、`id(b)`。再对小整数（`id(10)`）与大整数（`id(1000)`）进行测试，引出整数驻留机制。
  **示例代码**：

  ```python
  a = [1, 2]
  b = [1, 2]
  print("a == b?", a == b)
  print("a is b?", a is b)
  print("id(a)", id(a))
  print("id(b)", id(b))

  print("id(10)", id(10))
  print("id(1000)", id(1000))
  ```

  **运行结果**：

  ```text
  a == b? True
  a is b? False
  id(a) 139876543210128
  id(b) 139876543209984
  id(10) 9793376
  id(1000) 140082855348464
  ```
- **题 5**：构造 `a = []; b = a; c = a.copy()`，比较 `a is b` 与 `a is c`。进一步引入 `copy.deepcopy`，说明嵌套结构需要深拷贝。
  **示例代码**：

  ```python
  import copy

  a = [["剑"], ["符"]]
  b = a
  c = a.copy()
  d = copy.deepcopy(a)

  print("a is b?", a is b)
  print("a is c?", a is c)
  print("a == d?", a == d)

  a[0].append("阵")
  print("修改后 a:", a)
  print("浅拷贝 c:", c)
  print("深拷贝 d:", d)
  ```

  **运行结果**：

  ```text
  a is b? True
  a is c? False
  a == d? True
  修改后 a: [['剑', '阵'], ['符']]
  浅拷贝 c: [['剑', '阵'], ['符']]
  深拷贝 d: [['剑'], ['符']]
  ```

## 灵气入体 · 数值与字符串

### 题目回顾
1. 🌱 入门：计算 `(5 ** 2) // 3` 与 `5 ** (2 // 3)`，写出运算顺序解释。
2. 🌿 进阶：验证 `0.1 + 0.2 == 0.3`，并打印误差来源。
3. 🔥 突破：构造 `"剑"*3` 与 `"仙"*2` 的组合字符串，输出带分隔符的版本。
4. 🌟 圆满：使用 f-string 格式化输出修士信息，包含对齐与千分位。
5. 🛡️ 化神：统计咒语字符串中元音个数与占比，忽略大小写和空格。

### 逐题拆解
- **题 1**：手绘运算顺序树或使用 Python Tutor 可视化，帮助理解指数比整除优先。可要求写出 `5 ** 2 // 3` 做比较。
  **示例代码**：

  ```python
  print("(5 ** 2) // 3 =", (5 ** 2) // 3)
  print("5 ** (2 // 3) =", 5 ** (2 // 3))
  print("5 ** 2 // 3 =", 5 ** 2 // 3)
  ```

  **运行结果**：

  ```text
  (5 ** 2) // 3 = 8
  5 ** (2 // 3) = 1
  5 ** 2 // 3 = 8
  ```
- **题 2**：打印 `0.1 + 0.2 - 0.3` 观察 `5.551115123125783e-17`，再使用 `math.isclose` 与 `Decimal('0.1')` 解释浮点误差。
  **示例代码**：

  ```python
  import math
  from decimal import Decimal

  result = 0.1 + 0.2
  print("0.1 + 0.2 == 0.3?", result == 0.3)
  print("误差:", result - 0.3)
  print("math.isclose:", math.isclose(result, 0.3))

  precise = Decimal("0.1") + Decimal("0.2")
  print("Decimal 结果:", precise)
  ```

  **运行结果**：

  ```text
  0.1 + 0.2 == 0.3? False
  误差: 5.551115123125783e-17
  math.isclose: True
  Decimal 结果: 0.3
  ```
- **题 3**：建议先写出 `"剑" * 3 + "-" + "仙" * 2`，再尝试 `"剑".join(['剑', '剑', '剑'])` 以及 `"·".join(list('剑剑剑仙仙'))`，体会性能差异。
  **示例代码**：

  ```python
  combo = "剑" * 3 + "-" + "仙" * 2
  print(combo)

  parts = ["剑"] * 3 + ["仙"] * 2
  print("·".join(parts))
  ```

  **运行结果**：

  ```text
  剑剑剑-仙仙
  剑·剑·剑·仙·仙
  ```
- **题 4**：演示 `f"{name:^10}"`、`f"{realm:>4}"`、`f"{mana:,.2f}"` 等格式化；要求写注释说明各部分用途。
  **示例代码**：

  ```python
  name = "云岚"
  realm = "练气"
  mana = 10800.567

  card = f"{name:^10}|{realm:>4}|{mana:,.2f}"
  print(card)
  ```

  **运行结果**：

  ```text
    云岚    |  练气|10,800.57
  ```
- **题 5**：将字符串转小写后迭代统计。遇到标点符号可用 `str.isalpha()` 过滤，并最终输出百分比格式 `f"{ratio:.1%}"`。
  **示例代码**：

  ```python
  chant = "Fire Ball"  # 任意咒语
  vowels = "aeiou"

  cleaned = [ch.lower() for ch in chant if ch.isalpha()]
  count = sum(ch in vowels for ch in cleaned)
  ratio = count / len(cleaned)

  print("元音个数:", count)
  print("占比:", f"{ratio:.1%}")
  ```

  **运行结果**：

  ```text
  元音个数: 3
  占比: 60.0%
  ```

## 灵识初开 · 输入输出

### 题目回顾
1. 🌱 入门：通过 `input` 获取姓名，首字母大写后打印欢迎语。
2. 🌿 进阶：读取多组数字求平均，需处理空输入或非法字符。
3. 🔥 突破：把姓名与境界写入 `blessing.txt`，采用追加模式并换行。
4. 🌟 圆满：按行读取 `blessing.txt`，编号输出且跳过空行。
5. 🛡️ 化神：将错误信息输出到 `stderr`，并演示如何分离日志与正常输出。

### 逐题拆解
- **题 1**：`input().strip().title()` 可移除首尾空格并美化名称。提醒中文不一定适用 `.title()`，可改用自定义函数。
  **示例代码**（保存为 `greet.py`）：

  ```python
  name = input("请输入姓名：").strip()
  greeting = f"欢迎加入修仙之旅，{name.title()}!"
  print(greeting)
  ```

  **运行演示**：

  ```text
  请输入姓名：  ling hu
  欢迎加入修仙之旅，Ling Hu!
  ```
- **题 2**：使用 `while True` 与 `try/except` 捕获 `ValueError`。若输入为空列表，提示“至少输入一个数字”。
  **示例代码**（`avg.py`）：

  ```python
  numbers: list[float] = []
  while True:
      raw = input("请输入数字，留空结束：").strip()
      if not raw:
          break
      try:
          numbers.append(float(raw))
      except ValueError:
          print("⚠️ 请输入合法数字，例如 3.14 或 42")

  if not numbers:
      raise SystemExit("至少输入一个数字")

  avg = sum(numbers) / len(numbers)
  print(f"平均值：{avg:.2f}")
  ```

  **运行演示**：

  ```text
  请输入数字，留空结束：3
  请输入数字，留空结束：abc
  ⚠️ 请输入合法数字，例如 3.14 或 42
  请输入数字，留空结束：6
  请输入数字，留空结束：
  平均值：4.50
  ```
- **题 3**：`with open("blessing.txt", "a", encoding="utf-8") as f:` 写入 `f"{name},{realm}\n"`。强调 `a` 追加不会清空旧记录。
  **示例代码**：

  ```python
  name = "白砾"
  realm = "练气"

  with open("blessing.txt", "a", encoding="utf-8") as f:
      f.write(f"{name},{realm}\n")

  print("已写入 blessing.txt")
  ```

  **文件片段**（多次执行会追加）：

  ```text
  白砾,练气
  沐言,练气
  ```
- **题 4**：`with open(...) as f: for idx, line in enumerate(f, 1):` 输出 `f"{idx:02d} | {line.strip()}"`。`if not line.strip(): continue` 可跳过空行。
  **示例代码**：

  ```python
  from pathlib import Path

  path = Path("blessing.txt")
  if not path.exists():
      raise SystemExit("请先完成上一题生成 blessing.txt")

  with path.open(encoding="utf-8") as f:
      for idx, line in enumerate(f, 1):
          content = line.strip()
          if not content:
              continue
          print(f"{idx:02d} | {content}")
  ```

  **运行结果**：

  ```text
  01 | 白砾,练气
  02 | 沐言,练气
  ```
- **题 5**：导入 `sys`，使用 `print("格式错误", file=sys.stderr)`。在命令行演示 `python script.py 1>out.log 2>err.log` 分离输出。
  **示例代码**（`log_demo.py`）：

  ```python
  import sys

  print("正常输出: 练气完成")
  print("警告: 请检查灵力输入格式", file=sys.stderr)
  ```

  **命令行演示**：

  ```bash
  python log_demo.py 1>out.log 2>err.log
  cat out.log
  cat err.log
  ```

  **日志内容**：

  ```text
  # out.log
  正常输出: 练气完成

  # err.log
  警告: 请检查灵力输入格式
  ```

## 气脉调息 · 调试与笔记

### 题目回顾
1. 🌱 入门：将主要逻辑包裹在 `main()` 函数，并以 `if __name__ == "__main__"` 作为入口。
2. 🌿 进阶：在 REPL 中使用 `dir()` 与 `help()` 查阅模块说明。
3. 🔥 突破：理解交互式解释器的 `_` 变量，观察何时会更新。
4. 🌟 圆满：使用 `logging` 替换 `print`，自定义格式和级别。
5. 🛡️ 化神：设计 Markdown 学习日志模板，包含“问题-尝试-成果”。

### 逐题拆解
- **题 1**：编写 `def main():` 并在末尾调用。导入该模块到另一个脚本，观察无入口保护时会立即执行的差异。
  **示例代码**（`main_guard.py`）：

  ```python
  def main() -> None:
      print("正在运行主逻辑")


  if __name__ == "__main__":
      main()
  ```

  **验证步骤**：

  ```bash
  python main_guard.py
  python - <<'PY'
  import main_guard
  print("导入后不会自动执行 main()")
  PY
  ```

  **终端输出**：

  ```text
  正在运行主逻辑
  导入后不会自动执行 main()
  ```
- **题 2**：在 Python 中运行 `dir(math)`、`help(math.sqrt)`，截屏记录。补充 `pydoc` 命令行用法：`python -m pydoc math`。
  **示例操作**：

  ```bash
  python - <<'PY'
  import math
  print(dir(math)[:5])
  help(math.sqrt)
  PY

  python -m pydoc math | head -n 10
  ```

  **输出摘要**：

  ```text
  ['acos', 'acosh', 'asin', 'asinh', 'atan']
  Help on built-in function sqrt in module math:

  sqrt(x, /)
      Return the square root of x.
  ```
- **题 3**：在 REPL 依次执行表达式和赋值，观察 `_` 是否更新。强调 `_` 仅在交互式环境存在，脚本中不会自动生成。
  **示例操作**：

  ```bash
  python - <<'PY'
  2 + 3
  print("_ =", _)
  _ = "手动覆盖"
  print("再次计算:", 5 * 5)
  print("此时 _:", _)
  PY
  ```

  **输出摘要**：

  ```text
  _ = 5
  再次计算: 25
  此时 _: 手动覆盖
  ```
- **题 4**：通过 `logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')` 配置输出，比较 `print` 与 `logging` 的差异。
  **示例代码**（`log_config.py`）：

  ```python
  import logging

  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s %(levelname)s %(message)s",
  )

  print("使用 print 输出普通消息")
  logging.info("使用 logging 记录信息级别日志")
  logging.warning("发现灵力波动异常")
  ```

  **运行结果**：

  ```text
  使用 print 输出普通消息
  2024-03-01 10:00:00,123 INFO 使用 logging 记录信息级别日志
  2024-03-01 10:00:00,123 WARNING 发现灵力波动异常
  ```
- **题 5**：推荐日志模板：`### 今日问题` → `### 调试过程` → `### 收获与遗留`。建议同步至 GitHub Wiki、Notion 或 Obsidian 形成知识库。
  **示例模板**（`logs/2024-03-01.md`）：

  ```markdown
  # 2024-03-01 · 练气修炼日志

  ### 今日问题
  - 运行 `avg.py` 输入文本时报错 `ValueError`

  ### 调试过程
  - 重现：输入 `abc` → 捕获异常
  - 尝试：为 `float(raw)` 包裹 `try/except`
  - 结果：提示用户重新输入

  ### 收获与遗留
  - ✅ 学会区分 `print` 与 `logging`
  - 🔄 待办：整理虚拟环境搭建截图
  ```
