# 【炼气境】完整版速成教程：从凡人到炼气士

> 徒儿，你既已下定决心，为师便将这【炼气境】的完整心法倾囊相授。
> 此境乃万法之基，虽名“速成”，实则要你 **字字揣摩，行行亲历**。
> 记住：编程之道，“看懂”与“会用”之间隔着一道天堑，唯有亲手练习方能飞升。

---

## 第零步：开辟紫府 · 环境搭建

“工欲善其事，必先利其器。”在修炼之前，你需要一处“洞天福地”来汇聚灵气（代码）。

### 何为 Anaconda？
- **比喻**：它是官方发放的新手大礼包。
- **作用**：一次安装即可获得 Python（心法）、Jupyter Notebook（修炼场）以及 NumPy/Pandas 等常用法器。

### 三步入门（以 Windows 为例，macOS 与 Linux 流程类似）
1. 访问 [Anaconda 官网](https://www.anaconda.com/download)，下载与你系统对应的 **Python 3.x** 版本。
2. 双击安装包，保持默认设置即可。若看到 “Add Anaconda3 to my PATH environment variable”，勾选后命令行调用更方便。
3. 安装完成后，在开始菜单搜索并打开 **Anaconda Navigator**。

### 启动你的修炼场（Jupyter Notebook）
1. 在 Navigator 主界面找到 **Jupyter Notebook**，点击 “Launch”。
2. 浏览器会弹出一个新页面，即你的修炼主界面（文件管理器）。
3. 在右上角选择 `New -> Python 3 (ipykernel)`，即可创建一份 `.ipynb` 玉简。

### Jupyter 必背口诀
- **单元格 (Cell)**：灰色框就是符纸，在此书写咒语（代码）。
- **运行**：按下 `Shift + Enter` 施法，运行当前单元并切换到下一格。
- **模式切换**：
  - `Enter` → 编辑模式（绿色边框），可输入代码；
  - `Esc` → 命令模式（蓝色边框），可使用快捷键：`A` 向上插入，`B` 向下插入，`DD` 删除。

---

## 第一步：认识灵气 · 变量与数据类型

**变量 (Variable)** 是储存灵气的“储物袋”，你可为其贴上标签并放入不同种类的灵气（数据）。

### 基础心法
- `name = "小白"`：`name` 是储物袋标签；`=` 是注入灵气；`"小白"` 是存入的灵气。
- 常见灵气：
  - **整数 (int)**：`age = 18`
  - **浮点数 (float)**：`height = 175.5`
  - **字符串 (str)**：`spell = "火球术"`
  - **布尔值 (bool)**：`is_awake = True`

### 练功一：基本操作
```python
# 1. 变量赋值：创建储物袋
my_name = "小白"
my_age = 18
my_height = 175.5

# 2. 显形咒 (print)：查看袋中灵气
print("我的名字是:", my_name)
print("我的年龄是:", my_age)

# 3. 探查术 (type)：确认灵气属性
print("名字的类型是:", type(my_name))
print("年龄的类型是:", type(my_age))
print("身高的类型是:", type(my_height))

# 4. 灵气运算
one_year_later = my_age + 1
print("一年后我的年龄是:", one_year_later)

is_adult = my_age >= 18
print("我成年了吗?", is_adult)
print("成年的类型是:", type(is_adult))
```

### 运行示例
```text
我的名字是: 小白
我的年龄是: 18
名字的类型是: <class 'str'>
年龄的类型是: <class 'int'>
身高的类型是: <class 'float'>
一年后我的年龄是: 19
我成年了吗? True
成年的类型是: <class 'bool'>
```

### 常见走火入魔
- **变量名** 不能以数字开头，也不可与 `if`, `for` 等禁咒重名。
- **引号** 请使用英文双引号 `""` 或单引号 `''`，避免中文引号导致报错。
- **NameError**：调用未赋值的变量。运行前确认储物袋已装好灵气。

---

## 第二步：祭炼法宝 · 核心数据结构

当储物袋无法满足需要，就要祭出更高级的法宝来组织灵气。

### 2.1 列表（List）——乾坤袋
- **特性**：有序、可变，能装下任何类型的灵气。
- **理解**：每个物品都有编号，索引从 `0` 开始。

```python
# 创建：装备栏可混合存放
gears = ["布衣", "木剑", 100, 3.5]
scores = [90, 85, 92, 88]
print("我的装备:", gears)

# 读取：使用索引（从 0 开始）
print("我的武器是:", gears[1])
print("第一项分数:", scores[0])
print("最后一项分数:", scores[-1])

# 更新：通过索引替换
gears[0] = "铁甲"
print("升级装备:", gears)

# 增加：.append() 在末尾添加
scores.append(95)
print("追加后的分数:", scores)

# 删除：.pop() 弹出末尾元素
last_score = scores.pop()
print(f"弹出的分数: {last_score}, 剩余: {scores}")

# 切片：截取一段 [start:stop) 不含 stop
top_two = scores[0:2]
print("前两个分数:", top_two)
```

### 运行示例
```text
我的装备: ['布衣', '木剑', 100, 3.5]
我的武器是: 木剑
第一项分数: 90
最后一项分数: 88
升级装备: ['铁甲', '木剑', 100, 3.5]
追加后的分数: [90, 85, 92, 88, 95]
弹出的分数: 95, 剩余: [90, 85, 92, 88]
前两个分数: [90, 85]
```

### 2.2 字典（Dictionary）——丹方宝鉴
- **特性**：以键值对形式存储，通过 `key` 迅速找到对应的 `value`。
- **理解**：像翻阅丹方手册，用“丹药名”查找配方。

```python
# 创建：键唯一且常用字符串
student = {
    "name": "小白",
    "id": 1001,
    "major": "机器学习",
    "score": 95.5
}
print("学生信息:", student)

# 读取：通过 key 而非索引
print("学生姓名:", student["name"])
print("学生成绩:", student["score"])

# 更新或增加：直接赋值
student["score"] = 98.0
student["mentor"] = "Master Gemini"
print("更新后:", student)

# 删除：使用 del
del student["major"]
print("删除专业后:", student)
```

### 常见走火入魔
- `IndexError`：列表索引越界，例如 `scores[100]`。
- `KeyError`：访问了字典里不存在的键，如 `student["age"]`。

---

## 第三步：修炼心法 · 流程控制

让代码具备判断与重复执行的能力，是迈向高阶修士的关键。

### 3.1 条件判断（if / elif / else）——命运岔路
- 冒号 `:` 后记得换行并缩进四个空格，构成同一法阵。

```python
score = 85

print("开始评级...")

if score > 90:
    print("评级: 优秀 (A)")
elif score > 80:
    print("评级: 良好 (B)")
elif score >= 60:
    print("评级: 及格 (C)")
else:
    print("评级: 不及格 (F)")

print("评级结束。")
```

### 运行示例
```text
开始评级...
评级: 良好 (B)
评级结束。
```

### 3.2 `for` 循环——扫荡副本
- “对……中的每一项，施展同一段法术”。

```python
scores = [90, 85, 92, 88]
total_score = 0

for score in scores:
    print(f"当前分数: {score}")
    total_score = total_score + score

print(f"总分: {total_score}")

# 遍历字典默认取 key
for key in student:
    print(f"键: {key}, 值: {student[key]}")

# range(n) 用于重复 n 次
print("重复 3 次:")
for i in range(3):
    print(f"这是第 {i + 1} 次循环")
```

### 运行示例
```text
当前分数: 90
当前分数: 85
当前分数: 92
当前分数: 88
总分: 355
键: name, 值: 小白
键: id, 值: 1001
键: score, 值: 98.0
键: mentor, 值: Master Gemini
重复 3 次:
这是第 1 次循环
这是第 2 次循环
这是第 3 次循环
```

### 3.3 `while` 循环——闭关修炼
- 当你只知道“达到某条件后停止”时使用。

```python
qi = 0
day = 1

while qi < 100:
    print(f"第 {day} 天, 当前真气: {qi}")
    qi = qi + 10
    day = day + 1

print(f"修炼结束! 总用时: {day - 1} 天, 最终真气: {qi}")
```

### 运行示例
```text
第 1 天, 当前真气: 0
第 2 天, 当前真气: 10
第 3 天, 当前真气: 20
第 4 天, 当前真气: 30
第 5 天, 当前真气: 40
第 6 天, 当前真气: 50
第 7 天, 当前真气: 60
第 8 天, 当前真气: 70
第 9 天, 当前真气: 80
第 10 天, 当前真气: 90
修炼结束! 总用时: 10 天, 最终真气: 100
```

### 常见走火入魔
- **缩进错误 (`IndentationError`)**：未正确缩进或混用 Tab/空格。
- **死循环**：忘记在循环内改变条件，例如 `qi` 从未增加。

---

## 第四步：凝练神通 · 函数

当你多次重复同一段法术，便该将其凝练成可复用的“神通”。

### 函数要点
- `def` 定义神通名称与参数。
- `return` 将修炼成果传回。
- 文档字符串 `"""..."""` 用于说明用途。

```python
def greet(name):
    """给指定的修士打招呼。"""
    print(f"你好, {name}! 欢迎来到炼气境。")

# 施展神通
greet("小白")
greet("小黑")


def add_spells(a, b):
    """返回两段灵气之和。"""
    total = a + b
    return total

sum_result = add_spells(10, 5)
print(f"10 + 5 的结果是: {sum_result}")
print(f"100 + 200 的结果是: {add_spells(100, 200)}")


def get_grade(score):
    """根据分数返回评级。"""
    if score > 90:
        return "A"
    elif score > 80:
        return "B"
    elif score >= 60:
        return "C"
    else:
        return "F"

print(f"85 分的评级是: {get_grade(85)}")
print(f"55 分的评级是: {get_grade(55)}")
```

### 运行示例
```text
你好, 小白! 欢迎来到炼气境。
你好, 小黑! 欢迎来到炼气境。
10 + 5 的结果是: 15
100 + 200 的结果是: 300
85 分的评级是: B
55 分的评级是: F
```

---

## 第五步：炼气大成 · 综合试炼

现在把四大心法融会贯通，完成终极任务。

### 试炼任务
编写函数 `process_students`，接收学生名单（列表中每个元素为学生信息字典），并完成以下步骤：
1. 遍历名单，读取每位学生的 `score`。
2. 调用 `get_grade` 计算评级，并以键 `"grade"` 写回字典。
3. 若评级不及格 (`F`)，记录姓名并提醒；否则为其庆贺。
4. 最终返回一个仅包含不及格学生姓名的新列表。

### 演练代码
```python
def get_grade(score):
    if score > 90:
        return "A"
    elif score > 80:
        return "B"
    elif score >= 60:
        return "C"
    else:
        return "F"


def process_students(student_list):
    """处理学生名单，返回不及格名单。"""
    failed_students = []

    for student in student_list:
        score = student["score"]
        grade = get_grade(score)
        student["grade"] = grade

        if grade == "F":
            print(f"注意: {student['name']} 不及格。")
            failed_students.append(student["name"])
        else:
            print(f"恭喜 {student['name']}, 评级 {grade}")

    return failed_students


class_roster = [
    {"name": "小白", "score": 92},
    {"name": "小黑", "score": 81},
    {"name": "小红", "score": 55},
    {"name": "小绿", "score": 75}
]

failed_list = process_students(class_roster)

print("\n--- 试炼结果 ---")
print("不及格的学生名单:", failed_list)
print("\n处理后的完整名单:")
print(class_roster)
```

### 运行示例
```text
恭喜 小白, 评级 A
恭喜 小黑, 评级 B
注意: 小红 不及格。
恭喜 小绿, 评级 C

--- 试炼结果 ---
不及格的学生名单: ['小红']

处理后的完整名单:
[{'name': '小白', 'score': 92, 'grade': 'A'}, {'name': '小黑', 'score': 81, 'grade': 'B'}, {'name': '小红', 'score': 55, 'grade': 'F'}, {'name': '小绿', 'score': 75, 'grade': 'C'}]
```

---

## 炼气境小结
- 你已掌握 **变量**、**列表与字典**、**流程控制** 以及 **函数** 的要义。
- 每段代码都附带预期运行结果，务必亲手敲入并运行，感受灵气流动。
- 养成记录报错与解决方案的习惯，未来筑基时可回顾自身修为进境。

稳固修为之后，便可踏入下一境界——【筑基境】。那里你将结识 NumPy、Pandas 与 Matplotlib 等高阶法器，学习批量处理与灵图呈现之术。准备好，下一卷玉简正待开启！
