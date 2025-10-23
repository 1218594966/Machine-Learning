# 筑基境题库详解

筑基阶段着重打磨内功：控制流、数据结构、函数式编程、面向对象、文件与 Git 协作。本手册为每道题提供行动步骤、常见坑位与进阶扩展。

## 阵法初成 · 控制流与推导式

### 题目回顾
1. 🌱 入门：使用 `for` 循环打印 1-9 九九乘法表，要求对齐。
2. 🌿 进阶：利用 `while` 统计 1-100 中被 3 整除但不被 5 整除的数字之和，并打印累计过程。
3. 🔥 突破：分别编写含 `break` 与 `continue` 的循环，通过日志比较差异。
4. 🌟 圆满：将生成偶数平方表的推导式与常规循环互换，比较可读性。
5. 🛡️ 化神：将双层嵌套循环改写为生成器表达式，并对比内存占用。

### 逐题拆解
- **题 1**：外层 `for i in range(1, 10)`，内层 `for j in range(1, i+1)`；`print(f"{i}×{j}={i*j:2}", end="\t")` 控制对齐，内层结束后 `print()` 换行。建议尝试 `str.ljust()` 优化对齐。
- **题 2**：初始化 `total = 0; i = 1`，用 `while i <= 100:` 判断，满足条件时累加并 `print(f"命中 {i}, 当前总和 {total}")`。注意循环变量递增，否则会无限循环。
- **题 3**：编写两个函数 `test_break`、`test_continue`，在每个分支打印 `f"当前 i={i}"`。总结 `break` 直接跳出循环、`continue` 仅跳过本次迭代。
- **题 4**：先写推导式 `[i**2 for i in range(10) if i % 2 == 0]`，再写常规循环 `result = []; for i in range(10): ...`。比较两者可读性，提醒复杂逻辑仍建议使用常规循环。
- **题 5**：先写列表推导式 `[i*j for i in ... for j in ...]`，再改为生成器 `(i*j for ...)`；使用 `sys.getsizeof` 对比内存，强调生成器惰性求值。

## 灵兽驯养 · 列表、元组、集合

### 题目回顾
1. 🌱 入门：列表去重并保持原顺序。
2. 🌿 进阶：用元组表示坐标并解包，扩展到三维。
3. 🔥 突破：计算两位修士法术集合的交集、差集，并生成可读描述。
4. 🌟 圆满：按灵宠品阶与灵力排序嵌套列表。
5. 🛡️ 化神：结合 `enumerate` 与 `zip` 输出排位表。

### 逐题拆解
- **题 1**：使用 `unique = list(dict.fromkeys(elixirs))`，然后比较 `len(elixirs)` 与 `len(unique)`。说明为什么直接 `set(elixirs)` 会打乱顺序。
- **题 2**：声明 `position = (12, 8)`，通过 `x, y = position` 解包。若扩展至三维，使用 `x, y, z = position`。解释不可变对象可用作字典键。
- **题 3**：`shared = skills_a & skills_b`; `exclusive = skills_a ^ skills_b`; 输出 `print(f"共有法术: {', '.join(shared)}")`。强调集合运算返回新集合，不会修改原集合。
- **题 4**：`sorted(pets, key=lambda p: (p['grade'], -p['power']))`；说明 `key` 返回元组时按顺序比较。可尝试 `operator.itemgetter` 简化。
- **题 5**：`for idx, (name, power) in enumerate(zip(names, powers), 1): print(f"{idx:02} | {name:<8} | {power:>4}")`。解释 `zip` 会在最短列表结束。

## 宝库开启 · 字典与映射

### 题目回顾
1. 🌱 入门：初始化库存字典并使用 `get` 更新数量。
2. 🌿 进阶：使用字典推导式从列表生成长度映射，带过滤条件。
3. 🔥 突破：比较 `dict.update`、解包、`|` 合并字典的差异。
4. 🌟 圆满：利用 `collections.Counter` 统计灵石颜色并绘制条形图。
5. 🛡️ 化神：封装 `safe_get` 函数安全访问嵌套字典。

### 逐题拆解
- **题 1**：`inventory['flying-sword'] = inventory.get('flying-sword', 0) + 1`；强调键不存在时返回默认值 0，避免 `KeyError`。
- **题 2**：`name_length = {item: len(item) for item in items if len(item) > 2}`；提醒推导式末尾不加逗号，条件放在末尾。
- **题 3**：展示 `merged = {**base, **override}` 和 `base | override`（3.9+）。解释 `base.update(override)` 会原地修改并返回 `None`。
- **题 4**：`counts = Counter(stones)`；使用 `counts.most_common()` 获取排序结果，再用 Pandas 或 Matplotlib 绘制。说明如何限制前 N 项。
- **题 5**：编写函数 `def safe_get(data, *keys, default=None):` 循环调用 `dict.get`。处理非字典对象时使用 `isinstance` 判断并提前返回默认值。

## 灵术传承 · 函数与参数

### 题目回顾
1. 🌱 入门：编写 `recover(mana, rate=0.1)`，附带 docstring。
2. 🌿 进阶：以关键字参数任意调整顺序调用 `recover`。
3. 🔥 突破：编写 `log_event(*events, **metadata)` 统计参数数量。
4. 🌟 圆满：递归计算族谱深度，加入断言验证输入结构。
5. 🛡️ 化神：利用闭包实现冷却计时器，并提供重置方法。

### 逐题拆解
- **题 1**：`def recover(mana: float, rate: float = 0.1) -> float:` 在 docstring 中写明“rate 表示每次恢复的百分比”。提示不要使用可变对象做默认值。
- **题 2**：演示 `recover(rate=0.2, mana=100)` 与 `recover(mana=80)`。强调关键字调用可避免记错顺序。
- **题 3**：`def log_event(*events, **metadata): print(events, metadata); return len(events) + len(metadata)`。说明 `events` 是元组、`metadata` 是字典。
- **题 4**：`def depth(tree): assert isinstance(tree, dict); if not tree: return 1`；对子节点调用 `max(depth(child) for child in tree.values()) + 1`。提醒设置递归终止条件。
- **题 5**：```
def cooldown_timer(initial: int):
    remaining = initial
    def use(cost: int) -> int:
        nonlocal remaining
        remaining = max(remaining - cost, 0)
        return remaining
    def reset(value: int = initial) -> None:
        nonlocal remaining
        remaining = value
    return use, reset
```说明闭包如何记住状态。

## 气海调息 · 匿名函数与迭代器

### 题目回顾
1. 🌱 入门：使用 `map`+`lambda` 对价格打折并转换为列表。
2. 🌿 进阶：用 `filter` 筛选修为 ≥ 80 的弟子，并统计数量。
3. 🔥 突破：实现可迭代对象 `WeeklyLog`，支持 `for` 循环。
4. 🌟 圆满：`itertools.cycle` 配合 `islice` 输出有限次数。
5. 🛡️ 化神：比较生成器与列表在内存和构建时间上的差异。

### 逐题拆解
- **题 1**：`discounted = list(map(lambda price: price * 0.5, prices))`；解释 `map` 返回迭代器，如需多次遍历需转换为列表。
- **题 2**：`qualified = list(filter(lambda d: d['power'] >= 80, disciples))`; `print(len(qualified))`。可改写为列表推导式对比。
- **题 3**：实现 `__iter__` 返回 `self`，`__next__` 中维护索引；当超出范围时 `raise StopIteration`。提醒每次遍历后重置索引或返回新迭代器。
- **题 4**：`from itertools import cycle, islice`; `for spell in islice(cycle(['fire', 'ice']), 6): ...`。强调必须限制输出次数以免无限循环。
- **题 5**：`sys.getsizeof(list(range(100000)))` 与 `sys.getsizeof((i for i in range(100000)))`；再用 `timeit` 比较初始化耗时，说明惰性计算优势。

## 真经铭刻 · 面向对象

### 题目回顾
1. 🌱 入门：实现 `Cultivator` 类，`level_up` 时打印升级记录。
2. 🌿 进阶：设计 `Sword` → `FlyingSword` 继承，覆写 `attack`。
3. 🔥 突破：用 `@property` 控制灵力范围，非法值抛异常。
4. 🌟 圆满：添加类方法 `novice`、`from_dict` 创建角色。
5. 🛡️ 化神：使用 `@dataclass` 生成不可变修士对象。

### 逐题拆解
- **题 1**：`class Cultivator: def __init__(self, name, realm=1): ...` 在 `level_up` 中保存旧值并打印 `print(f"{self.name} 升至 {self.realm}")`。
- **题 2**：父类 `Sword` 定义 `attack`，子类调用 `super().attack()` 并附加额外描述。强调 `super().__init__` 初始化父类属性。
- **题 3**：在 setter 中验证 `0 <= value <= 100`，越界 `raise ValueError("灵力必须在 0-100")`。可使用 `property` 组合 getter、setter。
- **题 4**：`@classmethod def novice(cls): return cls("新弟子", realm=1)`；`@classmethod def from_dict(cls, data): return cls(**data)`，展示多种构造方式。
- **题 5**：`@dataclass(frozen=True)` 自动生成比较与哈希方法。解释 `frozen` 设为 True 后属性不可修改，需要使用 `replace` 或新实例。

## 火候把握 · 文件与上下文

### 题目回顾
1. 🌱 入门：使用 `with open` 写入日志并确认文件存在。
2. 🌿 进阶：用 `csv.DictReader` 读取 CSV，打印前两行。
3. 🔥 突破：利用 `pathlib` 创建目录并遍历指定扩展名。
4. 🌟 圆满：`shutil.make_archive` 打包 `logs` 目录。
5. 🛡️ 化神：用 `tempfile.NamedTemporaryFile` 写入、读取并清理临时文件。

### 逐题拆解
- **题 1**：`with open('elixir.log', 'w', encoding='utf-8') as f: f.write('...\n')`；结束后 `Path('elixir.log').exists()` 应返回 True。
- **题 2**：`with open('elixir.csv', newline='', encoding='utf-8') as f:`；`reader = csv.DictReader(f)`；使用 `itertools.islice(reader, 2)` 打印两行。
- **题 3**：`logs = Path('logs'); logs.mkdir(parents=True, exist_ok=True); for path in logs.glob('*.txt'):` 输出文件名。强调 `path.name` 与 `path.stem`。
- **题 4**：`shutil.make_archive('logs-backup', 'zip', root_dir='logs')`，解释生成的压缩文件路径。提醒 Windows 若路径含空格需加引号。
- **题 5**：```
with tempfile.NamedTemporaryFile(mode='w+', delete=False, encoding='utf-8') as tmp:
    tmp.write('test')
    tmp.seek(0)
    print(tmp.read())
Path(tmp.name).unlink()
```说明为何需要手动删除。

## 灵符调和 · Git 分支与冲突

### 题目回顾
1. 🌱 入门：新建并切换到 `feature/spirit-log` 分支，列出所有分支。
2. 🌿 进阶：在分支提交一次后合并到主线，书写规范提交信息。
3. 🔥 突破：在两个分支修改同一行，刻意制造冲突并截图记录。
4. 🌟 圆满：通过 `git status`、`git diff` 定位冲突块，理解标记含义。
5. 🛡️ 化神：解决冲突、继续合并，并撰写复盘总结。

### 逐题拆解
- **题 1**：`git switch -c feature/spirit-log` 后 `git branch` 查看所有分支；当前分支前有 `*` 标记。若需删除分支用 `git branch -d`。
- **题 2**：在分支 `git add`、`git commit -m "feat: add spirit log"`，切回主线 `git switch main` 后 `git merge feature/spirit-log`。解释快进合并与普通合并区别。
- **题 3**：分别在 `main` 与新分支编辑同一文件行，合并时观察冲突标记；建议保存 diff 截图以便复盘。
- **题 4**：`git status` 会显示 `both modified` 文件；`git diff --merge` 或 `git diff --ours --theirs` 查看冲突两侧差异。解释 `<<<<<<< HEAD` 等标记代表当前分支与目标分支。
- **题 5**：手动编辑冲突块后 `git add` 解决文件，再运行项目测试确保无回归，最后 `git merge --continue`。复盘中记录冲突原因、解决办法、避免措施。

## 天罚护身 · 异常处理与调试

### 题目回顾
1. 🌱 入门：捕获 `ValueError` 并提示重新输入。
2. 🌿 进阶：自定义 `InsufficientManaError` 以处理灵力不足场景。
3. 🔥 突破：示范 `try/except/else/finally` 结构，处理文件读写。
4. 🌟 圆满：使用 `pdb` 单步调试循环，掌握常见命令。
5. 🛡️ 化神：使用 `logging.exception` 记录带堆栈的错误日志。

### 逐题拆解
- **题 1**：`while True:` → `try: value = int(input())` → `except ValueError: print("请输入数字"); continue`。成功解析后 `break`。
- **题 2**：定义异常类 `class InsufficientManaError(Exception): ...`；在函数中 `if mana < cost: raise InsufficientManaError(...)`，调用方捕获并打印剩余灵力。
- **题 3**：```
try:
    with open('mana.log') as f:
        data = f.read()
except FileNotFoundError:
    print('缺少日志')
else:
    process(data)
finally:
    print('无论是否成功都会执行')
```强调 `finally` 总会运行。
- **题 4**：`import pdb; pdb.set_trace()`，常用命令 `n`（next）、`s`（step）、`c`（continue）、`l`（list）、`p`（print）。
- **题 5**：`logging.basicConfig(filename='app.log', level=logging.ERROR)`；在异常处 `logging.exception("mana calculation failed")`，日志中包含堆栈信息，便于复盘。
