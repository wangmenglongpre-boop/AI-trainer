# repair_sft_demo 中文拆解

原始 JSONL 供程序逐行读取，每行是一条完整样例。`\n` 表示内容中的换行；英文、转义和元数据挤在一起，不适合直接阅读。本文件是原样例的中文讲解，不是训练输入。

## 每条数据在哪里？

| 原字段 | 含义 | 是否通常送入模型 |
|---|---|---|
| id | 该样例的编号 | 否 |
| source_task_id | 来源题编号，用于防止同题变体跨集合 | 否 |
| split | train/dev/test 数据归属 | 否 |
| messages | 模型接收的对话与正确回答 | 是 |
| role=system 的 content | 固定规则 | 作为提示 |
| role=user 的 content | 需求、错误代码和公开失败摘要 | 作为提示 |
| role=assistant 的 content | 预先准备的正确修复代码 | 作为监督目标 |

这里 assistant 字段不是待训练模型刚生成的回答，而是训练者提供的标准答案。

SFT 的一般形式是“输入 → 正确输出”，不要求输入中一定有错误回答。本任务是代码修复，所以错误代码是题目的一部分。

## 样例一：不要遗漏末尾数字

system：修复 Python 函数，只输出完整代码，不带 Markdown 代码围栏。

user 的需求：给定非负整数 n，返回从 0 到 n 的整数列表，包含 n。

示例：n=2 时应返回 `[0, 1, 2]`。

user 提供的错误代码：

```python
def through(n):
    return list(range(n))
```

错误摘要：`through(2)` 返回 `[0, 1]`，期望 `[0, 1, 2]`。

原因：Python 的 `range(n)` 不包含结束值 n。例如 `range(2)` 只产生 0 和 1。`list(...)` 把这些数变成列表。

assistant 的标准答案：

```python
def through(n):
    return list(range(n + 1))
```

训练关系：给模型看需求和错误代码，让它更可能输出上面的正确代码。并非训练它背诵 `[0, 1, 2]` 这一具体结果。

## 样例二：空列表不能直接取第一项

system：与第一条相同。

user 的需求：返回列表的第一个元素；如果列表为空，返回 None。

user 提供的错误代码：

```python
def first(items):
    return items[0]
```

错误摘要：`first([])` 抛出 IndexError，期望 None。

原因：`items[0]` 表示取第一项，但空列表没有任何一项。

assistant 的标准答案（原文件写法）：

```python
def first(items):
    return items[0] if items else None
```

它等价于下面更容易阅读的写法：

```python
def first(items):
    if items:
        return items[0]
    else:
        return None
```

对列表而言，`if items` 在非空时成立，空列表时不成立。`None` 表示没有值，不是字符串 `"None"`。

## 训练时怎样使用？

```text
模型的提示：
system 固定规则 + user 需求/错误代码/错误摘要

监督目标：
assistant 正确修复代码

完整训练输入：
提示 + 正确答案（teacher forcing）

loss：
只计算正确答案与模板结束标记的预测误差
```

推理时只提供提示，不附上标准答案，让模型生成修复代码。两条示例只是用来理解结构，不能据此评估泛化。
