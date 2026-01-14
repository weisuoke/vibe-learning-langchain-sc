# Chain 链式调用

> 原子化知识点 | LangChain 使用 | LangChain 源码学习核心知识

---

## 1. 【30字核心】

**Chain 是 LangChain 中将多个组件串联的核心机制，通过 LCEL 的 pipe(|) 操作符实现数据的流水线处理。**

---

## 2. 【第一性原理】

### 什么是第一性原理？

**第一性原理**：回到事物最基本的真理，从源头思考问题

### Chain 链式调用的第一性原理 🎯

#### 1. 最基础的定义

**Chain = 多个处理步骤的有序组合**

仅此而已！没有更基础的了。

```python
# Chain 的本质就是：
# 输入 → 步骤1 → 步骤2 → 步骤3 → 输出

# 每一步的输出成为下一步的输入
def chain(input):
    result = step1(input)
    result = step2(result)
    result = step3(result)
    return result
```

#### 2. 为什么需要 Chain？

**核心问题：LLM 应用通常不是单一调用，而是多个步骤的组合**

```python
# 实际场景：一个简单的问答系统
# 步骤1：格式化用户问题（Prompt）
# 步骤2：调用 LLM 获取回答
# 步骤3：解析输出格式（Parser）

# 如果没有 Chain，代码会变成：
prompt_result = format_prompt(user_question)
llm_result = call_llm(prompt_result)
final_result = parse_output(llm_result)

# 有了 Chain，代码变成：
chain = prompt | llm | parser
result = chain.invoke(user_question)
```

#### 3. Chain 的三层价值

##### 价值1：组合性 - 像乐高积木一样组装

```python
# 任何 Runnable 都可以组合
chain1 = prompt1 | llm | parser1
chain2 = prompt2 | llm | parser2

# 组合成更大的 Chain
big_chain = chain1 | transform | chain2
```

##### 价值2：统一接口 - invoke/stream/batch 通用

```python
# 同一个 Chain，不同调用方式
result = chain.invoke(input)              # 单次调用
async_result = await chain.ainvoke(input) # 异步调用
stream = chain.stream(input)              # 流式输出
results = chain.batch([input1, input2])   # 批量调用
```

##### 价值3：配置传递 - 参数自动流转

```python
# config 会自动传递给 Chain 中的每个组件
result = chain.invoke(
    input,
    config={
        "callbacks": [my_callback],
        "tags": ["production"],
        "metadata": {"user_id": "123"}
    }
)
```

#### 4. 从第一性原理推导 LCEL

**推理链：**

```
1. LLM 应用需要多步骤处理
   ↓
2. 步骤之间需要数据传递
   ↓
3. 需要统一的组合方式
   ↓
4. Python 的 | 操作符可以重载
   ↓
5. 实现 __or__ 方法实现 pipe 语法
   ↓
6. 这就是 LCEL (LangChain Expression Language)
   ↓
7. chain = component1 | component2 | component3
```

#### 5. 一句话总结第一性原理

**Chain 是将多个处理步骤组合成流水线的机制，通过 LCEL 的 pipe 操作符实现优雅的声明式组合。**

---

## 3. 【核心概念（全面覆盖）】

### 核心概念1：RunnableSequence 序列执行 🔗

**RunnableSequence 是多个 Runnable 顺序执行的容器，前一个的输出是后一个的输入**

```python
from langchain_core.runnables import RunnableSequence, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 方式1：使用 pipe 操作符（推荐）
prompt = ChatPromptTemplate.from_template("翻译成英文：{text}")
chain = prompt | llm | StrOutputParser()

# 方式2：显式创建 RunnableSequence
chain = RunnableSequence(
    first=prompt,
    middle=[llm],
    last=StrOutputParser()
)

# 执行
result = chain.invoke({"text": "你好世界"})
print(result)  # "Hello World"
```

**数据流转示意：**

```
输入: {"text": "你好世界"}
    ↓
[ChatPromptTemplate] → ChatPromptValue
    ↓
[ChatOpenAI] → AIMessage(content="Hello World")
    ↓
[StrOutputParser] → "Hello World"
    ↓
输出: "Hello World"
```

**在 LangChain 源码中的实现：**

```python
# langchain_core/runnables/base.py
class RunnableSequence(RunnableSerializable):
    """顺序执行多个 Runnable"""

    first: Runnable  # 第一个
    middle: List[Runnable] = []  # 中间的
    last: Runnable  # 最后一个

    def invoke(self, input, config=None):
        # 依次执行每个步骤
        for step in self.steps:
            input = step.invoke(input, config)
        return input

    @property
    def steps(self):
        return [self.first] + self.middle + [self.last]
```

---

### 核心概念2：Pipe 操作符 | 📐

**pipe 操作符通过 Python 的 `__or__` 魔法方法实现，让 Chain 组合变得优雅**

```python
from langchain_core.runnables import Runnable, RunnableLambda

# | 操作符的本质
# a | b 等价于 a.__or__(b) 或 b.__ror__(a)

# 示例：自定义 Runnable
def add_one(x: int) -> int:
    return x + 1

def multiply_two(x: int) -> int:
    return x * 2

# 使用 RunnableLambda 包装函数
step1 = RunnableLambda(add_one)
step2 = RunnableLambda(multiply_two)

# 使用 pipe 组合
chain = step1 | step2

# 执行：(5 + 1) * 2 = 12
result = chain.invoke(5)
print(result)  # 12
```

**pipe 操作符支持的组合类型：**

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# 1. Runnable | Runnable
chain = prompt | llm

# 2. Runnable | dict (自动转换为 RunnableParallel)
chain = retriever | {"context": lambda x: x, "question": RunnablePassthrough()}

# 3. dict | Runnable
chain = {"a": step1, "b": step2} | combine_step

# 4. 函数也可以（自动包装为 RunnableLambda）
chain = prompt | llm | (lambda x: x.content.upper())
```

**在 LangChain 源码中的实现：**

```python
# langchain_core/runnables/base.py
class Runnable(ABC):
    def __or__(self, other):
        """实现 | 操作符"""
        return RunnableSequence(first=self, last=coerce_to_runnable(other))

    def __ror__(self, other):
        """实现反向 | 操作符"""
        return RunnableSequence(first=coerce_to_runnable(other), last=self)

def coerce_to_runnable(thing):
    """将各种类型转换为 Runnable"""
    if isinstance(thing, Runnable):
        return thing
    elif callable(thing):
        return RunnableLambda(thing)
    elif isinstance(thing, dict):
        return RunnableParallel(thing)
    else:
        raise TypeError(f"Cannot coerce {type(thing)} to Runnable")
```

---

### 核心概念3：RunnableParallel 并行执行 🔀

**RunnableParallel 让多个 Runnable 同时执行，结果合并为字典**

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# 场景：同时执行翻译和摘要
translate_chain = translate_prompt | llm | StrOutputParser()
summary_chain = summary_prompt | llm | StrOutputParser()

# 并行执行
parallel = RunnableParallel({
    "translation": translate_chain,
    "summary": summary_chain
})

# 执行
result = parallel.invoke({"text": "这是一段中文"})
# result = {
#     "translation": "This is a Chinese text",
#     "summary": "A short Chinese passage"
# }
```

**RAG 中的典型用法：**

```python
from langchain_core.runnables import RunnablePassthrough

# RAG Chain：检索和问题并行传递
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 执行
# 1. retriever 检索相关文档
# 2. RunnablePassthrough 直接传递原始问题
# 3. 两者合并后传给 prompt
result = rag_chain.invoke("什么是 LangChain？")
```

**数据流转示意：**

```
输入: "什么是 LangChain？"
         ↓
    ┌────┴────┐
    ↓         ↓
[retriever] [passthrough]
    ↓         ↓
 [docs]    "什么是..."
    └────┬────┘
         ↓
{"context": docs, "question": "什么是..."}
         ↓
    [prompt | llm | parser]
         ↓
    输出: "答案..."
```

---

### 扩展概念4：RunnableBranch 条件分支 🔀

**RunnableBranch 根据条件选择不同的执行路径**

```python
from langchain_core.runnables import RunnableBranch

# 定义条件分支
branch = RunnableBranch(
    # (条件, 执行的 Runnable)
    (lambda x: "翻译" in x["task"], translate_chain),
    (lambda x: "摘要" in x["task"], summary_chain),
    # 默认分支
    default_chain
)

# 根据输入选择分支
result = branch.invoke({"task": "翻译这段话", "text": "..."})
```

---

### 扩展概念5：RunnablePassthrough 透传 ➡️

**RunnablePassthrough 直接传递输入，常用于并行时保留原始数据**

```python
from langchain_core.runnables import RunnablePassthrough

# 基础用法：直接传递
passthrough = RunnablePassthrough()
result = passthrough.invoke("hello")  # "hello"

# 带赋值的透传
chain = RunnablePassthrough.assign(
    enhanced=lambda x: x["text"].upper()
)
result = chain.invoke({"text": "hello"})
# {"text": "hello", "enhanced": "HELLO"}
```

---

### 扩展概念6：RunnableLambda 函数包装 λ

**RunnableLambda 将普通函数包装为 Runnable**

```python
from langchain_core.runnables import RunnableLambda

# 包装同步函数
def process(text: str) -> str:
    return text.strip().lower()

runnable = RunnableLambda(process)
result = runnable.invoke("  HELLO  ")  # "hello"

# 包装异步函数
async def async_process(text: str) -> str:
    await asyncio.sleep(0.1)
    return text.upper()

async_runnable = RunnableLambda(async_process)
result = await async_runnable.ainvoke("hello")  # "HELLO"
```

---

## 4. 【最小可用】

掌握以下内容，就能在 LangChain 中构建 Chain：

### 4.1 基础 Chain 组合

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# 三步走：Prompt → LLM → Parser
prompt = ChatPromptTemplate.from_template("回答问题：{question}")
llm = ChatOpenAI()
parser = StrOutputParser()

# 使用 | 组合
chain = prompt | llm | parser

# 调用
result = chain.invoke({"question": "什么是 Python？"})
```

### 4.2 并行执行

```python
from langchain_core.runnables import RunnableParallel

# 多个任务并行
parallel = RunnableParallel({
    "answer": qa_chain,
    "sources": retriever
})

result = parallel.invoke({"question": "..."})
# result["answer"] 和 result["sources"] 同时计算
```

### 4.3 RAG Chain 模板

```python
from langchain_core.runnables import RunnablePassthrough

# 标准 RAG Chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

result = rag_chain.invoke("你的问题")
```

### 4.4 流式输出

```python
# 使用 stream 方法
for chunk in chain.stream({"question": "..."}):
    print(chunk, end="", flush=True)
```

**这些知识足以：**
- 构建 Prompt → LLM → Parser 的基础 Chain
- 实现并行执行提高效率
- 构建 RAG 应用
- 实现流式输出提升用户体验

---

## 5. 【1个类比】（双轨制）

### 类比1：Chain 是流水线

#### 🎨 前端视角：Redux Middleware / RxJS pipe

Chain 就像 Redux 中间件或 RxJS 的 pipe，数据依次经过每个处理环节。

```javascript
// Redux Middleware 链
const store = createStore(
  reducer,
  applyMiddleware(logger, thunk, api)  // 中间件链
);

// RxJS pipe
observable.pipe(
  map(x => x * 2),
  filter(x => x > 10),
  take(5)
);
```

```python
# LangChain Chain
chain = prompt | llm | parser

# 数据流：input → prompt → llm → parser → output
```

**关键相似点：**
- 都是数据的单向流动
- 每个环节只关心自己的处理
- 通过组合实现复杂功能

#### 🧒 小朋友视角：工厂流水线

Chain 就像工厂的流水线：

```
原材料 → [切割机] → [打磨机] → [上色机] → 成品

一个零件依次经过每台机器：
1. 切割机把原材料切成形状
2. 打磨机把边缘磨光滑
3. 上色机涂上漂亮的颜色
4. 最后变成成品！

每台机器只做一件事，但组合起来就能做出复杂的产品。
```

**生活例子：**
```
做三明治：
面包 → [切开] → [涂酱] → [放肉] → [放菜] → [合上] → 三明治

每一步都很简单，但按顺序组合就能做出美味的三明治！
```

---

### 类比2：pipe 操作符是传送带

#### 🎨 前端视角：Promise.then 链 / Unix 管道

pipe 操作符就像 Promise.then 或 Unix 管道，连接多个处理步骤。

```javascript
// Promise.then 链
fetch(url)
  .then(response => response.json())
  .then(data => process(data))
  .then(result => display(result));

// Unix 管道
// cat file.txt | grep "error" | wc -l
```

```python
# LangChain pipe
chain = prompt | llm | parser
# 等价于 Unix: prompt | llm | parser
```

#### 🧒 小朋友视角：传送带

pipe 就像超市的传送带：

```
你把商品放上传送带：
苹果 → 【扫码】 → 【称重】 → 【装袋】 → 拿走

商品自动从一个站点传到下一个，
每个站点做自己的事情，
最后你拿到处理好的商品。
```

---

### 类比3：RunnableParallel 是并行车道

#### 🎨 前端视角：Promise.all

RunnableParallel 就像 Promise.all，同时执行多个任务。

```javascript
// Promise.all 并行执行
const [users, posts, comments] = await Promise.all([
  fetchUsers(),
  fetchPosts(),
  fetchComments()
]);
```

```python
# LangChain RunnableParallel
parallel = RunnableParallel({
    "translation": translate_chain,
    "summary": summary_chain
})
```

#### 🧒 小朋友视角：多人同时做事

RunnableParallel 就像分工合作：

```
老师说：把这篇文章翻译成英文，然后写一个摘要

❌ 一个人做：先翻译（10分钟）→ 再写摘要（10分钟）= 20分钟

✅ 两个人做：
   小明翻译（10分钟）──┐
                      ├→ 合并结果 = 10分钟
   小红写摘要（10分钟）─┘

两个任务同时进行，时间减半！
```

---

### 类比总结表

| LangChain 概念 | 前端类比 | 小朋友类比 |
|---------------|---------|-----------|
| Chain | Redux middleware / RxJS pipe | 工厂流水线 |
| pipe 操作符 | Promise.then / Unix 管道 | 传送带 |
| RunnableSequence | 同步函数调用链 | 接力赛跑 |
| RunnableParallel | Promise.all | 多人同时做事 |
| RunnablePassthrough | 恒等函数 / identity | 直接传递不改变 |
| RunnableBranch | if-else / switch | 走不同的路 |
| RunnableLambda | 高阶函数 | 把普通事情变成标准流程 |

---

## 6. 【反直觉点】

### 误区1：Chain 只能顺序执行 ❌

**为什么错？**
- RunnableParallel 支持并行执行
- RunnableBranch 支持条件分支
- 可以组合出复杂的执行拓扑

**为什么人们容易这样错？**
"Chain"（链）这个词暗示线性结构，但实际上 LCEL 支持更复杂的组合。

**正确理解：**

```python
# 不只是顺序
# 1. 并行执行
parallel = RunnableParallel({
    "a": chain_a,
    "b": chain_b
})

# 2. 条件分支
branch = RunnableBranch(
    (condition1, chain1),
    (condition2, chain2),
    default_chain
)

# 3. 复杂拓扑
complex_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | RunnableParallel({
        "answer": qa_chain,
        "summary": summary_chain
    })
)
```

---

### 误区2：pipe 操作符只是语法糖 ❌

**为什么错？**
- pipe 不仅是语法糖，还实现了：
  - 自动类型转换（dict → RunnableParallel，function → RunnableLambda）
  - 配置传递（callbacks, tags, metadata）
  - 流式支持
  - 批量执行优化

**为什么人们容易这样错？**
因为 `a | b` 看起来和 `RunnableSequence(a, b)` 效果一样。

**正确理解：**

```python
# pipe 的隐藏能力

# 1. 自动类型转换
chain = prompt | llm | (lambda x: x.content)
# lambda 自动转换为 RunnableLambda

chain = {"a": step1} | step2
# dict 自动转换为 RunnableParallel

# 2. 配置自动传递
result = chain.invoke(
    input,
    config={"callbacks": [handler]}  # 传递给所有步骤
)

# 3. 流式自动串联
for chunk in chain.stream(input):
    # 每个步骤的流式输出自动连接
    print(chunk)
```

---

### 误区3：Chain 和 Agent 是一样的 ❌

**为什么错？**
- Chain：确定性执行，流程固定
- Agent：动态决策，LLM 决定下一步

**为什么人们容易这样错？**
两者都是"多步骤执行"，但执行逻辑完全不同。

**正确理解：**

```python
# Chain：流程固定，每次执行路径相同
chain = prompt | llm | parser
# 永远是：prompt → llm → parser

# Agent：动态决策，每次可能不同
# 循环：LLM决定 → 执行工具 → 观察结果 → LLM决定 → ...
# 可能调用1个工具，可能调用10个，由 LLM 决定

# 选择标准
# 固定流程 → Chain（可预测、易调试）
# 需要决策 → Agent（灵活、不可预测）
```

| 特性 | Chain | Agent |
|-----|-------|-------|
| 执行路径 | 固定 | 动态 |
| 可预测性 | 高 | 低 |
| 调试难度 | 低 | 高 |
| 适用场景 | 固定流程 | 需要决策 |
| 成本控制 | 易 | 难 |

---

## 7. 【实战代码】

```python
"""
示例：Chain 链式调用完整演示
展示 LangChain 中 Chain 的核心用法
"""

from typing import Dict, List, Any
from dataclasses import dataclass

# ===== 1. 模拟 LangChain 核心组件 =====
print("=== 1. 模拟 Runnable 基类 ===")

class Runnable:
    """Runnable 基类"""

    def invoke(self, input: Any) -> Any:
        raise NotImplementedError

    def __or__(self, other: "Runnable") -> "RunnableSequence":
        """实现 | 操作符"""
        return RunnableSequence([self, other])

    def __ror__(self, other) -> "RunnableSequence":
        """实现反向 | 操作符"""
        if isinstance(other, dict):
            return RunnableSequence([RunnableParallel(other), self])
        return RunnableSequence([other, self])


class RunnableSequence(Runnable):
    """顺序执行"""

    def __init__(self, steps: List[Runnable]):
        self.steps = steps

    def invoke(self, input: Any) -> Any:
        result = input
        for step in self.steps:
            result = step.invoke(result)
        return result

    def __or__(self, other: Runnable) -> "RunnableSequence":
        return RunnableSequence(self.steps + [other])


class RunnableLambda(Runnable):
    """包装普通函数"""

    def __init__(self, func):
        self.func = func

    def invoke(self, input: Any) -> Any:
        return self.func(input)


class RunnableParallel(Runnable):
    """并行执行"""

    def __init__(self, branches: Dict[str, Runnable]):
        self.branches = branches

    def invoke(self, input: Any) -> Dict[str, Any]:
        return {
            key: branch.invoke(input)
            for key, branch in self.branches.items()
        }


class RunnablePassthrough(Runnable):
    """直接传递"""

    def invoke(self, input: Any) -> Any:
        return input


# ===== 2. 基础 Chain 组合 =====
print("\n=== 2. 基础 Chain 组合 ===")

# 模拟 Prompt
class PromptTemplate(Runnable):
    def __init__(self, template: str):
        self.template = template

    def invoke(self, input: Dict[str, Any]) -> str:
        return self.template.format(**input)

# 模拟 LLM
class MockLLM(Runnable):
    def invoke(self, input: str) -> str:
        # 简单模拟 LLM 响应
        if "翻译" in input:
            return "Translation: Hello World"
        return f"LLM Response to: {input[:50]}..."

# 模拟 Parser
class StrOutputParser(Runnable):
    def invoke(self, input: str) -> str:
        return input.strip()

# 创建组件
prompt = PromptTemplate("请翻译以下内容：{text}")
llm = MockLLM()
parser = StrOutputParser()

# 使用 pipe 组合
chain = prompt | llm | parser

# 执行
result = chain.invoke({"text": "你好世界"})
print(f"基础 Chain 结果: {result}")

# ===== 3. 并行执行 =====
print("\n=== 3. 并行执行 ===")

# 创建两个处理分支
translate_chain = RunnableLambda(lambda x: f"翻译: {x}")
summary_chain = RunnableLambda(lambda x: f"摘要: {x[:20]}...")

# 并行执行
parallel = RunnableParallel({
    "translation": translate_chain,
    "summary": summary_chain
})

result = parallel.invoke("这是一段需要处理的中文文本内容")
print(f"并行执行结果: {result}")

# ===== 4. RAG Chain 模式 =====
print("\n=== 4. RAG Chain 模式 ===")

# 模拟 Retriever
class MockRetriever(Runnable):
    def __init__(self, docs: List[str]):
        self.docs = docs

    def invoke(self, query: str) -> str:
        # 简单返回所有文档
        return "\n".join(self.docs)

# 创建组件
retriever = MockRetriever([
    "LangChain 是一个 LLM 应用框架",
    "它提供了 Chain、Agent、Memory 等组件"
])

rag_prompt = PromptTemplate(
    "基于以下上下文回答问题:\n{context}\n\n问题: {question}"
)

# 构建 RAG Chain
# 注意：这里简化了实现，实际 LangChain 中 dict 会自动转换
context_and_question = RunnableParallel({
    "context": retriever,
    "question": RunnablePassthrough()
})

rag_chain = context_and_question | rag_prompt | llm | parser

result = rag_chain.invoke("什么是 LangChain？")
print(f"RAG Chain 结果: {result}")

# ===== 5. 复杂 Chain 组合 =====
print("\n=== 5. 复杂 Chain 组合 ===")

# 定义多个处理步骤
step1 = RunnableLambda(lambda x: x.strip())
step2 = RunnableLambda(lambda x: x.lower())
step3 = RunnableLambda(lambda x: x.replace(" ", "_"))
step4 = RunnableLambda(lambda x: f"processed_{x}")

# 组合成复杂 Chain
processing_chain = step1 | step2 | step3 | step4

result = processing_chain.invoke("  Hello World  ")
print(f"复杂处理结果: {result}")

# ===== 6. 数据流追踪 =====
print("\n=== 6. 数据流追踪 ===")

class TracingRunnable(Runnable):
    """带追踪的 Runnable"""

    def __init__(self, name: str, func):
        self.name = name
        self.func = func

    def invoke(self, input: Any) -> Any:
        print(f"  [{self.name}] 输入: {input}")
        result = self.func(input)
        print(f"  [{self.name}] 输出: {result}")
        return result

# 创建带追踪的 Chain
traced_chain = (
    TracingRunnable("Step1", lambda x: x + 10) |
    TracingRunnable("Step2", lambda x: x * 2) |
    TracingRunnable("Step3", lambda x: x - 5)
)

print("执行追踪:")
result = traced_chain.invoke(5)
print(f"最终结果: {result}")  # (5 + 10) * 2 - 5 = 25

# ===== 7. 条件分支模拟 =====
print("\n=== 7. 条件分支 ===")

class RunnableBranch(Runnable):
    """条件分支"""

    def __init__(self, branches, default):
        self.branches = branches  # [(condition, runnable), ...]
        self.default = default

    def invoke(self, input: Any) -> Any:
        for condition, runnable in self.branches:
            if condition(input):
                return runnable.invoke(input)
        return self.default.invoke(input)

# 创建分支
branch = RunnableBranch(
    branches=[
        (lambda x: x.get("type") == "translate",
         RunnableLambda(lambda x: f"翻译: {x['text']}")),
        (lambda x: x.get("type") == "summary",
         RunnableLambda(lambda x: f"摘要: {x['text'][:10]}...")),
    ],
    default=RunnableLambda(lambda x: f"默认处理: {x['text']}")
)

# 测试不同分支
print(branch.invoke({"type": "translate", "text": "Hello"}))
print(branch.invoke({"type": "summary", "text": "This is a long text"}))
print(branch.invoke({"type": "other", "text": "Unknown type"}))

# ===== 8. 批量执行模拟 =====
print("\n=== 8. 批量执行 ===")

class BatchableRunnable(Runnable):
    """支持批量执行的 Runnable"""

    def __init__(self, func):
        self.func = func

    def invoke(self, input: Any) -> Any:
        return self.func(input)

    def batch(self, inputs: List[Any]) -> List[Any]:
        return [self.invoke(input) for input in inputs]

# 创建 Chain
batch_chain = BatchableRunnable(lambda x: x * 2)

# 批量执行
inputs = [1, 2, 3, 4, 5]
results = batch_chain.batch(inputs)
print(f"批量执行结果: {results}")

# ===== 9. 实际 LangChain 风格的完整示例 =====
print("\n=== 9. 完整 RAG 示例 ===")

# 模拟完整的 RAG 场景
documents = [
    "Python 是一种解释型编程语言",
    "LangChain 使用 Python 开发",
    "LCEL 是 LangChain Expression Language"
]

class SimpleRAG:
    """简单的 RAG 实现"""

    def __init__(self, docs: List[str]):
        self.docs = docs

    def retrieve(self, query: str) -> List[str]:
        """简单的关键词匹配检索"""
        results = []
        query_words = query.lower().split()
        for doc in self.docs:
            if any(word in doc.lower() for word in query_words):
                results.append(doc)
        return results if results else self.docs[:1]

    def generate(self, context: str, question: str) -> str:
        """生成回答（模拟）"""
        return f"根据上下文 '{context[:30]}...'，回答问题 '{question}' 的答案是：这是一个模拟回答。"

    def query(self, question: str) -> str:
        """完整的 RAG 流程"""
        # 1. 检索
        relevant_docs = self.retrieve(question)
        context = "\n".join(relevant_docs)

        # 2. 生成
        answer = self.generate(context, question)

        return answer

rag = SimpleRAG(documents)
answer = rag.query("什么是 LangChain？")
print(f"RAG 回答: {answer}")

print("\n=== 完成！===")
```

**运行输出示例：**
```
=== 1. 模拟 Runnable 基类 ===

=== 2. 基础 Chain 组合 ===
基础 Chain 结果: Translation: Hello World

=== 3. 并行执行 ===
并行执行结果: {'translation': '翻译: 这是一段需要处理的中文文本内容', 'summary': '摘要: 这是一段需要处理的中文文本内容...'}

=== 4. RAG Chain 模式 ===
RAG Chain 结果: LLM Response to: 基于以下上下文回答问题:
LangChain 是一个 LLM 应用框...

=== 5. 复杂 Chain 组合 ===
复杂处理结果: processed_hello_world

=== 6. 数据流追踪 ===
执行追踪:
  [Step1] 输入: 5
  [Step1] 输出: 15
  [Step2] 输入: 15
  [Step2] 输出: 30
  [Step3] 输入: 30
  [Step3] 输出: 25
最终结果: 25

=== 7. 条件分支 ===
翻译: Hello
摘要: This is a ...
默认处理: Unknown type

=== 8. 批量执行 ===
批量执行结果: [2, 4, 6, 8, 10]

=== 9. 完整 RAG 示例 ===
RAG 回答: 根据上下文 'LangChain 使用 Python 开发...'，回答问题 '什么是 LangChain？' 的答案是：这是一个模拟回答。

=== 完成！===
```

---

## 8. 【面试必问】

### 问题1："什么是 LCEL？它和传统的 Chain 有什么区别？"

**普通回答（❌ 不出彩）：**
"LCEL 是 LangChain Expression Language，用 | 操作符连接组件。"

**出彩回答（✅ 推荐）：**

> **LCEL (LangChain Expression Language) 是 LangChain 0.1+ 版本引入的声明式 Chain 构建方式：**
>
> **1. 语法层面**
> ```python
> # 传统 Chain
> chain = LLMChain(llm=llm, prompt=prompt, output_parser=parser)
>
> # LCEL
> chain = prompt | llm | parser
> ```
>
> **2. 核心优势**
> - **统一接口**：所有组件实现 Runnable 接口，支持 invoke/stream/batch
> - **类型安全**：自动推断输入输出类型
> - **配置传递**：callbacks、tags 等自动流转
> - **流式原生**：天然支持流式输出
>
> **3. 架构差异**
> - 传统 Chain：继承式，每种 Chain 是独立类
> - LCEL：组合式，通过 pipe 组合 Runnable
>
> **4. 实际体验**
> 在我的项目中，用 LCEL 重构后代码量减少 40%，调试更直观，因为每个步骤都是独立的 Runnable。

**为什么这个回答出彩？**
1. ✅ 展示了新旧两种写法的对比
2. ✅ 总结了核心优势
3. ✅ 解释了架构层面的差异
4. ✅ 有实际项目经验

---

### 问题2："如何选择 Chain 和 Agent？"

**普通回答（❌ 不出彩）：**
"简单任务用 Chain，复杂任务用 Agent。"

**出彩回答（✅ 推荐）：**

> **选择依据是「流程是否确定」：**
>
> **使用 Chain 的场景：**
> - 流程固定：翻译、摘要、格式转换
> - 需要可预测性：生产环境、成本敏感
> - 易于调试：每个步骤输入输出明确
>
> ```python
> # Chain：固定流程
> chain = prompt | llm | parser  # 永远这三步
> ```
>
> **使用 Agent 的场景：**
> - 需要动态决策：智能助手、研究任务
> - 工具选择不确定：可能用搜索，可能用计算器
> - 任务复杂度未知：可能一步完成，可能十步
>
> ```python
> # Agent：动态决策
> # LLM 决定：要不要用工具？用哪个？什么时候结束？
> ```
>
> **我的经验法则：**
> 1. 如果你能画出确定的流程图 → Chain
> 2. 如果流程图有"根据情况决定" → Agent
> 3. 不确定时先用 Chain，遇到瓶颈再考虑 Agent
>
> **成本考虑：**
> Agent 的 LLM 调用次数不确定，生产环境要设置 max_iterations 和超时。

---

## 9. 【化骨绵掌】

### 卡片1：Chain 是什么？ 🎯

**一句话：** Chain 是多个处理步骤的有序组合，输入依次流经每个步骤。

**举例：**
```python
chain = prompt | llm | parser
# 输入 → prompt → llm → parser → 输出
```

**应用：** LangChain 中构建 LLM 应用的核心模式。

---

### 卡片2：LCEL 是什么？ 📝

**一句话：** LCEL (LangChain Expression Language) 是用 `|` 操作符声明式构建 Chain 的语法。

**举例：**
```python
# 用 | 连接组件
chain = component1 | component2 | component3
```

**应用：** LangChain 0.1+ 版本的标准写法。

---

### 卡片3：pipe 操作符 | 📐

**一句话：** `|` 通过 Python 的 `__or__` 魔法方法实现，创建 RunnableSequence。

**举例：**
```python
a | b  # 等价于 RunnableSequence([a, b])
```

**应用：** 让 Chain 组合像 Unix 管道一样直观。

---

### 卡片4：RunnableSequence 顺序执行 🔗

**一句话：** 多个 Runnable 按顺序执行，前一个的输出是后一个的输入。

**举例：**
```python
chain = step1 | step2 | step3
result = chain.invoke(input)
# step1(input) → step2 → step3 → result
```

**应用：** Prompt → LLM → Parser 的典型模式。

---

### 卡片5：RunnableParallel 并行执行 🔀

**一句话：** 多个 Runnable 同时执行，结果合并为字典。

**举例：**
```python
parallel = RunnableParallel({
    "a": chain_a,
    "b": chain_b
})
# result = {"a": ..., "b": ...}
```

**应用：** 同时执行翻译和摘要，提高效率。

---

### 卡片6：RunnablePassthrough 透传 ➡️

**一句话：** 直接传递输入不做修改，常用于并行时保留原始数据。

**举例：**
```python
chain = {"context": retriever, "question": RunnablePassthrough()}
# 问题原样传递，同时检索上下文
```

**应用：** RAG Chain 中保留原始问题。

---

### 卡片7：RunnableLambda 函数包装 λ

**一句话：** 将普通 Python 函数包装为 Runnable。

**举例：**
```python
step = RunnableLambda(lambda x: x.upper())
# 或直接在 Chain 中使用
chain = prompt | llm | (lambda x: x.content)
```

**应用：** 在 Chain 中插入自定义处理逻辑。

---

### 卡片8：RAG Chain 模板 📚

**一句话：** 检索增强生成的标准 Chain 模式：检索 + 问题 → 生成。

**举例：**
```python
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

**应用：** 知识库问答系统的核心模式。

---

### 卡片9：Chain vs Agent ⚖️

**一句话：** Chain 是确定性流水线，Agent 是动态决策循环。

**举例：**
```python
# Chain：固定路径
chain = prompt | llm | parser

# Agent：LLM 决定路径
# 观察 → 决策 → 行动 → 观察 → ...
```

**应用：** 固定流程用 Chain，需要决策用 Agent。

---

### 卡片10：Chain 在 LangChain 源码中的位置 ⭐

**一句话：** Chain 基于 Runnable 协议，是 LCEL 的核心实现。

**举例：**
```python
# langchain_core/runnables/base.py
class RunnableSequence(Runnable):
    def invoke(self, input, config=None):
        for step in self.steps:
            input = step.invoke(input, config)
        return input
```

**应用：** 理解 Chain 就理解了 LCEL 的执行机制。

---

## 10. 【一句话总结】

**Chain 是 LangChain 中将多个组件串联成流水线的核心机制，通过 LCEL 的 pipe(|) 操作符实现声明式组合，支持顺序执行、并行执行和条件分支，是构建 LLM 应用的基础模式。**

---

## 📚 学习检查清单

- [ ] 理解 Chain 的本质是多步骤的有序组合
- [ ] 会使用 `|` 操作符组合 Runnable
- [ ] 理解 RunnableSequence 和 RunnableParallel 的区别
- [ ] 能够构建 RAG Chain 模板
- [ ] 知道何时选择 Chain vs Agent
- [ ] 了解 LCEL 相比传统 Chain 的优势

## 🔗 下一步学习

- **Agent 代理模式**：动态决策的 LLM 应用
- **Runnable 协议**：LCEL 的底层实现
- **流式输出**：Chain 的 stream 方法深入

---

**版本：** v1.0
**最后更新：** 2025-01-14
