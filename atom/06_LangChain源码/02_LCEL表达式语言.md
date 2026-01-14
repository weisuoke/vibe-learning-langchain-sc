# LCEL 表达式语言

> 原子化知识点 | LangChain 源码 | LangChain Expression Language

---

## 1. 【30字核心】

**LCEL 是 LangChain 的声明式组合语法，通过 | 操作符将 Runnable 组件串联成可执行的处理管道。**

---

## 2. 【第一性原理】

### 什么是第一性原理？

**第一性原理**：回到事物最基本的真理，从源头思考问题

### LCEL 的第一性原理 🎯

#### 1. 最基础的定义

**LCEL = 组件 + 组合规则**

仅此而已！没有更基础的了。

- **组件 (Component)**：实现 Runnable 协议的任何对象
- **组合规则 (Composition)**：用 `|` 串联，用 `RunnableParallel` 并行

```python
# LCEL 的本质
chain = component_a | component_b | component_c
# 等价于：输入 → A处理 → B处理 → C处理 → 输出
```

#### 2. 为什么需要 LCEL？

**核心问题：如何简洁地表达复杂的 LLM 处理流程？**

```python
# 没有 LCEL 的写法（命令式）
def process(input_data):
    # 1. 格式化提示
    prompt_result = prompt_template.format(**input_data)

    # 2. 调用 LLM
    llm_result = llm.generate(prompt_result)

    # 3. 解析输出
    parsed_result = parser.parse(llm_result)

    # 4. 后处理
    final_result = postprocess(parsed_result)

    return final_result

# 问题：
# 1. 代码冗长，流程不直观
# 2. 难以复用和修改
# 3. 流式处理、批量处理需要重写
# 4. 错误处理和回调需要手动添加
```

```python
# 有了 LCEL（声明式）
chain = prompt | llm | parser | postprocess

result = chain.invoke(input_data)

# 优势：
# 1. 一行代码表达完整流程
# 2. 自动获得 stream/batch/ainvoke
# 3. 配置和回调自动传递
# 4. 易于修改和复用
```

#### 3. LCEL 的三层价值

##### 价值1：声明式语法 - 代码即文档

```python
# 代码本身就说明了处理流程
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)
# 一眼就能看出：检索 → 提示 → LLM → 解析
```

##### 价值2：自动能力继承 - 写一次，多种执行

```python
# 定义一次
chain = prompt | llm | parser

# 自动获得多种执行方式
chain.invoke(input)              # 同步
chain.stream(input)              # 流式
chain.batch([input1, input2])    # 批量
await chain.ainvoke(input)       # 异步
```

##### 价值3：可组合性 - 管道可以嵌套

```python
# 子管道
summarize = prompt1 | llm | parser1
translate = prompt2 | llm | parser2

# 组合成更大的管道
full_chain = (
    RunnableParallel(
        summary=summarize,
        translation=translate
    )
    | combine_results
)
```

#### 4. 从第一性原理推导 LCEL 设计

**推理链：**

```
1. LLM 应用本质是数据处理管道
   ↓
2. 管道由多个处理步骤组成
   ↓
3. 需要一种简洁的方式表达管道
   ↓
4. Unix 管道用 | 连接命令，直观易懂
   ↓
5. Python 支持操作符重载（__or__）
   ↓
6. 让所有组件实现 Runnable 协议
   ↓
7. 在 Runnable 中重载 | 操作符
   ↓
8. LCEL 诞生：prompt | llm | parser
```

#### 5. 一句话总结第一性原理

**LCEL 是用 | 操作符表达数据处理管道的声明式语法，让复杂的 LLM 应用流程变得简洁、可读、可组合。**

---

## 3. 【核心概念（全面覆盖）】

### 核心概念1：管道操作符 | 🔗

**| 操作符将两个 Runnable 串联成 RunnableSequence**

```python
from langchain_core.runnables import Runnable, RunnableSequence

# | 操作符的实现原理
class Runnable:
    def __or__(self, other: "Runnable") -> RunnableSequence:
        """a | b 时调用 a.__or__(b)"""
        return RunnableSequence(first=self, last=other)

    def __ror__(self, other) -> RunnableSequence:
        """当左操作数不是 Runnable 时调用"""
        # 将左操作数转换为 Runnable
        return RunnableSequence(
            first=coerce_to_runnable(other),
            last=self
        )

# 使用示例
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Tell me about {topic}")
llm = ChatOpenAI()
parser = StrOutputParser()

# 创建管道
chain = prompt | llm | parser

# 执行流程
# input → prompt.invoke() → llm.invoke() → parser.invoke() → output
```

**链式调用的展开：**

```python
# prompt | llm | parser 的展开过程

# 第一步：prompt | llm
step1 = prompt.__or__(llm)
# step1 = RunnableSequence(first=prompt, last=llm)

# 第二步：step1 | parser
step2 = step1.__or__(parser)
# step2 = RunnableSequence(
#     first=prompt,
#     middle=[llm],
#     last=parser
# )
```

**在 LangChain 源码中的位置：**

```python
# langchain_core/runnables/base.py
class Runnable(Generic[Input, Output], ABC):
    def __or__(
        self,
        other: Union[
            Runnable[Any, Other],
            Callable[[Any], Other],
            Callable[[Iterator[Any]], Iterator[Other]],
            Mapping[str, Union[Runnable[Any, Other], Callable[[Any], Other]]],
        ],
    ) -> RunnableSerializable[Input, Other]:
        return RunnableSequence(self, coerce_to_runnable(other))
```

---

### 核心概念2：RunnableSequence 串行组合 📐

**RunnableSequence 是 LCEL 管道的核心数据结构**

```python
from typing import List, Any, Optional, Iterator

class RunnableSequence(Runnable[Input, Output]):
    """串行执行多个 Runnable

    数据流：input → first → middle[0] → ... → middle[n] → last → output
    """

    first: Runnable[Input, Any]
    middle: List[Runnable[Any, Any]]
    last: Runnable[Any, Output]

    def __init__(
        self,
        *steps: Runnable,
        first: Runnable = None,
        middle: List[Runnable] = None,
        last: Runnable = None
    ):
        if steps:
            # 从位置参数构建
            self.first = steps[0]
            self.middle = list(steps[1:-1]) if len(steps) > 2 else []
            self.last = steps[-1] if len(steps) > 1 else steps[0]
        else:
            # 从关键字参数构建
            self.first = first
            self.middle = middle or []
            self.last = last

    @property
    def steps(self) -> List[Runnable]:
        """所有步骤的列表"""
        return [self.first] + self.middle + [self.last]

    def invoke(self, input: Input, config: Optional[dict] = None) -> Output:
        """串行执行所有步骤"""
        result = input
        for step in self.steps:
            result = step.invoke(result, config)
        return result

    def stream(self, input: Input, config: Optional[dict] = None) -> Iterator[Output]:
        """流式执行：只有最后一步流式输出"""
        # 前面的步骤正常执行
        result = input
        for step in self.steps[:-1]:
            result = step.invoke(result, config)

        # 最后一步流式输出
        for chunk in self.last.stream(result, config):
            yield chunk

    @property
    def input_schema(self):
        """输入 schema 由第一步决定"""
        return self.first.input_schema

    @property
    def output_schema(self):
        """输出 schema 由最后一步决定"""
        return self.last.output_schema

# 使用示例
chain = prompt | llm | parser

# 查看结构
print(chain.steps)  # [prompt, llm, parser]
print(chain.first)  # prompt
print(chain.last)   # parser
```

---

### 核心概念3：RunnableParallel 并行组合 🔀

**RunnableParallel 并行执行多个分支，结果合并为字典**

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

class RunnableParallel(Runnable[Input, Dict[str, Any]]):
    """并行执行多个 Runnable

    输入：同一个 input 传给所有分支
    输出：{"key1": result1, "key2": result2, ...}
    """

    steps: Dict[str, Runnable]

    def __init__(self, steps: Dict[str, Runnable] = None, **kwargs):
        self.steps = steps or kwargs

    def invoke(self, input: Input, config: Optional[dict] = None) -> Dict[str, Any]:
        """并行执行所有分支"""
        return {
            key: step.invoke(input, config)
            for key, step in self.steps.items()
        }

    async def ainvoke(self, input: Input, config: Optional[dict] = None) -> Dict[str, Any]:
        """真正的并行执行（异步）"""
        import asyncio

        async def run_step(key: str, step: Runnable):
            result = await step.ainvoke(input, config)
            return key, result

        results = await asyncio.gather(*[
            run_step(k, s) for k, s in self.steps.items()
        ])
        return dict(results)

# 使用示例

# 方式1：字典语法
parallel = RunnableParallel({
    "summary": summary_chain,
    "keywords": keyword_chain,
    "sentiment": sentiment_chain,
})

# 方式2：关键字参数
parallel = RunnableParallel(
    summary=summary_chain,
    keywords=keyword_chain,
    sentiment=sentiment_chain,
)

# 方式3：直接用字典（自动转换）
chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm
```

**RAG 经典模式：**

```python
from langchain_core.runnables import RunnablePassthrough

# RAG 管道的标准写法
rag_chain = (
    RunnableParallel(
        context=retriever | format_docs,    # 检索并格式化
        question=RunnablePassthrough()       # 原样传递问题
    )
    | prompt    # 使用 context 和 question
    | llm
    | parser
)

# 输入
input_data = {"question": "What is LangChain?"}

# 执行流程
# 1. RunnableParallel 并行执行：
#    - context: retriever.invoke(input) | format_docs.invoke(docs)
#    - question: RunnablePassthrough().invoke(input) → input
# 2. 结果合并：{"context": "...", "question": {"question": "..."}}
# 3. prompt.invoke(merged) → formatted_prompt
# 4. llm.invoke(formatted_prompt) → response
# 5. parser.invoke(response) → final_result
```

---

### 核心概念4：RunnablePassthrough 数据透传 ➡️

**RunnablePassthrough 原样传递输入，是构建复杂管道的关键工具**

```python
class RunnablePassthrough(Runnable[Input, Input]):
    """原样传递输入

    看起来什么都不做，但在构建管道时非常有用
    """

    def invoke(self, input: Input, config: Optional[dict] = None) -> Input:
        return input

    @classmethod
    def assign(cls, **kwargs: Runnable) -> "RunnableAssign":
        """在输入基础上添加新字段"""
        return RunnableAssign(mapper=RunnableParallel(kwargs))

# 使用场景1：在 RunnableParallel 中保留原始输入
chain = RunnableParallel(
    processed=some_processor,
    original=RunnablePassthrough()  # 保留原始输入
)

# 使用场景2：RAG 中保留问题
rag = RunnableParallel(
    context=retriever,
    question=RunnablePassthrough()  # 问题原样传递
) | prompt | llm

# 使用场景3：assign 添加字段
chain = RunnablePassthrough.assign(
    context=retriever  # 在原输入基础上添加 context 字段
) | prompt | llm

# assign 的效果
input_data = {"question": "What is AI?"}
# 经过 assign 后变成：
# {"question": "What is AI?", "context": "retrieved docs..."}
```

---

### 核心概念5：RunnableBranch 条件分支 🔀

**RunnableBranch 根据条件选择不同的执行路径**

```python
from langchain_core.runnables import RunnableBranch

class RunnableBranch(Runnable[Input, Output]):
    """条件分支执行

    类似 if-elif-else 逻辑
    """

    branches: List[Tuple[Callable[[Input], bool], Runnable]]
    default: Runnable

    def __init__(
        self,
        *branches: Tuple[Callable[[Input], bool], Runnable],
        default: Runnable
    ):
        self.branches = list(branches)
        self.default = default

    def invoke(self, input: Input, config: Optional[dict] = None) -> Output:
        for condition, runnable in self.branches:
            if condition(input):
                return runnable.invoke(input, config)
        return self.default.invoke(input, config)

# 使用示例
branch = RunnableBranch(
    # (条件函数, 对应的处理链)
    (lambda x: "code" in x["question"].lower(), code_chain),
    (lambda x: "translate" in x["question"].lower(), translate_chain),
    (lambda x: len(x["question"]) > 200, long_question_chain),
    default=general_chain
)

# 执行
result = branch.invoke({"question": "Write code for sorting"})
# 会选择 code_chain 执行
```

**使用 RunnableLambda 实现动态路由：**

```python
from langchain_core.runnables import RunnableLambda

def route(input: dict) -> Runnable:
    """根据输入动态选择链"""
    question = input.get("question", "").lower()
    if "code" in question:
        return code_chain
    elif "translate" in question:
        return translate_chain
    else:
        return general_chain

# 动态路由
chain = RunnableLambda(route) | RunnableLambda(lambda chain: chain.invoke)
# 或者更简洁
chain = RunnableLambda(lambda x: route(x).invoke(x))
```

---

### 核心概念6：RunnableLambda 函数包装 🔧

**RunnableLambda 将普通函数转换为 Runnable**

```python
from langchain_core.runnables import RunnableLambda

# 使用方式1：直接包装
def format_output(text: str) -> str:
    return f"[AI] {text.strip()}"

formatter = RunnableLambda(format_output)
chain = prompt | llm | parser | formatter

# 使用方式2：装饰器语法
@RunnableLambda
def add_timestamp(text: str) -> str:
    from datetime import datetime
    return f"[{datetime.now()}] {text}"

chain = prompt | llm | parser | add_timestamp

# 使用方式3：lambda 表达式
chain = prompt | llm | RunnableLambda(lambda x: x.content.upper())
```

---

### 核心概念7：bind 和 with_config ⚙️

**bind 预设参数，with_config 预设运行配置**

```python
# bind：预设调用参数
llm = ChatOpenAI()
llm_with_temp = llm.bind(temperature=0.9)
llm_with_tools = llm.bind(tools=[tool1, tool2])

# with_config：预设运行配置
chain = prompt | llm | parser
configured_chain = chain.with_config(
    tags=["production"],
    metadata={"user_id": "123"},
    run_name="my-chain"
)

# 配置会传递给所有组件
result = configured_chain.invoke(input)
```

---

## 4. 【最小可用】

掌握以下内容，就能使用 LCEL 构建 LLM 应用：

### 4.1 基本管道 |

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# 创建组件
prompt = ChatPromptTemplate.from_template("Tell me about {topic}")
llm = ChatOpenAI()
parser = StrOutputParser()

# 用 | 组合
chain = prompt | llm | parser

# 执行
result = chain.invoke({"topic": "Python"})
```

### 4.2 并行执行 RunnableParallel

```python
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# 并行获取多个结果
chain = RunnableParallel(
    summary=summary_chain,
    keywords=keyword_chain,
) | combine_chain
```

### 4.3 保留原始输入 RunnablePassthrough

```python
# RAG 标准模式
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
)
```

### 4.4 自定义函数 RunnableLambda

```python
from langchain_core.runnables import RunnableLambda

chain = prompt | llm | RunnableLambda(lambda x: x.content.upper())
```

**这些知识足以：**
- 构建基本的 LLM 处理管道
- 实现 RAG 应用
- 添加自定义处理逻辑

---

## 5. 【1个类比】（双轨制）

### 类比1：LCEL 管道

#### 🎨 前端视角：RxJS Pipe / Redux Middleware

```typescript
// RxJS pipe：数据流经一系列操作符
observable.pipe(
  map(x => x * 2),
  filter(x => x > 10),
  take(5)
);

// Redux middleware：action 流经中间件
const store = createStore(
  reducer,
  applyMiddleware(logger, thunk, api)
);
```

```python
# LCEL：数据流经一系列组件
chain = prompt | llm | parser
```

#### 🧒 小朋友视角：工厂流水线

```
LCEL 就像工厂的流水线：

原材料 → [切割机] → [打磨机] → [喷漆机] → 成品
   ↓         ↓          ↓          ↓
 input →  prompt  →   llm   →  parser → output

每个机器（组件）做一件事，
产品（数据）依次经过每个机器，
最后变成成品（结果）！
```

---

### 类比2：RunnableParallel

#### 🎨 前端视角：Promise.all

```typescript
const results = await Promise.all([
  fetchUser(id),
  fetchOrders(id),
  fetchReviews(id),
]);
```

#### 🧒 小朋友视角：分组做作业

```
RunnableParallel 就像分组做作业：

老师布置了三道题，你找三个朋友同时做：
- 小明做第1题 → 答案1
- 小红做第2题 → 答案2
- 小刚做第3题 → 答案3

最后把答案合在一起交给老师！
比一个人做三道题快多了！
```

---

### 类比总结表

| LCEL 概念 | 前端类比 | 小朋友类比 |
|----------|---------|-----------|
| `\|` 管道 | RxJS pipe | 工厂流水线 |
| RunnableSequence | middleware chain | 接力赛 |
| RunnableParallel | Promise.all | 分组做作业 |
| RunnablePassthrough | identity 函数 | 复印机 |
| RunnableBranch | if-else / switch | 走迷宫选路 |
| RunnableLambda | 高阶函数 | 万能转换器 |

---

## 6. 【反直觉点】

### 误区1：LCEL 只是语法糖 ❌

**为什么错？**
- LCEL 不仅是语法简化，还提供了完整的执行框架
- 自动获得 stream/batch/ainvoke 能力
- 配置和回调自动传递

**正确理解：**
```python
# LCEL 提供的不只是简洁语法
chain = prompt | llm | parser

# 自动获得这些能力
chain.stream(input)      # 流式
chain.batch(inputs)      # 批量
await chain.ainvoke(input)  # 异步
```

---

### 误区2：RunnablePassthrough 没什么用 ❌

**为什么错？**
- 在 RunnableParallel 中保留原始输入是关键操作
- assign 方法可以在原输入基础上添加字段

**正确理解：**
```python
# 没有 RunnablePassthrough，无法同时传递 context 和 question
chain = {
    "context": retriever,
    "question": RunnablePassthrough()  # 关键！
} | prompt | llm
```

---

### 误区3：| 操作符从左到右立即执行 ❌

**为什么错？**
- `|` 只是构建数据结构，不执行任何逻辑
- 只有调用 invoke/stream 时才真正执行

**正确理解：**
```python
# 这一行不执行任何 LLM 调用
chain = prompt | llm | parser  # 只是构建 RunnableSequence

# 这一行才真正执行
result = chain.invoke(input)  # 现在才调用 LLM
```

---

## 7. 【实战代码】

```python
"""
示例：LCEL 表达式语言实战
演示 LCEL 的核心用法
"""

from typing import Any, Dict, List, Iterator, Optional

# ===== 1. 基础管道 =====
print("=== 1. 基础管道 ===")

# 模拟组件
class FakePrompt:
    def __init__(self, template: str):
        self.template = template

    def invoke(self, input: Dict, config=None) -> str:
        return self.template.format(**input)

    def __or__(self, other):
        return Sequence(self, other)

class FakeLLM:
    def invoke(self, input: str, config=None) -> str:
        return f"LLM: {input[:30]}..."

    def stream(self, input: str, config=None) -> Iterator[str]:
        response = self.invoke(input)
        for char in response:
            yield char

    def __or__(self, other):
        return Sequence(self, other)

class Sequence:
    def __init__(self, first, last, middle=None):
        self.first = first
        self.middle = middle or []
        self.last = last

    def invoke(self, input, config=None):
        result = self.first.invoke(input, config)
        for step in self.middle:
            result = step.invoke(result, config)
        return self.last.invoke(result, config)

    def __or__(self, other):
        return Sequence(self.first, other, self.middle + [self.last])

prompt = FakePrompt("Tell me about {topic}")
llm = FakeLLM()

chain = prompt | llm
result = chain.invoke({"topic": "Python"})
print(f"Result: {result}")
```

---

## 8. 【面试必问】

### 问题："LCEL 是什么？有什么优势？"

**普通回答（❌ 不出彩）：**
"LCEL 是 LangChain Expression Language，用 | 连接组件。"

**出彩回答（✅ 推荐）：**

> **LCEL 有三个核心优势：**
>
> 1. **声明式语法**：`prompt | llm | parser` 一眼看出处理流程
>
> 2. **自动能力继承**：定义一次，自动获得 stream/batch/ainvoke
>
> 3. **可组合性**：管道可以嵌套，支持串行、并行、条件分支
>
> **实现原理**：通过 `__or__` 操作符重载，`a | b` 返回 `RunnableSequence`

---

## 9. 【化骨绵掌】

### 卡片1：LCEL 是什么 🎯

**一句话：** LCEL 是用 | 操作符组合 LangChain 组件的声明式语法。

**举例：**
```python
chain = prompt | llm | parser
```

**应用：** 一行代码构建完整的 LLM 处理管道。

---

### 卡片2：| 操作符原理 📐

**一句话：** | 通过 `__or__` 方法创建 RunnableSequence。

**举例：**
```python
# a | b 等价于
a.__or__(b)  # 返回 RunnableSequence(a, b)
```

**应用：** 理解 | 的本质是构建数据结构，不是立即执行。

---

### 卡片3：RunnableSequence 🔗

**一句话：** RunnableSequence 串行执行多个组件。

**举例：**
```python
chain = prompt | llm | parser
# 执行：input → prompt → llm → parser → output
```

**应用：** LCEL 管道的核心数据结构。

---

### 卡片4：RunnableParallel 🔀

**一句话：** RunnableParallel 并行执行多个分支。

**举例：**
```python
parallel = RunnableParallel(
    summary=chain1,
    keywords=chain2,
)
```

**应用：** 同时获取多个结果，如 RAG 中的 context 和 question。

---

### 卡片5：RunnablePassthrough ➡️

**一句话：** RunnablePassthrough 原样传递输入。

**举例：**
```python
{"context": retriever, "question": RunnablePassthrough()}
```

**应用：** 在并行执行时保留原始输入。

---

### 卡片6：RunnableLambda 🔧

**一句话：** RunnableLambda 将普通函数包装成 Runnable。

**举例：**
```python
formatter = RunnableLambda(lambda x: x.upper())
chain = prompt | llm | formatter
```

**应用：** 让任意自定义函数加入 LCEL 管道。

---

### 卡片7：RunnableBranch 🔀

**一句话：** RunnableBranch 根据条件选择执行路径。

**举例：**
```python
branch = RunnableBranch(
    (lambda x: "code" in x, code_chain),
    default=general_chain
)
```

**应用：** 实现动态路由，根据输入选择不同处理逻辑。

---

### 卡片8：bind 绑定参数 ⚙️

**一句话：** bind 预设组件的调用参数。

**举例：**
```python
llm_creative = llm.bind(temperature=0.9)
llm_with_tools = llm.bind(tools=[tool1])
```

**应用：** 创建预配置的组件变体。

---

### 卡片9：stream 流式输出 🌊

**一句话：** stream 方法实现流式输出。

**举例：**
```python
for chunk in chain.stream(input):
    print(chunk, end="")
```

**应用：** 实时显示 LLM 响应，提升用户体验。

---

### 卡片10：LCEL 最佳实践 ⭐

**一句话：** 掌握 LCEL 模式可以快速构建复杂 LLM 应用。

**RAG 标准模式：**
```python
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)
```

**应用：** 这是 LangChain 最常用的设计模式。

---

## 10. 【一句话总结】

**LCEL 是 LangChain 的声明式组合语法，通过 | 操作符将 Runnable 组件串联成处理管道，提供简洁的语法和自动的执行能力（stream/batch/async）。**

---

## 📚 学习检查清单

- [ ] 理解 LCEL 是什么以及为什么需要它
- [ ] 会使用 | 操作符组合组件
- [ ] 理解 RunnableSequence 的工作原理
- [ ] 会使用 RunnableParallel 并行执行
- [ ] 会使用 RunnablePassthrough 保留输入
- [ ] 能用 RunnableLambda 包装自定义函数
- [ ] 理解 RunnableBranch 条件分支
- [ ] 会使用 bind 预设参数
- [ ] 能够构建 RAG 风格的 LCEL 管道

## 🔗 下一步学习

- **BaseChatModel 实现**：理解 LLM 组件如何实现 Runnable
- **Agent 执行引擎**：理解 Agent 如何使用 LCEL
- **Callback 回调系统**：理解执行过程中的事件处理

---

**版本：** v1.0
**最后更新：** 2025-12-12