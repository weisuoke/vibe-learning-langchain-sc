# Runnable 协议

> 原子化知识点 | LangChain 源码 | 核心组件协议

---

## 1. 【30字核心】

**Runnable 是 LangChain 所有可执行组件的统一协议，定义了 invoke/stream/batch 等标准接口，是 LCEL 的基石。**

---

## 2. 【第一性原理】

### 什么是第一性原理？

**第一性原理**：回到事物最基本的真理，从源头思考问题

### Runnable 协议的第一性原理 🎯

#### 1. 最基础的定义

**Runnable = 输入 → 处理 → 输出**

仅此而已！没有更基础的了。

- **输入 (Input)**：组件接收的数据
- **处理 (Process)**：组件执行的逻辑
- **输出 (Output)**：组件产生的结果

```python
# 最简单的 Runnable 本质
def runnable(input: Input) -> Output:
    return process(input)
```

#### 2. 为什么需要 Runnable 协议？

**核心问题：如何让 LLM 应用的各种组件能够无缝协作？**

```python
# 没有统一协议的困境
prompt_template.format(query="Hello")      # 返回 str
llm.generate("Hello")                       # 返回 Generation
parser.parse("response")                    # 返回 dict
retriever.get_relevant_docs("query")        # 返回 List[Document]

# 问题：
# 1. 每个组件的调用方式不同
# 2. 无法简单地串联组件
# 3. 难以实现批量处理、流式输出、异步调用
```

```python
# 有了 Runnable 协议
prompt.invoke({"query": "Hello"})          # 统一的 invoke 接口
llm.invoke("Hello")                         # 统一的 invoke 接口
parser.invoke("response")                   # 统一的 invoke 接口
retriever.invoke("query")                   # 统一的 invoke 接口

# 优势：
# 1. 所有组件用相同方式调用
# 2. 可以用 | 操作符串联：prompt | llm | parser
# 3. 自动获得 batch/stream/ainvoke 能力
```

#### 3. Runnable 协议的三层价值

##### 价值1：统一接口 - 一致的调用方式

```python
from langchain_core.runnables import Runnable

# 无论是什么组件，都用 invoke 调用
result1 = prompt.invoke(input_data)
result2 = llm.invoke(input_data)
result3 = chain.invoke(input_data)

# 多态的威力：不关心具体类型
def process(runnable: Runnable, data):
    return runnable.invoke(data)
```

##### 价值2：可组合性 - LCEL 管道

```python
# Runnable 支持 | 操作符组合
chain = prompt | llm | parser

# 等价于
chain = RunnableSequence(
    first=prompt,
    middle=[llm],
    last=parser
)

# 执行时自动串联
result = chain.invoke({"query": "Hello"})
```

##### 价值3：多模式执行 - 同步/异步/批量/流式

```python
# 一个 Runnable 自动获得四种执行模式
runnable.invoke(input)              # 同步单次
runnable.batch([input1, input2])    # 批量处理
runnable.stream(input)              # 流式输出
await runnable.ainvoke(input)       # 异步单次
await runnable.astream(input)       # 异步流式
```

#### 4. 从第一性原理推导 LangChain 源码架构

**推理链：**

```
1. LLM 应用需要组合多种组件（提示、模型、解析器、检索器...）
   ↓
2. 每种组件都是"输入→处理→输出"的函数
   ↓
3. 需要一个统一的"可执行"抽象
   ↓
4. 定义 Runnable 协议：invoke(input) -> output
   ↓
5. 所有组件实现 Runnable 协议
   ↓
6. 利用 Python 的 __or__ 实现 | 操作符
   ↓
7. LCEL 诞生：prompt | llm | parser
   ↓
8. 在 Runnable 基类中实现 batch/stream/ainvoke
   ↓
9. 所有组件自动获得多模式执行能力
```

#### 5. 一句话总结第一性原理

**Runnable 是"输入→输出"的统一抽象，通过协议标准化让所有组件可组合、可替换、自动获得多模式执行能力。**

---

## 3. 【核心概念（全面覆盖）】

### 核心概念1：Runnable 抽象基类 🏗️

**Runnable 是所有可执行组件的抽象基类，定义了标准执行接口**

```python
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional, List, Iterator, Any

Input = TypeVar("Input")
Output = TypeVar("Output")

class Runnable(ABC, Generic[Input, Output]):
    """LangChain 核心抽象：可执行组件协议

    所有 LangChain 组件都实现这个协议：
    - PromptTemplate
    - ChatModel / LLM
    - OutputParser
    - Retriever
    - Chain
    - Agent
    """

    # ===== 核心执行方法 =====

    @abstractmethod
    def invoke(self, input: Input, config: Optional[dict] = None) -> Output:
        """同步执行（核心方法，子类必须实现）"""
        pass

    def batch(self, inputs: List[Input], config: Optional[dict] = None) -> List[Output]:
        """批量执行（默认实现：循环调用 invoke）"""
        return [self.invoke(x, config) for x in inputs]

    def stream(self, input: Input, config: Optional[dict] = None) -> Iterator[Output]:
        """流式执行（默认实现：yield 完整结果）"""
        yield self.invoke(input, config)

    # ===== 异步版本 =====

    async def ainvoke(self, input: Input, config: Optional[dict] = None) -> Output:
        """异步执行（默认实现：调用同步版本）"""
        return self.invoke(input, config)

    async def abatch(self, inputs: List[Input], config: Optional[dict] = None) -> List[Output]:
        """异步批量执行"""
        import asyncio
        return await asyncio.gather(*[self.ainvoke(x, config) for x in inputs])

    # ===== 组合操作符 =====

    def __or__(self, other: "Runnable") -> "RunnableSequence":
        """重载 | 操作符，实现 LCEL 管道"""
        return RunnableSequence(first=self, last=other)

    def __ror__(self, other: Any) -> "RunnableSequence":
        """处理左操作数不是 Runnable 的情况"""
        return RunnableSequence(first=coerce_to_runnable(other), last=self)
```

**核心方法对比：**

| 方法 | 说明 | 输入 | 输出 |
|------|------|------|------|
| `invoke` | 同步单次执行 | `Input` | `Output` |
| `batch` | 批量执行 | `List[Input]` | `List[Output]` |
| `stream` | 流式输出 | `Input` | `Iterator[Output]` |
| `ainvoke` | 异步执行 | `Input` | `Output` |
| `astream` | 异步流式 | `Input` | `AsyncIterator[Output]` |

**在 LangChain 源码中的位置：**

```python
# langchain_core/runnables/base.py
class Runnable(Generic[Input, Output], ABC):
    """所有 LangChain 组件的基类"""

    @property
    def InputType(self) -> Type[Input]:
        """输入类型"""
        # 通过泛型参数推断
        ...

    @property
    def OutputType(self) -> Type[Output]:
        """输出类型"""
        ...

    @property
    def input_schema(self) -> Type[BaseModel]:
        """输入的 Pydantic schema"""
        ...

    @property
    def output_schema(self) -> Type[BaseModel]:
        """输出的 Pydantic schema"""
        ...
```

---

### 核心概念2：RunnableSequence 序列组合 📐

**RunnableSequence 是 LCEL 管道的核心实现，串联多个 Runnable**

```python
from typing import List, Any

class RunnableSequence(Runnable[Input, Output]):
    """Runnable 序列：A | B | C 的实现

    执行流程：
    input → A.invoke() → B.invoke() → C.invoke() → output
    """

    first: Runnable       # 第一个组件
    middle: List[Runnable] # 中间组件（可以为空）
    last: Runnable        # 最后一个组件

    def __init__(self, first: Runnable, last: Runnable, middle: List[Runnable] = None):
        self.first = first
        self.middle = middle or []
        self.last = last

    def invoke(self, input: Input, config: Optional[dict] = None) -> Output:
        """串联执行所有组件"""
        # 1. 执行第一个
        result = self.first.invoke(input, config)

        # 2. 执行中间的
        for runnable in self.middle:
            result = runnable.invoke(result, config)

        # 3. 执行最后一个
        return self.last.invoke(result, config)

    def stream(self, input: Input, config: Optional[dict] = None) -> Iterator[Output]:
        """流式执行：只有最后一个组件流式输出"""
        # 1. 前面的组件正常执行
        result = self.first.invoke(input, config)
        for runnable in self.middle:
            result = runnable.invoke(result, config)

        # 2. 最后一个组件流式输出
        for chunk in self.last.stream(result, config):
            yield chunk

    @property
    def input_schema(self):
        """输入 schema 由第一个组件决定"""
        return self.first.input_schema

    @property
    def output_schema(self):
        """输出 schema 由最后一个组件决定"""
        return self.last.output_schema

    def __or__(self, other: Runnable) -> "RunnableSequence":
        """支持继续链接：(A | B) | C"""
        return RunnableSequence(
            first=self.first,
            middle=self.middle + [self.last],
            last=other
        )

# 使用示例
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("Tell me about {topic}")
llm = ChatOpenAI()
parser = StrOutputParser()

# 创建序列（三种等价方式）
chain1 = prompt | llm | parser                         # LCEL 语法
chain2 = RunnableSequence(first=prompt, last=parser, middle=[llm])  # 显式创建
chain3 = prompt.__or__(llm).__or__(parser)            # 手动调用

# 执行
result = chain1.invoke({"topic": "Python"})
```

**RunnableSequence 的流式行为：**

```python
# 流式输出时，只有最后一个支持流式的组件会产生多个 chunk
chain = prompt | llm | parser

for chunk in chain.stream({"topic": "AI"}):
    print(chunk, end="", flush=True)
    # 逐字输出 LLM 的响应
```

---

### 核心概念3：RunnableParallel 并行组合 🔀

**RunnableParallel 并行执行多个 Runnable，结果合并为字典**

```python
from typing import Dict, Any

class RunnableParallel(Runnable[Input, Dict[str, Any]]):
    """并行执行多个 Runnable

    输入：同一个 input 传给所有分支
    输出：{key1: result1, key2: result2, ...}
    """

    steps: Dict[str, Runnable]

    def __init__(self, steps: Dict[str, Runnable] = None, **kwargs):
        self.steps = steps or kwargs

    def invoke(self, input: Input, config: Optional[dict] = None) -> Dict[str, Any]:
        """并行执行（同步版本实际上是顺序执行）"""
        return {
            key: runnable.invoke(input, config)
            for key, runnable in self.steps.items()
        }

    async def ainvoke(self, input: Input, config: Optional[dict] = None) -> Dict[str, Any]:
        """真正的并行执行（异步版本）"""
        import asyncio

        async def run_one(key: str, runnable: Runnable):
            result = await runnable.ainvoke(input, config)
            return key, result

        results = await asyncio.gather(*[
            run_one(key, runnable)
            for key, runnable in self.steps.items()
        ])
        return dict(results)

# 使用示例
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

# 并行执行多个任务
parallel = RunnableParallel(
    summary=summary_chain,
    translation=translation_chain,
    keywords=keyword_chain,
)

# 同一个输入并行处理
result = parallel.invoke({"text": "Hello World"})
# result = {
#     "summary": "...",
#     "translation": "...",
#     "keywords": ["..."]
# }

# 常见用法：构造 context + question
chain = (
    RunnableParallel(
        context=retriever,           # 检索相关文档
        question=RunnablePassthrough() # 原样传递问题
    )
    | prompt  # 使用 context 和 question
    | llm
)
```

---

### 核心概念4：RunnableLambda 函数包装 🔧

**RunnableLambda 将普通函数包装为 Runnable**

```python
from typing import Callable

class RunnableLambda(Runnable[Input, Output]):
    """将普通函数包装为 Runnable

    让任意函数都能参与 LCEL 管道
    """

    func: Callable[[Input], Output]
    afunc: Optional[Callable[[Input], Awaitable[Output]]] = None

    def __init__(
        self,
        func: Callable[[Input], Output],
        afunc: Optional[Callable] = None
    ):
        self.func = func
        self.afunc = afunc

    def invoke(self, input: Input, config: Optional[dict] = None) -> Output:
        """调用包装的函数"""
        return self.func(input)

    async def ainvoke(self, input: Input, config: Optional[dict] = None) -> Output:
        """异步调用"""
        if self.afunc:
            return await self.afunc(input)
        return self.func(input)

# 使用示例
from langchain_core.runnables import RunnableLambda

# 包装普通函数
def format_output(text: str) -> str:
    return text.upper()

formatter = RunnableLambda(format_output)

# 参与 LCEL 管道
chain = prompt | llm | parser | formatter

# 使用装饰器语法
@RunnableLambda
def add_prefix(text: str) -> str:
    return f"[AI] {text}"

chain = prompt | llm | parser | add_prefix
```

---

### 核心概念5：RunnablePassthrough 数据透传 ➡️

**RunnablePassthrough 原样传递输入，常用于保留原始数据**

```python
class RunnablePassthrough(Runnable[Input, Input]):
    """原样传递输入

    看起来什么都不做，但在构造复杂管道时非常有用
    """

    def invoke(self, input: Input, config: Optional[dict] = None) -> Input:
        return input

    @classmethod
    def assign(cls, **kwargs) -> "RunnableAssign":
        """添加新字段到输入"""
        return RunnableAssign(mapper=RunnableParallel(kwargs))

# 使用示例
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# 场景1：保留原始输入
chain = RunnableParallel(
    original=RunnablePassthrough(),  # 保留原始输入
    processed=some_processor         # 处理后的结果
)

# 场景2：RAG 常见模式
rag_chain = (
    RunnableParallel(
        context=retriever | format_docs,  # 检索并格式化文档
        question=RunnablePassthrough()    # 原样传递问题
    )
    | prompt
    | llm
    | parser
)

# 场景3：assign 添加字段
chain = RunnablePassthrough.assign(
    context=retriever,  # 添加 context 字段
    # question 字段自动保留
) | prompt | llm
```

---

### 核心概念6：RunnableBranch 条件分支 🔀

**RunnableBranch 根据条件选择不同的执行路径**

```python
from typing import Tuple, Callable

class RunnableBranch(Runnable[Input, Output]):
    """条件分支：根据条件选择不同的 Runnable

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
        """根据条件选择分支执行"""
        for condition, runnable in self.branches:
            if condition(input):
                return runnable.invoke(input, config)
        return self.default.invoke(input, config)

# 使用示例
from langchain_core.runnables import RunnableBranch

# 根据问题类型选择不同的处理链
branch = RunnableBranch(
    # (条件函数, 对应的 Runnable)
    (lambda x: "代码" in x["question"], code_chain),
    (lambda x: "翻译" in x["question"], translation_chain),
    (lambda x: len(x["question"]) > 100, long_question_chain),
    default=general_chain  # 默认分支
)

# 执行
result = branch.invoke({"question": "帮我写一段代码"})
# 会选择 code_chain 执行
```

---

### 扩展概念7：RunnableConfig 运行配置 ⚙️

```python
from typing import TypedDict, Optional, List, Dict, Any

class RunnableConfig(TypedDict, total=False):
    """Runnable 执行时的配置"""

    # 回调处理器
    callbacks: Optional[List[BaseCallbackHandler]]

    # 标签（用于追踪）
    tags: Optional[List[str]]

    # 元数据
    metadata: Optional[Dict[str, Any]]

    # 运行名称
    run_name: Optional[str]

    # 最大并发数
    max_concurrency: Optional[int]

    # 递归深度限制
    recursion_limit: Optional[int]

    # 可配置字段
    configurable: Optional[Dict[str, Any]]

# 使用示例
config = {
    "callbacks": [MyCallbackHandler()],
    "tags": ["production", "user-123"],
    "metadata": {"user_id": "123"},
    "run_name": "my-chain-run",
    "max_concurrency": 5,
}

result = chain.invoke(input, config=config)

# 使用 with_config 预设配置
configured_chain = chain.with_config(
    tags=["production"],
    run_name="production-chain"
)
result = configured_chain.invoke(input)
```

---

## 4. 【最小可用】

掌握以下内容，就能开始使用和理解 Runnable 协议：

### 4.1 invoke 基本调用

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 所有 LangChain 组件都用 invoke 调用
prompt = ChatPromptTemplate.from_template("Tell me about {topic}")
llm = ChatOpenAI()

# 调用 prompt
formatted = prompt.invoke({"topic": "Python"})

# 调用 llm
response = llm.invoke(formatted)
```

### 4.2 使用 | 操作符组合

```python
from langchain_core.output_parsers import StrOutputParser

# 用 | 串联组件
chain = prompt | llm | StrOutputParser()

# 一次调用完成整个流程
result = chain.invoke({"topic": "Python"})
print(result)  # 直接得到字符串结果
```

### 4.3 stream 流式输出

```python
# 流式输出 LLM 响应
for chunk in chain.stream({"topic": "AI"}):
    print(chunk, end="", flush=True)
```

### 4.4 batch 批量处理

```python
# 批量处理多个输入
inputs = [
    {"topic": "Python"},
    {"topic": "JavaScript"},
    {"topic": "Rust"},
]

results = chain.batch(inputs)
# results = ["...", "...", "..."]
```

### 4.5 RunnableLambda 包装函数

```python
from langchain_core.runnables import RunnableLambda

# 将自定义函数加入管道
def postprocess(text: str) -> str:
    return text.strip().upper()

chain = prompt | llm | StrOutputParser() | RunnableLambda(postprocess)
```

**这些知识足以：**
- 理解 LangChain 组件的统一调用方式
- 使用 LCEL 构建处理管道
- 实现流式输出和批量处理
- 将自定义逻辑集成到管道中

---

## 5. 【1个类比】（双轨制）

### 类比1：Runnable 协议

#### 🎨 前端视角：React Component 协议

Runnable 就像 React 中所有组件都必须实现的 `render()` 方法。

```typescript
// React: 所有组件都必须实现 render
interface Component<Props, State> {
  render(): ReactNode;          // 类似 invoke
  componentDidMount?(): void;   // 生命周期
  componentDidUpdate?(): void;
}

// 函数组件也是一种 "Runnable"
function MyComponent(props: Props): ReactNode {
  return <div>{props.text}</div>;
}
```

```python
# LangChain: 所有组件都必须实现 invoke
class Runnable(ABC, Generic[Input, Output]):
    @abstractmethod
    def invoke(self, input: Input) -> Output:  # 核心方法
        pass

    def stream(self, input: Input): ...  # 额外能力
    def batch(self, inputs: List[Input]): ...
```

**相似点：**
- 都是定义组件的标准接口
- 都支持组合（React 组件嵌套 ≈ LCEL 管道）
- 都有输入输出（Props → ReactNode ≈ Input → Output）

#### 🧒 小朋友视角：USB 接口

Runnable 就像 **USB 接口**：

```
所有 USB 设备都遵守同一个标准：
- 鼠标 🖱️ ── USB 接口 ── 电脑
- 键盘 ⌨️ ── USB 接口 ── 电脑
- U盘 💾 ── USB 接口 ── 电脑
- 手柄 🎮 ── USB 接口 ── 电脑

不管是什么设备，只要有 USB 接口，就能插到电脑上用！
这就是 "协议" 的威力！

LangChain 的 Runnable 也是这样：
- 提示模板 📝 ── invoke ── 结果
- 语言模型 🤖 ── invoke ── 结果
- 解析器 🔍 ── invoke ── 结果

不管是什么组件，只要实现了 invoke，就能组合在一起！
```

---

### 类比2：RunnableSequence (| 管道)

#### 🎨 前端视角：Redux Middleware / RxJS Pipe

```typescript
// Redux middleware：请求经过一系列中间件
const middleware = applyMiddleware(
  logger,     // 第一个
  thunk,      // 第二个
  api         // 第三个
);
// 请求流：action → logger → thunk → api → store

// RxJS pipe：数据经过一系列操作符
observable.pipe(
  map(x => x * 2),
  filter(x => x > 10),
  take(5)
);
// 数据流：source → map → filter → take → subscribe
```

```python
# LangChain LCEL：数据经过一系列组件
chain = prompt | llm | parser
# 数据流：input → prompt → llm → parser → output
```

**相似点：**
- 都是数据管道模式
- 都是从左到右依次处理
- 上一步的输出是下一步的输入

#### 🧒 小朋友视角：接力赛跑

```
LCEL 管道就像接力赛跑：

第一棒（prompt）🏃
  ↓ 传递接力棒
第二棒（llm）🏃
  ↓ 传递接力棒
第三棒（parser）🏃
  ↓
终点！🏁

每个选手（组件）：
1. 接过接力棒（接收上一步的输出）
2. 跑自己的那段路（执行自己的逻辑）
3. 把接力棒传给下一个人（输出给下一步）

prompt | llm | parser
就像：小明 → 小红 → 小刚 → 终点
```

---

### 类比3：RunnableParallel (并行)

#### 🎨 前端视角：Promise.all

```typescript
// Promise.all：并行执行多个异步操作
const results = await Promise.all([
  fetchUser(id),
  fetchOrders(id),
  fetchReviews(id),
]);
// results = [user, orders, reviews]
```

```python
# RunnableParallel：并行执行多个组件
parallel = RunnableParallel(
    user=fetch_user_chain,
    orders=fetch_orders_chain,
    reviews=fetch_reviews_chain,
)
result = parallel.invoke({"id": "123"})
# result = {"user": ..., "orders": ..., "reviews": ...}
```

#### 🧒 小朋友视角：同时做作业

```
RunnableParallel 就像同时做不同科目的作业：

你有三个好朋友帮你同时做作业：
┌─────────────────────────┐
│        同一道题          │
│     "1+1等于几？"        │
└─────────────────────────┘
         ↓↓↓
    ┌────┴────┬────┴────┐
    ↓         ↓         ↓
  小明      小红       小刚
 (数学)    (语文)     (英语)
    ↓         ↓         ↓
  答案1     答案2      答案3
    └────┬────┴────┬────┘
         ↓
   {数学: 2, 语文: "二", 英语: "two"}

三个人同时做，比一个人做三遍快多了！
```

---

### 类比4：RunnableLambda

#### 🎨 前端视角：高阶函数 / Array.map

```typescript
// 任何函数都能参与 map 管道
const numbers = [1, 2, 3];
const doubled = numbers.map(x => x * 2);  // 普通函数包装进 map

// 任何函数都能变成中间件
const myMiddleware = (next) => (action) => {
  console.log(action);
  return next(action);
};
```

```python
# 任何函数都能变成 Runnable
def my_function(x):
    return x * 2

runnable = RunnableLambda(my_function)
chain = other_runnable | runnable | another_runnable
```

#### 🧒 小朋友视角：把普通工具变成乐高零件

```
RunnableLambda 就像给普通工具装上乐高接口：

你有一把普通的小锤子（普通函数）：
🔨 锤子

但乐高积木需要特殊接口才能拼接...

用 RunnableLambda 包装一下：
┌─────────────────┐
│  🔨 锤子        │
│  ○──────────○  │  ← 加上乐高接口
└─────────────────┘

现在它可以和其他乐高零件拼在一起了！
```

---

### 类比总结表

| Runnable 概念 | 前端类比 | 小朋友类比 |
|--------------|---------|-----------|
| Runnable 协议 | React Component 接口 | USB 接口标准 |
| invoke() | render() / 函数调用 | 按下开始按钮 |
| RunnableSequence (`\|`) | Redux middleware / RxJS pipe | 接力赛跑 |
| RunnableParallel | Promise.all | 同时做不同作业 |
| RunnableLambda | 高阶函数包装 | 给普通工具装乐高接口 |
| RunnablePassthrough | identity 函数 | 原样复印 |
| RunnableBranch | if-else / switch | 走迷宫选路 |
| stream() | Observable / Event Stream | 水龙头流水 |
| batch() | Promise.all + map | 批量生产 |
| with_config() | React Context | 贴标签 |

---

## 6. 【反直觉点】

### 误区1：Runnable 只是简单的函数包装 ❌

**为什么错？**
- Runnable 不仅仅是包装函数，它提供了**完整的执行能力矩阵**
- 实现 invoke 后自动获得 batch/stream/ainvoke/astream
- 支持配置传递、回调系统、类型推断

**为什么人们容易这样错？**
看到 `invoke(input) -> output`，很容易认为这只是给函数换了个名字。实际上 Runnable 是一套完整的执行框架。

**正确理解：**

```python
# ❌ 错误理解：只是换个名字
def my_func(x):
    return x * 2

# "这不就是 my_func(x) 吗？"

# ✅ 正确理解：获得了完整的执行能力
from langchain_core.runnables import RunnableLambda

runnable = RunnableLambda(my_func)

# 1. 同步单次
result = runnable.invoke(5)

# 2. 批量处理（自动并行优化）
results = runnable.batch([1, 2, 3, 4, 5])

# 3. 流式输出
for chunk in runnable.stream(5):
    print(chunk)

# 4. 异步执行
result = await runnable.ainvoke(5)

# 5. 配置和回调
result = runnable.invoke(5, config={
    "callbacks": [MyCallback()],
    "tags": ["production"]
})

# 6. 组合能力
chain = other | runnable | another
```

---

### 误区2：| 操作符只是语法糖，没什么特别的 ❌

**为什么错？**
- `|` 创建的 RunnableSequence 有智能的类型推断
- 流式执行时会自动优化（只有最后一个流式）
- 支持嵌套组合和分支逻辑

**为什么人们容易这样错？**
在 Unix shell 中 `|` 确实只是简单的管道，但 LangChain 的 `|` 更强大。

**正确理解：**

```python
# ❌ 错误理解：只是简单串联
# "prompt | llm 不就是先执行 prompt 再执行 llm 吗？"

# ✅ 正确理解：智能的序列组合

# 1. 类型推断
chain = prompt | llm | parser
# chain.input_schema 来自 prompt
# chain.output_schema 来自 parser

# 2. 流式优化
for chunk in chain.stream(input):
    # prompt 和 parser 不产生流
    # 只有 llm 产生流式输出
    print(chunk)

# 3. 嵌套组合
chain = (
    RunnableParallel(
        context=retriever,
        question=RunnablePassthrough()
    )  # 并行
    | prompt  # 然后串行
    | llm
    | parser
)

# 4. 错误传播和配置传递
# 配置会自动传递给所有组件
# 错误会带上完整的执行路径信息
```

---

### 误区3：batch 就是循环调用 invoke ❌

**为什么错？**
- 默认实现确实是循环，但可以被重写为并行执行
- 很多组件有优化的 batch 实现（如 LLM 的批量 API 调用）
- 异步版本 `abatch` 会自动并发执行

**为什么人们容易这样错？**
看到默认实现是 `[self.invoke(x) for x in inputs]`，就以为永远是这样。

**正确理解：**

```python
# ❌ 错误理解：batch 没有性能优势
# "batch([1,2,3]) 不就是 [invoke(1), invoke(2), invoke(3)] 吗？"

# ✅ 正确理解：batch 可以被优化

# 1. LLM 的 batch 会合并 API 调用
llm = ChatOpenAI()
results = llm.batch(["Hello", "Hi", "Hey"])
# 可能只发送一次 API 请求（取决于实现）

# 2. 异步 batch 自动并发
results = await chain.abatch(inputs, config={"max_concurrency": 10})
# 最多同时执行 10 个

# 3. 自定义 batch 优化
class OptimizedRunnable(Runnable):
    def batch(self, inputs, config=None):
        # 批量查询数据库，而不是逐个查询
        return self.db.bulk_query(inputs)
```

---

## 7. 【实战代码】

```python
"""
示例：深入理解 Runnable 协议
演示 Runnable 的核心概念和实际应用
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Iterator, Optional, TypeVar, Generic, Callable
import asyncio

# ===== 1. 实现自定义 Runnable =====
print("=== 1. 自定义 Runnable ===")

Input = TypeVar("Input")
Output = TypeVar("Output")

class Runnable(ABC, Generic[Input, Output]):
    """简化版 Runnable 基类"""

    @abstractmethod
    def invoke(self, input: Input, config: Optional[Dict] = None) -> Output:
        pass

    def batch(self, inputs: List[Input], config: Optional[Dict] = None) -> List[Output]:
        return [self.invoke(x, config) for x in inputs]

    def stream(self, input: Input, config: Optional[Dict] = None) -> Iterator[Output]:
        yield self.invoke(input, config)

    def __or__(self, other: "Runnable") -> "RunnableSequence":
        return RunnableSequence(first=self, last=other)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

class RunnableSequence(Runnable[Input, Output]):
    """Runnable 序列"""

    def __init__(self, first: Runnable, last: Runnable, middle: List[Runnable] = None):
        self.first = first
        self.middle = middle or []
        self.last = last

    def invoke(self, input: Input, config: Optional[Dict] = None) -> Output:
        result = self.first.invoke(input, config)
        for step in self.middle:
            result = step.invoke(result, config)
        return self.last.invoke(result, config)

    def stream(self, input: Input, config: Optional[Dict] = None) -> Iterator[Output]:
        # 前面的步骤正常执行
        result = self.first.invoke(input, config)
        for step in self.middle:
            result = step.invoke(result, config)
        # 最后一步流式执行
        for chunk in self.last.stream(result, config):
            yield chunk

    def __or__(self, other: Runnable) -> "RunnableSequence":
        return RunnableSequence(
            first=self.first,
            middle=self.middle + [self.last],
            last=other
        )

    def __repr__(self) -> str:
        steps = [self.first] + self.middle + [self.last]
        return " | ".join(repr(s) for s in steps)

class RunnableLambda(Runnable[Input, Output]):
    """包装普通函数"""

    def __init__(self, func: Callable[[Input], Output]):
        self.func = func

    def invoke(self, input: Input, config: Optional[Dict] = None) -> Output:
        return self.func(input)

    def __repr__(self) -> str:
        return f"RunnableLambda({self.func.__name__})"

# 创建具体的 Runnable
class PromptTemplate(Runnable[Dict[str, Any], str]):
    """提示模板"""

    def __init__(self, template: str):
        self.template = template

    def invoke(self, input: Dict[str, Any], config: Optional[Dict] = None) -> str:
        return self.template.format(**input)

class FakeLLM(Runnable[str, str]):
    """模拟 LLM"""

    def __init__(self, prefix: str = "AI:"):
        self.prefix = prefix

    def invoke(self, input: str, config: Optional[Dict] = None) -> str:
        return f"{self.prefix} Response to '{input[:20]}...'"

    def stream(self, input: str, config: Optional[Dict] = None) -> Iterator[str]:
        response = self.invoke(input, config)
        # 模拟流式输出：逐字输出
        for char in response:
            yield char

class OutputParser(Runnable[str, Dict[str, Any]]):
    """输出解析器"""

    def invoke(self, input: str, config: Optional[Dict] = None) -> Dict[str, Any]:
        return {"text": input, "length": len(input)}

# 演示
prompt = PromptTemplate("Tell me about {topic} in {style} style")
llm = FakeLLM()
parser = OutputParser()

# 使用 | 组合
chain = prompt | llm | parser
print(f"Chain: {chain}")

result = chain.invoke({"topic": "Python", "style": "simple"})
print(f"Result: {result}")

# ===== 2. 流式输出 =====
print("\n=== 2. 流式输出 ===")

simple_chain = prompt | llm
print("Streaming: ", end="")
for chunk in simple_chain.stream({"topic": "AI", "style": "fun"}):
    print(chunk, end="", flush=True)
print()

# ===== 3. RunnableParallel =====
print("\n=== 3. RunnableParallel ===")

class RunnableParallel(Runnable[Input, Dict[str, Any]]):
    """并行执行多个 Runnable"""

    def __init__(self, **steps: Runnable):
        self.steps = steps

    def invoke(self, input: Input, config: Optional[Dict] = None) -> Dict[str, Any]:
        return {
            key: runnable.invoke(input, config)
            for key, runnable in self.steps.items()
        }

    def __repr__(self) -> str:
        return f"RunnableParallel({list(self.steps.keys())})"

class RunnablePassthrough(Runnable[Input, Input]):
    """透传输入"""

    def invoke(self, input: Input, config: Optional[Dict] = None) -> Input:
        return input

# RAG 风格的管道
def fake_retriever(query: Dict) -> str:
    return f"Retrieved docs for: {query.get('question', query)}"

rag_chain = (
    RunnableParallel(
        context=RunnableLambda(fake_retriever),
        question=RunnablePassthrough()
    )
    | RunnableLambda(lambda x: f"Context: {x['context']}\nQuestion: {x['question']}")
    | llm
)

print(f"RAG Chain: {rag_chain}")
result = rag_chain.invoke({"question": "What is LangChain?"})
print(f"Result: {result}")

# ===== 4. 批量处理 =====
print("\n=== 4. 批量处理 ===")

inputs = [
    {"topic": "Python", "style": "technical"},
    {"topic": "JavaScript", "style": "casual"},
    {"topic": "Rust", "style": "detailed"},
]

results = chain.batch(inputs)
for inp, res in zip(inputs, results):
    print(f"  {inp['topic']}: {res['length']} chars")

# ===== 5. RunnableBranch =====
print("\n=== 5. RunnableBranch ===")

class RunnableBranch(Runnable[Input, Output]):
    """条件分支"""

    def __init__(self, *branches, default: Runnable):
        self.branches = branches  # List of (condition, runnable)
        self.default = default

    def invoke(self, input: Input, config: Optional[Dict] = None) -> Output:
        for condition, runnable in self.branches:
            if condition(input):
                return runnable.invoke(input, config)
        return self.default.invoke(input, config)

# 创建分支
branch = RunnableBranch(
    (lambda x: "code" in x.get("question", "").lower(),
     RunnableLambda(lambda x: f"[CODE MODE] {x}")),
    (lambda x: "translate" in x.get("question", "").lower(),
     RunnableLambda(lambda x: f"[TRANSLATE MODE] {x}")),
    default=RunnableLambda(lambda x: f"[GENERAL MODE] {x}")
)

test_inputs = [
    {"question": "Write code for sorting"},
    {"question": "Translate hello to Chinese"},
    {"question": "What is the weather?"},
]

for inp in test_inputs:
    result = branch.invoke(inp)
    print(f"  {inp['question'][:25]}... -> {result[:30]}...")

# ===== 6. with_config 模式 =====
print("\n=== 6. 配置传递 ===")

class ConfigAwareRunnable(Runnable[str, str]):
    """配置感知的 Runnable"""

    def invoke(self, input: str, config: Optional[Dict] = None) -> str:
        config = config or {}
        tags = config.get("tags", [])
        run_name = config.get("run_name", "unnamed")
        return f"[{run_name}][tags:{tags}] Processed: {input}"

aware = ConfigAwareRunnable()

# 无配置
print(aware.invoke("Hello"))

# 有配置
print(aware.invoke("Hello", config={
    "run_name": "production",
    "tags": ["important", "user-123"]
}))

# ===== 7. 组合复杂管道 =====
print("\n=== 7. 复杂管道示例 ===")

# 模拟一个完整的 RAG 管道
def format_docs(docs: str) -> str:
    return f"<docs>{docs}</docs>"

def extract_answer(response: Dict) -> str:
    return response.get("text", "")[:50]

complex_chain = (
    # 第一步：并行获取 context 和保留 question
    RunnableParallel(
        context=RunnableLambda(fake_retriever) | RunnableLambda(format_docs),
        question=RunnablePassthrough()
    )
    # 第二步：格式化为 prompt
    | RunnableLambda(lambda x: f"Context: {x['context']}\n\nQuestion: {x['question']}\n\nAnswer:")
    # 第三步：调用 LLM
    | llm
    # 第四步：解析输出
    | parser
    # 第五步：提取答案
    | RunnableLambda(extract_answer)
)

print(f"Complex chain structure:")
print(f"  {complex_chain}")
print()

final_result = complex_chain.invoke({"question": "How does LangChain work?"})
print(f"Final answer: {final_result}")

print("\n=== 完成 ===")
```

**运行输出示例：**

```
=== 1. 自定义 Runnable ===
Chain: PromptTemplate() | FakeLLM() | OutputParser()
Result: {'text': "AI: Response to 'Tell me about Pytho...'", 'length': 42}

=== 2. 流式输出 ===
Streaming: AI: Response to 'Tell me about AI i...'

=== 3. RunnableParallel ===
RAG Chain: RunnableParallel(['context', 'question']) | RunnableLambda(<lambda>) | FakeLLM()
Result: AI: Response to 'Context: Retrieved...'

=== 4. 批量处理 ===
  Python: 45 chars
  JavaScript: 49 chars
  Rust: 45 chars

=== 5. RunnableBranch ===
  Write code for sorting... -> [CODE MODE] {'question': 'Wri...
  Translate hello to Chine... -> [TRANSLATE MODE] {'question...
  What is the weather?... -> [GENERAL MODE] {'question':...

=== 6. 配置传递 ===
[unnamed][tags:[]] Processed: Hello
[production][tags:['important', 'user-123']] Processed: Hello

=== 7. 复杂管道示例 ===
Complex chain structure:
  RunnableParallel(['context', 'question']) | RunnableLambda(<lambda>) | FakeLLM() | OutputParser() | RunnableLambda(extract_answer)

Final answer: AI: Response to 'Context: <docs>Ret

=== 完成 ===
```

---

## 8. 【面试必问】

### 问题："LangChain 的 Runnable 协议是什么？为什么要设计这个？"

**普通回答（❌ 不出彩）：**
"Runnable 是 LangChain 的基类，所有组件都继承它，可以用 invoke 方法调用。"

**出彩回答（✅ 推荐）：**

> **Runnable 协议有三个层面的意义：**
>
> 1. **统一接口层面**：
>    - 所有组件（Prompt、LLM、Parser、Retriever）都实现 `invoke(input) -> output`
>    - 这让组件可以像乐高积木一样随意组合
>    - 类似于 React 组件都要实现 `render()`
>
> 2. **执行能力层面**：
>    - 实现 `invoke` 后自动获得 `batch`、`stream`、`ainvoke`、`astream`
>    - 这是模板方法模式的典型应用
>    - 子类只需关注核心逻辑，执行框架由基类提供
>
> 3. **组合能力层面**：
>    - 通过 `__or__` 重载实现 `|` 操作符（LCEL 语法）
>    - `prompt | llm | parser` 创建 `RunnableSequence`
>    - 支持串行、并行、条件分支等复杂组合
>
> **为什么这样设计？**
> - **问题**：LLM 应用需要组合多种组件，每种组件 API 不同
> - **解决**：定义统一协议，让所有组件可互换、可组合
> - **好处**：用户可以用声明式语法（LCEL）构建复杂管道
>
> **实际例子：**
> ```python
> # 一行代码构建 RAG 管道
> rag = retriever | prompt | llm | parser
> ```

**为什么这个回答出彩？**
1. ✅ 分层次解释（接口、执行、组合）
2. ✅ 说明了设计动机和解决的问题
3. ✅ 联系了设计模式（模板方法）
4. ✅ 给出了简洁的代码示例

---

### 问题："LCEL 的 | 操作符是怎么实现的？"

**普通回答（❌ 不出彩）：**
"| 是 Python 的位或操作符，LangChain 重载了它来连接组件。"

**出彩回答（✅ 推荐）：**

> **LCEL 的 `|` 实现涉及三个要点：**
>
> 1. **Python 魔术方法**：
>    - `|` 操作符对应 `__or__` 方法
>    - `a | b` 实际调用 `a.__or__(b)`
>    - 如果 `a` 没有 `__or__`，会调用 `b.__ror__(a)`
>
> 2. **RunnableSequence 创建**：
>    ```python
>    class Runnable:
>        def __or__(self, other: Runnable) -> RunnableSequence:
>            return RunnableSequence(first=self, last=other)
>    ```
>    - `prompt | llm` 返回 `RunnableSequence(prompt, llm)`
>
> 3. **链式调用支持**：
>    - `RunnableSequence` 也是 `Runnable`
>    - `(prompt | llm) | parser` 仍然有效
>    - 通过重写 `__or__` 支持任意长度的链
>
> **执行流程：**
> ```python
> chain = prompt | llm | parser
> # 等价于
> chain = RunnableSequence(
>     first=prompt,
>     middle=[llm],
>     last=parser
> )
>
> # invoke 执行
> result = chain.invoke(input)
> # -> prompt.invoke(input)
> # -> llm.invoke(result1)
> # -> parser.invoke(result2)
> ```

---

## 9. 【化骨绵掌】

### 卡片1：Runnable 是什么 🎯

**一句话：** Runnable 是 LangChain 所有可执行组件的统一接口。

**举例：**
```python
from langchain_core.runnables import Runnable

# 所有这些都是 Runnable
prompt: Runnable      # 提示模板
llm: Runnable         # 语言模型
parser: Runnable      # 输出解析器
retriever: Runnable   # 检索器
chain: Runnable       # 链
```

**应用：** 只要是 Runnable，就能用 `invoke()` 调用，就能用 `|` 组合。

---

### 卡片2：invoke 核心方法 📐

**一句话：** `invoke(input) -> output` 是 Runnable 的核心执行方法。

**举例：**
```python
# 所有组件都用 invoke 调用
result = prompt.invoke({"topic": "AI"})
result = llm.invoke("Hello")
result = parser.invoke(text)
result = chain.invoke(input)
```

**应用：** 不管什么组件，只要知道输入格式，就能用 `invoke` 执行。

---

### 卡片3：| 操作符（LCEL）🔗

**一句话：** `|` 操作符将多个 Runnable 串联成管道。

**举例：**
```python
# 创建处理管道
chain = prompt | llm | parser

# 等价于
def chain(input):
    x = prompt.invoke(input)
    x = llm.invoke(x)
    return parser.invoke(x)
```

**应用：** 用 `|` 可以一行代码构建复杂的处理流程。

---

### 卡片4：RunnableSequence 🔄

**一句话：** RunnableSequence 是 `|` 操作符的返回值，表示串行执行序列。

**举例：**
```python
chain = prompt | llm  # 返回 RunnableSequence

# 内部结构
class RunnableSequence:
    first: Runnable   # prompt
    last: Runnable    # llm

    def invoke(self, input):
        x = self.first.invoke(input)
        return self.last.invoke(x)
```

**应用：** 理解 RunnableSequence 才能理解 LCEL 的工作原理。

---

### 卡片5：RunnableParallel 并行 🔀

**一句话：** RunnableParallel 并行执行多个 Runnable，结果合并为字典。

**举例：**
```python
from langchain_core.runnables import RunnableParallel

parallel = RunnableParallel(
    summary=summary_chain,
    keywords=keyword_chain,
)

result = parallel.invoke(text)
# result = {"summary": "...", "keywords": [...]}
```

**应用：** RAG 中常用来同时获取 context 和传递 question。

---

### 卡片6：RunnableLambda 包装 🔧

**一句话：** RunnableLambda 将普通函数包装成 Runnable。

**举例：**
```python
from langchain_core.runnables import RunnableLambda

def postprocess(text: str) -> str:
    return text.upper()

# 包装成 Runnable
wrapped = RunnableLambda(postprocess)

# 现在可以参与管道
chain = prompt | llm | wrapped
```

**应用：** 让任何自定义函数都能加入 LCEL 管道。

---

### 卡片7：stream 流式输出 🌊

**一句话：** `stream()` 方法实现流式输出，逐步返回结果。

**举例：**
```python
# 流式获取 LLM 响应
for chunk in chain.stream({"topic": "AI"}):
    print(chunk, end="", flush=True)
    # 逐字打印，不等完整响应
```

**应用：** 实时显示 LLM 输出，提升用户体验。

---

### 卡片8：batch 批量处理 📦

**一句话：** `batch()` 方法批量处理多个输入。

**举例：**
```python
inputs = [
    {"topic": "Python"},
    {"topic": "JavaScript"},
    {"topic": "Rust"},
]

# 批量处理
results = chain.batch(inputs)
# results = [result1, result2, result3]
```

**应用：** 批量处理数据，部分组件会优化为单次 API 调用。

---

### 卡片9：ainvoke 异步执行 ⚡

**一句话：** `ainvoke()` 是 `invoke()` 的异步版本。

**举例：**
```python
import asyncio

async def main():
    # 异步执行
    result = await chain.ainvoke({"topic": "AI"})

    # 异步并发批处理
    results = await chain.abatch(inputs)

    # 异步流式
    async for chunk in chain.astream(input):
        print(chunk)

asyncio.run(main())
```

**应用：** 在异步应用（如 FastAPI）中使用 LangChain。

---

### 卡片10：Runnable 生态全景 ⭐

**一句话：** 掌握 Runnable 协议就掌握了 LangChain 的核心。

**核心组件都是 Runnable：**
```python
# 全部实现 Runnable 协议
ChatPromptTemplate    # 提示模板
ChatOpenAI            # 聊天模型
StrOutputParser       # 输出解析器
VectorStoreRetriever  # 向量检索器
AgentExecutor         # Agent 执行器
```

**统一的使用方式：**
```python
# 所有组件都能这样用
component.invoke(input)
component.stream(input)
component.batch(inputs)
await component.ainvoke(input)

# 所有组件都能组合
chain = a | b | c
```

**应用：** 理解 Runnable 是阅读 LangChain 源码的金钥匙。

---

## 10. 【一句话总结】

**Runnable 是 LangChain 的核心协议，通过统一的 invoke/stream/batch 接口和 | 操作符组合能力，让所有组件可以像乐高积木一样自由组合，是 LCEL 表达式语言的基石。**

---

## 📚 学习检查清单

- [ ] 理解 Runnable 是什么以及为什么需要它
- [ ] 能够使用 invoke 调用各种 LangChain 组件
- [ ] 会使用 | 操作符组合组件
- [ ] 理解 RunnableSequence 的工作原理
- [ ] 会使用 RunnableParallel 并行执行
- [ ] 能用 RunnableLambda 包装自定义函数
- [ ] 会使用 stream 实现流式输出
- [ ] 理解 batch 和 ainvoke 的用途
- [ ] 能够阅读 LangChain 源码中的 Runnable 相关代码
- [ ] 能够实现自定义 Runnable

## 🔗 下一步学习

- **LCEL 表达式语言**：深入学习 LCEL 的高级用法
- **BaseChatModel 实现**：理解 LLM 如何实现 Runnable
- **Callback 回调系统**：理解执行过程中的事件处理

---

**版本：** v1.0
**最后更新：** 2025-12-12
