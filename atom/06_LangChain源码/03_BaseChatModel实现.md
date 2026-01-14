# BaseChatModel 实现

> 原子化知识点 | LangChain 源码 | 聊天模型基类实现

---

## 1. 【30字核心】

**BaseChatModel 是 LangChain 聊天模型的抽象基类，定义了消息输入输出和 _generate 核心方法，是所有 LLM 的统一接口。**

---

## 2. 【第一性原理】

### 什么是第一性原理？

**第一性原理**：回到事物最基本的真理，从源头思考问题

### BaseChatModel 的第一性原理 🎯

#### 1. 最基础的定义

**BaseChatModel = 消息列表 → LLM → AI消息**

仅此而已！没有更基础的了。

- **输入**：消息列表 `List[BaseMessage]`
- **处理**：调用底层 LLM API
- **输出**：AI 消息 `AIMessage`

```python
# BaseChatModel 的本质
def chat_model(messages: List[BaseMessage]) -> AIMessage:
    return call_llm_api(messages)
```

#### 2. 为什么需要 BaseChatModel？

**核心问题：如何统一不同 LLM 提供商的接口？**

```python
# 没有统一基类的困境
# OpenAI
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Hello"}]
)
result = response.choices[0].message.content

# Anthropic
from anthropic import Anthropic
client = Anthropic()
response = client.messages.create(
    model="claude-3-opus",
    messages=[{"role": "user", "content": "Hello"}]
)
result = response.content[0].text

# 问题：
# 1. 每个 API 调用方式不同
# 2. 响应格式不同
# 3. 消息格式不同
# 4. 难以切换模型
```

```python
# 有了 BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# 统一的调用方式
openai_llm = ChatOpenAI(model="gpt-4")
anthropic_llm = ChatAnthropic(model="claude-3-opus")

# 相同的 invoke 接口
result1 = openai_llm.invoke("Hello")
result2 = anthropic_llm.invoke("Hello")

# 优势：
# 1. 统一的调用方式
# 2. 统一的消息格式
# 3. 可以无缝切换模型
# 4. 自动获得 stream/batch/ainvoke
```

#### 3. BaseChatModel 的三层价值

##### 价值1：统一接口 - 屏蔽 API 差异

```python
# 不管是什么模型，都用相同方式调用
def process_with_llm(llm: BaseChatModel, query: str) -> str:
    return llm.invoke(query).content

# 可以传入任何实现
process_with_llm(ChatOpenAI(), "Hello")
process_with_llm(ChatAnthropic(), "Hello")
process_with_llm(ChatOllama(), "Hello")
```

##### 价值2：消息标准化 - 统一消息格式

```python
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 统一的消息类型
messages = [
    SystemMessage(content="You are helpful"),
    HumanMessage(content="Hello"),
    AIMessage(content="Hi there!"),
]

# 所有模型都接受这种格式
result = llm.invoke(messages)
```

##### 价值3：Runnable 集成 - 参与 LCEL 管道

```python
# BaseChatModel 实现了 Runnable 协议
chain = prompt | llm | parser

# 自动获得这些能力
chain.invoke(input)
chain.stream(input)
chain.batch(inputs)
await chain.ainvoke(input)
```

#### 4. 从第一性原理推导 BaseChatModel 架构

**推理链：**

```
1. LLM 应用需要调用各种语言模型
   ↓
2. 不同模型有不同的 API 和消息格式
   ↓
3. 需要一个统一的抽象层
   ↓
4. 定义 BaseChatModel 抽象基类
   ↓
5. 抽象方法 _generate：子类实现具体 API 调用
   ↓
6. 公开方法 invoke：处理输入转换和结果包装
   ↓
7. 继承 Runnable：自动获得组合能力
   ↓
8. 各模型提供商实现具体子类
```

#### 5. 一句话总结第一性原理

**BaseChatModel 是"消息→AI响应"的统一抽象，通过模板方法模式让不同 LLM 提供统一接口，同时集成 Runnable 协议参与 LCEL 管道。**

---

## 3. 【核心概念（全面覆盖）】

### 核心概念1：BaseChatModel 类层次 🏗️

**BaseChatModel 继承自 BaseLanguageModel 和 Runnable**

```python
from abc import ABC, abstractmethod
from typing import List, Optional, Any, Iterator, AsyncIterator
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.runnables import Runnable

class BaseChatModel(BaseLanguageModel, Runnable[LanguageModelInput, BaseMessage]):
    """聊天模型抽象基类

    继承关系：
    - BaseLanguageModel：语言模型的基础能力
    - Runnable：LCEL 组合能力

    子类需要实现：
    - _generate：核心生成方法
    - _llm_type：模型类型标识
    """

    # ===== 抽象方法：子类必须实现 =====

    @abstractmethod
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManager] = None,
        **kwargs
    ) -> ChatResult:
        """核心生成方法：子类实现具体的 API 调用"""
        pass

    @property
    @abstractmethod
    def _llm_type(self) -> str:
        """模型类型标识"""
        pass

    # ===== 公开接口方法 =====

    def invoke(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        **kwargs
    ) -> BaseMessage:
        """同步调用（Runnable 接口）"""
        # 1. 转换输入为消息列表
        messages = self._convert_input(input)

        # 2. 调用 _generate
        result = self._generate(messages, **kwargs)

        # 3. 返回第一个生成结果
        return result.generations[0].message

    def stream(
        self,
        input: LanguageModelInput,
        config: Optional[RunnableConfig] = None,
        **kwargs
    ) -> Iterator[BaseMessageChunk]:
        """流式输出"""
        messages = self._convert_input(input)

        # 调用 _stream 方法
        for chunk in self._stream(messages, **kwargs):
            yield chunk
```

**类层次图：**

```
Runnable[Input, Output]
    ↑
BaseLanguageModel
    ↑
BaseChatModel
    ↑
├── ChatOpenAI
├── ChatAnthropic
├── ChatOllama
├── ChatGoogleGenerativeAI
└── ...更多实现
```

---

### 核心概念2：_generate 核心方法 📐

**_generate 是模板方法模式的核心，子类实现具体 API 调用**

```python
from langchain_core.outputs import ChatResult, ChatGeneration

class ChatResult:
    """聊天结果"""
    generations: List[ChatGeneration]  # 生成的消息列表
    llm_output: Optional[dict] = None  # LLM 额外输出（token 使用量等）

class ChatGeneration:
    """单个生成结果"""
    message: BaseMessage  # 生成的消息
    generation_info: Optional[dict] = None  # 生成信息

# ChatOpenAI 的 _generate 实现（简化版）
class ChatOpenAI(BaseChatModel):
    model: str = "gpt-4"
    temperature: float = 0.7
    client: Any = None  # OpenAI client

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> ChatResult:
        # 1. 转换消息格式
        openai_messages = self._convert_messages(messages)

        # 2. 调用 OpenAI API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,
            temperature=self.temperature,
            stop=stop,
            **kwargs
        )

        # 3. 转换响应为 ChatResult
        return self._create_chat_result(response)

    def _convert_messages(self, messages: List[BaseMessage]) -> List[dict]:
        """将 LangChain 消息转换为 OpenAI 格式"""
        result = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                result.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                result.append({"role": "system", "content": msg.content})
        return result

    def _create_chat_result(self, response) -> ChatResult:
        """将 OpenAI 响应转换为 ChatResult"""
        generations = []
        for choice in response.choices:
            message = AIMessage(content=choice.message.content)
            generations.append(ChatGeneration(message=message))

        return ChatResult(
            generations=generations,
            llm_output={
                "token_usage": response.usage.model_dump(),
                "model": response.model,
            }
        )

    @property
    def _llm_type(self) -> str:
        return "openai-chat"
```

---

### 核心概念3：消息类型系统 💬

**LangChain 定义了统一的消息类型**

```python
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
    FunctionMessage,
)

# BaseMessage 基类
class BaseMessage:
    """消息基类"""
    content: str                          # 消息内容
    type: str                             # 消息类型
    additional_kwargs: dict = {}          # 额外参数
    response_metadata: dict = {}          # 响应元数据

# 具体消息类型
class HumanMessage(BaseMessage):
    """用户消息"""
    type: str = "human"

class AIMessage(BaseMessage):
    """AI 消息"""
    type: str = "ai"
    tool_calls: List[ToolCall] = []       # 工具调用

class SystemMessage(BaseMessage):
    """系统消息"""
    type: str = "system"

class ToolMessage(BaseMessage):
    """工具返回消息"""
    type: str = "tool"
    tool_call_id: str                     # 对应的工具调用 ID

# 使用示例
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is 2+2?"),
    AIMessage(content="2+2 equals 4."),
    HumanMessage(content="Thanks!"),
]

result = llm.invoke(messages)
# result 是 AIMessage
```

**消息类型对照表：**

| LangChain 类型 | OpenAI role | Anthropic role |
|---------------|-------------|----------------|
| SystemMessage | system | system |
| HumanMessage | user | user |
| AIMessage | assistant | assistant |
| ToolMessage | tool | tool_result |

---

### 核心概念4：流式输出 _stream 🌊

**流式输出逐块返回 LLM 响应**

```python
from langchain_core.messages import AIMessageChunk

class AIMessageChunk(BaseMessageChunk):
    """AI 消息块：流式输出的单个片段"""
    type: str = "AIMessageChunk"

class BaseChatModel:
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> Iterator[ChatGenerationChunk]:
        """流式生成（子类可重写）

        默认实现：调用 _generate 然后一次性返回
        优化实现：真正的流式 API 调用
        """
        # 默认实现（非流式）
        result = self._generate(messages, stop=stop, **kwargs)
        yield ChatGenerationChunk(
            message=AIMessageChunk(content=result.generations[0].message.content)
        )

# ChatOpenAI 的流式实现
class ChatOpenAI(BaseChatModel):
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> Iterator[ChatGenerationChunk]:
        # 调用 OpenAI 流式 API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._convert_messages(messages),
            stream=True,  # 启用流式
            **kwargs
        )

        # 逐块返回
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=chunk.choices[0].delta.content
                    )
                )

# 使用流式输出
for chunk in llm.stream("Tell me a story"):
    print(chunk.content, end="", flush=True)
```

---

### 核心概念5：Callback 回调集成 📞

**BaseChatModel 集成回调系统追踪执行过程**

```python
from langchain_core.callbacks import CallbackManager, BaseCallbackHandler

class BaseChatModel:
    callbacks: Optional[List[BaseCallbackHandler]] = None

    def invoke(self, input, config=None, **kwargs):
        # 获取回调管理器
        callback_manager = CallbackManager.configure(
            inheritable_callbacks=self.callbacks,
            local_callbacks=config.get("callbacks") if config else None,
        )

        # 开始运行回调
        run_manager = callback_manager.on_chat_model_start(
            serialized=self._serialized,
            messages=messages,
        )

        try:
            # 执行生成
            result = self._generate(messages, run_manager=run_manager, **kwargs)

            # 成功回调
            run_manager.on_llm_end(result)
            return result.generations[0].message
        except Exception as e:
            # 错误回调
            run_manager.on_llm_error(e)
            raise

# 自定义回调处理器
class MyCallbackHandler(BaseCallbackHandler):
    def on_chat_model_start(self, serialized, messages, **kwargs):
        print(f"Starting LLM call with {len(messages)} messages")

    def on_llm_end(self, response, **kwargs):
        print(f"LLM finished, generated {len(response.generations)} responses")

    def on_llm_error(self, error, **kwargs):
        print(f"LLM error: {error}")

# 使用回调
llm = ChatOpenAI(callbacks=[MyCallbackHandler()])
result = llm.invoke("Hello")
```

---

### 核心概念6：bind_tools 工具绑定 🔧

**bind_tools 让模型可以调用工具**

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    return f"{city}: 晴天，25°C"

@tool
def search(query: str) -> str:
    """搜索信息"""
    return f"搜索结果：{query}"

# 绑定工具到模型
llm = ChatOpenAI()
llm_with_tools = llm.bind_tools([get_weather, search])

# 调用时模型可能返回工具调用
result = llm_with_tools.invoke("北京今天天气怎么样？")

# 检查是否有工具调用
if result.tool_calls:
    for tool_call in result.tool_calls:
        print(f"Tool: {tool_call['name']}")
        print(f"Args: {tool_call['args']}")

# bind_tools 的实现
class BaseChatModel:
    def bind_tools(
        self,
        tools: List[BaseTool],
        **kwargs
    ) -> "BaseChatModel":
        """绑定工具到模型"""
        # 将工具转换为模型需要的格式
        formatted_tools = self._convert_tools(tools)
        # 返回绑定了工具的新模型
        return self.bind(tools=formatted_tools, **kwargs)
```

---

### 扩展概念7：with_structured_output 结构化输出 📋

```python
from pydantic import BaseModel, Field

class Person(BaseModel):
    """人员信息"""
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")
    hobbies: List[str] = Field(description="爱好列表")

# 获取结构化输出
llm = ChatOpenAI()
structured_llm = llm.with_structured_output(Person)

result = structured_llm.invoke("介绍一下张三，他今年25岁，喜欢编程和读书")
# result 是 Person 对象
print(result.name)     # "张三"
print(result.age)      # 25
print(result.hobbies)  # ["编程", "读书"]
```

---

## 4. 【最小可用】

掌握以下内容，就能使用和理解 BaseChatModel：

### 4.1 基本调用

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# 字符串输入
result = llm.invoke("Hello")
print(result.content)

# 消息列表输入
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You are helpful"),
    HumanMessage(content="Hello"),
]
result = llm.invoke(messages)
```

### 4.2 流式输出

```python
for chunk in llm.stream("Tell me a story"):
    print(chunk.content, end="", flush=True)
```

### 4.3 绑定工具

```python
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """搜索信息"""
    return f"Result for: {query}"

llm_with_tools = llm.bind_tools([search])
result = llm_with_tools.invoke("Search for Python tutorials")
```

### 4.4 结构化输出

```python
from pydantic import BaseModel

class Answer(BaseModel):
    answer: str
    confidence: float

structured_llm = llm.with_structured_output(Answer)
result = structured_llm.invoke("What is 2+2?")
```

**这些知识足以：**
- 使用任何 LangChain 支持的聊天模型
- 实现流式输出提升用户体验
- 让模型调用工具（Function Calling）
- 获取结构化的 JSON 输出

---

## 5. 【1个类比】（双轨制）

### 类比1：BaseChatModel 抽象基类

#### 🎨 前端视角：React 组件基类

```typescript
// React：所有组件继承 Component
abstract class Component<Props, State> {
  abstract render(): ReactNode;  // 子类必须实现
  setState(state: State): void;  // 基类提供
}

class MyComponent extends Component {
  render() {  // 具体实现
    return <div>Hello</div>;
  }
}
```

```python
# LangChain：所有聊天模型继承 BaseChatModel
class BaseChatModel(ABC):
    @abstractmethod
    def _generate(self, messages): pass  # 子类必须实现

    def invoke(self, input):  # 基类提供
        messages = self._convert_input(input)
        return self._generate(messages)

class ChatOpenAI(BaseChatModel):
    def _generate(self, messages):  # 具体实现
        return self.client.chat.completions.create(...)
```

#### 🧒 小朋友视角：餐厅的标准菜谱

```
BaseChatModel 就像餐厅的标准菜谱模板：

标准菜谱（BaseChatModel）说：
1. 接收顾客点的菜（messages）
2. 按照某种方式做菜（_generate）
3. 把菜端给顾客（返回结果）

不同餐厅（子类）的做法不同：
- 中餐厅（ChatOpenAI）：用炒锅做
- 西餐厅（ChatAnthropic）：用烤箱做
- 日本料理（ChatOllama）：用寿司手法

但对顾客来说：
"我要一份宫保鸡丁"（invoke）
不管哪个餐厅，都是同样的点菜方式！
```

---

### 类比2：消息类型系统

#### 🎨 前端视角：TypeScript Union Types

```typescript
// TypeScript：消息类型联合
type Message =
  | { type: "user"; content: string }
  | { type: "assistant"; content: string }
  | { type: "system"; content: string };

function processMessage(msg: Message) {
  switch (msg.type) {
    case "user": ...
    case "assistant": ...
    case "system": ...
  }
}
```

#### 🧒 小朋友视角：对话中的角色

```
消息类型就像对话中的不同角色：

SystemMessage = 老师（设定规则）
"你要认真听讲，回答问题要完整"

HumanMessage = 学生（提问）
"老师，1+1 等于几？"

AIMessage = AI 助手（回答）
"1+1 等于 2"

ToolMessage = 小助手（查资料后报告）
"我查了一下，答案是 2"
```

---

### 类比总结表

| BaseChatModel 概念 | 前端类比 | 小朋友类比 |
|-------------------|---------|-----------|
| BaseChatModel | Component 基类 | 标准菜谱模板 |
| _generate | abstract render() | 做菜的具体方法 |
| invoke | 调用组件 | 点菜 |
| ChatOpenAI | 具体组件实现 | 中餐厅 |
| messages | props | 顾客的要求 |
| AIMessage | 返回的 ReactNode | 端上来的菜 |
| stream | Progressive Rendering | 一道道上菜 |
| bind_tools | 添加事件处理器 | 加配菜选项 |

---

## 6. 【反直觉点】

### 误区1：BaseChatModel 只是 API 包装 ❌

**为什么错？**
- BaseChatModel 提供了完整的执行框架
- 集成了回调系统、配置传递、错误处理
- 实现了 Runnable 协议，可参与 LCEL 管道

**正确理解：**
```python
# 不只是包装 API
llm = ChatOpenAI()

# 自动获得这些能力
llm.invoke(input)           # 同步
llm.stream(input)           # 流式
llm.batch(inputs)           # 批量
await llm.ainvoke(input)    # 异步
llm.with_config(...)        # 配置
llm.bind_tools(tools)       # 工具绑定

# 参与 LCEL 管道
chain = prompt | llm | parser
```

---

### 误区2：所有模型的消息格式相同 ❌

**为什么错？**
- 不同 LLM 提供商的消息格式不同
- BaseChatModel 负责格式转换
- LangChain 消息是中间层抽象

**正确理解：**
```python
# LangChain 统一格式
messages = [HumanMessage(content="Hello")]

# OpenAI 格式
[{"role": "user", "content": "Hello"}]

# Anthropic 格式
[{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]

# BaseChatModel 自动转换
result = llm.invoke(messages)  # 内部会转换格式
```

---

### 误区3：stream 和 invoke 完全不同 ❌

**为什么错？**
- stream 最终结果和 invoke 相同
- stream 只是分块返回
- 可以合并 stream 结果

**正确理解：**
```python
# invoke 一次性返回
result = llm.invoke("Hello")
print(result.content)

# stream 分块返回，但最终内容相同
chunks = list(llm.stream("Hello"))
full_content = "".join(chunk.content for chunk in chunks)
# full_content == result.content
```

---

## 7. 【实战代码】

```python
"""
示例：实现简化版 BaseChatModel
演示聊天模型的核心架构
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Iterator, Any, Dict
from dataclasses import dataclass

# ===== 1. 消息类型 =====
print("=== 1. 消息类型系统 ===")

@dataclass
class BaseMessage:
    content: str
    type: str = "base"

@dataclass
class HumanMessage(BaseMessage):
    type: str = "human"

@dataclass
class AIMessage(BaseMessage):
    type: str = "ai"

@dataclass
class SystemMessage(BaseMessage):
    type: str = "system"

@dataclass
class AIMessageChunk(BaseMessage):
    type: str = "ai_chunk"

# ===== 2. ChatResult =====
@dataclass
class ChatGeneration:
    message: BaseMessage

@dataclass
class ChatResult:
    generations: List[ChatGeneration]
    llm_output: Optional[Dict] = None

# ===== 3. BaseChatModel =====
print("\n=== 2. BaseChatModel 基类 ===")

class BaseChatModel(ABC):
    """聊天模型抽象基类"""

    @abstractmethod
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> ChatResult:
        """子类必须实现的核心方法"""
        pass

    @property
    @abstractmethod
    def _llm_type(self) -> str:
        """模型类型标识"""
        pass

    def invoke(self, input: Any, **kwargs) -> BaseMessage:
        """统一调用接口"""
        messages = self._convert_input(input)
        result = self._generate(messages, **kwargs)
        return result.generations[0].message

    def stream(self, input: Any, **kwargs) -> Iterator[AIMessageChunk]:
        """流式输出"""
        messages = self._convert_input(input)
        for chunk in self._stream(messages, **kwargs):
            yield chunk

    def _stream(
        self,
        messages: List[BaseMessage],
        **kwargs
    ) -> Iterator[AIMessageChunk]:
        """默认流式实现（子类可重写）"""
        result = self._generate(messages, **kwargs)
        content = result.generations[0].message.content
        for char in content:
            yield AIMessageChunk(content=char)

    def _convert_input(self, input: Any) -> List[BaseMessage]:
        """输入转换"""
        if isinstance(input, str):
            return [HumanMessage(content=input)]
        elif isinstance(input, list):
            return input
        elif isinstance(input, BaseMessage):
            return [input]
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")

    def batch(self, inputs: List[Any], **kwargs) -> List[BaseMessage]:
        """批量处理"""
        return [self.invoke(inp, **kwargs) for inp in inputs]

# ===== 4. 具体实现 =====
print("\n=== 3. 具体模型实现 ===")

class FakeChatOpenAI(BaseChatModel):
    """模拟 ChatOpenAI"""

    model: str = "gpt-4"
    temperature: float = 0.7

    def __init__(self, model: str = "gpt-4", temperature: float = 0.7):
        self.model = model
        self.temperature = temperature

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> ChatResult:
        # 模拟 API 调用
        last_message = messages[-1].content
        response_content = f"[{self.model}] Response to: {last_message[:30]}..."

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=response_content))],
            llm_output={"model": self.model, "tokens": len(response_content)}
        )

    @property
    def _llm_type(self) -> str:
        return "fake-openai"

class FakeChatAnthropic(BaseChatModel):
    """模拟 ChatAnthropic"""

    model: str = "claude-3"

    def __init__(self, model: str = "claude-3"):
        self.model = model

    def _generate(
        self,
        messages: List[BaseMessage],
        **kwargs
    ) -> ChatResult:
        last_message = messages[-1].content
        response_content = f"[Claude] I understand: {last_message[:30]}..."

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=response_content))]
        )

    @property
    def _llm_type(self) -> str:
        return "fake-anthropic"

# ===== 5. 使用示例 =====
print("\n=== 4. 统一接口调用 ===")

def process_query(llm: BaseChatModel, query: str) -> str:
    """统一处理函数：不关心具体模型"""
    return llm.invoke(query).content

# 使用不同模型
openai_llm = FakeChatOpenAI(model="gpt-4")
anthropic_llm = FakeChatAnthropic(model="claude-3-opus")

print(f"OpenAI: {process_query(openai_llm, 'Hello')}")
print(f"Anthropic: {process_query(anthropic_llm, 'Hello')}")

# ===== 6. 消息列表输入 =====
print("\n=== 5. 消息列表输入 ===")

messages = [
    SystemMessage(content="You are helpful"),
    HumanMessage(content="What is Python?"),
]

result = openai_llm.invoke(messages)
print(f"Result: {result.content}")

# ===== 7. 流式输出 =====
print("\n=== 6. 流式输出 ===")

print("Streaming: ", end="")
for chunk in openai_llm.stream("Tell me about AI"):
    print(chunk.content, end="", flush=True)
print()

# ===== 8. 批量处理 =====
print("\n=== 7. 批量处理 ===")

queries = ["Hello", "What is Python?", "How are you?"]
results = openai_llm.batch(queries)
for query, result in zip(queries, results):
    print(f"  Q: {query[:20]}... -> A: {result.content[:30]}...")

print("\n=== 完成 ===")
```

---

## 8. 【面试必问】

### 问题："LangChain 的 BaseChatModel 是什么？为什么这样设计？"

**普通回答（❌ 不出彩）：**
"BaseChatModel 是聊天模型的基类，所有模型都继承它。"

**出彩回答（✅ 推荐）：**

> **BaseChatModel 有三个设计目标：**
>
> 1. **统一接口**：屏蔽不同 LLM 提供商的 API 差异
>    - OpenAI、Anthropic、Ollama 等都有不同的 API
>    - BaseChatModel 提供统一的 invoke/stream 方法
>
> 2. **模板方法模式**：
>    - 抽象方法 `_generate`：子类实现具体 API 调用
>    - 公开方法 `invoke`：处理输入转换、回调、错误处理
>
> 3. **Runnable 集成**：
>    - 继承 Runnable 协议，可参与 LCEL 管道
>    - 自动获得 batch/stream/ainvoke 等能力
>
> **实际例子**：
> ```python
> # 可以无缝切换模型
> chain = prompt | llm | parser
> # llm 可以是 ChatOpenAI、ChatAnthropic 等任何实现
> ```

---

## 9. 【化骨绵掌】

### 卡片1：BaseChatModel 是什么 🎯

**一句话：** BaseChatModel 是所有聊天模型的抽象基类。

**核心方法：**
- `_generate`：子类必须实现
- `invoke`：统一调用接口

**应用：** ChatOpenAI、ChatAnthropic 都继承自它。

---

### 卡片2：_generate 核心方法 📐

**一句话：** _generate 是子类必须实现的核心生成方法。

**签名：**
```python
def _generate(self, messages, stop, **kwargs) -> ChatResult
```

**应用：** 子类在这里调用具体的 LLM API。

---

### 卡片3：消息类型 💬

**一句话：** LangChain 定义了统一的消息类型。

**类型：**
- `HumanMessage`：用户消息
- `AIMessage`：AI 消息
- `SystemMessage`：系统消息

**应用：** 所有模型使用相同的消息格式。

---

### 卡片4：invoke 方法 🔧

**一句话：** invoke 是统一的调用接口。

**流程：**
1. 转换输入为消息列表
2. 调用 _generate
3. 返回 AI 消息

**应用：** `llm.invoke("Hello")` 或 `llm.invoke(messages)`

---

### 卡片5：stream 流式输出 🌊

**一句话：** stream 逐块返回 LLM 响应。

**用法：**
```python
for chunk in llm.stream("Hello"):
    print(chunk.content, end="")
```

**应用：** 实时显示 AI 响应，提升用户体验。

---

### 卡片6：bind_tools 工具绑定 🔧

**一句话：** bind_tools 让模型可以调用工具。

**用法：**
```python
llm_with_tools = llm.bind_tools([search, calculator])
```

**应用：** 实现 Function Calling / Tool Use。

---

### 卡片7：with_structured_output 📋

**一句话：** 获取结构化的 JSON 输出。

**用法：**
```python
structured_llm = llm.with_structured_output(Person)
result = structured_llm.invoke("...")  # 返回 Person 对象
```

**应用：** 从 LLM 输出中提取结构化数据。

---

### 卡片8：Callback 回调 📞

**一句话：** BaseChatModel 集成回调系统追踪执行。

**事件：**
- `on_chat_model_start`
- `on_llm_end`
- `on_llm_error`

**应用：** 监控、日志、追踪 LLM 调用。

---

### 卡片9：Runnable 集成 🔗

**一句话：** BaseChatModel 实现 Runnable 协议。

**能力：**
- 可以用 `|` 组合
- 自动获得 batch/ainvoke

**应用：** `chain = prompt | llm | parser`

---

### 卡片10：模型切换 ⭐

**一句话：** 统一接口让模型切换变得简单。

**示例：**
```python
# 只需改一行
llm = ChatOpenAI()  # 切换为
llm = ChatAnthropic()

# 其他代码不变
chain = prompt | llm | parser
```

**应用：** 灵活选择最适合的模型。

---

## 10. 【一句话总结】

**BaseChatModel 是 LangChain 聊天模型的抽象基类，通过模板方法模式统一不同 LLM 的接口，集成 Runnable 协议参与 LCEL 管道，是构建 LLM 应用的核心组件。**

---

## 📚 学习检查清单

- [ ] 理解 BaseChatModel 的设计目的
- [ ] 会使用 invoke 调用聊天模型
- [ ] 理解 _generate 抽象方法的作用
- [ ] 掌握消息类型系统（Human/AI/System）
- [ ] 会使用 stream 实现流式输出
- [ ] 会使用 bind_tools 绑定工具
- [ ] 会使用 with_structured_output 获取结构化输出
- [ ] 理解 BaseChatModel 与 Runnable 的关系

## 🔗 下一步学习

- **Agent 执行引擎**：理解 Agent 如何使用 ChatModel
- **Callback 回调系统**：深入理解执行追踪机制
- **Tool Use**：学习如何实现工具调用

---

**版本：** v1.0
**最后更新：** 2025-12-12
