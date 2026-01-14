# 流式输出 Streaming

> 原子化知识点 | LLM领域知识 | LangChain 源码学习前置知识

---

## 1. 【30字核心】

**流式输出让 LLM 逐 Token 返回响应，实现打字机效果，显著提升用户体验，是 LangChain 异步编程的核心应用。**

---

## 2. 【第一性原理】

### 什么是第一性原理？

**第一性原理**：回到事物最基本的真理，从源头思考问题

### 流式输出的第一性原理 🎯

#### 1. 最基础的定义

**流式输出 = 数据生成时就立即发送，而不是等全部生成完再发送**

仅此而已！没有更基础的了。

```python
# 非流式：等待全部生成
response = llm.invoke("写一篇文章")  # 等 10 秒，一次性返回完整文章

# 流式：边生成边发送
for chunk in llm.stream("写一篇文章"):  # 立即开始，每 0.1 秒返回几个字
    print(chunk, end="")
```

#### 2. 为什么需要流式输出？

**核心问题：LLM 生成响应需要时间，用户等待体验差**

```python
# 非流式的用户体验
# 用户：发送问题
# [等待 5 秒...什么都看不到...]
# [等待 10 秒...还是什么都看不到...]
# [突然] 完整响应一次性出现

# 流式的用户体验
# 用户：发送问题
# [0.2秒后] "让"
# [0.3秒后] "让我"
# [0.4秒后] "让我来"
# [0.5秒后] "让我来解"
# ... 像打字机一样逐字出现
```

#### 3. 流式输出的三层价值

##### 价值1：感知延迟降低

```python
# 同样生成 1000 字
# 非流式：用户感知延迟 = 10 秒（等待全部生成）
# 流式：用户感知延迟 = 0.2 秒（第一个字出现的时间）

# 用户心理：
# 非流式："卡住了？出错了？"
# 流式："正在思考，正在回答..."
```

##### 价值2：早期错误检测

```python
# 流式可以在生成过程中检测问题
for chunk in llm.stream(prompt):
    if is_toxic(chunk):  # 发现有害内容
        break  # 立即停止，不等到生成完
    yield chunk
```

##### 价值3：实时交互能力

```python
# 流式使实时应用成为可能
async def chat_with_streaming(websocket, message):
    async for chunk in llm.astream(message):
        await websocket.send(chunk)  # 实时推送给前端
```

#### 4. 从第一性原理推导 LangChain 应用

**推理链：**

```
1. LLM 按 Token 逐个生成输出
   ↓
2. 可以在生成每个 Token 后立即返回
   ↓
3. 用户可以实时看到"思考过程"
   ↓
4. 需要异步编程支持（async/await）
   ↓
5. 需要标准的流式接口（stream/astream）
   ↓
6. LangChain 的 Runnable 协议统一了流式接口
   ↓
7. 所有 Chain、Agent 都支持 stream() 方法
```

#### 5. 一句话总结第一性原理

**流式输出的本质是"边生产边消费"的数据流模式，它将 LLM 的逐 Token 生成特性暴露给用户，大幅提升交互体验，是现代 LLM 应用的标配能力。**

---

## 3. 【核心概念（全面覆盖）】

### 核心概念1：同步流式 stream() 🌊

**stream() 方法返回一个生成器，同步地逐个产出响应片段**

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 创建 LLM
llm = ChatOpenAI(model="gpt-4", streaming=True)

# 同步流式调用
for chunk in llm.stream([HumanMessage(content="写一首关于编程的短诗")]):
    print(chunk.content, end="", flush=True)
```

**chunk 的结构：**

```python
# 每个 chunk 是一个 AIMessageChunk
chunk = AIMessageChunk(
    content="让",  # 本次返回的文本片段
    additional_kwargs={},
    response_metadata={
        "finish_reason": None  # 生成未完成
    }
)

# 最后一个 chunk
final_chunk = AIMessageChunk(
    content="",
    response_metadata={
        "finish_reason": "stop"  # 生成完成
    }
)
```

**在 LangChain 源码中的应用：**

```python
# langchain_core/runnables/base.py
class Runnable(Generic[Input, Output], ABC):
    """Runnable 基类定义了 stream 接口"""

    def stream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs
    ) -> Iterator[Output]:
        """同步流式输出"""
        yield self.invoke(input, config, **kwargs)

    # 子类可以重写实现真正的流式
```

---

### 核心概念2：异步流式 astream() ⚡

**astream() 是 stream() 的异步版本，返回 AsyncIterator**

```python
import asyncio
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")

async def stream_response():
    """异步流式输出"""
    async for chunk in llm.astream("解释什么是异步编程"):
        print(chunk.content, end="", flush=True)

# 运行异步函数
asyncio.run(stream_response())
```

**异步流式的优势：**

```python
import asyncio
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")

async def handle_multiple_users():
    """同时处理多个用户的流式请求"""

    async def user_stream(user_id: str, message: str):
        print(f"\n[User {user_id}] ", end="")
        async for chunk in llm.astream(message):
            print(chunk.content, end="", flush=True)

    # 并发处理多个用户
    await asyncio.gather(
        user_stream("1", "什么是 Python？"),
        user_stream("2", "什么是 JavaScript？"),
        user_stream("3", "什么是 Rust？"),
    )

# 三个流式响应同时进行！
asyncio.run(handle_multiple_users())
```

**在 LangChain 源码中的应用：**

```python
# langchain_core/runnables/base.py
class Runnable(Generic[Input, Output], ABC):

    async def astream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
        **kwargs
    ) -> AsyncIterator[Output]:
        """异步流式输出"""
        yield await self.ainvoke(input, config, **kwargs)
```

---

### 核心概念3：流式事件 astream_events() 📡

**astream_events() 提供更细粒度的流式事件，包括中间步骤**

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4")
prompt = ChatPromptTemplate.from_template("翻译成英文：{text}")

chain = prompt | llm

async def stream_with_events():
    """流式事件输出"""
    async for event in chain.astream_events(
        {"text": "你好世界"},
        version="v2"
    ):
        kind = event["event"]

        if kind == "on_chat_model_stream":
            # LLM 输出的 token
            content = event["data"]["chunk"].content
            print(content, end="", flush=True)

        elif kind == "on_chain_start":
            print(f"\n[Chain 开始] {event['name']}")

        elif kind == "on_chain_end":
            print(f"\n[Chain 结束] {event['name']}")
```

**事件类型：**

| 事件类型 | 触发时机 | 包含数据 |
|---------|---------|---------|
| `on_chain_start` | Chain 开始执行 | 输入数据 |
| `on_chain_end` | Chain 执行完成 | 输出数据 |
| `on_chat_model_start` | LLM 开始生成 | 消息 |
| `on_chat_model_stream` | LLM 输出 token | chunk |
| `on_chat_model_end` | LLM 生成完成 | 完整响应 |
| `on_tool_start` | 工具开始执行 | 工具输入 |
| `on_tool_end` | 工具执行完成 | 工具输出 |
| `on_retriever_start` | 检索器开始 | 查询 |
| `on_retriever_end` | 检索器完成 | 文档 |

---

### 核心概念4：回调处理器 Callbacks 🔔

**回调是另一种处理流式输出的方式，通过事件钩子响应**

```python
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI

class StreamingHandler(BaseCallbackHandler):
    """自定义流式回调处理器"""

    def on_llm_new_token(self, token: str, **kwargs):
        """每生成一个 token 时调用"""
        print(token, end="", flush=True)

    def on_llm_start(self, serialized, prompts, **kwargs):
        """LLM 开始生成时调用"""
        print("\n[开始生成]")

    def on_llm_end(self, response, **kwargs):
        """LLM 生成结束时调用"""
        print("\n[生成完成]")

    def on_llm_error(self, error, **kwargs):
        """LLM 出错时调用"""
        print(f"\n[错误] {error}")

# 使用回调
llm = ChatOpenAI(model="gpt-4", callbacks=[StreamingHandler()])
response = llm.invoke("写一个笑话")
```

**在 LangChain 源码中的应用：**

```python
# langchain_core/callbacks/base.py
class BaseCallbackHandler(ABC):
    """回调处理器基类"""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """LLM 生成新 token 时触发"""
        pass

    def on_chain_start(self, serialized, inputs, **kwargs) -> None:
        """Chain 开始时触发"""
        pass

    def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        """Tool 开始时触发"""
        pass
```

---

### 扩展概念5：SSE（Server-Sent Events） 📤

**SSE 是 Web 应用中实现流式输出的标准协议**

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_openai import ChatOpenAI

app = FastAPI()
llm = ChatOpenAI(model="gpt-4")

@app.get("/chat/stream")
async def stream_chat(message: str):
    """SSE 流式响应接口"""

    async def generate():
        async for chunk in llm.astream(message):
            # SSE 格式：data: {content}\n\n
            yield f"data: {chunk.content}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )

# 前端 JavaScript 消费
# const eventSource = new EventSource('/chat/stream?message=Hello');
# eventSource.onmessage = (e) => console.log(e.data);
```

**前端消费 SSE：**

```javascript
// 使用 EventSource（原生支持）
const eventSource = new EventSource('/chat/stream?message=Hello');

eventSource.onmessage = (event) => {
    if (event.data === '[DONE]') {
        eventSource.close();
        return;
    }
    document.getElementById('output').textContent += event.data;
};

// 或使用 fetch + ReadableStream
async function streamChat(message) {
    const response = await fetch(`/chat/stream?message=${message}`);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const text = decoder.decode(value);
        console.log(text);
    }
}
```

---

## 4. 【最小可用】

掌握以下内容，就能在 LangChain 应用中实现流式输出：

### 4.1 基础同步流式

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")

# 同步流式
for chunk in llm.stream("你好"):
    print(chunk.content, end="", flush=True)
```

### 4.2 基础异步流式

```python
import asyncio
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")

async def main():
    async for chunk in llm.astream("你好"):
        print(chunk.content, end="", flush=True)

asyncio.run(main())
```

### 4.3 Chain 的流式输出

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("翻译成英文：{text}")
llm = ChatOpenAI(model="gpt-4")
chain = prompt | llm | StrOutputParser()

# Chain 也支持流式
for chunk in chain.stream({"text": "你好世界"}):
    print(chunk, end="", flush=True)
```

### 4.4 收集完整响应

```python
# 流式输出同时收集完整响应
full_response = ""
for chunk in llm.stream("写一首诗"):
    print(chunk.content, end="", flush=True)
    full_response += chunk.content

print(f"\n\n完整响应长度: {len(full_response)}")
```

**这些知识足以：**
- 实现基本的打字机效果
- 构建流式 API 接口
- 在 LangChain Chain 中使用流式输出

---

## 5. 【1个类比】（双轨制）

### 类比1：流式 vs 非流式

#### 🎨 前端视角：fetch vs EventSource

流式输出就像 EventSource 或 WebSocket，实时推送数据。

```javascript
// 非流式：像传统 fetch，等待完整响应
const response = await fetch('/api/generate');
const data = await response.json();  // 等待全部数据
console.log(data);

// 流式：像 EventSource，实时接收数据
const eventSource = new EventSource('/api/stream');
eventSource.onmessage = (e) => {
    console.log(e.data);  // 实时收到每一小块数据
};
```

```python
# LangChain 非流式
response = llm.invoke("写一篇文章")  # 等待完整响应

# LangChain 流式
for chunk in llm.stream("写一篇文章"):  # 实时收到每个 token
    print(chunk.content)
```

**关键相似点：**
- 都是"推送"模式而非"拉取"模式
- 都需要保持连接
- 都适合实时数据场景

#### 🧒 小朋友视角：一次给 vs 一点一点给

流式就像妈妈一口一口喂饭：

```
非流式（一次给）：
- 等厨师做完整盘菜（10分钟）
- 然后一次端上来
- 你要等很久才能吃到第一口

流式（一点一点给）：
- 厨师做一点就送一点（每30秒一小盘）
- 你马上就能开始吃
- 边做边吃，不用等
```

**生活例子：**
```
下载电影 vs 在线播放：

下载电影（非流式）：
- 等整个电影下载完（30分钟）
- 然后才能开始看
- "好无聊，还要等多久？"

在线播放（流式）：
- 缓冲几秒就开始播放
- 边下载边看
- "太好了，马上就能看！"
```

---

### 类比2：异步流式 astream()

#### 🎨 前端视角：async/await + ReadableStream

异步流式就像使用 ReadableStream 处理大文件。

```javascript
// 异步读取大文件流
async function readStream(stream) {
    const reader = stream.getReader();

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        console.log('收到数据块:', value);
    }
}

// fetch 获取流式响应
const response = await fetch('/large-file');
await readStream(response.body);
```

```python
# LangChain 异步流式
async def read_llm_stream():
    async for chunk in llm.astream("生成一篇长文章"):
        print(chunk.content, end="")

asyncio.run(read_llm_stream())
```

#### 🧒 小朋友视角：边做作业边听故事

异步流式就像一边做作业一边听妈妈讲故事：

```
同步方式：
- 先听完整个故事（10分钟）
- 然后再做作业
- 作业开始得很晚

异步方式：
- 妈妈一边讲故事，你一边做作业
- 听一句话，写一个字
- 两件事同时进行，都不耽误！
```

---

### 类比3：流式事件 astream_events()

#### 🎨 前端视角：Redux 事件流 / 调试工具

流式事件就像 Redux DevTools 实时显示 action 流。

```javascript
// Redux 中间件：记录所有 action
const loggerMiddleware = store => next => action => {
    console.log('dispatching', action);
    let result = next(action);
    console.log('next state', store.getState());
    return result;
};

// 你可以看到：
// dispatching { type: 'FETCH_START' }
// dispatching { type: 'FETCH_PROGRESS', payload: 30 }
// dispatching { type: 'FETCH_PROGRESS', payload: 60 }
// dispatching { type: 'FETCH_SUCCESS', payload: data }
```

```python
# LangChain 流式事件
async for event in chain.astream_events(input, version="v2"):
    print(f"事件: {event['event']}, 数据: {event['data']}")

# 输出：
# 事件: on_chain_start, 数据: {...}
# 事件: on_chat_model_start, 数据: {...}
# 事件: on_chat_model_stream, 数据: {"chunk": "你"}
# 事件: on_chat_model_stream, 数据: {"chunk": "好"}
# 事件: on_chat_model_end, 数据: {...}
# 事件: on_chain_end, 数据: {...}
```

#### 🧒 小朋友视角：直播做蛋糕

流式事件就像看厨师直播做蛋糕：

```
不是直播（只看结果）：
- 厨师做完蛋糕后发一张照片
- 你只能看到最终的蛋糕

直播（看全过程）：
- "厨师开始了！"
- "正在打鸡蛋..."
- "加入面粉..."
- "放进烤箱..."
- "蛋糕出炉了！"

流式事件让你看到每一步：
on_chain_start → "厨师开始了"
on_chat_model_stream → "正在做..."
on_chain_end → "完成了！"
```

---

### 类比4：回调处理器 Callbacks

#### 🎨 前端视角：事件监听器

回调就像 DOM 事件监听器。

```javascript
// DOM 事件监听
const button = document.getElementById('btn');

button.addEventListener('click', () => console.log('点击了'));
button.addEventListener('mouseenter', () => console.log('鼠标进入'));
button.addEventListener('mouseleave', () => console.log('鼠标离开'));

// 事件发生时，对应的回调被调用
```

```python
# LangChain 回调
class MyHandler(BaseCallbackHandler):
    def on_llm_start(self, *args):
        print("LLM 开始")

    def on_llm_new_token(self, token):
        print(f"新 token: {token}")

    def on_llm_end(self, *args):
        print("LLM 结束")

llm = ChatOpenAI(callbacks=[MyHandler()])
```

#### 🧒 小朋友视角：闹钟提醒

回调就像设置不同时间的闹钟：

```
设置闹钟：
- 7:00 闹钟：起床！
- 7:30 闹钟：吃早餐！
- 8:00 闹钟：上学！

当时间到了，对应的闹钟响起

LangChain 回调：
- on_llm_start：开始了！
- on_llm_new_token：生成了一个字！
- on_llm_end：结束了！
```

---

### 类比总结表

| 流式概念 | 前端类比 | 小朋友类比 |
|---------|---------|-----------|
| stream() | EventSource | 一口一口喂饭 |
| astream() | async ReadableStream | 边做作业边听故事 |
| astream_events() | Redux DevTools | 直播做蛋糕 |
| Callbacks | addEventListener | 闹钟提醒 |
| SSE | Server-Sent Events | 电视直播 |
| WebSocket | 双向实时通信 | 打电话聊天 |

---

## 6. 【反直觉点】

### 误区1：流式输出总是更快 ❌

**为什么错？**
- 流式的**总时间**和非流式相同（都要生成完整响应）
- 流式只是**感知延迟**更短（第一个 token 到达更快）
- 流式有额外的连接开销

**为什么人们容易这样错？**
因为流式"看起来"更快——用户很快就能看到内容开始出现。但实际上，获取完整响应的时间是一样的。

**正确理解：**

```python
import time

# 非流式
start = time.time()
response = llm.invoke("写一篇 500 字的文章")
total_time = time.time() - start
print(f"非流式总时间: {total_time:.2f}s")

# 流式
start = time.time()
first_token_time = None
for chunk in llm.stream("写一篇 500 字的文章"):
    if first_token_time is None:
        first_token_time = time.time() - start
        print(f"流式首字时间: {first_token_time:.2f}s")
total_time = time.time() - start
print(f"流式总时间: {total_time:.2f}s")

# 结果：
# 非流式总时间: 10.23s
# 流式首字时间: 0.34s  # ← 用户感知到的延迟大幅降低
# 流式总时间: 10.45s   # ← 总时间差不多
```

**经验法则：** 流式改善的是"感知延迟"，不是"总时间"

---

### 误区2：所有 LangChain 组件都支持真正的流式 ❌

**为什么错？**
- 只有 LLM 层真正支持 token 级流式
- 中间的 OutputParser 可能需要完整响应才能解析
- 有些 Chain 会缓冲全部内容再输出

**为什么人们容易这样错？**
因为 Runnable 协议定义了 `stream()` 方法，人们以为所有组件都能真正流式输出。

**正确理解：**

```python
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()

# ❌ JsonOutputParser 无法真正流式（需要完整 JSON 才能解析）
chain = llm | JsonOutputParser()
for chunk in chain.stream("生成一个 JSON"):
    print(chunk)  # 可能一次性输出完整结果，不是真正流式

# ✅ StrOutputParser 支持真正流式
from langchain_core.output_parsers import StrOutputParser

chain = llm | StrOutputParser()
for chunk in chain.stream("写一首诗"):
    print(chunk, end="")  # 真正的 token 级流式

# ✅ 使用 streaming JSON parser 实现部分流式
from langchain_core.output_parsers import JsonOutputParser

chain = llm | JsonOutputParser()
# astream_events 可以获取原始 token 流
async for event in chain.astream_events(input, version="v2"):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content, end="")
```

**经验法则：** 检查每个组件是否真正支持流式，必要时使用 `astream_events`

---

### 误区3：流式输出不需要考虑错误处理 ❌

**为什么错？**
- 流式过程中可能断连
- 可能生成有害内容需要中断
- 客户端可能提前关闭连接

**为什么人们容易这样错？**
简单的流式 demo 通常忽略错误处理，给人"流式很简单"的错觉。

**正确理解：**

```python
async def safe_stream(llm, message):
    """带错误处理的流式输出"""
    full_response = ""
    try:
        async for chunk in llm.astream(message):
            content = chunk.content

            # 检查有害内容
            if contains_harmful_content(content):
                yield "[内容已过滤]"
                break

            full_response += content
            yield content

    except asyncio.CancelledError:
        # 客户端断开连接
        print(f"客户端断开，已生成: {len(full_response)} 字符")
        raise

    except Exception as e:
        # 其他错误
        yield f"\n[错误: {str(e)}]"
        raise

    finally:
        # 清理资源
        print(f"流式完成，总长度: {len(full_response)}")
```

**经验法则：** 生产环境必须处理断连、超时、内容过滤等场景

---

## 7. 【实战代码】

```python
"""
示例：流式输出的完整应用
演示 LangChain 中流式输出的核心用法
"""

import asyncio
from typing import AsyncIterator, Iterator
from dataclasses import dataclass

# ===== 1. 模拟 LLM 流式输出 =====
print("=== 1. 模拟 LLM 流式输出 ===")

@dataclass
class MockChunk:
    """模拟的响应块"""
    content: str
    finish_reason: str = None

class MockStreamingLLM:
    """模拟流式 LLM"""

    def __init__(self, delay: float = 0.1):
        self.delay = delay

    def stream(self, prompt: str) -> Iterator[MockChunk]:
        """同步流式输出"""
        import time

        # 模拟生成响应
        response = f"这是对「{prompt}」的回复。LangChain 是一个强大的框架！"

        for char in response:
            time.sleep(self.delay)
            yield MockChunk(content=char)

        yield MockChunk(content="", finish_reason="stop")

    async def astream(self, prompt: str) -> AsyncIterator[MockChunk]:
        """异步流式输出"""
        response = f"异步回复「{prompt}」：Hello from async streaming!"

        for char in response:
            await asyncio.sleep(self.delay)
            yield MockChunk(content=char)

        yield MockChunk(content="", finish_reason="stop")

# 同步流式演示
llm = MockStreamingLLM(delay=0.05)

print("同步流式输出：")
for chunk in llm.stream("你好"):
    print(chunk.content, end="", flush=True)
print("\n")

# ===== 2. 异步流式输出 =====
print("=== 2. 异步流式输出 ===")

async def async_stream_demo():
    """异步流式演示"""
    llm = MockStreamingLLM(delay=0.03)

    print("异步流式输出：")
    async for chunk in llm.astream("异步测试"):
        print(chunk.content, end="", flush=True)
    print("\n")

asyncio.run(async_stream_demo())

# ===== 3. 并发流式处理 =====
print("=== 3. 并发流式处理 ===")

async def concurrent_streams():
    """并发处理多个流式请求"""
    llm = MockStreamingLLM(delay=0.05)

    async def process_stream(stream_id: str, prompt: str):
        """处理单个流"""
        result = []
        async for chunk in llm.astream(prompt):
            result.append(chunk.content)
        return f"[{stream_id}] {''.join(result)}"

    # 并发执行三个流式请求
    results = await asyncio.gather(
        process_stream("A", "问题A"),
        process_stream("B", "问题B"),
        process_stream("C", "问题C"),
    )

    print("并发结果：")
    for r in results:
        print(f"  {r[:50]}...")

asyncio.run(concurrent_streams())

# ===== 4. 流式回调处理器 =====
print("\n=== 4. 流式回调处理器 ===")

class StreamingCallback:
    """流式回调处理器"""

    def __init__(self):
        self.tokens = []
        self.start_time = None
        self.end_time = None

    def on_llm_start(self):
        import time
        self.start_time = time.time()
        print("[回调] LLM 开始生成")

    def on_new_token(self, token: str):
        self.tokens.append(token)
        print(token, end="", flush=True)

    def on_llm_end(self):
        import time
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        print(f"\n[回调] 生成完成，耗时: {duration:.2f}s，Token数: {len(self.tokens)}")

# 使用回调
callback = StreamingCallback()
callback.on_llm_start()

for chunk in llm.stream("回调测试"):
    callback.on_new_token(chunk.content)

callback.on_llm_end()

# ===== 5. 流式事件模拟 =====
print("\n=== 5. 流式事件模拟 ===")

async def stream_events_demo():
    """模拟 astream_events"""

    events = [
        {"event": "on_chain_start", "name": "MyChain", "data": {}},
        {"event": "on_chat_model_start", "name": "ChatOpenAI", "data": {}},
        {"event": "on_chat_model_stream", "data": {"chunk": "你"}},
        {"event": "on_chat_model_stream", "data": {"chunk": "好"}},
        {"event": "on_chat_model_stream", "data": {"chunk": "！"}},
        {"event": "on_chat_model_end", "data": {"output": "你好！"}},
        {"event": "on_chain_end", "name": "MyChain", "data": {"output": "你好！"}},
    ]

    print("流式事件：")
    for event in events:
        await asyncio.sleep(0.1)

        if event["event"] == "on_chat_model_stream":
            print(event["data"]["chunk"], end="", flush=True)
        else:
            print(f"\n[事件] {event['event']}", end="")

    print("\n")

asyncio.run(stream_events_demo())

# ===== 6. SSE 响应生成器 =====
print("=== 6. SSE 响应生成器 ===")

async def sse_generator(prompt: str):
    """生成 SSE 格式的响应"""
    llm = MockStreamingLLM(delay=0.05)

    async for chunk in llm.astream(prompt):
        if chunk.finish_reason == "stop":
            yield "data: [DONE]\n\n"
            break
        # SSE 格式
        yield f"data: {chunk.content}\n\n"

async def sse_demo():
    print("SSE 格式输出：")
    async for sse_chunk in sse_generator("SSE测试"):
        print(sse_chunk, end="")

asyncio.run(sse_demo())

# ===== 7. 带超时的流式处理 =====
print("\n=== 7. 带超时的流式处理 ===")

async def stream_with_timeout(prompt: str, timeout: float = 2.0):
    """带超时的流式处理"""
    llm = MockStreamingLLM(delay=0.1)
    result = []

    try:
        async with asyncio.timeout(timeout):
            async for chunk in llm.astream(prompt):
                result.append(chunk.content)
                print(chunk.content, end="", flush=True)
    except asyncio.TimeoutError:
        print(f"\n[超时] 已获取 {len(result)} 个字符")
        return "".join(result) + "..."

    return "".join(result)

async def timeout_demo():
    print("带超时的流式（2秒超时）：")
    result = await stream_with_timeout("一个很长的提示词", timeout=1.0)
    print(f"\n结果: {result[:30]}...")

asyncio.run(timeout_demo())

# ===== 8. 流式内容过滤 =====
print("\n=== 8. 流式内容过滤 ===")

async def filtered_stream(prompt: str, banned_words: list):
    """带内容过滤的流式输出"""
    llm = MockStreamingLLM(delay=0.05)

    buffer = ""
    async for chunk in llm.astream(prompt):
        buffer += chunk.content

        # 检查是否包含违禁词
        for word in banned_words:
            if word in buffer:
                print("\n[警告] 检测到违禁内容，已停止")
                return

        print(chunk.content, end="", flush=True)

    print()

async def filter_demo():
    print("带内容过滤的流式：")
    await filtered_stream(
        "这是一段包含测试的文本",
        banned_words=["禁止"]  # 示例违禁词
    )

asyncio.run(filter_demo())

# ===== 9. 统计信息收集 =====
print("\n=== 9. 统计信息收集 ===")

class StreamingStats:
    """流式统计收集器"""

    def __init__(self):
        self.token_count = 0
        self.char_count = 0
        self.first_token_time = None
        self.last_token_time = None
        self._start_time = None

    def start(self):
        import time
        self._start_time = time.time()

    def record_token(self, token: str):
        import time
        now = time.time()

        if self.first_token_time is None:
            self.first_token_time = now - self._start_time

        self.last_token_time = now - self._start_time
        self.token_count += 1
        self.char_count += len(token)

    def get_stats(self):
        return {
            "token_count": self.token_count,
            "char_count": self.char_count,
            "time_to_first_token": f"{self.first_token_time:.3f}s" if self.first_token_time else None,
            "total_time": f"{self.last_token_time:.3f}s" if self.last_token_time else None,
            "tokens_per_second": f"{self.token_count / self.last_token_time:.1f}" if self.last_token_time else None
        }

# 收集统计
stats = StreamingStats()
stats.start()

print("带统计的流式输出：")
for chunk in llm.stream("统计测试"):
    stats.record_token(chunk.content)
    print(chunk.content, end="", flush=True)

print("\n\n统计信息：")
for k, v in stats.get_stats().items():
    print(f"  {k}: {v}")

print("\n=== 完成！===")
```

**运行输出示例：**
```
=== 1. 模拟 LLM 流式输出 ===
同步流式输出：
这是对「你好」的回复。LangChain 是一个强大的框架！

=== 2. 异步流式输出 ===
异步流式输出：
异步回复「异步测试」：Hello from async streaming!

=== 3. 并发流式处理 ===
并发结果：
  [A] 异步回复「问题A」：Hello from async streaming!...
  [B] 异步回复「问题B」：Hello from async streaming!...
  [C] 异步回复「问题C」：Hello from async streaming!...

=== 4. 流式回调处理器 ===
[回调] LLM 开始生成
这是对「回调测试」的回复。LangChain 是一个强大的框架！
[回调] 生成完成，耗时: 2.45s，Token数: 38

=== 5. 流式事件模拟 ===
流式事件：
[事件] on_chain_start
[事件] on_chat_model_start
你好！
[事件] on_chat_model_end
[事件] on_chain_end

=== 6. SSE 响应生成器 ===
SSE 格式输出：
data: 异
data: 步
...
data: [DONE]

=== 7. 带超时的流式处理 ===
带超时的流式（2秒超时）：
异步回复「一...
[超时] 已获取 10 个字符
结果: 异步回复「一...

=== 9. 统计信息收集 ===
带统计的流式输出：
这是对「统计测试」的回复...

统计信息：
  token_count: 38
  char_count: 38
  time_to_first_token: 0.050s
  total_time: 1.900s
  tokens_per_second: 20.0

=== 完成！===
```

---

## 8. 【面试必问】

### 问题："什么是流式输出？为什么 ChatGPT 使用流式输出？"

**普通回答（❌ 不出彩）：**
"流式输出就是一个字一个字地输出，这样用户体验更好。"

**出彩回答（✅ 推荐）：**

> **流式输出是一种"边生成边发送"的数据传输模式：**
>
> **为什么 LLM 适合流式？**
> - LLM 本身就是逐 Token 生成的（自回归模型）
> - 每生成一个 Token，就可以立即返回给用户
> - 不需要等待完整响应生成完毕
>
> **流式输出的三个核心价值：**
>
> 1. **降低感知延迟**
>    - 非流式：用户等待 10 秒看到第一个字
>    - 流式：用户等待 0.2 秒看到第一个字
>    - 心理体验完全不同
>
> 2. **实时交互能力**
>    - 用户可以边看边决定是否继续
>    - 可以在发现问题时提前中断
>    - 支持实时对话界面
>
> 3. **资源优化**
>    - 不需要缓存完整响应
>    - 可以及早释放连接
>    - 适合长响应场景
>
> **在 LangChain 中的实现：**
> - `stream()`：同步流式，返回 Iterator
> - `astream()`：异步流式，返回 AsyncIterator
> - `astream_events()`：事件级流式，包含中间步骤
> - 所有实现 Runnable 协议的组件都支持流式
>
> **实际应用注意点：**
> - 流式改善"感知延迟"，不改善"总时间"
> - 不是所有组件都支持真正的 Token 级流式
> - 生产环境需要处理断连、超时、内容过滤

**为什么这个回答出彩？**
1. ✅ 解释了"为什么"而不只是"是什么"
2. ✅ 明确了三个核心价值
3. ✅ 联系了 LangChain 具体实现
4. ✅ 提到了实际应用的注意点

---

### 问题："如何在 Web 应用中实现 LLM 的流式输出？"

**普通回答（❌ 不出彩）：**
"用 WebSocket 或者 SSE 把数据推送到前端。"

**出彩回答（✅ 推荐）：**

> **Web 应用实现流式输出有两种主要方案：**
>
> **方案1：SSE（Server-Sent Events）** - 推荐
> ```python
> # 后端（FastAPI）
> @app.get("/chat/stream")
> async def stream_chat(message: str):
>     async def generate():
>         async for chunk in llm.astream(message):
>             yield f"data: {chunk.content}\n\n"
>         yield "data: [DONE]\n\n"
>
>     return StreamingResponse(generate(), media_type="text/event-stream")
> ```
> ```javascript
> // 前端
> const es = new EventSource('/chat/stream?message=Hello');
> es.onmessage = (e) => {
>     if (e.data === '[DONE]') { es.close(); return; }
>     output.textContent += e.data;
> };
> ```
>
> **方案2：WebSocket** - 适合双向通信
> - 更复杂，但支持用户中途发送消息
> - 适合需要实时交互的场景
>
> **选择建议：**
> | 场景 | 推荐方案 |
> |------|---------|
> | 简单问答 | SSE |
> | 长对话 | WebSocket |
> | 只读流式 | SSE |
> | 需要中断 | WebSocket |
>
> **生产环境考虑：**
> - 连接超时处理
> - 断线重连机制
> - 内容安全过滤
> - 负载均衡支持（SSE 友好，WebSocket 需要 sticky session）

---

## 9. 【化骨绵掌】

### 卡片1：流式输出是什么？ 🎯

**一句话：** 流式输出让 LLM 边生成边返回，而不是等全部生成完。

**举例：**
```python
# 非流式：等 10 秒，一次性返回
response = llm.invoke("写一篇文章")

# 流式：立即开始，逐字返回
for chunk in llm.stream("写一篇文章"):
    print(chunk.content, end="")
```

**应用：** ChatGPT 的打字机效果就是流式输出。

---

### 卡片2：stream() 方法 🌊

**一句话：** 同步流式输出，返回一个生成器，逐个产出响应片段。

**举例：**
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
for chunk in llm.stream("你好"):
    print(chunk.content, end="", flush=True)
```

**应用：** 简单脚本和同步代码中使用。

---

### 卡片3：astream() 异步流式 ⚡

**一句话：** 异步版本的流式输出，支持并发处理多个请求。

**举例：**
```python
async def stream_response():
    async for chunk in llm.astream("你好"):
        print(chunk.content, end="")

asyncio.run(stream_response())
```

**应用：** Web 应用、高并发场景必备。

---

### 卡片4：astream_events() 事件流 📡

**一句话：** 提供更细粒度的事件，包括 Chain 和 Tool 的中间状态。

**举例：**
```python
async for event in chain.astream_events(input, version="v2"):
    if event["event"] == "on_chat_model_stream":
        print(event["data"]["chunk"].content)
```

**应用：** 调试复杂 Chain、显示中间步骤。

---

### 卡片5：回调处理器 Callbacks 🔔

**一句话：** 通过事件钩子响应流式输出，不改变调用方式。

**举例：**
```python
class MyHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token, **kwargs):
        print(token, end="")

llm = ChatOpenAI(callbacks=[MyHandler()])
```

**应用：** 日志记录、UI 更新、指标收集。

---

### 卡片6：SSE 协议 📤

**一句话：** Server-Sent Events 是 Web 应用实现流式输出的标准协议。

**举例：**
```python
# 后端返回 SSE 格式
yield f"data: {chunk.content}\n\n"

# 前端使用 EventSource
new EventSource('/stream')
```

**应用：** ChatGPT Web 版就是用 SSE 实现的。

---

### 卡片7：感知延迟 vs 总时间 ⏱️

**一句话：** 流式降低感知延迟（首字时间），但不改变总生成时间。

**举例：**
```
非流式：用户等 10 秒看到内容
流式：用户等 0.2 秒看到第一个字，总共还是 10 秒

感知上快了 50 倍！
```

**应用：** 设计用户体验时要理解这个区别。

---

### 卡片8：流式错误处理 🛡️

**一句话：** 生产环境必须处理断连、超时、内容过滤等异常。

**举例：**
```python
try:
    async for chunk in llm.astream(prompt):
        if is_harmful(chunk):
            break
        yield chunk
except asyncio.CancelledError:
    # 客户端断开
    pass
```

**应用：** 稳定的生产服务必备。

---

### 卡片9：Chain 流式支持 🔗

**一句话：** LCEL Chain 也支持流式，但中间组件可能缓冲内容。

**举例：**
```python
chain = prompt | llm | StrOutputParser()

# StrOutputParser 支持真流式
for chunk in chain.stream({"text": "你好"}):
    print(chunk, end="")

# JsonOutputParser 可能需要缓冲
```

**应用：** 选择合适的 OutputParser。

---

### 卡片10：流式在 LangChain 中的位置 ⭐

**一句话：** 流式是 Runnable 协议的核心能力，所有组件都支持。

**举例：**
```python
# Runnable 协议定义
class Runnable:
    def invoke(self, input) -> Output
    def stream(self, input) -> Iterator[Output]
    async def astream(self, input) -> AsyncIterator[Output]
```

**应用：** 理解 Runnable 就理解了 LangChain 的流式能力。

---

## 10. 【一句话总结】

**流式输出让 LLM 边生成边返回，通过 stream()、astream()、astream_events() 三种方式实现，大幅降低用户感知延迟，是现代 LLM 应用的标配能力。**

---

## 📚 学习检查清单

- [ ] 理解流式输出的原理和价值
- [ ] 能够使用 stream() 实现同步流式
- [ ] 能够使用 astream() 实现异步流式
- [ ] 理解 astream_events() 的事件类型
- [ ] 会使用回调处理器处理流式事件
- [ ] 了解 SSE 协议的基本格式

## 🔗 下一步学习

- **Function Calling 与 Tool Use**：流式场景下的工具调用
- **异步编程 async/await**：深入异步流式处理
- **Callback 回调系统**：LangChain 事件系统详解

---

**版本：** v1.0
**最后更新：** 2025-12-12
