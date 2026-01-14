# Callback 回调系统

> 原子化知识点 | LangChain 使用 | LangChain 源码学习核心知识

---

## 1. 【30字核心】

**Callback 是 LangChain 的事件监听机制，通过 Handler 可以实现日志记录、流式输出、性能监控和调试追踪。**

---

## 2. 【第一性原理】

### 什么是第一性原理？

**第一性原理**：回到事物最基本的真理，从源头思考问题

### Callback 回调系统的第一性原理 🎯

#### 1. 最基础的定义

**Callback = 当事件发生时执行的函数**

仅此而已！没有更基础的了。

```python
# Callback 的本质
def on_event(event_data):
    # 事件发生时执行这个函数
    print(f"事件发生了: {event_data}")

# 注册 callback
llm.on("start", on_event)

# 当 LLM 开始时，on_event 被调用
```

#### 2. 为什么需要 Callback？

**核心问题：需要在执行过程中插入自定义逻辑**

```python
# 没有 Callback 的问题
result = chain.invoke(input)
# 只能看到最终结果
# 不知道中间发生了什么
# 无法：
# ❌ 实时看到生成过程（流式输出）
# ❌ 记录执行日志
# ❌ 监控性能和成本
# ❌ 调试问题

# 有 Callback 的解决方案
# ✅ on_llm_new_token：每生成一个 token 就通知
# ✅ on_llm_start/end：记录开始和结束时间
# ✅ on_chain_error：捕获错误
```

#### 3. Callback 的三层价值

##### 价值1：流式输出 - 实时反馈

```python
# 不等 LLM 完成，边生成边显示
class StreamHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token, **kwargs):
        print(token, end="", flush=True)  # 实时打印

# 用户看到：
# "你" → "你好" → "你好！" → "你好！我" → ...
# 而不是等 5 秒后看到完整回复
```

##### 价值2：监控追踪 - 了解执行过程

```python
# 记录每一步的执行
class MonitorHandler(BaseCallbackHandler):
    def on_chain_start(self, serialized, inputs, **kwargs):
        print(f"Chain 开始: {inputs}")

    def on_llm_start(self, serialized, prompts, **kwargs):
        print(f"LLM 调用: {len(prompts)} 个 prompt")

    def on_llm_end(self, response, **kwargs):
        print(f"LLM 返回: {response.generations[0][0].text[:50]}...")
```

##### 价值3：成本控制 - 统计 Token 使用

```python
# 统计 Token 消耗
class CostHandler(BaseCallbackHandler):
    def __init__(self):
        self.total_tokens = 0

    def on_llm_end(self, response, **kwargs):
        usage = response.llm_output.get("token_usage", {})
        self.total_tokens += usage.get("total_tokens", 0)
        print(f"本次: {usage}, 累计: {self.total_tokens}")
```

#### 4. 从第一性原理推导 Callback 设计

**推理链：**

```
1. 执行过程是黑盒
   ↓
2. 需要观察内部状态
   ↓
3. 在关键节点插入钩子
   ↓
4. 定义标准的事件类型
   ↓
5. on_llm_start, on_llm_end, on_chain_start...
   ↓
6. 用户实现 Handler 处理事件
   ↓
7. 通过 CallbackManager 管理多个 Handler
```

#### 5. 一句话总结第一性原理

**Callback 是在执行过程的关键节点插入的钩子，让开发者能够观察、记录、控制 LLM 应用的执行过程。**

---

## 3. 【核心概念（全面覆盖）】

### 核心概念1：BaseCallbackHandler 回调处理器 📡

**BaseCallbackHandler 定义了所有可监听的事件钩子**

```python
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage
from typing import Any, Dict, List

class MyCallbackHandler(BaseCallbackHandler):
    """自定义回调处理器"""

    # ===== LLM 相关事件 =====
    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs
    ) -> None:
        """LLM 开始调用时"""
        print(f"LLM 开始，prompt 数量: {len(prompts)}")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """生成新 token 时（流式）"""
        print(token, end="", flush=True)

    def on_llm_end(self, response, **kwargs) -> None:
        """LLM 调用结束时"""
        print(f"\nLLM 结束")

    def on_llm_error(self, error: Exception, **kwargs) -> None:
        """LLM 调用出错时"""
        print(f"LLM 错误: {error}")

    # ===== Chain 相关事件 =====
    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs
    ) -> None:
        """Chain 开始执行时"""
        print(f"Chain 开始: {inputs}")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Chain 执行结束时"""
        print(f"Chain 结束: {outputs}")

    def on_chain_error(self, error: Exception, **kwargs) -> None:
        """Chain 执行出错时"""
        print(f"Chain 错误: {error}")

    # ===== Tool 相关事件 =====
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs
    ) -> None:
        """Tool 开始执行时"""
        print(f"Tool 开始: {input_str}")

    def on_tool_end(self, output: str, **kwargs) -> None:
        """Tool 执行结束时"""
        print(f"Tool 结束: {output}")

    def on_tool_error(self, error: Exception, **kwargs) -> None:
        """Tool 执行出错时"""
        print(f"Tool 错误: {error}")

    # ===== Agent 相关事件 =====
    def on_agent_action(self, action, **kwargs) -> None:
        """Agent 执行动作时"""
        print(f"Agent 动作: {action.tool}")

    def on_agent_finish(self, finish, **kwargs) -> None:
        """Agent 完成时"""
        print(f"Agent 完成: {finish.return_values}")

    # ===== Retriever 相关事件 =====
    def on_retriever_start(self, serialized, query, **kwargs) -> None:
        """Retriever 开始检索时"""
        print(f"检索: {query}")

    def on_retriever_end(self, documents, **kwargs) -> None:
        """Retriever 检索结束时"""
        print(f"检索到 {len(documents)} 个文档")
```

**事件类型速查表：**

| 组件 | 开始事件 | 结束事件 | 错误事件 | 特殊事件 |
|-----|---------|---------|---------|---------|
| LLM | `on_llm_start` | `on_llm_end` | `on_llm_error` | `on_llm_new_token` |
| Chain | `on_chain_start` | `on_chain_end` | `on_chain_error` | - |
| Tool | `on_tool_start` | `on_tool_end` | `on_tool_error` | - |
| Agent | - | - | - | `on_agent_action`, `on_agent_finish` |
| Retriever | `on_retriever_start` | `on_retriever_end` | `on_retriever_error` | - |

---

### 核心概念2：CallbackManager 回调管理器 🎛️

**CallbackManager 管理多个 Handler，负责事件的分发**

```python
from langchain_core.callbacks import CallbackManager

# 创建多个 Handler
stream_handler = StreamingHandler()
monitor_handler = MonitorHandler()
cost_handler = CostHandler()

# 通过 CallbackManager 管理
callback_manager = CallbackManager(
    handlers=[stream_handler, monitor_handler, cost_handler]
)

# 或者直接传递列表
llm = ChatOpenAI(callbacks=[stream_handler, monitor_handler])
```

**两种传递 Callback 的方式：**

```python
# 方式1：构造时传递（全局生效）
llm = ChatOpenAI(callbacks=[handler1, handler2])

# 方式2：调用时传递（单次生效）
result = chain.invoke(
    {"input": "你好"},
    config={"callbacks": [handler3]}
)

# 两种方式可以组合使用
# 构造时的是"常驻"，调用时的是"临时"
```

**在 LangChain 源码中的应用：**

```python
# langchain_core/callbacks/manager.py
class CallbackManager:
    """回调管理器"""

    def __init__(
        self,
        handlers: List[BaseCallbackHandler] = None,
        inheritable_handlers: List[BaseCallbackHandler] = None,
    ):
        self.handlers = handlers or []
        self.inheritable_handlers = inheritable_handlers or []

    def on_llm_start(self, serialized, prompts, **kwargs):
        """广播 LLM 开始事件"""
        for handler in self.handlers:
            handler.on_llm_start(serialized, prompts, **kwargs)

    # ... 其他事件方法类似
```

---

### 核心概念3：流式输出 Streaming 🌊

**流式输出是 Callback 最常用的应用场景**

```python
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_openai import ChatOpenAI

# 方式1：使用内置的 StreamingStdOutCallbackHandler
llm = ChatOpenAI(
    streaming=True,  # 开启流式
    callbacks=[StreamingStdOutCallbackHandler()]
)

response = llm.invoke("写一首诗")
# 实时打印每个 token

# 方式2：自定义流式处理
class CustomStreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.tokens = []

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.tokens.append(token)
        # 可以发送到 WebSocket、更新 UI 等
        print(token, end="", flush=True)

# 方式3：使用 astream 方法
async for chunk in llm.astream("写一首诗"):
    print(chunk.content, end="", flush=True)

# 方式4：使用 astream_events 获取详细事件
async for event in chain.astream_events({"input": "你好"}, version="v2"):
    kind = event["event"]
    if kind == "on_llm_stream":
        print(event["data"]["chunk"].content, end="")
```

**流式输出的三种方式对比：**

| 方式 | 优点 | 缺点 | 适用场景 |
|-----|------|------|---------|
| Callback | 灵活，可自定义处理 | 需要实现 Handler | 复杂的流式处理 |
| `astream` | 简单，async for | 只能获取内容 | 简单的流式显示 |
| `astream_events` | 信息最全 | 复杂 | 需要详细事件信息 |

---

### 核心概念4：RunnableConfig 配置传递 ⚙️

**通过 config 参数传递 callbacks 和其他配置**

```python
from langchain_core.runnables import RunnableConfig

# 定义配置
config = RunnableConfig(
    callbacks=[MyHandler()],
    tags=["production", "user_123"],
    metadata={"user_id": "123", "session_id": "abc"},
    max_concurrency=5,
)

# 调用时传递
result = chain.invoke(input, config=config)

# 或者使用字典形式
result = chain.invoke(
    input,
    config={
        "callbacks": [handler],
        "tags": ["test"],
        "metadata": {"key": "value"}
    }
)

# config 会自动传递给 Chain 中的所有组件
chain = prompt | llm | parser
# 三个组件都会收到相同的 config
```

---

### 扩展概念5：异步 Callback 🔄

**AsyncCallbackHandler 用于异步场景**

```python
from langchain_core.callbacks import AsyncCallbackHandler
import asyncio

class AsyncStreamHandler(AsyncCallbackHandler):
    """异步流式处理器"""

    async def on_llm_new_token(self, token: str, **kwargs) -> None:
        # 可以进行异步操作
        await send_to_websocket(token)

    async def on_llm_end(self, response, **kwargs) -> None:
        await notify_completion()

# 使用
handler = AsyncStreamHandler()
async for chunk in llm.astream("你好", config={"callbacks": [handler]}):
    pass
```

---

### 扩展概念6：内置 Callback Handler 📦

**LangChain 提供多个内置的 Handler**

```python
from langchain_core.callbacks import (
    StreamingStdOutCallbackHandler,  # 标准输出流式
    StdOutCallbackHandler,           # 标准输出（非流式）
    FileCallbackHandler,             # 写入文件
)

# 1. 流式输出到终端
streaming_handler = StreamingStdOutCallbackHandler()

# 2. 详细日志到终端
stdout_handler = StdOutCallbackHandler()

# 3. 日志写入文件
file_handler = FileCallbackHandler("output.log")

# 组合使用
llm = ChatOpenAI(
    streaming=True,
    callbacks=[streaming_handler, file_handler]
)
```

---

## 4. 【最小可用】

掌握以下内容，就能在 LangChain 中使用 Callback：

### 4.1 流式输出

```python
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

response = llm.invoke("写一首诗")
# 实时输出每个 token
```

### 4.2 自定义 Handler

```python
from langchain_core.callbacks import BaseCallbackHandler

class MyHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print("开始...")

    def on_llm_end(self, response, **kwargs):
        print("结束！")

llm = ChatOpenAI(callbacks=[MyHandler()])
```

### 4.3 通过 config 传递

```python
result = chain.invoke(
    {"input": "你好"},
    config={"callbacks": [MyHandler()]}
)
```

### 4.4 异步流式

```python
async for chunk in llm.astream("你好"):
    print(chunk.content, end="")
```

**这些知识足以：**
- 实现流式输出提升用户体验
- 添加日志记录和监控
- 追踪执行过程和调试
- 统计 Token 使用和成本

---

## 5. 【1个类比】（双轨制）

### 类比1：Callback 是事件监听器

#### 🎨 前端视角：addEventListener / React Hooks

Callback 就像前端的事件监听，在特定时机触发。

```javascript
// DOM 事件监听
button.addEventListener('click', (event) => {
  console.log('按钮被点击了');
});

// React useEffect
useEffect(() => {
  console.log('组件挂载了');
  return () => console.log('组件卸载了');
}, []);

// 自定义 Hook
function useLoading() {
  const [loading, setLoading] = useState(false);
  const onStart = () => setLoading(true);
  const onEnd = () => setLoading(false);
  return { loading, onStart, onEnd };
}
```

```python
# LangChain Callback
class MyHandler(BaseCallbackHandler):
    def on_llm_start(self, **kwargs):  # 类似 onStart
        print("开始")

    def on_llm_end(self, **kwargs):    # 类似 onEnd
        print("结束")
```

**关键相似点：**
- 都是事件驱动
- 都在特定时机触发
- 都可以有多个监听器

#### 🧒 小朋友视角：闹钟提醒

Callback 就像设置的各种闹钟：

```
你设置了几个闹钟：
- 7:00 起床闹钟 → on_llm_start（开始了提醒我）
- 7:30 吃早饭闹钟 → on_llm_new_token（每次有新东西提醒我）
- 8:00 上学闹钟 → on_llm_end（结束了提醒我）

闹钟响了，你就知道该做什么了！
```

---

### 类比2：流式输出是水龙头

#### 🎨 前端视角：Server-Sent Events (SSE) / WebSocket

流式输出就像 SSE，服务器不断推送数据。

```javascript
// Server-Sent Events
const eventSource = new EventSource('/stream');

eventSource.onmessage = (event) => {
  // 收到一小块数据
  appendToUI(event.data);
};

// WebSocket
const ws = new WebSocket('ws://server');

ws.onmessage = (event) => {
  // 收到消息
  updateUI(event.data);
};
```

```python
# LangChain 流式
class StreamHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token, **kwargs):
        # 收到一个 token
        send_to_client(token)
```

#### 🧒 小朋友视角：水龙头流水

流式输出就像水龙头：

```
普通方式（非流式）：
打开水龙头 → 等水装满一桶 → 关水龙头 → 拿走整桶水
（等很久才能用水）

流式方式：
打开水龙头 → 水一直流 → 需要多少接多少 → 随时能用水
（马上就能用水）

LLM 流式输出：
问问题 → AI 一个字一个字回答 → 你边看边读
（不用等全部生成完）
```

---

### 类比3：CallbackManager 是广播站

#### 🎨 前端视角：EventEmitter / 发布订阅

CallbackManager 就像一个事件广播系统。

```javascript
// Node.js EventEmitter
const emitter = new EventEmitter();

// 订阅者1
emitter.on('message', (data) => console.log('订阅者1:', data));

// 订阅者2
emitter.on('message', (data) => saveToLog(data));

// 发布消息
emitter.emit('message', 'Hello');
// 两个订阅者都收到消息
```

```python
# LangChain CallbackManager
manager = CallbackManager(handlers=[handler1, handler2, handler3])

# 当事件发生时，所有 handler 都被通知
manager.on_llm_start(...)  # handler1, handler2, handler3 都被调用
```

#### 🧒 小朋友视角：学校广播站

CallbackManager 就像学校的广播站：

```
广播站（CallbackManager）广播一条消息

所有教室（Handler）都能听到：
- 一年级教室 → 记录到日志
- 二年级教室 → 更新大屏幕
- 三年级教室 → 通知家长

一条广播，多个地方响应！
```

---

### 类比总结表

| LangChain 概念 | 前端类比 | 小朋友类比 |
|---------------|---------|-----------|
| Callback | addEventListener | 闹钟提醒 |
| CallbackHandler | 事件处理函数 | 听到闹钟后做的事 |
| CallbackManager | EventEmitter | 广播站 |
| on_llm_new_token | SSE onmessage | 水龙头流水 |
| streaming | Server Push | 边做边看 |
| config | Context | 传递设置 |

---

## 6. 【反直觉点】

### 误区1：Callback 会阻塞主流程 ❌

**为什么错？**
- Callback 是旁路处理，不影响主流程的返回值
- 即使 Callback 出错，主流程也可以继续
- Callback 的执行时间不计入主流程

**为什么人们容易这样错？**
以为 Callback 是串行执行的"中间件"。

**正确理解：**

```python
# Callback 是旁路，不影响返回值
class SlowHandler(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs):
        time.sleep(10)  # 慢处理
        print("处理完成")

# 主流程不受影响
result = llm.invoke("你好")  # 正常返回
# 之后 SlowHandler.on_llm_end 才执行

# 即使 Callback 出错
class BuggyHandler(BaseCallbackHandler):
    def on_llm_end(self, response, **kwargs):
        raise Exception("出错了！")

# 主流程仍然可以得到结果
result = llm.invoke("你好")  # 依然返回结果
# Callback 错误被记录但不影响主流程
```

---

### 误区2：流式输出只是打印 ❌

**为什么错？**
- 流式涉及 LLM 的流式生成模式
- 需要正确的 streaming=True 设置
- Token 的传递、累积、展示是完整链路
- 前端集成需要 WebSocket/SSE

**为什么人们容易这样错？**
只看到 print() 这一步。

**正确理解：**

```python
# 完整的流式链路

# 1. LLM 层：开启流式生成
llm = ChatOpenAI(streaming=True)

# 2. Callback 层：处理 token
class WebSocketHandler(BaseCallbackHandler):
    def __init__(self, websocket):
        self.ws = websocket

    def on_llm_new_token(self, token, **kwargs):
        # 发送到前端
        self.ws.send(token)

# 3. 前端层：接收和显示
# JavaScript
ws.onmessage = (event) => {
    appendToChat(event.data);
};

# 4. 还需要考虑：
# - 错误处理
# - 连接断开
# - 多用户隔离
# - 超时处理
```

---

### 误区3：所有事件都会触发 Callback ❌

**为什么错？**
- 只有 Handler 实现的方法才会被调用
- 未实现的方法默认空操作
- 需要根据需求选择性实现

**为什么人们容易这样错？**
以为继承 BaseCallbackHandler 就会收到所有事件。

**正确理解：**

```python
# 只实现需要的方法
class MinimalHandler(BaseCallbackHandler):
    """只关心 LLM 开始和结束"""

    def on_llm_start(self, serialized, prompts, **kwargs):
        print("开始")  # 这个会被调用

    def on_llm_end(self, response, **kwargs):
        print("结束")  # 这个会被调用

    # on_llm_new_token 没实现
    # → 流式 token 事件不会被处理

    # on_chain_start 没实现
    # → Chain 开始事件不会被处理

# 如果需要所有事件
class FullHandler(BaseCallbackHandler):
    def on_llm_start(self, ...): pass
    def on_llm_new_token(self, ...): pass
    def on_llm_end(self, ...): pass
    def on_chain_start(self, ...): pass
    def on_chain_end(self, ...): pass
    # ... 实现所有需要的方法
```

---

## 7. 【实战代码】

```python
"""
示例：Callback 回调系统完整演示
展示 LangChain 中 Callback 的核心用法
"""

from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import time

# ===== 1. 基础事件和消息结构 =====
print("=== 1. 基础结构 ===")

@dataclass
class LLMResult:
    """LLM 返回结果"""
    text: str
    token_count: int = 0

@dataclass
class Event:
    """事件对象"""
    type: str
    data: Any
    timestamp: datetime = field(default_factory=datetime.now)

# ===== 2. BaseCallbackHandler 实现 =====
print("\n=== 2. BaseCallbackHandler ===")

class BaseCallbackHandler:
    """回调处理器基类"""

    def on_llm_start(self, serialized: Dict, prompts: List[str], **kwargs) -> None:
        """LLM 开始"""
        pass

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """新 token"""
        pass

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """LLM 结束"""
        pass

    def on_llm_error(self, error: Exception, **kwargs) -> None:
        """LLM 错误"""
        pass

    def on_chain_start(self, serialized: Dict, inputs: Dict, **kwargs) -> None:
        """Chain 开始"""
        pass

    def on_chain_end(self, outputs: Dict, **kwargs) -> None:
        """Chain 结束"""
        pass

    def on_tool_start(self, serialized: Dict, input_str: str, **kwargs) -> None:
        """Tool 开始"""
        pass

    def on_tool_end(self, output: str, **kwargs) -> None:
        """Tool 结束"""
        pass

# ===== 3. 自定义 Handler 实现 =====
print("\n=== 3. 自定义 Handler ===")

class StreamingHandler(BaseCallbackHandler):
    """流式输出处理器"""

    def __init__(self):
        self.tokens = []

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.tokens.append(token)
        print(token, end="", flush=True)

    def get_full_response(self) -> str:
        return "".join(self.tokens)

class MonitorHandler(BaseCallbackHandler):
    """监控处理器"""

    def __init__(self):
        self.events = []
        self.start_time = None

    def on_llm_start(self, serialized: Dict, prompts: List[str], **kwargs) -> None:
        self.start_time = time.time()
        self.events.append(Event("llm_start", {"prompt_count": len(prompts)}))
        print(f"\n[Monitor] LLM 开始，{len(prompts)} 个 prompt")

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        duration = time.time() - self.start_time if self.start_time else 0
        self.events.append(Event("llm_end", {"duration": duration}))
        print(f"[Monitor] LLM 结束，耗时 {duration:.2f}s")

    def on_chain_start(self, serialized: Dict, inputs: Dict, **kwargs) -> None:
        self.events.append(Event("chain_start", inputs))
        print(f"[Monitor] Chain 开始: {list(inputs.keys())}")

    def on_chain_end(self, outputs: Dict, **kwargs) -> None:
        self.events.append(Event("chain_end", outputs))
        print(f"[Monitor] Chain 结束: {list(outputs.keys())}")

class CostHandler(BaseCallbackHandler):
    """成本统计处理器"""

    def __init__(self, price_per_1k_tokens: float = 0.002):
        self.total_tokens = 0
        self.total_cost = 0.0
        self.price_per_1k = price_per_1k_tokens

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        tokens = response.token_count
        cost = (tokens / 1000) * self.price_per_1k
        self.total_tokens += tokens
        self.total_cost += cost
        print(f"[Cost] 本次: {tokens} tokens (${cost:.4f}), "
              f"累计: {self.total_tokens} tokens (${self.total_cost:.4f})")

# ===== 4. CallbackManager 实现 =====
print("\n=== 4. CallbackManager ===")

class CallbackManager:
    """回调管理器"""

    def __init__(self, handlers: List[BaseCallbackHandler] = None):
        self.handlers = handlers or []

    def add_handler(self, handler: BaseCallbackHandler):
        self.handlers.append(handler)

    def on_llm_start(self, serialized: Dict, prompts: List[str], **kwargs):
        for handler in self.handlers:
            try:
                handler.on_llm_start(serialized, prompts, **kwargs)
            except Exception as e:
                print(f"Handler 错误: {e}")

    def on_llm_new_token(self, token: str, **kwargs):
        for handler in self.handlers:
            try:
                handler.on_llm_new_token(token, **kwargs)
            except Exception as e:
                pass  # 流式输出不打断

    def on_llm_end(self, response: LLMResult, **kwargs):
        for handler in self.handlers:
            try:
                handler.on_llm_end(response, **kwargs)
            except Exception as e:
                print(f"Handler 错误: {e}")

    def on_chain_start(self, serialized: Dict, inputs: Dict, **kwargs):
        for handler in self.handlers:
            try:
                handler.on_chain_start(serialized, inputs, **kwargs)
            except Exception as e:
                print(f"Handler 错误: {e}")

    def on_chain_end(self, outputs: Dict, **kwargs):
        for handler in self.handlers:
            try:
                handler.on_chain_end(outputs, **kwargs)
            except Exception as e:
                print(f"Handler 错误: {e}")

# ===== 5. 模拟 LLM 和 Chain =====
print("\n=== 5. 模拟 LLM ===")

class MockLLM:
    """模拟 LLM（支持流式）"""

    def __init__(self, callbacks: List[BaseCallbackHandler] = None, streaming: bool = False):
        self.callback_manager = CallbackManager(callbacks or [])
        self.streaming = streaming

    def invoke(self, prompt: str) -> str:
        # 通知开始
        self.callback_manager.on_llm_start({}, [prompt])

        # 模拟生成
        response_text = f"这是对「{prompt}」的回答。"

        if self.streaming:
            # 流式：逐字输出
            for char in response_text:
                self.callback_manager.on_llm_new_token(char)
                time.sleep(0.05)  # 模拟生成延迟
        else:
            time.sleep(0.5)  # 模拟非流式延迟

        # 通知结束
        result = LLMResult(text=response_text, token_count=len(response_text) * 2)
        self.callback_manager.on_llm_end(result)

        return response_text

# 测试流式输出
print("\n测试流式输出:")
streaming_handler = StreamingHandler()
llm = MockLLM(callbacks=[streaming_handler], streaming=True)
result = llm.invoke("你好")
print(f"\n完整响应: {streaming_handler.get_full_response()}")

# ===== 6. 多 Handler 组合 =====
print("\n=== 6. 多 Handler 组合 ===")

# 创建多个 Handler
stream_handler = StreamingHandler()
monitor_handler = MonitorHandler()
cost_handler = CostHandler(price_per_1k_tokens=0.002)

# 创建 LLM（组合多个 Handler）
llm = MockLLM(
    callbacks=[stream_handler, monitor_handler, cost_handler],
    streaming=True
)

print("\n组合使用多个 Handler:")
result = llm.invoke("写一首诗")

print(f"\n\n事件记录: {len(monitor_handler.events)} 个事件")
for event in monitor_handler.events:
    print(f"  - {event.type}: {event.data}")

# ===== 7. Chain 中使用 Callback =====
print("\n=== 7. Chain 中使用 Callback ===")

class MockChain:
    """模拟 Chain"""

    def __init__(self, llm: MockLLM, callbacks: List[BaseCallbackHandler] = None):
        self.llm = llm
        self.callback_manager = CallbackManager(callbacks or [])

    def invoke(self, inputs: Dict) -> Dict:
        # Chain 开始
        self.callback_manager.on_chain_start({}, inputs)

        # 执行 LLM
        prompt = inputs.get("input", "")
        response = self.llm.invoke(prompt)

        # Chain 结束
        outputs = {"output": response}
        self.callback_manager.on_chain_end(outputs)

        return outputs

# 测试
monitor = MonitorHandler()
chain = MockChain(
    llm=MockLLM(callbacks=[], streaming=False),
    callbacks=[monitor]
)

print("\nChain 执行:")
result = chain.invoke({"input": "什么是 Python？"})
print(f"结果: {result['output'][:30]}...")

# ===== 8. 通过 config 传递 Callback =====
print("\n=== 8. 通过 config 传递 ===")

class ConfigurableLLM:
    """支持 config 传递的 LLM"""

    def invoke(self, prompt: str, config: Dict = None) -> str:
        config = config or {}
        callbacks = config.get("callbacks", [])
        manager = CallbackManager(callbacks)

        manager.on_llm_start({}, [prompt])

        response = f"回答: {prompt}"
        time.sleep(0.2)

        result = LLMResult(text=response, token_count=50)
        manager.on_llm_end(result)

        return response

# 测试
llm = ConfigurableLLM()

# 调用时传递 callback
handler = MonitorHandler()
result = llm.invoke("测试", config={"callbacks": [handler]})
print(f"结果: {result}")

# ===== 9. 错误处理 =====
print("\n=== 9. 错误处理 ===")

class ErrorHandler(BaseCallbackHandler):
    """错误处理器"""

    def __init__(self):
        self.errors = []

    def on_llm_error(self, error: Exception, **kwargs) -> None:
        self.errors.append(error)
        print(f"[Error] LLM 错误: {error}")

class MockLLMWithError:
    """可能出错的 LLM"""

    def __init__(self, callbacks: List[BaseCallbackHandler] = None):
        self.callback_manager = CallbackManager(callbacks or [])

    def invoke(self, prompt: str) -> str:
        self.callback_manager.on_llm_start({}, [prompt])

        if "错误" in prompt:
            error = Exception("模拟的 LLM 错误")
            # 通知错误
            for handler in self.callback_manager.handlers:
                if hasattr(handler, 'on_llm_error'):
                    handler.on_llm_error(error)
            raise error

        return f"回答: {prompt}"

# 测试错误处理
error_handler = ErrorHandler()
llm = MockLLMWithError(callbacks=[error_handler])

try:
    llm.invoke("触发错误")
except:
    pass

print(f"记录的错误: {error_handler.errors}")

# ===== 10. 实际应用：Token 统计 =====
print("\n=== 10. Token 统计应用 ===")

class TokenCounter(BaseCallbackHandler):
    """Token 计数器"""

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.call_count = 0

    def on_llm_start(self, serialized: Dict, prompts: List[str], **kwargs) -> None:
        # 估算 prompt tokens
        tokens = sum(len(p) // 4 for p in prompts)
        self.prompt_tokens += tokens

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        self.completion_tokens += response.token_count
        self.total_tokens = self.prompt_tokens + self.completion_tokens
        self.call_count += 1

    def report(self):
        print(f"=== Token 使用报告 ===")
        print(f"调用次数: {self.call_count}")
        print(f"Prompt Tokens: {self.prompt_tokens}")
        print(f"Completion Tokens: {self.completion_tokens}")
        print(f"Total Tokens: {self.total_tokens}")
        print(f"预估成本: ${self.total_tokens * 0.002 / 1000:.4f}")

# 测试
counter = TokenCounter()
llm = MockLLM(callbacks=[counter], streaming=False)

# 多次调用
for i in range(3):
    llm.invoke(f"问题 {i+1}: 这是一个测试问题")

# 打印报告
counter.report()

print("\n=== 完成！===")
```

**运行输出示例：**
```
=== 1. 基础结构 ===

=== 2. BaseCallbackHandler ===

=== 3. 自定义 Handler ===

=== 4. CallbackManager ===

=== 5. 模拟 LLM ===

测试流式输出:
[Monitor] LLM 开始，1 个 prompt
这是对「你好」的回答。
[Monitor] LLM 结束，耗时 0.75s
完整响应: 这是对「你好」的回答。

=== 6. 多 Handler 组合 ===

组合使用多个 Handler:
[Monitor] LLM 开始，1 个 prompt
这是对「写一首诗」的回答。
[Monitor] LLM 结束，耗时 0.85s
[Cost] 本次: 26 tokens ($0.0001), 累计: 26 tokens ($0.0001)

事件记录: 2 个事件
  - llm_start: {'prompt_count': 1}
  - llm_end: {'duration': 0.85}

=== 10. Token 统计应用 ===
=== Token 使用报告 ===
调用次数: 3
Prompt Tokens: 21
Completion Tokens: 138
Total Tokens: 159
预估成本: $0.0003

=== 完成！===
```

---

## 8. 【面试必问】

### 问题1："LangChain 中如何实现流式输出？"

**普通回答（❌ 不出彩）：**
"使用 Callback 的 on_llm_new_token 方法。"

**出彩回答（✅ 推荐）：**

> **LangChain 流式输出有三种方式：**
>
> **1. Callback 方式**
> ```python
> class StreamHandler(BaseCallbackHandler):
>     def on_llm_new_token(self, token, **kwargs):
>         print(token, end="")
>
> llm = ChatOpenAI(streaming=True, callbacks=[StreamHandler()])
> ```
>
> **2. astream 方式**
> ```python
> async for chunk in llm.astream("你好"):
>     print(chunk.content, end="")
> ```
>
> **3. astream_events 方式（最详细）**
> ```python
> async for event in chain.astream_events(input, version="v2"):
>     if event["event"] == "on_llm_stream":
>         print(event["data"]["chunk"].content, end="")
> ```
>
> **关键配置：**
> - LLM 需要 `streaming=True`
> - Callback 需要实现 `on_llm_new_token`
>
> **实际应用**：在 Web 应用中，我用 WebSocket 配合 Callback 实现实时显示。用户提问后立即看到 AI 逐字回答，体验大幅提升。

**为什么这个回答出彩？**
1. ✅ 三种方式对比
2. ✅ 有代码示例
3. ✅ 提到关键配置
4. ✅ 有实际应用场景

---

### 问题2："如何监控 LangChain 应用的执行过程？"

**出彩回答（✅ 推荐）：**

> **监控主要通过 Callback 系统实现：**
>
> **1. 执行追踪**
> ```python
> class TraceHandler(BaseCallbackHandler):
>     def on_chain_start(self, serialized, inputs, **kwargs):
>         log.info(f"Chain 开始: {inputs}")
>     def on_llm_start(self, serialized, prompts, **kwargs):
>         log.info(f"LLM 调用: {prompts}")
> ```
>
> **2. 性能监控**
> ```python
> def on_llm_start(...):
>     self.start_time = time.time()
> def on_llm_end(...):
>     duration = time.time() - self.start_time
>     metrics.record("llm_latency", duration)
> ```
>
> **3. 成本统计**
> ```python
> def on_llm_end(self, response, **kwargs):
>     tokens = response.llm_output.get("token_usage", {})
>     self.total_cost += tokens.get("total_tokens", 0) * price
> ```
>
> **4. 集成 LangSmith**
> ```python
> # 官方追踪平台
> os.environ["LANGCHAIN_TRACING_V2"] = "true"
> os.environ["LANGCHAIN_API_KEY"] = "..."
> ```

---

## 9. 【化骨绵掌】

### 卡片1：Callback 是什么？ 🎯

**一句话：** Callback 是在执行过程中触发的钩子函数，用于监控和扩展。

**举例：**
```python
def on_llm_start(...):
    print("开始")  # LLM 开始时触发
```

**应用：** 日志、监控、流式输出。

---

### 卡片2：BaseCallbackHandler 基类 📡

**一句话：** 定义所有可监听事件的处理器基类。

**举例：**
```python
class MyHandler(BaseCallbackHandler):
    def on_llm_start(self, ...): pass
    def on_llm_end(self, ...): pass
```

**应用：** 继承并实现需要的方法。

---

### 卡片3：流式输出 on_llm_new_token 🌊

**一句话：** 每生成一个 token 就触发，实现实时显示。

**举例：**
```python
def on_llm_new_token(self, token, **kwargs):
    print(token, end="", flush=True)
```

**应用：** 提升用户体验的核心功能。

---

### 卡片4：CallbackManager 管理器 🎛️

**一句话：** 管理多个 Handler，负责事件的广播分发。

**举例：**
```python
manager = CallbackManager([handler1, handler2])
# 事件会通知所有 handler
```

**应用：** 同时使用多个 Handler。

---

### 卡片5：两种传递 Callback 的方式 📤

**一句话：** 构造时传递（全局）或调用时传递（单次）。

**举例：**
```python
# 构造时
llm = ChatOpenAI(callbacks=[handler])

# 调用时
result = chain.invoke(input, config={"callbacks": [handler]})
```

**应用：** 灵活控制 Callback 范围。

---

### 卡片6：astream 异步流式 🔄

**一句话：** 使用 async for 直接获取流式输出。

**举例：**
```python
async for chunk in llm.astream("你好"):
    print(chunk.content)
```

**应用：** 最简单的流式方式。

---

### 卡片7：RunnableConfig 配置传递 ⚙️

**一句话：** 通过 config 传递 callbacks、tags、metadata。

**举例：**
```python
config = {"callbacks": [handler], "tags": ["test"]}
result = chain.invoke(input, config=config)
```

**应用：** 统一的配置传递机制。

---

### 卡片8：事件类型速查 📋

**一句话：** LLM/Chain/Tool/Agent 各有 start/end/error 事件。

**举例：**
```
on_llm_start, on_llm_end, on_llm_error
on_chain_start, on_chain_end
on_tool_start, on_tool_end
on_agent_action, on_agent_finish
```

**应用：** 根据需求选择监听的事件。

---

### 卡片9：Callback 不阻塞主流程 ⚡

**一句话：** Callback 是旁路处理，不影响主流程的返回值。

**举例：**
```python
# 即使 Callback 很慢或出错
# 主流程依然正常返回结果
```

**应用：** 安全地添加监控逻辑。

---

### 卡片10：Callback 在 LangChain 源码中的位置 ⭐

**一句话：** 所有 Runnable 组件都支持 Callback，通过 config 传递。

**举例：**
```python
# langchain_core/runnables/base.py
class Runnable:
    def invoke(self, input, config=None):
        # config 中包含 callbacks
```

**应用：** 理解 Callback 与 Runnable 的集成。

---

## 10. 【一句话总结】

**Callback 是 LangChain 的事件监听机制，通过在 LLM/Chain/Tool 执行过程中触发钩子函数，实现流式输出、日志记录、性能监控和成本统计等功能，是构建可观测 LLM 应用的关键组件。**

---

## 📚 学习检查清单

- [ ] 理解 Callback 的事件驱动机制
- [ ] 会实现自定义 CallbackHandler
- [ ] 掌握流式输出的实现方式
- [ ] 了解 CallbackManager 的作用
- [ ] 知道两种传递 Callback 的方式
- [ ] 能够用 Callback 实现监控和日志

## 🔗 下一步学习

- **LangSmith**：官方的追踪和监控平台
- **Runnable 协议**：深入理解 Callback 与 Runnable 的集成
- **生产部署**：如何在生产环境使用 Callback

---

**版本：** v1.0
**最后更新：** 2025-01-14
