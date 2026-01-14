# 异步编程 async/await

> 原子化知识点 | Python高级特性 | LangChain 源码学习前置知识

---

## 1. 【30字核心】

**async/await 是 Python 的异步编程语法，让程序在等待 I/O 时可以做其他事，是 LangChain 流式输出的基础。**

---

## 2. 【第一性原理】

### 什么是第一性原理？

**第一性原理**：回到事物最基本的真理，从源头思考问题

### 异步编程的第一性原理 🎯

#### 1. 最基础的定义

**异步 = 不等待，先去做别的事，好了再回来**

仅此而已！没有更基础的了。

- **同步**：排队打饭，前一个人打完你才能打（等待）
- **异步**：点外卖，下单后你可以做别的事，送到了再去拿（不等待）

#### 2. 为什么需要异步编程？

**核心问题：I/O 操作很慢，同步等待浪费 CPU 时间**

```python
# 同步代码：发送3个请求需要3秒（串行等待）
import time

def fetch_url(url):
    time.sleep(1)  # 模拟网络请求
    return f"Response from {url}"

start = time.time()
results = [
    fetch_url("url1"),  # 等1秒
    fetch_url("url2"),  # 等1秒
    fetch_url("url3"),  # 等1秒
]
print(f"同步耗时: {time.time() - start:.1f}秒")  # 3秒
```

问题：
- CPU 99% 时间在等待 I/O
- 请求数量多时性能极差
- 无法利用等待时间做其他事

#### 3. 异步编程的三层价值

##### 价值1：高效利用等待时间

```python
import asyncio

async def fetch_url(url):
    await asyncio.sleep(1)  # 异步等待，可以做别的事
    return f"Response from {url}"

async def main():
    start = time.time()
    # 并发执行3个请求，只需要1秒
    results = await asyncio.gather(
        fetch_url("url1"),
        fetch_url("url2"),
        fetch_url("url3"),
    )
    print(f"异步耗时: {time.time() - start:.1f}秒")  # 1秒

asyncio.run(main())
```

##### 价值2：流式处理（边产生边消费）

```python
async def stream_tokens():
    """模拟 LLM 流式输出"""
    for token in ["Hello", " ", "World", "!"]:
        await asyncio.sleep(0.1)
        yield token

async def main():
    async for token in stream_tokens():
        print(token, end="", flush=True)  # 边收到边显示
```

##### 价值3：高并发处理

```python
# 同时处理1000个请求，不需要1000个线程
async def handle_request(request_id):
    await asyncio.sleep(0.1)
    return f"Done {request_id}"

async def main():
    tasks = [handle_request(i) for i in range(1000)]
    results = await asyncio.gather(*tasks)  # 并发执行
```

#### 4. 从第一性原理推导 LangChain 源码应用

**推理链：**

```
1. LLM API 调用是 I/O 密集型操作（网络请求）
   ↓
2. 同步调用会阻塞，用户体验差
   ↓
3. 异步调用可以在等待 LLM 响应时做其他事
   ↓
4. LLM 支持流式输出（一个字一个字返回）
   ↓
5. 需要异步迭代器来处理流式响应
   ↓
6. LangChain Runnable 定义 ainvoke/astream 异步方法
   ↓
7. 实现流式输出、并发调用、异步回调
```

#### 5. 一句话总结第一性原理

**异步编程让程序在等待 I/O 时不阻塞，通过协程实现高效的并发处理，是 LangChain 实现流式输出和高性能调用的基础。**

---

## 3. 【核心概念（全面覆盖）】

### 核心概念1：async def 和 await 语法 🏷️

**async def 定义协程函数，await 等待异步操作完成**

```python
import asyncio

# async def 定义协程函数
async def greet(name: str) -> str:
    print(f"开始问候 {name}")
    await asyncio.sleep(1)  # await 等待异步操作
    print(f"问候完成 {name}")
    return f"Hello, {name}!"

# 调用协程函数返回协程对象（不会执行）
coro = greet("Alice")
print(type(coro))  # <class 'coroutine'>

# 必须用 await 或 asyncio.run() 执行
async def main():
    result = await greet("Alice")
    print(result)

asyncio.run(main())
```

**关键规则：**

| 规则 | 说明 |
|-----|------|
| `async def` | 定义协程函数 |
| `await` | 只能在 `async def` 内部使用 |
| `await` 后面 | 必须是 awaitable 对象（协程、Task、Future）|
| 协程不会自动执行 | 必须 await 或用 asyncio.run() |

**在 LangChain 源码中的应用：**

```python
# langchain_core/runnables/base.py
class Runnable(Generic[Input, Output], ABC):

    @abstractmethod
    def invoke(self, input: Input) -> Output:
        """同步调用"""
        ...

    @abstractmethod
    async def ainvoke(self, input: Input) -> Output:
        """异步调用"""
        ...
```

---

### 核心概念2：asyncio 事件循环 📐

**事件循环是异步程序的核心，负责调度和执行协程**

```python
import asyncio

async def task1():
    print("Task 1 开始")
    await asyncio.sleep(2)
    print("Task 1 完成")
    return "Result 1"

async def task2():
    print("Task 2 开始")
    await asyncio.sleep(1)
    print("Task 2 完成")
    return "Result 2"

async def main():
    # 并发执行两个任务
    results = await asyncio.gather(task1(), task2())
    print(f"Results: {results}")

# 运行事件循环
asyncio.run(main())
```

**输出顺序：**
```
Task 1 开始
Task 2 开始
Task 2 完成  # Task 2 先完成（只等1秒）
Task 1 完成  # Task 1 后完成（等2秒）
Results: ['Result 1', 'Result 2']
```

**事件循环工作原理：**

```
┌─────────────────────────────────────────┐
│              事件循环                    │
├─────────────────────────────────────────┤
│  1. 执行 task1 直到 await              │
│  2. task1 暂停，切换到 task2           │
│  3. 执行 task2 直到 await              │
│  4. task2 暂停，检查 I/O 完成情况      │
│  5. task2 I/O 完成，恢复执行           │
│  6. task1 I/O 完成，恢复执行           │
│  7. 所有任务完成，返回结果             │
└─────────────────────────────────────────┘
```

**常用 asyncio 函数：**

| 函数 | 作用 | 示例 |
|-----|------|------|
| `asyncio.run(coro)` | 运行顶层协程 | `asyncio.run(main())` |
| `asyncio.gather(*coros)` | 并发执行多个协程 | `await asyncio.gather(t1(), t2())` |
| `asyncio.create_task(coro)` | 创建后台任务 | `task = asyncio.create_task(foo())` |
| `asyncio.sleep(seconds)` | 异步等待 | `await asyncio.sleep(1)` |
| `asyncio.wait_for(coro, timeout)` | 带超时的等待 | `await asyncio.wait_for(foo(), 5.0)` |
| `asyncio.Queue()` | 异步队列 | `queue = asyncio.Queue()` |

---

### 核心概念3：AsyncIterator 异步迭代器 🔧

**异步迭代器用于流式数据处理，LangChain 流式输出的基础**

```python
import asyncio
from typing import AsyncIterator

# async def + yield = 异步生成器
async def stream_numbers(n: int) -> AsyncIterator[int]:
    """异步生成器：产生0到n-1的数字"""
    for i in range(n):
        await asyncio.sleep(0.1)  # 模拟异步操作
        yield i  # yield 产生值

async def main():
    # async for 消费异步迭代器
    async for num in stream_numbers(5):
        print(f"Received: {num}")

asyncio.run(main())
```

**输出：**
```
Received: 0
Received: 1
Received: 2
Received: 3
Received: 4
```

**在 LangChain 源码中的应用：**

```python
# langchain_core/runnables/base.py
class Runnable(Generic[Input, Output], ABC):

    async def astream(
        self,
        input: Input,
        config: Optional[RunnableConfig] = None,
    ) -> AsyncIterator[Output]:
        """异步流式输出"""
        yield await self.ainvoke(input, config)

# 实际使用：流式接收 LLM 输出
async def stream_chat():
    async for chunk in chat_model.astream("Hello!"):
        print(chunk.content, end="", flush=True)
```

**自定义异步迭代器类：**

```python
class AsyncTokenStream:
    """模拟 LLM token 流"""

    def __init__(self, tokens: list[str]):
        self.tokens = tokens
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        if self.index >= len(self.tokens):
            raise StopAsyncIteration
        token = self.tokens[self.index]
        self.index += 1
        await asyncio.sleep(0.05)  # 模拟网络延迟
        return token

async def main():
    stream = AsyncTokenStream(["Hello", " ", "World", "!"])
    async for token in stream:
        print(token, end="", flush=True)
```

---

### 扩展概念4：asyncio.gather 并发执行 📋

```python
import asyncio
import time

async def fetch_data(source: str, delay: float) -> str:
    print(f"Fetching from {source}...")
    await asyncio.sleep(delay)
    return f"Data from {source}"

async def main():
    start = time.time()

    # 并发执行多个协程
    results = await asyncio.gather(
        fetch_data("API-1", 1.0),
        fetch_data("API-2", 2.0),
        fetch_data("API-3", 1.5),
    )

    print(f"耗时: {time.time() - start:.1f}秒")  # 2秒（最长的那个）
    print(f"结果: {results}")

asyncio.run(main())
```

**gather vs create_task：**

```python
async def main():
    # gather: 等待所有任务完成
    results = await asyncio.gather(task1(), task2())

    # create_task: 创建后台任务，立即返回
    t1 = asyncio.create_task(task1())
    t2 = asyncio.create_task(task2())

    # 可以做其他事...
    print("Tasks are running in background")

    # 后面再等待结果
    result1 = await t1
    result2 = await t2
```

---

### 扩展概念5：异步上下文管理器 🔄

```python
import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator

# 方式1：使用 @asynccontextmanager 装饰器
@asynccontextmanager
async def async_timer(name: str) -> AsyncIterator[None]:
    """异步计时器上下文管理器"""
    print(f"[{name}] 开始")
    start = asyncio.get_event_loop().time()
    try:
        yield
    finally:
        elapsed = asyncio.get_event_loop().time() - start
        print(f"[{name}] 完成，耗时 {elapsed:.2f}秒")

async def main():
    async with async_timer("LLM调用"):
        await asyncio.sleep(1)

# 方式2：实现 __aenter__ 和 __aexit__
class AsyncConnection:
    """模拟异步数据库连接"""

    def __init__(self, host: str):
        self.host = host

    async def __aenter__(self):
        print(f"Connecting to {self.host}...")
        await asyncio.sleep(0.1)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print(f"Disconnecting from {self.host}...")
        await asyncio.sleep(0.1)

    async def query(self, sql: str) -> str:
        await asyncio.sleep(0.1)
        return f"Result of: {sql}"

async def main():
    async with AsyncConnection("localhost") as conn:
        result = await conn.query("SELECT * FROM users")
        print(result)
```

---

### 扩展概念6：asyncio.Queue 异步队列 📬

```python
import asyncio

async def producer(queue: asyncio.Queue, items: list):
    """生产者：向队列添加数据"""
    for item in items:
        await asyncio.sleep(0.1)
        await queue.put(item)
        print(f"Produced: {item}")
    await queue.put(None)  # 结束信号

async def consumer(queue: asyncio.Queue):
    """消费者：从队列取数据"""
    while True:
        item = await queue.get()
        if item is None:
            break
        print(f"Consumed: {item}")
        queue.task_done()

async def main():
    queue = asyncio.Queue()

    # 并发运行生产者和消费者
    await asyncio.gather(
        producer(queue, ["a", "b", "c", "d"]),
        consumer(queue)
    )

asyncio.run(main())
```

**在 LangChain 源码中的应用：**
- 流式输出的 token 缓冲
- 异步回调队列
- 并发任务调度

---

### 扩展概念7：异步回调 AsyncCallback 📞

```python
import asyncio
from typing import Callable, Awaitable, Any

# 定义异步回调类型
AsyncCallback = Callable[[str], Awaitable[None]]

async def on_token(token: str) -> None:
    """异步回调：处理每个 token"""
    print(f"Token received: {token}")

async def stream_with_callback(
    text: str,
    callback: AsyncCallback
) -> None:
    """流式输出，每个 token 触发回调"""
    for char in text:
        await asyncio.sleep(0.05)
        await callback(char)

async def main():
    await stream_with_callback("Hello!", on_token)

asyncio.run(main())
```

**在 LangChain 源码中的应用：**

```python
# langchain_core/callbacks/base.py 简化版
class AsyncCallbackHandler:
    """异步回调处理器"""

    async def on_llm_start(self, prompts: list[str]) -> None:
        """LLM 开始调用时触发"""
        pass

    async def on_llm_new_token(self, token: str) -> None:
        """收到新 token 时触发"""
        pass

    async def on_llm_end(self, response: str) -> None:
        """LLM 调用结束时触发"""
        pass
```

---

## 4. 【最小可用】

掌握以下内容，就能开始阅读 LangChain 异步源码：

### 4.1 定义异步函数

```python
import asyncio

async def fetch_data() -> str:
    await asyncio.sleep(1)  # 异步等待
    return "data"
```

### 4.2 运行异步函数

```python
# 顶层运行
asyncio.run(fetch_data())

# 在异步函数内部
async def main():
    result = await fetch_data()
```

### 4.3 并发执行

```python
async def main():
    # 同时执行多个异步操作
    results = await asyncio.gather(
        fetch_data(),
        fetch_data(),
        fetch_data()
    )
```

### 4.4 异步迭代（流式处理）

```python
async def stream_tokens():
    for token in ["Hello", " ", "World"]:
        await asyncio.sleep(0.1)
        yield token

async def main():
    async for token in stream_tokens():
        print(token, end="")
```

### 4.5 理解 LangChain 的异步方法

```python
# LangChain Runnable 的异步接口
class Runnable:
    async def ainvoke(self, input) -> Output:
        """异步调用"""
        ...

    async def astream(self, input) -> AsyncIterator[Output]:
        """异步流式输出"""
        ...

    async def abatch(self, inputs: list) -> list[Output]:
        """异步批量处理"""
        ...
```

**这些知识足以：**
- 阅读 LangChain 源码中的 `ainvoke`、`astream` 方法
- 理解流式输出的实现原理
- 编写高性能的异步 LangChain 应用

---

## 5. 【1个类比】（双轨制）

### 类比1：async/await 异步语法

#### 🎨 前端视角：Promise 和 async/await

Python 的 async/await 和 JavaScript 的 async/await 几乎一模一样！

```javascript
// JavaScript
async function fetchData() {
  const response = await fetch('/api/data');
  const data = await response.json();
  return data;
}

// 并发执行
const results = await Promise.all([
  fetchData(),
  fetchData(),
  fetchData()
]);
```

```python
# Python
async def fetch_data():
    response = await aiohttp.get('/api/data')
    data = await response.json()
    return data

# 并发执行
results = await asyncio.gather(
    fetch_data(),
    fetch_data(),
    fetch_data()
)
```

**对应关系：**

| JavaScript | Python | 作用 |
|------------|--------|------|
| `async function` | `async def` | 定义异步函数 |
| `await` | `await` | 等待异步操作 |
| `Promise.all()` | `asyncio.gather()` | 并发执行 |
| `new Promise()` | `asyncio.Future()` | 底层 Promise/Future |

#### 🧒 小朋友视角：点外卖

async/await 就像点外卖：

- **同步（synchronous）= 去餐厅吃饭**
  - 走到餐厅 → 排队 → 点餐 → 等待 → 吃饭
  - 全程你只能等着，不能做别的事

- **异步（async）= 点外卖**
  - 下单（async def）→ 回家做别的事（不阻塞）
  - 外卖到了（await）→ 取餐吃饭
  - 等待期间你可以写作业、玩游戏！

**生活例子：**
```
点外卖流程：
1. async def 下单() -> 返回「订单」
2. 回家（不等待，做别的事）
3. await 下单()  # 外卖到了，取餐
4. 吃饭

同时点3份外卖（gather）：
- 不是等第1份到了再点第2份
- 而是同时下3个单，哪个先到吃哪个
- 总时间 = 最慢那份的时间
```

---

### 类比2：asyncio.gather 并发执行

#### 🎨 前端视角：Promise.all

```javascript
// JavaScript: Promise.all
const results = await Promise.all([
  fetch('/api/user'),
  fetch('/api/posts'),
  fetch('/api/comments')
]);
// 等待所有请求完成，返回结果数组
```

```python
# Python: asyncio.gather
results = await asyncio.gather(
    fetch_user(),
    fetch_posts(),
    fetch_comments()
)
# 等待所有协程完成，返回结果列表
```

#### 🧒 小朋友视角：同时做多件事

gather 就像妈妈同时安排多个任务：

- **串行（不用 gather）**：
  - 先洗碗（10分钟）
  - 再扫地（10分钟）
  - 再洗衣服（30分钟）
  - 总共：50分钟

- **并行（用 gather）**：
  - 洗衣机洗衣服（自动，30分钟）
  - 同时你洗碗（10分钟）
  - 然后扫地（10分钟）
  - 总共：30分钟（等洗衣机）

**生活例子：**
```
gather([洗衣服, 洗碗, 扫地])
↓
- 洗衣机开始转（不用人盯着）
- 你去洗碗
- 洗完碗去扫地
- 扫完地洗衣机也好了
↓
全部完成！只花了最长任务的时间
```

---

### 类比3：AsyncIterator 异步迭代器

#### 🎨 前端视角：ReadableStream / Async Generator

```javascript
// JavaScript: Async Generator
async function* streamTokens() {
  for (const token of ["Hello", " ", "World"]) {
    await delay(100);
    yield token;
  }
}

// 消费
for await (const token of streamTokens()) {
  console.log(token);
}
```

```python
# Python: Async Generator
async def stream_tokens():
    for token in ["Hello", " ", "World"]:
        await asyncio.sleep(0.1)
        yield token

# 消费
async for token in stream_tokens():
    print(token)
```

#### 🧒 小朋友视角：传送带

AsyncIterator 就像工厂的传送带：

- **普通列表**：一次把所有玩具都搬过来（要等全部做好）
- **异步迭代器**：传送带一个一个传过来（做好一个传一个）

**生活例子：**
```
LLM 回答问题就像传送带：

普通方式（不用流式）：
- 等 LLM 想完整个答案（10秒）
- 一次性显示全部
- 用户：等得好无聊...

流式方式（async for）：
- LLM 想一个字，传一个字
- 用户立刻看到字出现
- 就像看打字机一样，一个字一个字蹦出来！
```

---

### 类比4：await 等待

#### 🎨 前端视角：await Promise

```javascript
// JavaScript
async function main() {
  // await 等待 Promise resolve
  const user = await fetchUser();
  console.log(user);
}
```

```python
# Python
async def main():
    # await 等待协程完成
    user = await fetch_user()
    print(user)
```

#### 🧒 小朋友视角：排号等叫号

await 就像在医院排号：

- 拿个号（调用 async 函数）
- 去休息区坐着（不用一直站着等）
- 等叫到你的号（await）
- 去看医生（继续执行）

**关键：** 等叫号的时候你可以玩手机，不用一直盯着叫号屏！

---

### 类比总结表

| Python 异步概念 | JavaScript 类比 | 小朋友类比 |
|---------------|----------------|-----------|
| `async def` | `async function` | 下外卖订单 |
| `await` | `await` | 等外卖到了取餐 |
| `asyncio.gather()` | `Promise.all()` | 同时点多份外卖 |
| `asyncio.run()` | 顶层 await | 开始执行订单 |
| `async for` | `for await` | 传送带一个个传 |
| `yield` (async) | `yield` (async) | 做好一个送一个 |
| `asyncio.Queue` | 消息队列 | 排队取餐口 |
| `asyncio.sleep()` | `setTimeout` (Promise) | 定时器 |
| 协程 Coroutine | Promise | 「待完成」的任务 |
| 事件循环 | Event Loop | 外卖调度中心 |

---

## 6. 【反直觉点】

### 误区1：async/await 就是多线程 ❌

**为什么错？**
- async/await 是**单线程**的协程
- 协程在**同一个线程**内切换，没有线程切换开销
- 不能利用多核 CPU（和多线程不同）

**为什么人们容易这样错？**
因为 async/await 可以"同时"做多件事，看起来像多线程。但实际上是在等待 I/O 时切换执行，本质是单线程。

**正确理解：**

```python
import asyncio
import threading

async def show_thread():
    print(f"协程运行在线程: {threading.current_thread().name}")
    await asyncio.sleep(0.1)
    print(f"还是同一个线程: {threading.current_thread().name}")

async def main():
    # 所有协程都在同一个线程
    await asyncio.gather(
        show_thread(),
        show_thread(),
        show_thread()
    )

asyncio.run(main())
# 输出：全部都是 MainThread
```

**适用场景对比：**

| 场景 | 推荐方式 | 原因 |
|-----|---------|------|
| I/O 密集（网络、文件）| async/await | 等待时可以做别的事 |
| CPU 密集（计算）| 多进程 multiprocessing | 利用多核 |
| 简单并发 | 多线程 threading | 实现简单 |

---

### 误区2：调用 async 函数就会自动执行 ❌

**为什么错？**
- 调用 async 函数返回**协程对象**，不会执行
- 必须用 `await` 或 `asyncio.run()` 才会执行
- 不 await 的协程会被垃圾回收，代码不会运行

**为什么人们容易这样错？**
因为普通函数调用就会执行。async 函数看起来语法一样，容易误以为调用就会执行。

**正确理解：**

```python
import asyncio

async def greet():
    print("Hello!")
    return "Done"

# ❌ 错误：只创建协程对象，不执行
coro = greet()  # 没有任何输出
print(type(coro))  # <class 'coroutine'>
# 警告：RuntimeWarning: coroutine 'greet' was never awaited

# ✅ 正确：用 await 执行
async def main():
    result = await greet()  # 输出 "Hello!"
    print(result)

asyncio.run(main())

# ✅ 正确：用 asyncio.run() 执行
asyncio.run(greet())
```

---

### 误区3：await asyncio.sleep(0) 没有意义 ❌

**为什么错？**
- `await asyncio.sleep(0)` 会**让出控制权**给事件循环
- 其他等待的协程有机会执行
- 是实现"协作式多任务"的关键

**为什么人们容易这样错？**
因为 `time.sleep(0)` 确实没意义。但 asyncio.sleep(0) 是异步的，会触发任务切换。

**正确理解：**

```python
import asyncio

async def task1():
    for i in range(3):
        print(f"Task 1: {i}")
        await asyncio.sleep(0)  # 让出控制权

async def task2():
    for i in range(3):
        print(f"Task 2: {i}")
        await asyncio.sleep(0)  # 让出控制权

async def main():
    await asyncio.gather(task1(), task2())

asyncio.run(main())
```

**输出（交替执行）：**
```
Task 1: 0
Task 2: 0
Task 1: 1
Task 2: 1
Task 1: 2
Task 2: 2
```

如果没有 `await asyncio.sleep(0)`，task1 会一次性执行完。

---

## 7. 【实战代码】

```python
"""
示例：使用 async/await 构建 LangChain 风格的异步组件
演示异步调用、流式输出、并发处理
"""

import asyncio
from typing import AsyncIterator, Optional, List
from dataclasses import dataclass
import time

# ===== 1. 基础异步函数 =====
print("=== 1. 基础异步函数 ===")

async def fetch_completion(prompt: str, delay: float = 0.5) -> str:
    """模拟异步 LLM API 调用"""
    print(f"Fetching completion for: {prompt[:20]}...")
    await asyncio.sleep(delay)
    return f"Response to: {prompt}"

async def demo_basic():
    start = time.time()
    result = await fetch_completion("Hello, how are you?")
    print(f"Result: {result}")
    print(f"耗时: {time.time() - start:.2f}秒")

asyncio.run(demo_basic())

# ===== 2. 并发执行 gather =====
print("\n=== 2. 并发执行 gather ===")

async def demo_gather():
    start = time.time()

    # 并发执行3个请求
    results = await asyncio.gather(
        fetch_completion("Question 1", 1.0),
        fetch_completion("Question 2", 1.0),
        fetch_completion("Question 3", 1.0),
    )

    print(f"Results: {len(results)} responses")
    print(f"耗时: {time.time() - start:.2f}秒")  # 约1秒，不是3秒

asyncio.run(demo_gather())

# ===== 3. 异步流式输出 =====
print("\n=== 3. 异步流式输出 ===")

async def stream_tokens(text: str, delay: float = 0.05) -> AsyncIterator[str]:
    """模拟 LLM 流式输出"""
    for char in text:
        await asyncio.sleep(delay)
        yield char

async def demo_stream():
    print("Streaming: ", end="")
    async for token in stream_tokens("Hello, I am an AI assistant!"):
        print(token, end="", flush=True)
    print()  # 换行

asyncio.run(demo_stream())

# ===== 4. 异步 Runnable 接口 =====
print("\n=== 4. 异步 Runnable 接口 ===")

from abc import ABC, abstractmethod
from typing import TypeVar, Generic

Input = TypeVar('Input')
Output = TypeVar('Output')

class AsyncRunnable(Generic[Input, Output], ABC):
    """异步可运行组件 - 模拟 LangChain Runnable"""

    @abstractmethod
    async def ainvoke(self, input: Input) -> Output:
        """异步调用"""
        ...

    async def astream(self, input: Input) -> AsyncIterator[Output]:
        """异步流式输出（默认实现）"""
        yield await self.ainvoke(input)

    async def abatch(self, inputs: List[Input]) -> List[Output]:
        """异步批量处理"""
        return await asyncio.gather(
            *[self.ainvoke(inp) for inp in inputs]
        )

@dataclass
class Message:
    role: str
    content: str

class MockChatModel(AsyncRunnable[str, Message]):
    """模拟异步聊天模型"""

    def __init__(self, model_name: str = "mock-gpt"):
        self.model_name = model_name

    async def ainvoke(self, input: str) -> Message:
        await asyncio.sleep(0.5)  # 模拟 API 延迟
        return Message(role="assistant", content=f"Echo: {input}")

    async def astream(self, input: str) -> AsyncIterator[str]:
        """重写流式输出"""
        response = f"Echo: {input}"
        for char in response:
            await asyncio.sleep(0.03)
            yield char

async def demo_runnable():
    model = MockChatModel()

    # 单次调用
    result = await model.ainvoke("Hello!")
    print(f"Single invoke: {result.content}")

    # 流式输出
    print("Streaming: ", end="")
    async for chunk in model.astream("World!"):
        print(chunk, end="", flush=True)
    print()

    # 批量处理（并发）
    start = time.time()
    results = await model.abatch(["Q1", "Q2", "Q3"])
    print(f"Batch results: {[r.content for r in results]}")
    print(f"Batch 耗时: {time.time() - start:.2f}秒")  # 约0.5秒

asyncio.run(demo_runnable())

# ===== 5. 异步回调处理 =====
print("\n=== 5. 异步回调处理 ===")

class AsyncStreamHandler:
    """异步流式回调处理器"""

    def __init__(self):
        self.tokens: List[str] = []

    async def on_token(self, token: str) -> None:
        """收到新 token 时的回调"""
        self.tokens.append(token)
        print(f"[Handler] Token: '{token}'")

    async def on_complete(self) -> None:
        """完成时的回调"""
        full_text = "".join(self.tokens)
        print(f"[Handler] Complete! Full text: {full_text}")

async def stream_with_handler(
    text: str,
    handler: AsyncStreamHandler
) -> str:
    """带回调的流式输出"""
    for char in text:
        await asyncio.sleep(0.02)
        await handler.on_token(char)
    await handler.on_complete()
    return "".join(handler.tokens)

async def demo_handler():
    handler = AsyncStreamHandler()
    result = await stream_with_handler("Hi!", handler)
    print(f"Final result: {result}")

asyncio.run(demo_handler())

# ===== 6. 异步队列生产者消费者 =====
print("\n=== 6. 异步队列 ===")

async def token_producer(queue: asyncio.Queue, tokens: List[str]) -> None:
    """生产者：模拟 LLM 产生 token"""
    for token in tokens:
        await asyncio.sleep(0.05)
        await queue.put(token)
        print(f"[Producer] Put: {token}")
    await queue.put(None)  # 结束信号

async def token_consumer(queue: asyncio.Queue) -> str:
    """消费者：收集 token"""
    result = []
    while True:
        token = await queue.get()
        if token is None:
            break
        result.append(token)
        print(f"[Consumer] Got: {token}")
    return "".join(result)

async def demo_queue():
    queue = asyncio.Queue()

    # 并发运行生产者和消费者
    producer_task = asyncio.create_task(
        token_producer(queue, list("Hello!"))
    )
    consumer_task = asyncio.create_task(
        token_consumer(queue)
    )

    await producer_task
    result = await consumer_task
    print(f"Final assembled: {result}")

asyncio.run(demo_queue())

# ===== 7. 超时处理 =====
print("\n=== 7. 超时处理 ===")

async def slow_operation() -> str:
    await asyncio.sleep(5)  # 模拟很慢的操作
    return "Done"

async def demo_timeout():
    try:
        # 设置2秒超时
        result = await asyncio.wait_for(slow_operation(), timeout=2.0)
        print(f"Result: {result}")
    except asyncio.TimeoutError:
        print("Operation timed out!")

asyncio.run(demo_timeout())

# ===== 8. 实际应用：并发调用多个模型 =====
print("\n=== 8. 并发调用多个模型 ===")

async def call_model(model_name: str, prompt: str) -> dict:
    """调用单个模型"""
    delay = {"gpt-4": 1.0, "claude": 0.8, "gemini": 0.6}[model_name]
    await asyncio.sleep(delay)
    return {
        "model": model_name,
        "response": f"[{model_name}] Response to: {prompt}"
    }

async def demo_multi_model():
    prompt = "What is AI?"
    start = time.time()

    # 并发调用多个模型
    results = await asyncio.gather(
        call_model("gpt-4", prompt),
        call_model("claude", prompt),
        call_model("gemini", prompt),
    )

    for r in results:
        print(f"{r['model']}: {r['response'][:50]}...")

    print(f"总耗时: {time.time() - start:.2f}秒")  # 约1秒（最慢的）

asyncio.run(demo_multi_model())
```

**运行输出示例：**
```
=== 1. 基础异步函数 ===
Fetching completion for: Hello, how are you?...
Result: Response to: Hello, how are you?
耗时: 0.50秒

=== 2. 并发执行 gather ===
Fetching completion for: Question 1...
Fetching completion for: Question 2...
Fetching completion for: Question 3...
Results: 3 responses
耗时: 1.00秒

=== 3. 异步流式输出 ===
Streaming: Hello, I am an AI assistant!

=== 4. 异步 Runnable 接口 ===
Single invoke: Echo: Hello!
Streaming: Echo: World!
Batch results: ['Echo: Q1', 'Echo: Q2', 'Echo: Q3']
Batch 耗时: 0.50秒

=== 5. 异步回调处理 ===
[Handler] Token: 'H'
[Handler] Token: 'i'
[Handler] Token: '!'
[Handler] Complete! Full text: Hi!
Final result: Hi!

=== 6. 异步队列 ===
[Producer] Put: H
[Consumer] Got: H
...
Final assembled: Hello!

=== 7. 超时处理 ===
Operation timed out!

=== 8. 并发调用多个模型 ===
gpt-4: [gpt-4] Response to: What is AI?...
claude: [claude] Response to: What is AI?...
gemini: [gemini] Response to: What is AI?...
总耗时: 1.00秒
```

---

## 8. 【面试必问】

### 问题："Python 的 async/await 和多线程有什么区别？"

**普通回答（❌ 不出彩）：**
"async/await 是异步编程，多线程是多线程编程，async/await 效率更高。"

**出彩回答（✅ 推荐）：**

> **async/await 和多线程有三个核心区别：**
>
> 1. **执行方式**：
>    - async/await 是**协程**，单线程内切换执行
>    - 多线程是**真正的并行**，操作系统调度
>
> 2. **切换开销**：
>    - 协程切换只是函数调用，开销极小
>    - 线程切换需要保存/恢复上下文，开销大
>
> 3. **适用场景**：
>    - async/await 适合 **I/O 密集型**（网络请求、文件读写）
>    - 多线程/多进程适合 **CPU 密集型**（大量计算）
>
> **关键原理**：async/await 在遇到 `await` 时会**让出控制权**给事件循环，事件循环可以调度其他协程执行。这样在等待网络响应时，CPU 不会闲着。
>
> **在 LangChain 中的应用**：LLM API 调用是 I/O 密集型操作，使用 async/await 可以：
> - 同时向多个模型发送请求（`asyncio.gather`）
> - 流式输出时不阻塞主线程（`async for`）
> - 提高应用的并发性能

**为什么这个回答出彩？**
1. ✅ 分三点对比，结构清晰
2. ✅ 说明了原理（让出控制权）
3. ✅ 明确了适用场景
4. ✅ 联系了 LangChain 实际应用

---

### 问题："如何在 Python 中实现流式输出？"

**普通回答（❌ 不出彩）：**
"用 yield 生成器，或者用 async for 异步迭代。"

**出彩回答（✅ 推荐）：**

> **Python 实现流式输出有两种方式：**
>
> 1. **同步生成器**（简单场景）：
> ```python
> def stream_tokens(text):
>     for char in text:
>         yield char
>
> for token in stream_tokens("Hello"):
>     print(token, end="")
> ```
>
> 2. **异步生成器**（I/O 场景）：
> ```python
> async def stream_tokens(text):
>     for char in text:
>         await asyncio.sleep(0.01)  # 模拟网络延迟
>         yield char
>
> async for token in stream_tokens("Hello"):
>     print(token, end="")
> ```
>
> **LangChain 的实现**：
> ```python
> # Runnable 接口
> async def astream(self, input) -> AsyncIterator[Output]:
>     yield await self.ainvoke(input)
>
> # 使用
> async for chunk in model.astream("Hello"):
>     print(chunk.content, end="")
> ```
>
> **异步生成器的优势**：在等待下一个 token 时，事件循环可以处理其他任务（比如更新 UI、处理其他请求）。

---

## 9. 【化骨绵掌】

### 卡片1：什么是异步编程？ 🎯

**一句话：** 异步编程让程序在等待 I/O 时可以做其他事，不会傻等。

**举例：**
```python
# 同步：等1秒什么都不能做
time.sleep(1)

# 异步：等1秒的同时可以做别的
await asyncio.sleep(1)
```

**应用：** LangChain 的 `ainvoke` 让等待 LLM 响应时不阻塞。

---

### 卡片2：async def 定义协程 📐

**一句话：** `async def` 定义协程函数，调用它返回协程对象而不是执行。

**举例：**
```python
async def greet():
    return "Hello"

coro = greet()  # 协程对象，不执行
result = await coro  # 执行并获取结果
```

**应用：** LangChain 的 `ainvoke`, `astream` 都是 async def。

---

### 卡片3：await 等待异步操作 ⏳

**一句话：** `await` 等待协程完成，同时让出控制权给其他协程。

**举例：**
```python
async def main():
    result = await fetch_data()  # 等待完成
    print(result)
```

**应用：** `await model.ainvoke("Hello")` 等待 LLM 响应。

---

### 卡片4：asyncio.run() 运行协程 🚀

**一句话：** `asyncio.run()` 是运行顶层协程的入口。

**举例：**
```python
async def main():
    print("Hello, Async!")

# 运行入口
asyncio.run(main())
```

**应用：** 在普通 Python 脚本中运行 LangChain 异步代码。

---

### 卡片5：asyncio.gather() 并发执行 🔄

**一句话：** `gather` 同时执行多个协程，总时间等于最长的那个。

**举例：**
```python
results = await asyncio.gather(
    fetch_user(),      # 1秒
    fetch_posts(),     # 2秒
    fetch_comments()   # 1.5秒
)
# 总耗时：2秒（不是4.5秒）
```

**应用：** 同时调用多个 LLM 模型进行对比。

---

### 卡片6：async for 异步迭代 📦

**一句话：** `async for` 用于消费异步迭代器，实现流式处理。

**举例：**
```python
async def stream_tokens():
    for token in ["Hello", " ", "World"]:
        await asyncio.sleep(0.1)
        yield token

async for token in stream_tokens():
    print(token, end="")
```

**应用：** `async for chunk in model.astream("Hi"):` 流式输出。

---

### 卡片7：异步生成器 async + yield 🌊

**一句话：** `async def` + `yield` = 异步生成器，边产生边消费。

**举例：**
```python
async def stream():
    for i in range(3):
        await asyncio.sleep(0.1)
        yield i  # 产生一个值
```

**应用：** LangChain `astream` 方法返回 `AsyncIterator`。

---

### 卡片8：asyncio.create_task() 后台任务 🎭

**一句话：** `create_task` 创建后台运行的任务，不立即等待结果。

**举例：**
```python
task = asyncio.create_task(long_operation())
# 做其他事...
result = await task  # 需要结果时再等待
```

**应用：** 在后台预加载数据或预热模型。

---

### 卡片9：超时处理 wait_for ⏰

**一句话：** `wait_for` 给异步操作加上超时限制。

**举例：**
```python
try:
    result = await asyncio.wait_for(
        slow_llm_call(),
        timeout=30.0
    )
except asyncio.TimeoutError:
    print("LLM 响应超时!")
```

**应用：** 避免 LLM 调用无限等待。

---

### 卡片10：在 LangChain 源码中的应用 ⭐

**一句话：** LangChain Runnable 定义了完整的异步接口：ainvoke, astream, abatch。

**举例：**
```python
# langchain_core/runnables/base.py
class Runnable(Generic[Input, Output], ABC):
    async def ainvoke(self, input: Input) -> Output: ...
    async def astream(self, input: Input) -> AsyncIterator[Output]: ...
    async def abatch(self, inputs: List[Input]) -> List[Output]: ...
```

**应用：** 理解这个模式就能编写高性能的 LangChain 应用。

---

## 10. 【一句话总结】

**async/await 是 Python 的协程语法，通过事件循环在单线程内实现高效并发，asyncio.gather 实现并发调用，AsyncIterator 实现流式输出，是 LangChain ainvoke/astream 的底层基础。**

---

## 📚 学习检查清单

- [ ] 理解 `async def` 定义协程函数
- [ ] 会使用 `await` 等待异步操作
- [ ] 知道 `asyncio.run()` 是运行入口
- [ ] 会使用 `asyncio.gather()` 并发执行
- [ ] 理解 `async for` 消费异步迭代器
- [ ] 能写 `async def` + `yield` 异步生成器
- [ ] 理解协程和线程的区别

## 🔗 下一步学习

- **上下文管理器**：异步上下文管理器 `async with`
- **Runnable 协议**：LangChain 的 ainvoke/astream 实现
- **Callback 回调系统**：AsyncCallbackHandler

---

**版本：** v1.0
**最后更新：** 2025-01-14
