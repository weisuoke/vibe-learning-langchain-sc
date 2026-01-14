# Memory 记忆系统

> 原子化知识点 | LangChain 使用 | LangChain 源码学习核心知识

---

## 1. 【30字核心】

**Memory 让 LLM 能够记住对话历史，通过消息列表维护上下文，ConversationBufferMemory 是最基础的实现。**

---

## 2. 【第一性原理】

### 什么是第一性原理？

**第一性原理**：回到事物最基本的真理，从源头思考问题

### Memory 记忆系统的第一性原理 🎯

#### 1. 最基础的定义

**Memory = 存储和检索对话历史的机制**

仅此而已！没有更基础的了。

```python
# Memory 的本质
memory = []

# 保存对话
memory.append({"role": "user", "content": "你好"})
memory.append({"role": "assistant", "content": "你好！"})

# 下次对话时，把历史也传给 LLM
messages = memory + [{"role": "user", "content": "我叫什么？"}]
# LLM 看到历史，知道之前的上下文
```

#### 2. 为什么需要 Memory？

**核心问题：LLM 本身是无状态的，每次调用都是独立的**

```python
# 没有 Memory 的问题
llm.invoke("我叫小明")  # LLM: 你好小明！
llm.invoke("我叫什么？")  # LLM: 我不知道你叫什么...

# LLM 不记得上一次对话！

# 有 Memory 的解决方案
memory.save("user", "我叫小明")
memory.save("assistant", "你好小明！")

# 下次对话时带上历史
history = memory.load()
llm.invoke(history + ["我叫什么？"])  # LLM: 你叫小明
```

#### 3. Memory 的三层价值

##### 价值1：上下文连续 - 多轮对话

```python
# 多轮对话需要记住之前说过什么
对话1: "推荐一本书" → "《Python编程》"
对话2: "这本书多少钱？" → 需要知道"这本书"指的是《Python编程》
对话3: "有电子版吗？" → 还是在问同一本书
```

##### 价值2：个性化 - 记住用户信息

```python
# 记住用户偏好
memory.save("用户喜欢科幻小说")
memory.save("用户是程序员")

# 后续推荐基于这些信息
"推荐一本书" → 基于偏好推荐科幻相关技术书
```

##### 价值3：效率 - 避免重复

```python
# 不需要每次都重复说明
用户: "我在做一个电商项目"
...10轮对话后...
用户: "订单模块怎么设计？"
# LLM 知道这是电商项目的订单模块
```

#### 4. 从第一性原理推导 Memory 设计

**推理链：**

```
1. LLM 是无状态的
   ↓
2. 多轮对话需要上下文
   ↓
3. 需要存储对话历史
   ↓
4. 历史可能很长，超出上下文窗口
   ↓
5. 需要策略：全量保存/窗口限制/摘要压缩
   ↓
6. 不同场景需要不同策略
   ↓
7. LangChain 提供多种 Memory 类型
```

#### 5. 一句话总结第一性原理

**Memory 是为无状态的 LLM 提供有状态对话能力的机制，核心是存储历史并在适当时机注入到 prompt 中。**

---

## 3. 【核心概念（全面覆盖）】

### 核心概念1：ChatMessageHistory 消息历史 📜

**ChatMessageHistory 是存储对话消息的基础容器**

```python
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# 创建消息历史
history = InMemoryChatMessageHistory()

# 添加消息
history.add_message(SystemMessage(content="你是一个助手"))
history.add_message(HumanMessage(content="你好"))
history.add_message(AIMessage(content="你好！有什么可以帮助你？"))

# 获取所有消息
messages = history.messages
for msg in messages:
    print(f"{msg.type}: {msg.content}")

# 清空历史
history.clear()
```

**消息类型：**

| 类型 | 用途 | 示例 |
|-----|------|------|
| `SystemMessage` | 系统指令 | "你是一个翻译助手" |
| `HumanMessage` | 用户输入 | "翻译这句话" |
| `AIMessage` | AI 回复 | "Here is the translation" |
| `ToolMessage` | 工具结果 | "天气：晴天 25度" |

**在 LangChain 源码中的应用：**

```python
# langchain_core/chat_history.py
class BaseChatMessageHistory(ABC):
    """消息历史基类"""

    @abstractmethod
    def add_message(self, message: BaseMessage) -> None:
        """添加消息"""
        pass

    @abstractmethod
    def clear(self) -> None:
        """清空历史"""
        pass

    @property
    def messages(self) -> List[BaseMessage]:
        """获取所有消息"""
        pass

    # 便捷方法
    def add_user_message(self, message: str) -> None:
        self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        self.add_message(AIMessage(content=message))
```

---

### 核心概念2：ConversationBufferMemory 缓冲记忆 💾

**ConversationBufferMemory 保存完整的对话历史，是最简单的 Memory 实现**

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

# 创建 Memory
memory = ConversationBufferMemory(
    return_messages=True,  # 返回消息对象（推荐）
    memory_key="history",  # 变量名
    input_key="input",     # 输入变量名
    output_key="output"    # 输出变量名
)

# 手动保存对话
memory.save_context(
    {"input": "你好，我叫小明"},
    {"output": "你好小明！很高兴认识你"}
)

# 加载历史
history = memory.load_memory_variables({})
print(history)
# {'history': [HumanMessage(...), AIMessage(...)]}

# 与 Chain 集成
llm = ChatOpenAI()
chain = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

# 多轮对话
chain.invoke({"input": "我叫什么？"})
# LLM 可以看到之前的对话，知道你叫小明
```

**ConversationBufferMemory 的特点：**

| 特点 | 说明 |
|-----|------|
| 简单 | 无需配置，直接存储 |
| 完整 | 保留所有对话历史 |
| 局限 | 历史过长会超出上下文窗口 |
| 适用 | 短对话、测试开发 |

---

### 核心概念3：历史管理策略 📊

**不同的 Memory 类型对应不同的历史管理策略**

#### 3.1 ConversationBufferWindowMemory - 窗口限制

```python
from langchain.memory import ConversationBufferWindowMemory

# 只保留最近 5 轮对话
window_memory = ConversationBufferWindowMemory(
    k=5,  # 保留最近 5 轮
    return_messages=True
)

# 即使对话了 100 轮，也只保留最近 5 轮
```

#### 3.2 ConversationSummaryMemory - 摘要压缩

```python
from langchain.memory import ConversationSummaryMemory

# 用 LLM 对历史进行摘要
summary_memory = ConversationSummaryMemory(
    llm=llm,  # 用于生成摘要的 LLM
    return_messages=True
)

# 历史被压缩成摘要
# 原始：10 轮对话 = 2000 tokens
# 摘要后：200 tokens
```

#### 3.3 ConversationTokenBufferMemory - Token 限制

```python
from langchain.memory import ConversationTokenBufferMemory

# 按 token 数量限制
token_memory = ConversationTokenBufferMemory(
    llm=llm,
    max_token_limit=1000  # 最多 1000 tokens
)

# 超过限制时，删除最早的消息
```

#### 3.4 选择策略对比

| Memory 类型 | 策略 | 优点 | 缺点 | 适用场景 |
|------------|------|------|------|---------|
| Buffer | 全量保存 | 信息完整 | 可能超限 | 短对话 |
| BufferWindow | 窗口限制 | 简单可控 | 丢失早期信息 | 一般对话 |
| Summary | 摘要压缩 | 保留要点 | 需要额外 LLM 调用 | 长对话 |
| TokenBuffer | Token 限制 | 精确控制 | 可能截断重要信息 | 生产环境 |

---

### 核心概念4：RunnableWithMessageHistory - LCEL 风格 🔗

**新版 LangChain 推荐使用 RunnableWithMessageHistory 管理对话历史**

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

# 存储会话历史的字典
store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """根据 session_id 获取或创建历史"""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# 创建带历史占位符的 prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手"),
    MessagesPlaceholder(variable_name="history"),  # 历史消息占位
    ("human", "{input}")
])

# 创建 Chain
llm = ChatOpenAI()
chain = prompt | llm

# 包装为带历史的 Chain
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# 使用（指定 session_id）
response = chain_with_history.invoke(
    {"input": "我叫小明"},
    config={"configurable": {"session_id": "user_123"}}
)

# 同一个 session_id 的对话会共享历史
response = chain_with_history.invoke(
    {"input": "我叫什么？"},
    config={"configurable": {"session_id": "user_123"}}
)
# LLM 知道你叫小明
```

**RunnableWithMessageHistory 的优势：**

- 与 LCEL 无缝集成
- 支持多会话（通过 session_id）
- 自动管理历史的加载和保存
- 更灵活的存储后端选择

---

### 扩展概念5：持久化存储后端 💽

**ChatMessageHistory 支持多种存储后端**

```python
# 1. 内存存储（默认）
from langchain_core.chat_history import InMemoryChatMessageHistory
memory = InMemoryChatMessageHistory()

# 2. Redis 存储
from langchain_community.chat_message_histories import RedisChatMessageHistory
memory = RedisChatMessageHistory(
    url="redis://localhost:6379",
    session_id="user_123"
)

# 3. PostgreSQL 存储
from langchain_community.chat_message_histories import PostgresChatMessageHistory
memory = PostgresChatMessageHistory(
    connection_string="postgresql://...",
    session_id="user_123"
)

# 4. 文件存储
from langchain_community.chat_message_histories import FileChatMessageHistory
memory = FileChatMessageHistory(file_path="chat_history.json")

# 使用方式相同
memory.add_user_message("你好")
memory.add_ai_message("你好！")
messages = memory.messages
```

---

## 4. 【最小可用】

掌握以下内容，就能在 LangChain 中使用 Memory：

### 4.1 基础对话记忆

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI

memory = ConversationBufferMemory(return_messages=True)
chain = ConversationChain(llm=ChatOpenAI(), memory=memory)

chain.invoke({"input": "我叫小明"})
chain.invoke({"input": "我叫什么？"})  # 能记住
```

### 4.2 窗口限制

```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=5, return_messages=True)
# 只保留最近 5 轮对话
```

### 4.3 LCEL 风格（推荐）

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

store = {}

def get_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history"
)

# 使用 session_id 区分不同对话
response = chain_with_history.invoke(
    {"input": "你好"},
    config={"configurable": {"session_id": "user_1"}}
)
```

### 4.4 手动管理历史

```python
from langchain_core.messages import HumanMessage, AIMessage

messages = [
    HumanMessage(content="你好"),
    AIMessage(content="你好！"),
    HumanMessage(content="我叫什么？")
]

response = llm.invoke(messages)
```

**这些知识足以：**
- 实现基础的多轮对话
- 控制历史长度避免超限
- 支持多用户会话
- 与 LCEL Chain 集成

---

## 5. 【1个类比】（双轨制）

### 类比1：Memory 是日记本

#### 🎨 前端视角：React Context / Redux Store

Memory 就像前端的全局状态管理，存储应用的"记忆"。

```javascript
// React Context
const ChatContext = createContext();

function ChatProvider({ children }) {
  const [messages, setMessages] = useState([]);

  const addMessage = (msg) => {
    setMessages(prev => [...prev, msg]);
  };

  return (
    <ChatContext.Provider value={{ messages, addMessage }}>
      {children}
    </ChatContext.Provider>
  );
}

// 组件中使用
function Chat() {
  const { messages, addMessage } = useContext(ChatContext);
  // messages 包含所有历史消息
}
```

```python
# LangChain Memory
memory = ConversationBufferMemory()
memory.save_context({"input": "你好"}, {"output": "你好！"})

# 历史自动注入到 Chain
chain = ConversationChain(llm=llm, memory=memory)
```

**关键相似点：**
- 都是存储状态/历史
- 都需要考虑存储位置（内存、持久化）
- 都要管理状态的生命周期

#### 🧒 小朋友视角：日记本

Memory 就像你的日记本：

```
日记本记录了每天发生的事：

1月1日：今天认识了新朋友小明
1月2日：和小明一起玩游戏
1月3日：小明教我画画

现在有人问："你最近和谁玩？"
你翻看日记本，就知道是小明！

如果没有日记本：
"你最近和谁玩？"
"我不记得了..." 😢
```

---

### 类比2：不同 Memory 类型是不同的记录方式

#### 🎨 前端视角：不同的缓存策略

不同的 Memory 类型就像不同的缓存策略。

```javascript
// BufferMemory - 保存所有
const cache = [];
cache.push(item);  // 无限增长

// WindowMemory - LRU 缓存
const cache = new LRUCache({ max: 100 });

// SummaryMemory - 压缩存储
const cache = {
  summary: "用户A，讨论了电商项目",
  recent: [最近几条]
};

// TokenBufferMemory - 按大小限制
const cache = {
  items: [],
  maxSize: 1000,
  currentSize: 0
};
```

#### 🧒 小朋友视角：不同的记录方式

不同的 Memory 就像不同的记东西方式：

```
BufferMemory = 全部记下来
├── 把每件事都写在日记本上
├── 优点：什么都不会忘
└── 缺点：日记本太厚了

WindowMemory = 只记最近的
├── 只记最近 7 天的事
├── 优点：日记本不会太厚
└── 缺点：一周前的事忘了

SummaryMemory = 写总结
├── 把一周的事写成一段总结
├── 优点：内容精简
└── 缺点：细节可能丢失

TokenBufferMemory = 按页数限制
├── 日记本只有 10 页
├── 写满了就撕掉最早的
└── 优点：大小固定
```

---

### 类比3：session_id 是不同的日记本

#### 🎨 前端视角：用户会话管理

session_id 就像前端的用户会话 ID。

```javascript
// 不同用户不同状态
const sessions = {
  "user_1": { messages: [...] },
  "user_2": { messages: [...] },
};

function getSession(userId) {
  if (!sessions[userId]) {
    sessions[userId] = { messages: [] };
  }
  return sessions[userId];
}
```

```python
# LangChain session 管理
store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]
```

#### 🧒 小朋友视角：每个人有自己的日记本

session_id 就像每个人有自己的日记本：

```
班级里：
- 小明的日记本（session_id: xiaoming）
- 小红的日记本（session_id: xiaohong）
- 小刚的日记本（session_id: xiaogang）

每个人记自己的事，互不干扰。

老师（LLM）看小明的日记本，只知道小明的事。
看小红的日记本，只知道小红的事。
```

---

### 类比总结表

| LangChain 概念 | 前端类比 | 小朋友类比 |
|---------------|---------|-----------|
| Memory | React Context / Redux Store | 日记本 |
| ChatMessageHistory | 消息数组 | 对话记录 |
| BufferMemory | 无限数组 | 记下所有事 |
| WindowMemory | LRU Cache | 只记最近的 |
| SummaryMemory | 压缩存储 | 写总结 |
| session_id | 用户会话 ID | 每个人的日记本 |
| load_memory_variables | getState() | 翻看日记 |
| save_context | setState() | 写日记 |

---

## 6. 【反直觉点】

### 误区1：Memory 会自动管理上下文长度 ❌

**为什么错？**
- ConversationBufferMemory 会无限增长
- 历史过长会超出 LLM 的上下文窗口
- 需要手动选择限制策略

**为什么人们容易这样错？**
以为 Memory 是"智能"的，会自动处理所有情况。

**正确理解：**

```python
# ❌ 错误：使用默认 BufferMemory 进行长对话
memory = ConversationBufferMemory()
# 对话 1000 轮后，历史可能有 100000 tokens
# 超出 GPT-4 的 128k 上下文限制

# ✅ 正确：选择合适的限制策略
# 方案1：窗口限制
memory = ConversationBufferWindowMemory(k=10)

# 方案2：Token 限制
memory = ConversationTokenBufferMemory(max_token_limit=4000)

# 方案3：摘要压缩
memory = ConversationSummaryMemory(llm=llm)
```

---

### 误区2：所有 Chain 都支持 Memory ❌

**为什么错？**
- 传统 Chain (LLMChain, ConversationChain) 原生支持 memory 参数
- LCEL 风格的 Chain 需要使用 RunnableWithMessageHistory
- 两种方式的集成方法不同

**为什么人们容易这样错？**
看到旧教程用 `memory=memory`，在新版 LCEL Chain 上不起作用。

**正确理解：**

```python
# 传统 Chain（原生支持 memory）
from langchain.chains import ConversationChain
chain = ConversationChain(llm=llm, memory=memory)  # ✅ 直接使用

# LCEL Chain（需要包装）
from langchain_core.runnables.history import RunnableWithMessageHistory

# 基础 LCEL Chain
chain = prompt | llm | parser

# ❌ 错误：直接添加 memory
# chain = prompt | llm | parser  # 没有 memory 参数！

# ✅ 正确：使用 RunnableWithMessageHistory 包装
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)
```

---

### 误区3：Memory 只是存储对话 ❌

**为什么错？**
Memory 还涉及：
- 变量注入（将历史注入到 prompt）
- 格式转换（消息对象 vs 字符串）
- 存储后端选择（内存、Redis、数据库）
- 会话隔离（不同用户不同历史）

**为什么人们容易这样错？**
只看到"保存对话"这一个功能。

**正确理解：**

```python
# Memory 的完整职责

# 1. 存储
memory.save_context({"input": "你好"}, {"output": "你好！"})

# 2. 检索（带格式转换）
history = memory.load_memory_variables({})
# 返回：{"history": [HumanMessage(...), AIMessage(...)]}
# 或者：{"history": "Human: 你好\nAI: 你好！"}

# 3. 与 Prompt 配合
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="history"),  # 历史占位
    ("human", "{input}")
])

# 4. 会话隔离
def get_history(session_id):
    # 不同 session_id 返回不同的历史
    return store[session_id]

# 5. 持久化
memory = RedisChatMessageHistory(
    url="redis://localhost:6379",
    session_id="user_123"
)
```

---

## 7. 【实战代码】

```python
"""
示例：Memory 记忆系统完整演示
展示 LangChain 中 Memory 的核心用法
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

# ===== 1. 基础消息结构 =====
print("=== 1. 基础消息结构 ===")

@dataclass
class Message:
    """消息基类"""
    content: str
    type: str  # human, ai, system
    timestamp: datetime = field(default_factory=datetime.now)

    def __str__(self):
        return f"[{self.type}]: {self.content}"

class HumanMessage(Message):
    def __init__(self, content: str):
        super().__init__(content=content, type="human")

class AIMessage(Message):
    def __init__(self, content: str):
        super().__init__(content=content, type="ai")

class SystemMessage(Message):
    def __init__(self, content: str):
        super().__init__(content=content, type="system")

# 测试
msg1 = HumanMessage("你好")
msg2 = AIMessage("你好！有什么可以帮助你？")
print(msg1)
print(msg2)

# ===== 2. ChatMessageHistory 基础实现 =====
print("\n=== 2. ChatMessageHistory ===")

class InMemoryChatMessageHistory:
    """内存消息历史"""

    def __init__(self):
        self._messages: List[Message] = []

    @property
    def messages(self) -> List[Message]:
        return self._messages

    def add_message(self, message: Message) -> None:
        self._messages.append(message)

    def add_user_message(self, content: str) -> None:
        self.add_message(HumanMessage(content))

    def add_ai_message(self, content: str) -> None:
        self.add_message(AIMessage(content))

    def clear(self) -> None:
        self._messages = []

    def __len__(self):
        return len(self._messages)

# 测试
history = InMemoryChatMessageHistory()
history.add_user_message("你好，我叫小明")
history.add_ai_message("你好小明！很高兴认识你")
history.add_user_message("今天天气怎么样？")
history.add_ai_message("今天天气很好，晴天，适合出行")

print(f"消息数量: {len(history)}")
for msg in history.messages:
    print(f"  {msg}")

# ===== 3. ConversationBufferMemory 实现 =====
print("\n=== 3. ConversationBufferMemory ===")

class ConversationBufferMemory:
    """对话缓冲记忆"""

    def __init__(
        self,
        return_messages: bool = False,
        memory_key: str = "history",
        input_key: str = "input",
        output_key: str = "output"
    ):
        self.chat_history = InMemoryChatMessageHistory()
        self.return_messages = return_messages
        self.memory_key = memory_key
        self.input_key = input_key
        self.output_key = output_key

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, Any]) -> None:
        """保存对话上下文"""
        input_str = inputs.get(self.input_key, "")
        output_str = outputs.get(self.output_key, "")

        self.chat_history.add_user_message(input_str)
        self.chat_history.add_ai_message(output_str)

    def load_memory_variables(self, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """加载记忆变量"""
        if self.return_messages:
            return {self.memory_key: self.chat_history.messages}
        else:
            # 转换为字符串格式
            history_str = "\n".join([
                f"{'Human' if msg.type == 'human' else 'AI'}: {msg.content}"
                for msg in self.chat_history.messages
            ])
            return {self.memory_key: history_str}

    def clear(self) -> None:
        """清空记忆"""
        self.chat_history.clear()

# 测试
memory = ConversationBufferMemory(return_messages=True)
memory.save_context(
    {"input": "我叫小红"},
    {"output": "你好小红！"}
)
memory.save_context(
    {"input": "我喜欢编程"},
    {"output": "编程很有趣！你喜欢什么语言？"}
)

print("消息格式:")
result = memory.load_memory_variables({})
for msg in result["history"]:
    print(f"  {msg}")

# 字符串格式
memory_str = ConversationBufferMemory(return_messages=False)
memory_str.save_context({"input": "你好"}, {"output": "你好！"})
print("\n字符串格式:")
print(memory_str.load_memory_variables({})["history"])

# ===== 4. ConversationBufferWindowMemory 实现 =====
print("\n=== 4. ConversationBufferWindowMemory ===")

class ConversationBufferWindowMemory(ConversationBufferMemory):
    """窗口限制的对话记忆"""

    def __init__(self, k: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.k = k  # 保留最近 k 轮对话

    def load_memory_variables(self, inputs: Dict[str, Any] = None) -> Dict[str, Any]:
        """只返回最近 k 轮对话"""
        messages = self.chat_history.messages
        # 每轮对话包含 user + ai 两条消息
        recent_messages = messages[-(self.k * 2):]

        if self.return_messages:
            return {self.memory_key: recent_messages}
        else:
            history_str = "\n".join([
                f"{'Human' if msg.type == 'human' else 'AI'}: {msg.content}"
                for msg in recent_messages
            ])
            return {self.memory_key: history_str}

# 测试
window_memory = ConversationBufferWindowMemory(k=2, return_messages=True)

# 添加 5 轮对话
for i in range(5):
    window_memory.save_context(
        {"input": f"问题{i+1}"},
        {"output": f"回答{i+1}"}
    )

print(f"总共添加 5 轮，只保留最近 2 轮:")
result = window_memory.load_memory_variables({})
for msg in result["history"]:
    print(f"  {msg}")

# ===== 5. 模拟 LLM 对话 =====
print("\n=== 5. 模拟对话链 ===")

class MockLLM:
    """模拟 LLM"""

    def invoke(self, messages: List[Message]) -> str:
        # 分析最后一条消息
        last_msg = messages[-1].content if messages else ""

        # 简单的规则响应
        if "叫什么" in last_msg or "名字" in last_msg:
            # 从历史中找名字
            for msg in messages:
                if msg.type == "human" and "叫" in msg.content:
                    # 提取名字
                    import re
                    match = re.search(r'叫(\w+)', msg.content)
                    if match:
                        return f"根据我们之前的对话，你叫{match.group(1)}"
            return "我还不知道你的名字"

        if "你好" in last_msg:
            return "你好！有什么可以帮助你的吗？"

        return f"我理解你说的是：{last_msg}"

class ConversationChain:
    """对话链"""

    def __init__(self, llm: MockLLM, memory: ConversationBufferMemory):
        self.llm = llm
        self.memory = memory

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        input_text = inputs.get("input", "")

        # 1. 加载历史
        history = self.memory.load_memory_variables({})
        messages = history.get(self.memory.memory_key, [])

        # 2. 添加当前输入
        if isinstance(messages, list):
            all_messages = messages + [HumanMessage(input_text)]
        else:
            all_messages = [HumanMessage(messages + "\n" + input_text)]

        # 3. 调用 LLM
        response = self.llm.invoke(all_messages)

        # 4. 保存到记忆
        self.memory.save_context(
            {"input": input_text},
            {"output": response}
        )

        return {"output": response}

# 测试对话链
llm = MockLLM()
memory = ConversationBufferMemory(return_messages=True)
chain = ConversationChain(llm, memory)

print("多轮对话测试:")
print(f"用户: 你好，我叫小刚")
result = chain.invoke({"input": "你好，我叫小刚"})
print(f"AI: {result['output']}")

print(f"\n用户: 我喜欢Python")
result = chain.invoke({"input": "我喜欢Python"})
print(f"AI: {result['output']}")

print(f"\n用户: 我叫什么名字？")
result = chain.invoke({"input": "我叫什么名字？"})
print(f"AI: {result['output']}")

# ===== 6. 多会话管理 =====
print("\n=== 6. 多会话管理 ===")

class SessionManager:
    """会话管理器"""

    def __init__(self):
        self.sessions: Dict[str, InMemoryChatMessageHistory] = {}

    def get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        """获取或创建会话历史"""
        if session_id not in self.sessions:
            self.sessions[session_id] = InMemoryChatMessageHistory()
            print(f"  创建新会话: {session_id}")
        return self.sessions[session_id]

    def list_sessions(self) -> List[str]:
        """列出所有会话"""
        return list(self.sessions.keys())

# 测试多会话
session_manager = SessionManager()

# 用户 A 的对话
history_a = session_manager.get_session_history("user_a")
history_a.add_user_message("我是用户A")
history_a.add_ai_message("你好用户A！")

# 用户 B 的对话
history_b = session_manager.get_session_history("user_b")
history_b.add_user_message("我是用户B")
history_b.add_ai_message("你好用户B！")

print("\n所有会话:")
for session_id in session_manager.list_sessions():
    history = session_manager.get_session_history(session_id)
    print(f"\n{session_id}:")
    for msg in history.messages:
        print(f"  {msg}")

# ===== 7. RunnableWithMessageHistory 模拟 =====
print("\n=== 7. RunnableWithMessageHistory 模拟 ===")

class RunnableWithMessageHistory:
    """带消息历史的 Runnable"""

    def __init__(
        self,
        runnable,  # 基础 Chain
        get_session_history,  # 获取历史的函数
        input_messages_key: str = "input",
        history_messages_key: str = "history"
    ):
        self.runnable = runnable
        self.get_session_history = get_session_history
        self.input_messages_key = input_messages_key
        self.history_messages_key = history_messages_key

    def invoke(self, input_dict: Dict[str, Any], config: Dict = None) -> Any:
        config = config or {}
        session_id = config.get("configurable", {}).get("session_id", "default")

        # 1. 获取历史
        history = self.get_session_history(session_id)

        # 2. 构建输入（包含历史）
        full_input = {
            self.input_messages_key: input_dict.get(self.input_messages_key, ""),
            self.history_messages_key: history.messages
        }

        # 3. 调用基础 Chain
        response = self.runnable(full_input)

        # 4. 保存到历史
        history.add_user_message(input_dict.get(self.input_messages_key, ""))
        history.add_ai_message(response)

        return response

# 模拟基础 Chain
def simple_chain(inputs: Dict) -> str:
    input_text = inputs.get("input", "")
    history = inputs.get("history", [])

    # 检查历史中是否有名字
    for msg in history:
        if "叫" in msg.content and msg.type == "human":
            import re
            match = re.search(r'叫(\w+)', msg.content)
            if match and "叫什么" in input_text:
                return f"你叫{match.group(1)}！"

    return f"收到: {input_text}"

# 测试
session_mgr = SessionManager()
chain_with_history = RunnableWithMessageHistory(
    simple_chain,
    session_mgr.get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

print("\n测试 RunnableWithMessageHistory:")

# 用户 C 的多轮对话
print("\n用户 C 的对话:")
result = chain_with_history.invoke(
    {"input": "你好，我叫小芳"},
    config={"configurable": {"session_id": "user_c"}}
)
print(f"  输入: 你好，我叫小芳")
print(f"  输出: {result}")

result = chain_with_history.invoke(
    {"input": "我叫什么名字？"},
    config={"configurable": {"session_id": "user_c"}}
)
print(f"  输入: 我叫什么名字？")
print(f"  输出: {result}")

# ===== 8. Token 计数（简化版） =====
print("\n=== 8. Token 计数 ===")

def count_tokens(text: str) -> int:
    """简单的 token 计数（实际应该用 tiktoken）"""
    # 简化：按字符数 / 4 估算
    return len(text) // 4 + 1

class TokenBufferMemory:
    """按 token 限制的记忆"""

    def __init__(self, max_token_limit: int = 100):
        self.messages: List[Message] = []
        self.max_token_limit = max_token_limit

    def add_message(self, message: Message):
        self.messages.append(message)
        self._prune()

    def _prune(self):
        """裁剪消息以满足 token 限制"""
        while self._total_tokens() > self.max_token_limit and len(self.messages) > 1:
            self.messages.pop(0)

    def _total_tokens(self) -> int:
        return sum(count_tokens(msg.content) for msg in self.messages)

    def get_messages(self) -> List[Message]:
        return self.messages

# 测试
token_memory = TokenBufferMemory(max_token_limit=50)

for i in range(10):
    token_memory.add_message(HumanMessage(f"这是第{i+1}条很长的测试消息，包含一些内容"))
    print(f"添加消息 {i+1}，当前 token 数: {token_memory._total_tokens()}")

print(f"\n最终保留 {len(token_memory.get_messages())} 条消息")

print("\n=== 完成！===")
```

**运行输出示例：**
```
=== 1. 基础消息结构 ===
[human]: 你好
[ai]: 你好！有什么可以帮助你？

=== 2. ChatMessageHistory ===
消息数量: 4
  [human]: 你好，我叫小明
  [ai]: 你好小明！很高兴认识你
  [human]: 今天天气怎么样？
  [ai]: 今天天气很好，晴天，适合出行

=== 3. ConversationBufferMemory ===
消息格式:
  [human]: 我叫小红
  [ai]: 你好小红！
  [human]: 我喜欢编程
  [ai]: 编程很有趣！你喜欢什么语言？

字符串格式:
Human: 你好
AI: 你好！

=== 4. ConversationBufferWindowMemory ===
总共添加 5 轮，只保留最近 2 轮:
  [human]: 问题4
  [ai]: 回答4
  [human]: 问题5
  [ai]: 回答5

=== 5. 模拟对话链 ===
多轮对话测试:
用户: 你好，我叫小刚
AI: 你好！有什么可以帮助你的吗？

用户: 我喜欢Python
AI: 我理解你说的是：我喜欢Python

用户: 我叫什么名字？
AI: 根据我们之前的对话，你叫小刚

=== 完成！===
```

---

## 8. 【面试必问】

### 问题1："LangChain 中如何实现多轮对话？"

**普通回答（❌ 不出彩）：**
"使用 Memory 保存对话历史。"

**出彩回答（✅ 推荐）：**

> **多轮对话的核心是管理对话历史，LangChain 提供两种方式：**
>
> **1. 传统方式：ConversationBufferMemory**
> ```python
> memory = ConversationBufferMemory(return_messages=True)
> chain = ConversationChain(llm=llm, memory=memory)
> # memory 自动保存和注入历史
> ```
>
> **2. LCEL 方式：RunnableWithMessageHistory（推荐）**
> ```python
> chain_with_history = RunnableWithMessageHistory(
>     chain,
>     get_session_history,  # 获取历史的函数
>     input_messages_key="input",
>     history_messages_key="history"
> )
> # 支持多会话（通过 session_id）
> ```
>
> **关键考虑：**
> - **上下文长度**：历史过长会超出窗口，需要限制策略
> - **多用户隔离**：使用 session_id 区分不同用户
> - **持久化**：生产环境用 Redis/数据库存储
>
> **我的实践经验**：在智能客服项目中，使用 WindowMemory(k=10) 保留最近 10 轮，超出部分用摘要压缩，Redis 持久化。

**为什么这个回答出彩？**
1. ✅ 两种方式对比
2. ✅ 考虑了关键问题
3. ✅ 有实际项目经验

---

### 问题2："如何处理对话历史过长的问题？"

**普通回答（❌ 不出彩）：**
"用 WindowMemory 限制消息数量。"

**出彩回答（✅ 推荐）：**

> **这是多轮对话的核心挑战，有几种策略：**
>
> | 策略 | 实现 | 适用场景 |
> |-----|-----|---------|
> | 窗口限制 | `WindowMemory(k=10)` | 最近上下文重要 |
> | Token 限制 | `TokenBufferMemory(limit=4000)` | 精确控制成本 |
> | 摘要压缩 | `SummaryMemory` | 需要长期记忆 |
> | 混合策略 | 窗口 + 摘要 | 生产环境 |
>
> **混合策略示例：**
> ```python
> # 保留最近 5 轮完整对话
> # + 更早对话的摘要
> class HybridMemory:
>     def __init__(self, llm, k=5):
>         self.recent = WindowMemory(k=k)
>         self.summary = ""
>
>     def _update_summary(self, old_messages):
>         self.summary = llm.invoke(f"总结: {old_messages}")
> ```
>
> **实际考虑：**
> - GPT-4 有 128k 上下文，但 token 越多成本越高
> - 太长的历史可能引入噪音
> - 不同场景需要不同策略

---

## 9. 【化骨绵掌】

### 卡片1：Memory 是什么？ 🎯

**一句话：** Memory 让无状态的 LLM 能够记住对话历史，实现多轮对话。

**举例：**
```python
memory.save("我叫小明")
memory.save("你好小明！")
# 下次 LLM 知道用户叫小明
```

**应用：** 智能客服、聊天机器人、对话助手。

---

### 卡片2：ChatMessageHistory 消息历史 📜

**一句话：** 存储对话消息的基础容器，支持多种后端（内存、Redis、数据库）。

**举例：**
```python
history = InMemoryChatMessageHistory()
history.add_user_message("你好")
history.add_ai_message("你好！")
```

**应用：** 所有 Memory 类型的底层存储。

---

### 卡片3：ConversationBufferMemory 缓冲记忆 💾

**一句话：** 最简单的 Memory，保存完整的对话历史。

**举例：**
```python
memory = ConversationBufferMemory()
memory.save_context({"input": "你好"}, {"output": "你好！"})
```

**应用：** 短对话、测试开发场景。

---

### 卡片4：WindowMemory 窗口限制 📊

**一句话：** 只保留最近 k 轮对话，避免历史过长。

**举例：**
```python
memory = ConversationBufferWindowMemory(k=5)
# 只保留最近 5 轮
```

**应用：** 大多数生产场景的首选。

---

### 卡片5：SummaryMemory 摘要压缩 📝

**一句话：** 用 LLM 对历史进行摘要，保留要点节省 token。

**举例：**
```python
memory = ConversationSummaryMemory(llm=llm)
# 100 轮对话 → 一段摘要
```

**应用：** 需要长期记忆的场景。

---

### 卡片6：RunnableWithMessageHistory 🔗

**一句话：** LCEL 风格的 Memory 管理，支持多会话。

**举例：**
```python
chain_with_history = RunnableWithMessageHistory(
    chain, get_session_history
)
# 通过 session_id 区分不同用户
```

**应用：** LangChain 0.1+ 版本推荐方式。

---

### 卡片7：session_id 会话标识 🔖

**一句话：** 区分不同用户/对话的标识，实现会话隔离。

**举例：**
```python
config={"configurable": {"session_id": "user_123"}}
# 不同 session_id 的历史互不影响
```

**应用：** 多用户系统必备。

---

### 卡片8：持久化存储 💽

**一句话：** 将对话历史持久化到 Redis/数据库，程序重启不丢失。

**举例：**
```python
history = RedisChatMessageHistory(
    url="redis://localhost:6379",
    session_id="user_123"
)
```

**应用：** 生产环境必备。

---

### 卡片9：return_messages 参数 📤

**一句话：** 决定历史返回格式：True 返回消息对象，False 返回字符串。

**举例：**
```python
# return_messages=True
[HumanMessage(...), AIMessage(...)]

# return_messages=False
"Human: 你好\nAI: 你好！"
```

**应用：** 根据 prompt 格式选择。

---

### 卡片10：Memory 在 LangChain 源码中的位置 ⭐

**一句话：** Memory 通过 load_memory_variables 注入历史，与 prompt 的占位符配合。

**举例：**
```python
# Prompt 中的占位符
MessagesPlaceholder(variable_name="history")

# Memory 提供数据
{"history": [HumanMessage(...), AIMessage(...)]}
```

**应用：** 理解 Memory 和 Prompt 的配合。

---

## 10. 【一句话总结】

**Memory 是为无状态的 LLM 提供对话上下文的机制，通过存储和注入历史消息实现多轮对话，不同的 Memory 类型（Buffer、Window、Summary）对应不同的历史管理策略。**

---

## 📚 学习检查清单

- [ ] 理解 LLM 无状态的问题和 Memory 的解决方案
- [ ] 会使用 ConversationBufferMemory
- [ ] 知道何时使用 WindowMemory vs SummaryMemory
- [ ] 掌握 RunnableWithMessageHistory 的用法
- [ ] 理解 session_id 的作用
- [ ] 了解持久化存储的选项

## 🔗 下一步学习

- **Retriever 检索器**：结合 Memory 实现 RAG
- **Callback 回调系统**：监控 Memory 的使用
- **Agent 与 Memory**：让 Agent 记住历史

---

**版本：** v1.0
**最后更新：** 2025-01-14
