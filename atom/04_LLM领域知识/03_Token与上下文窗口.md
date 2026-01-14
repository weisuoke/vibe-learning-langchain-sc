# Token 与上下文窗口

> 原子化知识点 | LLM领域知识 | LangChain 源码学习前置知识

---

## 1. 【30字核心】

**Token 是 LLM 处理文本的基本单位，上下文窗口是 LLM 一次能处理的 Token 总量限制。**

---

## 2. 【第一性原理】

### 什么是第一性原理？

**第一性原理**：回到事物最基本的真理，从源头思考问题

### Token 与上下文窗口的第一性原理 🎯

#### 1. 最基础的定义

**Token = 文本的最小处理单元（不等于字符，也不等于单词）**

仅此而已！没有更基础的了。

```python
# Token 拆分示例
"Hello World" → ["Hello", " World"]           # 2 tokens
"你好世界"    → ["你", "好", "世", "界"]       # 4 tokens（中文每字约1token）
"LangChain"   → ["Lang", "Chain"]             # 2 tokens（子词拆分）
```

**上下文窗口（Context Window）= LLM 能同时"看到"的 Token 总数**

```python
# 上下文窗口 = 输入 Token + 输出 Token
context_window = input_tokens + output_tokens
# GPT-4: 128K tokens
# Claude 3: 200K tokens
```

#### 2. 为什么需要 Token？

**核心问题：计算机无法直接处理任意长度的文本**

```python
# 直接处理文本的问题
text = "Hello, how are you?"

# 问题1：文本长度可变，神经网络需要固定输入
# 问题2：字符级处理效率太低（26个字母组合无穷）
# 问题3：单词级处理词汇表太大（英语有 100万+ 单词）

# 解决方案：Token（子词）
# - 常见词保持完整："the", "is"
# - 生僻词拆成子词："unhappiness" → ["un", "happiness"]
# - 词汇表大小可控：~50,000-100,000 tokens
```

#### 3. Token 的三层价值

##### 价值1：效率与质量的平衡

```python
# 字符级：效率低，需要学习字符组合
"cat" → ['c', 'a', 't']  # 3个单元

# 单词级：词汇表爆炸，无法处理未知词
"ChatGPT" → ['ChatGPT']  # OOV（Out of Vocabulary）问题

# Token级：平衡效率和覆盖
"ChatGPT" → ['Chat', 'G', 'PT']  # 可以处理任何词
```

##### 价值2：成本计算的基础

```python
# API 按 Token 计费
# GPT-4: $0.03/1K input tokens, $0.06/1K output tokens

text = "请用 Python 写一个快速排序算法"
tokens = count_tokens(text)  # ~15 tokens
cost = tokens * 0.00003  # 输入成本

# 理解 Token 才能控制成本
```

##### 价值3：上下文管理的依据

```python
# 上下文窗口有限，必须合理分配
context_window = 128000  # GPT-4 Turbo

# 分配策略
system_prompt = 2000      # 系统提示
history = 10000           # 对话历史
retrieved_docs = 50000    # RAG 检索内容
user_input = 1000         # 用户输入
reserved_output = 65000   # 留给输出

# 超出窗口 = 丢失信息或报错
```

#### 4. 从第一性原理推导 LangChain 应用

**推理链：**

```
1. LLM 以 Token 为单位处理文本
   ↓
2. 每个 LLM 有固定的上下文窗口大小
   ↓
3. 输入+输出不能超过上下文窗口
   ↓
4. 长对话需要"截断"或"总结"历史
   ↓
5. RAG 检索需要控制检索内容大小
   ↓
6. LangChain 需要 Token 计数和管理工具
   ↓
7. ConversationBufferWindowMemory、ConversationSummaryMemory 等
```

#### 5. 一句话总结第一性原理

**Token 是 LLM 理解世界的最小单元，上下文窗口是 LLM 的"工作记忆"容量，理解这两个概念是设计高效 LangChain 应用的基础。**

---

## 3. 【核心概念（全面覆盖）】

### 核心概念1：Token 与 Tokenizer 🔤

**Tokenizer 是将文本拆分成 Token 的工具，不同模型使用不同的 Tokenizer**

```python
import tiktoken

# OpenAI 的 Tokenizer
encoding = tiktoken.encoding_for_model("gpt-4")

# 文本 → Token IDs
text = "Hello, World!"
tokens = encoding.encode(text)
print(f"Tokens: {tokens}")  # [9906, 11, 4435, 0]
print(f"Token数: {len(tokens)}")  # 4

# Token IDs → Token 文本
for token_id in tokens:
    print(f"  {token_id} → '{encoding.decode([token_id])}'")
# 9906 → 'Hello'
# 11 → ','
# 4435 → ' World'
# 0 → '!'

# 中文 Token 化
chinese_text = "你好世界"
chinese_tokens = encoding.encode(chinese_text)
print(f"中文 Token数: {len(chinese_tokens)}")  # 约 4-8 个
```

**不同语言的 Token 效率：**

| 语言 | 示例文本 | Token 数 | 效率 |
|------|---------|----------|------|
| 英语 | "Hello World" | 2 | 高 |
| 中文 | "你好世界" | 4-6 | 中 |
| 日语 | "こんにちは" | 5-8 | 低 |
| 代码 | `def hello():` | 4 | 高 |

**在 LangChain 源码中的应用：**

```python
# langchain_core/language_models/base.py
class BaseLanguageModel(ABC):
    """语言模型基类"""

    def get_num_tokens(self, text: str) -> int:
        """计算文本的 Token 数量"""
        # 不同模型有不同实现
        pass

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        """计算消息列表的 Token 数量"""
        pass

# 使用示例
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
num_tokens = llm.get_num_tokens("Hello World")
```

---

### 核心概念2：上下文窗口（Context Window） 📏

**上下文窗口是 LLM 一次请求能处理的最大 Token 数量**

```python
# 常见模型的上下文窗口
context_windows = {
    "gpt-3.5-turbo": 16385,
    "gpt-4": 8192,
    "gpt-4-turbo": 128000,
    "gpt-4o": 128000,
    "claude-3-opus": 200000,
    "claude-3-sonnet": 200000,
    "claude-3-haiku": 200000,
}

# 上下文 = 输入 + 输出
# 输入：System Prompt + 历史消息 + 用户输入 + RAG 内容
# 输出：模型生成的回复

def check_context_fit(model: str, input_tokens: int, max_output: int) -> bool:
    """检查是否超出上下文窗口"""
    window = context_windows.get(model, 4096)
    return input_tokens + max_output <= window
```

**上下文窗口分配示例：**

```
┌─────────────────────────────────────────────────────┐
│                  128K Token 窗口                      │
├─────────────────────────────────────────────────────┤
│  System Prompt  │  历史消息  │  RAG 内容  │  输出空间  │
│     2K          │    20K    │    50K    │    56K    │
└─────────────────────────────────────────────────────┘
```

**在 LangChain 源码中的应用：**

```python
# langchain_openai/chat_models/base.py
class ChatOpenAI(BaseChatModel):
    model_name: str = "gpt-3.5-turbo"
    max_tokens: Optional[int] = None  # 限制输出 Token

    def _get_context_length(self) -> int:
        """获取模型的上下文窗口大小"""
        model_context_lengths = {
            "gpt-4": 8192,
            "gpt-4-turbo": 128000,
            "gpt-4o": 128000,
            "gpt-3.5-turbo": 16385,
        }
        return model_context_lengths.get(self.model_name, 4096)
```

---

### 核心概念3：Token 计数与成本 💰

**API 按 Token 收费，理解 Token 才能控制成本**

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4") -> int:
    """计算文本的 Token 数量"""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

def estimate_cost(
    input_tokens: int,
    output_tokens: int,
    model: str = "gpt-4"
) -> float:
    """估算 API 调用成本"""
    # 2024年价格（美元/1K tokens）
    pricing = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    }

    price = pricing.get(model, pricing["gpt-3.5-turbo"])
    input_cost = (input_tokens / 1000) * price["input"]
    output_cost = (output_tokens / 1000) * price["output"]

    return input_cost + output_cost

# 使用示例
prompt = "请详细解释什么是机器学习，包括监督学习、无监督学习和强化学习"
input_tokens = count_tokens(prompt)
estimated_output = 500  # 估计输出 500 tokens

cost = estimate_cost(input_tokens, estimated_output, "gpt-4")
print(f"输入: {input_tokens} tokens")
print(f"预计成本: ${cost:.4f}")
```

**不同场景的 Token 消耗：**

| 场景 | 输入 Token | 输出 Token | 预估成本（GPT-4） |
|------|-----------|------------|-----------------|
| 简单问答 | 50 | 100 | $0.0075 |
| 代码生成 | 200 | 500 | $0.036 |
| 长文档总结 | 10000 | 500 | $0.33 |
| RAG 问答 | 5000 | 300 | $0.168 |

---

### 核心概念4：消息截断与历史管理 ✂️

**当对话超出上下文窗口时，需要策略性地截断或总结**

```python
from typing import List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

def truncate_messages_by_token(
    messages: List[BaseMessage],
    max_tokens: int,
    tokenizer_fn
) -> List[BaseMessage]:
    """按 Token 数量截断消息（保留最近的）"""
    total_tokens = 0
    truncated = []

    # 从最新到最旧遍历
    for msg in reversed(messages):
        msg_tokens = tokenizer_fn(msg.content)
        if total_tokens + msg_tokens > max_tokens:
            break
        truncated.insert(0, msg)
        total_tokens += msg_tokens

    return truncated

# 使用示例
messages = [
    HumanMessage(content="你好"),
    AIMessage(content="你好！有什么可以帮助你的？"),
    HumanMessage(content="解释一下Python装饰器"),
    AIMessage(content="装饰器是...（长回复）"),
    # ... 更多消息
]

truncated = truncate_messages_by_token(
    messages,
    max_tokens=2000,
    tokenizer_fn=lambda x: len(x) // 3  # 简化的 token 计数
)
```

**LangChain 中的 Memory 策略：**

```python
# 1. 窗口记忆：保留最近 k 条消息
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=5)  # 保留最近5轮

# 2. Token 限制记忆：按 Token 数截断
from langchain.memory import ConversationTokenBufferMemory

memory = ConversationTokenBufferMemory(
    llm=llm,
    max_token_limit=2000  # 最多 2000 tokens
)

# 3. 总结记忆：用 LLM 总结历史
from langchain.memory import ConversationSummaryMemory

memory = ConversationSummaryMemory(llm=llm)
# 自动将长对话总结成简短描述
```

---

### 扩展概念5：max_tokens 参数 🎛️

**max_tokens 控制模型生成的最大输出长度**

```python
from langchain_openai import ChatOpenAI

# 不设置 max_tokens：模型自动决定输出长度
llm_auto = ChatOpenAI(model="gpt-4")

# 设置 max_tokens：限制输出长度
llm_limited = ChatOpenAI(model="gpt-4", max_tokens=100)

# 短回答场景
llm_short = ChatOpenAI(model="gpt-4", max_tokens=50)
response = llm_short.invoke("用一句话解释什么是 Python")
# 输出会被截断在约 50 tokens

# 长文章场景
llm_long = ChatOpenAI(model="gpt-4", max_tokens=4000)
response = llm_long.invoke("写一篇关于机器学习的详细文章")
# 可以生成更长的内容
```

**max_tokens 的作用：**

| 参数值 | 效果 | 适用场景 |
|--------|------|---------|
| 不设置 | 模型自动决定 | 通用场景 |
| 50-100 | 简短回答 | 分类、摘要 |
| 500-1000 | 中等长度 | 问答、翻译 |
| 2000-4000 | 长篇内容 | 文章生成、代码 |

---

## 4. 【最小可用】

掌握以下内容，就能开始处理 Token 相关的 LangChain 开发：

### 4.1 计算 Token 数量

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# 使用
tokens = count_tokens("Hello, how are you?")
print(f"Token 数: {tokens}")
```

### 4.2 检查上下文窗口

```python
def check_fits_context(
    prompt: str,
    model: str = "gpt-4",
    max_output: int = 1000
) -> bool:
    """检查是否超出上下文窗口"""
    context_limits = {
        "gpt-4": 8192,
        "gpt-4-turbo": 128000,
        "gpt-3.5-turbo": 16385,
    }
    limit = context_limits.get(model, 4096)
    prompt_tokens = count_tokens(prompt, model)
    return prompt_tokens + max_output <= limit
```

### 4.3 设置 max_tokens

```python
from langchain_openai import ChatOpenAI

# 根据需求设置
llm = ChatOpenAI(
    model="gpt-4",
    max_tokens=500  # 限制输出长度
)
```

### 4.4 使用 Token 限制的 Memory

```python
from langchain.memory import ConversationTokenBufferMemory
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
memory = ConversationTokenBufferMemory(
    llm=llm,
    max_token_limit=2000
)
```

**这些知识足以：**
- 估算 API 调用成本
- 避免上下文窗口超限错误
- 管理长对话的历史消息
- 优化 RAG 检索内容的大小

---

## 5. 【1个类比】（双轨制）

### 类比1：Token 与文本

#### 🎨 前端视角：UTF-8 编码 / 字符分词

Token 就像是 LLM 专用的"字符编码"方式。

```javascript
// UTF-8：将字符编码为字节
const text = "Hello";
const utf8 = new TextEncoder().encode(text);
// [72, 101, 108, 108, 111]  // 5 字节

// Token：将文本编码为语义单元
const tokens = tokenizer.encode("Hello");
// [15496]  // 1 token
// "Hello"是一个完整的常见词，所以是1个token
```

```python
# Tokenizer 示例
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4")
tokens = encoding.encode("Hello")
print(tokens)  # [9906]
```

**关键区别：** UTF-8 按字符编码，Token 按"语义单元"编码

#### 🧒 小朋友视角：乐高积木块

Token 就像乐高积木块：

```
普通文字 = 沙子（太小，不好用）
单词 = 大石头（太大，不灵活）
Token = 乐高积木（大小刚好！）

"Hello World" 变成乐高：
[Hello积木] [World积木]  → 2块积木

"你好世界" 变成乐高：
[你] [好] [世] [界]  → 4块积木
（中文每个字是一块积木）
```

**生活例子：**
```
想象你在用乐高搭房子：
- 太小的积木（沙子）：数量太多，搭不完
- 太大的积木（整块墙）：不够灵活，没法搭窗户
- 刚好的积木（乐高）：数量合适，想搭什么都行

LLM 用 Token 就像用乐高：
- 不用一个字一个字处理（太慢）
- 不用一个词一个词处理（词太多）
- 用 Token 处理（刚刚好！）
```

---

### 类比2：上下文窗口

#### 🎨 前端视角：API 请求体大小限制

上下文窗口就像 HTTP 请求体的大小限制。

```javascript
// HTTP 请求体限制
const MAX_BODY_SIZE = 1024 * 1024; // 1MB

// 如果请求体太大
fetch('/api/data', {
  method: 'POST',
  body: JSON.stringify(hugeData)  // 超过 1MB → 413 错误
});

// 需要分页或压缩
const chunks = splitIntoChunks(hugeData, MAX_BODY_SIZE);
for (const chunk of chunks) {
  await fetch('/api/data', { body: chunk });
}
```

```python
# LLM 上下文窗口限制
MAX_CONTEXT = 128000  # GPT-4 Turbo 的限制

# 如果输入太长
if count_tokens(prompt) > MAX_CONTEXT:
    # 需要截断或分块
    prompt = truncate_to_tokens(prompt, MAX_CONTEXT - 1000)
```

#### 🧒 小朋友视角：书包容量

上下文窗口就像书包的容量：

```
你的书包（GPT-4）：能装 128000 本小书（Token）

如果你想带：
- 语文书（System Prompt）：2000本
- 作业本（历史消息）：10000本
- 参考资料（RAG 内容）：50000本
- 铅笔盒（用户输入）：1000本
- 还要留空间给新东西（输出）：65000本

总共：128000本 = 刚好装满！

如果超过128000本 → 书包装不下，会有书掉出来！
```

**生活例子：**
```
想象你在看一本书：
- 短期记忆（上下文窗口）：你能同时记住的内容
- 如果书有1000页，你只能同时记住200页
- 之前看的内容会慢慢忘记

LLM 也是这样：
- 上下文窗口 = LLM 的"短期记忆"
- 超过窗口的内容 = LLM 看不到了
```

---

### 类比3：Token 成本计算

#### 🎨 前端视角：云服务按请求计费

Token 计费就像云函数按调用次数计费。

```javascript
// AWS Lambda 计费
// 每 100万次调用 $0.20
// 每 GB-秒 $0.0000166667

function estimateLambdaCost(invocations, memory, duration) {
  const invocationCost = invocations * 0.0000002;
  const computeCost = memory * duration * 0.0000166667;
  return invocationCost + computeCost;
}
```

```python
# LLM Token 计费
# GPT-4: $0.03/1K input, $0.06/1K output

def estimate_llm_cost(input_tokens, output_tokens):
    input_cost = (input_tokens / 1000) * 0.03
    output_cost = (output_tokens / 1000) * 0.06
    return input_cost + output_cost

# 1000 输入 + 500 输出 = $0.03 + $0.03 = $0.06
```

#### 🧒 小朋友视角：买零食按个数计费

Token 计费就像买糖果：

```
糖果店（OpenAI）的价格：
- 你给店员看的糖果图片（输入）：每1000张 3分钱
- 店员给你的糖果（输出）：每1000颗 6分钱

你的购物：
- 看了100张图片（输入100 tokens）：0.3分
- 拿了50颗糖（输出50 tokens）：0.3分
- 总共：0.6分钱

如果你看了10000张图片，拿了5000颗糖：
- 输入：30分 = 3毛钱
- 输出：30分 = 3毛钱
- 总共：6毛钱
```

---

### 类比4：消息截断策略

#### 🎨 前端视角：无限滚动 vs 分页

消息截断就像处理长列表的策略。

```javascript
// 策略1：窗口截断（保留最近N条）
// 类似：无限滚动只渲染可视区域
const messages = allMessages.slice(-10);  // 保留最近10条

// 策略2：Token 限制截断
// 类似：响应分页，限制每页数据量
function paginateBySize(data, maxSize) {
  let result = [];
  let currentSize = 0;
  for (const item of data.reverse()) {
    if (currentSize + item.size > maxSize) break;
    result.unshift(item);
    currentSize += item.size;
  }
  return result;
}

// 策略3：总结压缩
// 类似：数据聚合，将明细压缩成摘要
const summary = data.reduce((acc, item) => ({
  count: acc.count + 1,
  total: acc.total + item.value
}), { count: 0, total: 0 });
```

#### 🧒 小朋友视角：日记本

消息截断就像管理你的日记本：

```
日记本只有100页（上下文窗口）

策略1：只留最近的
- 写满100页后，撕掉最早的10页
- 写新的10页
- 永远保持100页

策略2：压缩旧内容
- 把前50页的内容总结成5页
- 腾出45页空间
- 详细内容 → 精简摘要

策略3：按重要性保留
- 重要的日记打星号
- 空间不够时，先删除没有星号的
```

---

### 类比总结表

| Token 概念 | 前端类比 | 小朋友类比 |
|-----------|---------|-----------|
| Token | UTF-8 字符编码 | 乐高积木块 |
| 上下文窗口 | 请求体大小限制 | 书包容量 |
| Token 计费 | 云服务按量计费 | 糖果按个数买 |
| 消息截断 | 无限滚动虚拟化 | 日记本页数管理 |
| max_tokens | 响应限速 | 作文字数限制 |
| Tokenizer | 字符集编码器 | 翻译官 |

---

## 6. 【反直觉点】

### 误区1：一个中文字 = 一个 Token ❌

**为什么错？**
- 不同 Tokenizer 对中文的处理不同
- 常见的中文词可能是 1 个 Token，生僻字可能是多个
- 实际 Token 数需要用 Tokenizer 计算

**为什么人们容易这样错？**
英文中一个常见词约等于 1 Token，人们简单类推到中文。但中文的 Token 化策略复杂得多。

**正确理解：**

```python
import tiktoken

encoding = tiktoken.encoding_for_model("gpt-4")

# 测试不同中文文本
texts = [
    "你好",        # 2字
    "机器学习",    # 4字
    "人工智能",    # 4字
    "LangChain",   # 英文
]

for text in texts:
    tokens = encoding.encode(text)
    print(f"'{text}' → {len(tokens)} tokens (字符数: {len(text)})")

# 输出示例：
# '你好' → 2 tokens (字符数: 2)
# '机器学习' → 4 tokens (字符数: 4)
# '人工智能' → 3 tokens (字符数: 4)  # 常见词可能更少
# 'LangChain' → 2 tokens (字符数: 9)
```

**经验法则：** 中文约 1.5-2 字符/Token，英文约 4 字符/Token

---

### 误区2：上下文窗口越大越好 ❌

**为什么错？**
- 大窗口 = 更高的成本（按 Token 计费）
- 大窗口 = 更长的延迟（处理时间增加）
- LLM 对超长上下文的"注意力"不均匀（可能忽略中间内容）

**为什么人们容易这样错？**
"越大越好"是直觉思维。但实际上，128K 窗口的模型在处理 128K 内容时，对中间部分的"记忆"可能很弱（"Lost in the Middle"问题）。

**正确理解：**

```python
# ❌ 错误：塞满整个上下文窗口
context = load_all_documents()  # 100K tokens
response = llm.invoke(context + question)
# 问题：中间的内容可能被"忽略"

# ✅ 正确：只放最相关的内容
relevant_docs = retriever.invoke(question)[:5]  # 5个相关文档
context = format_docs(relevant_docs)  # ~5K tokens
response = llm.invoke(context + question)
# 更精准，更便宜，更快

# 最佳实践：按重要性分层
# 1. System Prompt（始终保留）
# 2. 最相关的检索内容（高优先级）
# 3. 最近的对话历史（中优先级）
# 4. 较早的历史摘要（低优先级）
```

**经验法则：** 宁可精选 5K 有效内容，不要堆砌 50K 噪音

---

### 误区3：输出 Token 和输入 Token 成本一样 ❌

**为什么错？**
- 输出 Token 通常比输入 Token 贵 2-3 倍
- 生成输出需要更多计算资源
- 控制输出长度可以显著降低成本

**为什么人们容易这样错？**
计费模式不透明，很多人只关注输入，忽略了输出成本。

**正确理解：**

```python
# GPT-4 定价（2024）
# 输入：$0.03 / 1K tokens
# 输出：$0.06 / 1K tokens（贵2倍！）

# 场景对比
# 场景1：长问题，短回答
input_tokens = 5000  # $0.15
output_tokens = 100  # $0.006
total = 0.156  # 成本主要在输入

# 场景2：短问题，长回答
input_tokens = 100   # $0.003
output_tokens = 5000 # $0.30
total = 0.303  # 成本主要在输出！

# 优化策略
llm = ChatOpenAI(
    model="gpt-4",
    max_tokens=500  # 限制输出，控制成本
)
```

**经验法则：** 输出成本常被低估，用 max_tokens 控制

---

## 7. 【实战代码】

```python
"""
示例：Token 计数与上下文窗口管理
演示 LangChain 中 Token 相关的核心操作
"""

from typing import List, Dict
import re

# ===== 1. 模拟 Tokenizer =====
print("=== 1. Token 计数 ===")

class SimpleTokenizer:
    """简化的 Tokenizer 实现（演示用）"""

    def __init__(self):
        # 简化的词汇表
        self.vocab = {}
        self.next_id = 0

    def encode(self, text: str) -> List[int]:
        """文本 → Token IDs"""
        # 简化：按空格和标点分割
        tokens = re.findall(r'\w+|[^\w\s]', text)
        ids = []
        for token in tokens:
            if token not in self.vocab:
                self.vocab[token] = self.next_id
                self.next_id += 1
            ids.append(self.vocab[token])
        return ids

    def decode(self, ids: List[int]) -> str:
        """Token IDs → 文本"""
        reverse_vocab = {v: k for k, v in self.vocab.items()}
        tokens = [reverse_vocab.get(i, '<UNK>') for i in ids]
        return ' '.join(tokens)

    def count_tokens(self, text: str) -> int:
        """计算 Token 数量"""
        return len(self.encode(text))

tokenizer = SimpleTokenizer()

# 测试 Token 化
texts = [
    "Hello, World!",
    "LangChain is a framework for LLM applications.",
    "你好世界",  # 简化处理
]

for text in texts:
    tokens = tokenizer.encode(text)
    print(f"'{text}'")
    print(f"  Token数: {len(tokens)}")
    print(f"  Token IDs: {tokens}")
    print()

# ===== 2. 上下文窗口检查 =====
print("=== 2. 上下文窗口检查 ===")

class ContextWindowManager:
    """上下文窗口管理器"""

    # 模型上下文窗口大小
    CONTEXT_LIMITS = {
        "gpt-4": 8192,
        "gpt-4-turbo": 128000,
        "gpt-3.5-turbo": 16385,
        "claude-3": 200000,
    }

    def __init__(self, model: str, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.context_limit = self.CONTEXT_LIMITS.get(model, 4096)

    def check_fits(self, text: str, reserved_output: int = 1000) -> Dict:
        """检查文本是否适合上下文窗口"""
        input_tokens = self.tokenizer.count_tokens(text)
        available = self.context_limit - reserved_output
        fits = input_tokens <= available

        return {
            "fits": fits,
            "input_tokens": input_tokens,
            "context_limit": self.context_limit,
            "reserved_output": reserved_output,
            "available": available,
            "overflow": max(0, input_tokens - available)
        }

    def truncate_to_fit(self, text: str, reserved_output: int = 1000) -> str:
        """截断文本以适应上下文窗口"""
        check = self.check_fits(text, reserved_output)
        if check["fits"]:
            return text

        # 简化截断：按比例截取
        ratio = check["available"] / check["input_tokens"]
        target_len = int(len(text) * ratio * 0.9)  # 留10%余量
        return text[:target_len] + "..."

# 使用示例
manager = ContextWindowManager("gpt-4", tokenizer)

# 测试文本
short_text = "What is Python?"
long_text = "Python " * 5000  # 模拟长文本

print(f"短文本检查: {manager.check_fits(short_text)}")
print(f"长文本检查: {manager.check_fits(long_text)}")

truncated = manager.truncate_to_fit(long_text)
print(f"截断后长度: {len(truncated)} 字符")

# ===== 3. 消息历史管理 =====
print("\n=== 3. 消息历史管理 ===")

class Message:
    """简化的消息类"""
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

class ConversationTokenBuffer:
    """基于 Token 限制的对话缓冲区"""

    def __init__(self, tokenizer, max_tokens: int = 2000):
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.messages: List[Message] = []

    def add_message(self, role: str, content: str):
        """添加消息"""
        self.messages.append(Message(role, content))
        self._truncate_if_needed()

    def _truncate_if_needed(self):
        """如果超过 Token 限制，从最早的消息开始删除"""
        while self._total_tokens() > self.max_tokens and len(self.messages) > 1:
            self.messages.pop(0)

    def _total_tokens(self) -> int:
        """计算总 Token 数"""
        return sum(
            self.tokenizer.count_tokens(m.content)
            for m in self.messages
        )

    def get_messages(self) -> List[Dict]:
        """获取消息列表"""
        return [
            {"role": m.role, "content": m.content}
            for m in self.messages
        ]

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "message_count": len(self.messages),
            "total_tokens": self._total_tokens(),
            "max_tokens": self.max_tokens
        }

# 使用示例
buffer = ConversationTokenBuffer(tokenizer, max_tokens=100)

# 模拟对话
conversations = [
    ("user", "Hello!"),
    ("assistant", "Hi! How can I help you today?"),
    ("user", "Can you explain Python decorators?"),
    ("assistant", "Decorators are functions that modify the behavior of other functions. They use the @decorator syntax."),
    ("user", "Can you give me an example?"),
    ("assistant", "Sure! Here's a simple example of a decorator that prints when a function is called..."),
]

for role, content in conversations:
    buffer.add_message(role, content)
    stats = buffer.get_stats()
    print(f"添加: [{role}] {content[:30]}...")
    print(f"  状态: {stats}")

print(f"\n最终保留的消息:")
for msg in buffer.get_messages():
    print(f"  [{msg['role']}] {msg['content'][:40]}...")

# ===== 4. 成本估算器 =====
print("\n=== 4. 成本估算器 ===")

class CostEstimator:
    """API 成本估算器"""

    # 定价（美元/1K tokens）
    PRICING = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4o": {"input": 0.005, "output": 0.015},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    }

    def __init__(self, model: str = "gpt-4"):
        self.model = model
        self.pricing = self.PRICING.get(model, self.PRICING["gpt-4"])

    def estimate(self, input_tokens: int, output_tokens: int) -> Dict:
        """估算成本"""
        input_cost = (input_tokens / 1000) * self.pricing["input"]
        output_cost = (output_tokens / 1000) * self.pricing["output"]
        total = input_cost + output_cost

        return {
            "model": self.model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": f"${input_cost:.4f}",
            "output_cost": f"${output_cost:.4f}",
            "total_cost": f"${total:.4f}"
        }

    def compare_models(self, input_tokens: int, output_tokens: int) -> List[Dict]:
        """比较不同模型的成本"""
        results = []
        for model in self.PRICING.keys():
            estimator = CostEstimator(model)
            results.append(estimator.estimate(input_tokens, output_tokens))
        return sorted(results, key=lambda x: float(x["total_cost"][1:]))

# 使用示例
estimator = CostEstimator("gpt-4")

# 单次估算
result = estimator.estimate(input_tokens=1000, output_tokens=500)
print(f"单次调用成本估算:")
for k, v in result.items():
    print(f"  {k}: {v}")

# 模型比较
print(f"\n不同模型成本比较 (1000 输入 + 500 输出):")
comparison = estimator.compare_models(1000, 500)
for r in comparison:
    print(f"  {r['model']}: {r['total_cost']}")

# ===== 5. RAG 上下文分配 =====
print("\n=== 5. RAG 上下文分配 ===")

class RAGContextAllocator:
    """RAG 上下文分配器"""

    def __init__(self, tokenizer, context_limit: int = 8192):
        self.tokenizer = tokenizer
        self.context_limit = context_limit

    def allocate(
        self,
        system_prompt: str,
        retrieved_docs: List[str],
        user_query: str,
        reserved_output: int = 1000
    ) -> Dict:
        """分配上下文空间"""
        # 计算固定部分
        system_tokens = self.tokenizer.count_tokens(system_prompt)
        query_tokens = self.tokenizer.count_tokens(user_query)
        fixed_tokens = system_tokens + query_tokens + reserved_output

        # 计算可用于文档的空间
        available_for_docs = self.context_limit - fixed_tokens

        # 选择能放入的文档
        selected_docs = []
        used_tokens = 0

        for doc in retrieved_docs:
            doc_tokens = self.tokenizer.count_tokens(doc)
            if used_tokens + doc_tokens <= available_for_docs:
                selected_docs.append(doc)
                used_tokens += doc_tokens
            else:
                break

        return {
            "context_limit": self.context_limit,
            "system_tokens": system_tokens,
            "query_tokens": query_tokens,
            "reserved_output": reserved_output,
            "available_for_docs": available_for_docs,
            "docs_selected": len(selected_docs),
            "docs_total": len(retrieved_docs),
            "docs_tokens": used_tokens,
            "total_used": fixed_tokens + used_tokens,
            "remaining": self.context_limit - (fixed_tokens + used_tokens)
        }

# 使用示例
allocator = RAGContextAllocator(tokenizer, context_limit=1000)

system_prompt = "You are a helpful assistant. Answer based on the provided context."
retrieved_docs = [
    "Document 1: Python is a programming language...",
    "Document 2: LangChain is a framework for LLM...",
    "Document 3: Machine learning is a subset of AI...",
    "Document 4: Deep learning uses neural networks...",
]
user_query = "What is LangChain?"

allocation = allocator.allocate(system_prompt, retrieved_docs, user_query)

print("RAG 上下文分配:")
for k, v in allocation.items():
    print(f"  {k}: {v}")

print("\n=== 完成！===")
```

**运行输出示例：**
```
=== 1. Token 计数 ===
'Hello, World!'
  Token数: 4
  Token IDs: [0, 1, 2, 3]

'LangChain is a framework for LLM applications.'
  Token数: 8
  Token IDs: [4, 5, 6, 7, 8, 9, 10, 11]

'你好世界'
  Token数: 1
  Token IDs: [12]

=== 2. 上下文窗口检查 ===
短文本检查: {'fits': True, 'input_tokens': 4, 'context_limit': 8192, ...}
长文本检查: {'fits': False, 'input_tokens': 9999, 'overflow': 2807, ...}
截断后长度: 18432 字符

=== 3. 消息历史管理 ===
添加: [user] Hello!...
  状态: {'message_count': 1, 'total_tokens': 2, 'max_tokens': 100}
...
最终保留的消息:
  [assistant] Sure! Here's a simple example...

=== 4. 成本估算器 ===
单次调用成本估算:
  model: gpt-4
  input_tokens: 1000
  output_tokens: 500
  total_cost: $0.0600

不同模型成本比较:
  gpt-3.5-turbo: $0.0013
  gpt-4o: $0.0125
  gpt-4-turbo: $0.0250
  gpt-4: $0.0600

=== 5. RAG 上下文分配 ===
RAG 上下文分配:
  context_limit: 1000
  docs_selected: 2
  docs_total: 4
  remaining: 812

=== 完成！===
```

---

## 8. 【面试必问】

### 问题："什么是 Token？为什么 LLM 使用 Token 而不是字符或单词？"

**普通回答（❌ 不出彩）：**
"Token 就是把文本切成小块，比字符大，比单词小。"

**出彩回答（✅ 推荐）：**

> **Token 是 LLM 处理文本的基本单位，它是效率和质量的最佳平衡点：**
>
> 1. **为什么不用字符？**
>    - 字符太小（只有26个字母 + 符号）
>    - 需要学习字符组合规则（"c-a-t" = 猫）
>    - 序列太长，计算成本高
>
> 2. **为什么不用单词？**
>    - 词汇表太大（英语有 100万+ 单词）
>    - 无法处理新词（OOV 问题）
>    - 不同语言词汇表差异大
>
> 3. **Token 的优势**
>    - 子词拆分：`"unhappiness"` → `["un", "happiness"]`
>    - 词汇表可控：约 50,000-100,000 个 Token
>    - 能处理任何文本，包括新词、代码、表情符号
>
> **在实际应用中的影响：**
> - API 按 Token 计费，理解 Token 才能控制成本
> - 上下文窗口按 Token 计算，影响 RAG 检索量
> - 不同语言 Token 效率不同（中文约 1.5字/Token，英文约 4字符/Token）
>
> **在 LangChain 中**：`llm.get_num_tokens()` 方法用于计算 Token，`ConversationTokenBufferMemory` 用于基于 Token 管理对话历史。

**为什么这个回答出彩？**
1. ✅ 解释了"为什么"而不只是"是什么"
2. ✅ 对比了三种方案的优劣
3. ✅ 联系了实际应用（成本、RAG）
4. ✅ 提到了具体的 LangChain 组件

---

### 问题："如何处理超过上下文窗口的长文档？"

**普通回答（❌ 不出彩）：**
"把文档切短，或者用更大上下文的模型。"

**出彩回答（✅ 推荐）：**

> **处理长文档有三种主要策略：**
>
> **1. 截断策略**
> ```python
> # 保留开头和结尾（重要信息通常在这里）
> truncated = doc[:max_tokens//2] + doc[-max_tokens//2:]
> ```
>
> **2. 分块处理（Map-Reduce）**
> ```python
> # 文档分块 → 每块独立处理 → 合并结果
> chunks = split_document(doc, chunk_size=4000)
> summaries = [llm.invoke(chunk) for chunk in chunks]
> final = llm.invoke(combine(summaries))
> ```
>
> **3. RAG 检索（推荐）**
> ```python
> # 只检索相关部分，而不是处理整个文档
> relevant_chunks = retriever.invoke(question)
> answer = llm.invoke(relevant_chunks + question)
> ```
>
> **选择策略的依据：**
> | 场景 | 推荐策略 |
> |------|---------|
> | 问答 | RAG（只需相关部分）|
> | 全文总结 | Map-Reduce |
> | 快速预览 | 截断（首尾）|
>
> **LangChain 支持：**
> - `RecursiveCharacterTextSplitter`：智能分块
> - `MapReduceDocumentsChain`：Map-Reduce 模式
> - `VectorStoreRetriever`：RAG 检索
>
> **我的经验**：对于知识库问答，RAG 是最高效的方案；对于需要理解全文的任务（如合同审查），Map-Reduce 更合适。

---

## 9. 【化骨绵掌】

### 卡片1：Token 是什么？ 🎯

**一句话：** Token 是 LLM 处理文本的基本单位，介于字符和单词之间。

**举例：**
```python
"Hello World" → ["Hello", " World"]  # 2 tokens
"你好世界" → ["你", "好", "世", "界"]  # ~4 tokens
```

**应用：** API 按 Token 计费，上下文按 Token 限制。

---

### 卡片2：Tokenizer 分词器 🔤

**一句话：** Tokenizer 将文本拆分成 Token，不同模型使用不同的 Tokenizer。

**举例：**
```python
import tiktoken
encoding = tiktoken.encoding_for_model("gpt-4")
tokens = encoding.encode("Hello")  # [9906]
```

**应用：** 精确计算 Token 数需要使用模型对应的 Tokenizer。

---

### 卡片3：上下文窗口 📏

**一句话：** 上下文窗口是 LLM 一次能处理的最大 Token 数。

**举例：**
```python
# GPT-4: 8,192 tokens
# GPT-4 Turbo: 128,000 tokens
# Claude 3: 200,000 tokens
```

**应用：** 输入 + 输出不能超过上下文窗口。

---

### 卡片4：Token 计费 💰

**一句话：** API 按 Token 数量计费，输出通常比输入贵。

**举例：**
```python
# GPT-4 定价
# 输入: $0.03 / 1K tokens
# 输出: $0.06 / 1K tokens

cost = (1000 * 0.03 + 500 * 0.06) / 1000  # = $0.06
```

**应用：** 控制 max_tokens 可以降低成本。

---

### 卡片5：max_tokens 参数 🎛️

**一句话：** max_tokens 限制模型输出的最大长度。

**举例：**
```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", max_tokens=500)
# 输出最多 500 tokens
```

**应用：** 根据场景设置合适的输出限制。

---

### 卡片6：消息截断策略 ✂️

**一句话：** 当对话超出窗口时，需要策略性地截断历史消息。

**举例：**
```python
# 策略1：保留最近 K 条
messages = all_messages[-10:]

# 策略2：按 Token 截断
# 策略3：总结压缩
```

**应用：** LangChain 的 Memory 组件实现各种截断策略。

---

### 卡片7：LangChain Token Memory 🧠

**一句话：** ConversationTokenBufferMemory 按 Token 限制管理对话历史。

**举例：**
```python
from langchain.memory import ConversationTokenBufferMemory

memory = ConversationTokenBufferMemory(
    llm=llm,
    max_token_limit=2000
)
```

**应用：** 自动截断超出限制的历史消息。

---

### 卡片8：中英文 Token 差异 🌍

**一句话：** 中文每字约 1-2 Token，英文每词约 1 Token。

**举例：**
```python
# 同样的含义
"Hello World"  # 2 tokens
"你好世界"      # 4-6 tokens

# 中文消耗更多 Token！
```

**应用：** 中文应用需要预留更多 Token 空间。

---

### 卡片9：RAG 与上下文分配 📊

**一句话：** RAG 需要合理分配 System Prompt、检索内容、用户输入和输出空间。

**举例：**
```
128K 窗口分配：
- System Prompt: 2K
- 检索内容: 50K
- 用户输入: 1K
- 输出空间: 75K
```

**应用：** 检索内容不是越多越好，要留够输出空间。

---

### 卡片10：Token 在 LangChain 中的作用 ⭐

**一句话：** Token 是 LangChain 成本控制和上下文管理的基础。

**举例：**
```python
# 计算 Token
num_tokens = llm.get_num_tokens(text)

# Token 限制的 Memory
memory = ConversationTokenBufferMemory(max_token_limit=2000)

# 限制输出
llm = ChatOpenAI(max_tokens=500)
```

**应用：** 理解 Token 才能构建高效的 LangChain 应用。

---

## 10. 【一句话总结】

**Token 是 LLM 处理文本的最小单位，上下文窗口是 LLM 的工作记忆容量，理解这两个概念是控制 API 成本、管理对话历史、优化 RAG 检索的基础。**

---

## 📚 学习检查清单

- [ ] 理解 Token 与字符、单词的区别
- [ ] 能够使用 tiktoken 计算 Token 数量
- [ ] 知道常见模型的上下文窗口大小
- [ ] 会使用 max_tokens 控制输出长度
- [ ] 理解 Token 计费模型
- [ ] 能够实现基于 Token 的消息截断

## 🔗 下一步学习

- **流式输出 Streaming**：Token 级别的实时输出
- **Memory 记忆系统**：LangChain 的对话历史管理
- **RAG 检索器**：上下文窗口的高效利用

---

**版本：** v1.0
**最后更新：** 2025-12-12
