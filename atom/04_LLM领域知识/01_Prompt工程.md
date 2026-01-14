# Prompt 工程

> 原子化知识点 | LLM领域知识 | LangChain 源码学习前置知识

---

## 1. 【30字核心】

**Prompt 工程是设计和优化与 LLM 交互输入的技术，决定了 AI 输出质量，是 LangChain 链式调用的基础。**

---

## 2. 【第一性原理】

### 什么是第一性原理？

**第一性原理**：回到事物最基本的真理，从源头思考问题

### Prompt 工程的第一性原理 🎯

#### 1. 最基础的定义

**Prompt = 给 LLM 的指令 + 上下文 + 输入**

仅此而已！没有更基础的了。

- **指令**：告诉 LLM 要做什么（角色、任务、格式）
- **上下文**：提供背景信息（知识、示例、约束）
- **输入**：用户的具体问题或数据

```python
# Prompt 的三要素
prompt = """
[指令] 你是一个专业的代码审查专家。请分析以下代码的问题。

[上下文] 代码应该遵循 PEP8 规范，注重可读性和性能。

[输入]
def calc(x,y):return x+y
"""
```

#### 2. 为什么需要 Prompt 工程？

**核心问题：LLM 是"通才"，需要精确的指令才能变成"专家"**

```python
# ❌ 糟糕的 Prompt：模糊、缺乏上下文
response = llm("写代码")
# 可能输出：任何语言、任何功能的随机代码

# ✅ 好的 Prompt：明确、有上下文
response = llm("""
你是 Python 专家。请写一个函数：
- 功能：计算两个数的最大公约数
- 要求：使用递归实现，包含类型注解
- 格式：只输出代码，不要解释
""")
# 输出：精确符合要求的代码
```

#### 3. Prompt 工程的三层价值

##### 价值1：控制输出质量

```python
# 通过 Prompt 控制输出格式
prompt = """
分析以下文本的情感，输出 JSON 格式：
{"sentiment": "positive/negative/neutral", "confidence": 0.0-1.0}

文本：这个产品太棒了！
"""
# 输出：{"sentiment": "positive", "confidence": 0.95}
```

##### 价值2：注入领域知识

```python
# 通过 Prompt 注入专业知识
prompt = """
你是一名资深的 LangChain 开发者。

关于 Runnable 协议：
- Runnable 是 LangChain 的核心抽象
- 提供 invoke/stream/batch 三种调用方式
- 所有组件都实现 Runnable 接口

请回答：Runnable 的主要优势是什么？
"""
```

##### 价值3：实现复杂推理

```python
# Chain of Thought：让 LLM 逐步思考
prompt = """
问题：小明有 5 个苹果，给了小红 2 个，又买了 3 个，最后有多少个？

请逐步思考：
1. 初始数量
2. 每一步变化
3. 最终结果
"""
```

#### 4. 从第一性原理推导 LangChain 应用

**推理链：**

```
1. LLM 需要 Prompt 才能工作
   ↓
2. Prompt 经常需要动态插入变量（用户输入、检索结果）
   ↓
3. 需要模板系统管理 Prompt（PromptTemplate）
   ↓
4. 不同场景需要不同 Prompt 结构（ChatPromptTemplate、FewShotPromptTemplate）
   ↓
5. Prompt 需要与 LLM、输出解析器组合（LCEL 链式调用）
   ↓
6. LangChain 提供完整的 Prompt 管理和组合能力
```

#### 5. 一句话总结第一性原理

**Prompt 是人与 LLM 的接口协议，Prompt 工程是设计这个接口的艺术，LangChain 的 PromptTemplate 将这门艺术工程化。**

---

## 3. 【核心概念（全面覆盖）】

### 核心概念1：System Prompt（系统提示） 🎭

**System Prompt 是设定 LLM 角色和行为规则的指令，在整个对话中持续生效**

```python
from langchain_core.messages import SystemMessage, HumanMessage

# System Prompt 设定角色
messages = [
    SystemMessage(content="""
你是一个 Python 代码审查专家。
规则：
1. 只回答与 Python 相关的问题
2. 发现问题时给出修改建议
3. 使用中文回答
4. 输出格式：问题 -> 建议 -> 修改后代码
    """),
    HumanMessage(content="帮我检查这段代码：def add(a,b):return a+b")
]
```

**System Prompt 的关键要素：**

| 要素 | 作用 | 示例 |
|------|------|------|
| 角色定义 | 设定专业背景 | "你是资深 Python 开发者" |
| 行为规则 | 约束回答边界 | "只回答技术问题" |
| 输出格式 | 规范输出结构 | "使用 JSON 格式输出" |
| 语言风格 | 控制表达方式 | "使用简洁专业的语言" |

**在 LangChain 源码中的应用：**

```python
# langchain_core/prompts/chat.py
from langchain_core.prompts import ChatPromptTemplate

# 使用 ChatPromptTemplate 管理 System Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}，专门帮助用户{task}。"),
    ("human", "{input}")
])

# 动态填充变量
messages = prompt.invoke({
    "role": "代码助手",
    "task": "编写和优化 Python 代码",
    "input": "写一个快速排序函数"
})
```

---

### 核心概念2：Few-shot Learning（少样本学习） 📚

**通过在 Prompt 中提供示例，让 LLM 学习任务模式**

```python
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

# 定义示例
examples = [
    {"input": "开心", "output": "positive"},
    {"input": "难过", "output": "negative"},
    {"input": "还行", "output": "neutral"},
]

# 示例格式模板
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="输入：{input}\n输出：{output}"
)

# Few-shot Prompt 模板
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix="判断以下文本的情感倾向：",
    suffix="输入：{text}\n输出：",
    input_variables=["text"]
)

# 生成 Prompt
prompt = few_shot_prompt.format(text="这个产品真不错")
print(prompt)
```

**输出的 Prompt：**
```
判断以下文本的情感倾向：
输入：开心
输出：positive
输入：难过
输出：negative
输入：还行
输出：neutral
输入：这个产品真不错
输出：
```

**Few-shot 的核心技巧：**

| 技巧 | 说明 | 示例数量 |
|------|------|---------|
| Zero-shot | 不提供示例，直接指令 | 0 |
| One-shot | 提供 1 个示例 | 1 |
| Few-shot | 提供 2-5 个示例 | 2-5 |
| Many-shot | 提供大量示例 | 10+ |

**在 LangChain 源码中的应用：**

```python
# langchain_core/prompts/few_shot.py
from langchain_core.prompts import FewShotChatMessagePromptTemplate

# 聊天格式的 Few-shot
examples = [
    {"input": "2+2", "output": "4"},
    {"input": "3*4", "output": "12"},
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}")
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)
```

---

### 核心概念3：Chain of Thought（思维链） 🧠

**让 LLM 展示推理过程，逐步解决复杂问题**

```python
# Chain of Thought Prompt 结构
cot_prompt = """
问题：一个商店有 100 个苹果，第一天卖出 30%，第二天卖出剩余的 50%，还剩多少？

让我们一步一步思考：

步骤1：计算第一天卖出的数量
- 第一天卖出：100 × 30% = 30 个
- 第一天剩余：100 - 30 = 70 个

步骤2：计算第二天卖出的数量
- 第二天卖出：70 × 50% = 35 个
- 第二天剩余：70 - 35 = 35 个

答案：还剩 35 个苹果

---
现在请用同样的方式解决这个问题：
{question}

让我们一步一步思考：
"""
```

**Chain of Thought 的变体：**

```python
# 1. Zero-shot CoT：简单触发词
prompt = f"{question}\n\n让我们一步一步思考。"

# 2. Few-shot CoT：带推理示例
prompt = f"""
示例问题：[问题]
推理过程：[步骤1] -> [步骤2] -> [结论]
答案：[答案]

当前问题：{question}
推理过程：
"""

# 3. Self-Consistency CoT：多次推理取多数
responses = [llm(cot_prompt) for _ in range(5)]
final_answer = majority_vote(responses)
```

**在 LangChain 源码中的应用：**

```python
# 使用 LCEL 构建 CoT 链
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

cot_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个善于逻辑推理的助手。请逐步分析问题。"),
    ("human", """
问题：{question}

请按以下格式回答：
思考过程：
1. [第一步分析]
2. [第二步分析]
...
最终答案：[答案]
    """)
])

# CoT 链
cot_chain = cot_template | llm | StrOutputParser()
```

---

### 核心概念4：Prompt Template（提示模板） 📝

**使用变量占位符创建可复用的 Prompt 模板**

```python
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

# 1. 基础 PromptTemplate
simple_template = PromptTemplate(
    input_variables=["product"],
    template="请为{product}写一段产品描述。"
)

# 2. 带验证的 PromptTemplate
validated_template = PromptTemplate(
    input_variables=["language", "task"],
    template="用{language}语言{task}",
    validate_template=True  # 验证变量是否正确
)

# 3. ChatPromptTemplate（对话格式）
chat_template = ChatPromptTemplate.from_messages([
    ("system", "你是{role}专家"),
    ("human", "{question}"),
])

# 4. 使用 MessagesPlaceholder 插入历史消息
from langchain_core.prompts import MessagesPlaceholder

chat_with_history = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手"),
    MessagesPlaceholder(variable_name="history"),  # 插入历史消息
    ("human", "{input}"),
])
```

**PromptTemplate 的高级用法：**

```python
# 1. Partial：预填充部分变量
partial_prompt = simple_template.partial(product="iPhone")
final_prompt = partial_prompt.format()  # 不需要再传 product

# 2. 组合多个模板
from langchain_core.prompts import PipelinePromptTemplate

full_template = PromptTemplate.from_template("""
{introduction}

{example}

{question}
""")

pipeline_prompt = PipelinePromptTemplate(
    final_prompt=full_template,
    pipeline_prompts=[
        ("introduction", intro_template),
        ("example", example_template),
        ("question", question_template),
    ]
)
```

**在 LangChain 源码中的应用：**

```python
# langchain_core/prompts/base.py 核心结构
class BasePromptTemplate(ABC):
    """所有 Prompt 模板的基类"""
    input_variables: List[str]

    @abstractmethod
    def format_prompt(self, **kwargs) -> PromptValue:
        """格式化为 PromptValue"""
        pass

    def invoke(self, input: Dict) -> PromptValue:
        """Runnable 接口实现"""
        return self.format_prompt(**input)
```

---

### 扩展概念5：Output Instructions（输出指令） 📋

**在 Prompt 中明确指定输出格式和结构**

```python
# JSON 格式输出指令
json_prompt = """
分析以下代码并输出 JSON 格式的结果：

代码：
{code}

输出格式要求：
{{
    "language": "编程语言",
    "lines": 代码行数,
    "complexity": "low/medium/high",
    "issues": ["问题1", "问题2"],
    "suggestions": ["建议1", "建议2"]
}}

请严格按照上述 JSON 格式输出，不要包含其他内容。
"""
```

**配合 Output Parser 使用：**

```python
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# 定义输出结构
class CodeAnalysis(BaseModel):
    language: str = Field(description="编程语言")
    lines: int = Field(description="代码行数")
    complexity: str = Field(description="复杂度等级")
    issues: list[str] = Field(description="发现的问题")

# 创建解析器
parser = JsonOutputParser(pydantic_object=CodeAnalysis)

# 获取格式指令
format_instructions = parser.get_format_instructions()

# 组合到 Prompt
prompt = PromptTemplate(
    template="分析代码：{code}\n\n{format_instructions}",
    input_variables=["code"],
    partial_variables={"format_instructions": format_instructions}
)
```

---

## 4. 【最小可用】

掌握以下内容，就能开始编写有效的 LangChain Prompt：

### 4.1 使用 ChatPromptTemplate

```python
from langchain_core.prompts import ChatPromptTemplate

# 最常用的模式：system + human
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}"),
    ("human", "{question}")
])

# 格式化
messages = prompt.invoke({"role": "Python专家", "question": "什么是装饰器？"})
```

### 4.2 使用变量占位符

```python
from langchain_core.prompts import PromptTemplate

# 使用 {variable} 作为占位符
template = PromptTemplate.from_template(
    "将以下{source_lang}翻译成{target_lang}：\n{text}"
)

# 填充变量
prompt = template.format(
    source_lang="英文",
    target_lang="中文",
    text="Hello World"
)
```

### 4.3 插入历史消息

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

template = ChatPromptTemplate.from_messages([
    ("system", "你是一个助手"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# 传入历史消息
messages = template.invoke({
    "history": [
        HumanMessage(content="你好"),
        AIMessage(content="你好！有什么可以帮你？")
    ],
    "input": "继续上次的话题"
})
```

### 4.4 LCEL 链式组合

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Prompt + LLM + Parser 组合
chain = (
    ChatPromptTemplate.from_messages([
        ("system", "你是翻译专家"),
        ("human", "翻译：{text}")
    ])
    | ChatOpenAI()
    | StrOutputParser()
)

# 一行调用
result = chain.invoke({"text": "Hello"})
```

**这些知识足以：**
- 构建 90% 的 LangChain 应用 Prompt
- 阅读和理解 LangChain 示例代码
- 实现对话系统、翻译、问答等常见任务

---

## 5. 【1个类比】（双轨制）

### 类比1：PromptTemplate 模板系统

#### 🎨 前端视角：JSX 模板 / 模板字符串

PromptTemplate 就像前端的模板系统，使用占位符动态生成内容。

```javascript
// JavaScript 模板字符串
const template = (name, role) => `
  你好，${name}！
  你是一个${role}。
`;

const prompt = template("Alice", "助手");
```

```python
# LangChain PromptTemplate
from langchain_core.prompts import PromptTemplate

template = PromptTemplate.from_template("""
你好，{name}！
你是一个{role}。
""")

prompt = template.format(name="Alice", role="助手")
```

**关键相似点：**
- 都使用占位符（JS: `${}`，LangChain: `{}`）
- 都支持变量插值
- 都可以复用模板

#### 🧒 小朋友视角：填空题

PromptTemplate 就像填空题：

```
题目：我叫____，今年____岁，我喜欢____。

填写答案：
- 第一个空：小明
- 第二个空：10
- 第三个空：画画

完成后：我叫小明，今年10岁，我喜欢画画。
```

**生活例子：**
```
妈妈给你一张请假条模板：

"敬爱的____老师：
  我因____，需要请假____天。
  请批准。
                    学生：____"

你只需要填上空白的地方，就能完成请假条！
```

---

### 类比2：System Prompt 角色设定

#### 🎨 前端视角：应用配置 / 主题设置

System Prompt 就像前端应用的全局配置，一次设定，处处生效。

```javascript
// React Context 全局配置
const AppContext = createContext({
  theme: 'dark',
  language: 'zh-CN',
  role: 'admin'
});

// 所有子组件都受这个配置影响
function App() {
  return (
    <AppContext.Provider value={config}>
      <Dashboard />  {/* 自动使用 dark 主题 */}
      <Settings />   {/* 自动使用中文 */}
    </AppContext.Provider>
  );
}
```

```python
# LangChain System Prompt
messages = [
    SystemMessage(content="你是一个Python专家，只回答编程问题"),  # 全局设定
    HumanMessage(content="什么是闭包？"),  # 会按专家角色回答
    HumanMessage(content="天气怎么样？"),  # 会拒绝回答（不是编程问题）
]
```

#### 🧒 小朋友视角：游戏角色选择

System Prompt 就像游戏开始时选择角色：

```
开始游戏前：
- 选择角色：魔法师
- 角色特点：会魔法，血量低，攻击高

游戏中：
- 所有技能都是魔法类型
- 行为符合魔法师设定
- 不能使用战士的技能
```

**生活例子：**
```
老师说："今天我们玩角色扮演游戏，你是医生！"

这意味着：
- 你要穿白大褂（角色设定）
- 你只能给病人看病（行为限制）
- 你不能去抓小偷（超出角色范围）
- 整个游戏你都是医生（持续生效）
```

---

### 类比3：Few-shot Learning 示例学习

#### 🎨 前端视角：Storybook / 组件示例

Few-shot 就像 Storybook 中的组件示例，通过例子说明如何使用。

```javascript
// Storybook: 通过示例展示组件用法
export default {
  title: 'Button',
  component: Button,
};

// 示例1：Primary 按钮
export const Primary = () => <Button variant="primary">Primary</Button>;

// 示例2：Secondary 按钮
export const Secondary = () => <Button variant="secondary">Secondary</Button>;

// 用户看了示例就知道怎么用 Button 了
```

```python
# Few-shot: 通过示例展示任务模式
examples = [
    {"input": "开心", "output": "positive"},
    {"input": "难过", "output": "negative"},
]

# LLM 看了示例就知道怎么分类情感了
```

#### 🧒 小朋友视角：看例题学做题

Few-shot 就像数学作业本上的例题：

```
【例题】
问题：3 + 5 = ?
解答：3 + 5 = 8

问题：2 + 4 = ?
解答：2 + 4 = 6

【练习题】
问题：7 + 2 = ?
解答：____（你来填）
```

**生活例子：**
```
学写字：
老师先写一个"大"字（示例）
然后你照着写（学习）

学画画：
老师先画一个太阳（示例1）
老师再画一朵云（示例2）
然后让你画一棵树（你来做）
```

---

### 类比4：Chain of Thought 逐步推理

#### 🎨 前端视角：Debug 断点调试

Chain of Thought 就像设置断点一步步调试代码。

```javascript
// 不用 CoT：直接得结果，不知道哪里错了
const result = complexCalculation(data);  // 结果错误，无从下手

// 用 CoT：每一步都打印，清晰看到过程
function complexCalculation(data) {
  console.log("Step 1: 解析输入", data);
  const parsed = parse(data);

  console.log("Step 2: 验证数据", parsed);
  const validated = validate(parsed);

  console.log("Step 3: 计算结果", validated);
  const result = calculate(validated);

  return result;
}
```

```python
# LLM Chain of Thought
prompt = """
问题：计算 15% 的 200 是多少？

让我们一步步思考：
步骤1：理解问题 - 需要计算 200 的 15%
步骤2：转换百分比 - 15% = 0.15
步骤3：执行乘法 - 200 × 0.15 = 30
答案：30
"""
```

#### 🧒 小朋友视角：应用题的"解题步骤"

Chain of Thought 就像数学应用题要求写的"解题步骤"：

```
题目：小明有 10 块糖，给了小红 3 块，又买了 5 块，现在有多少块？

❌ 错误答案（不写步骤）：
答：12 块

✅ 正确答案（写清步骤）：
解：
（1）小明原来有 10 块糖
（2）给小红后剩余：10 - 3 = 7 块
（3）买了 5 块后：7 + 5 = 12 块
答：小明现在有 12 块糖
```

---

### 类比总结表

| Prompt 概念 | 前端类比 | 小朋友类比 |
|------------|---------|-----------|
| PromptTemplate | 模板字符串 / JSX | 填空题 |
| System Prompt | 全局配置 / Context | 游戏角色选择 |
| Few-shot Learning | Storybook 示例 | 看例题学做题 |
| Chain of Thought | Debug 断点 | 写解题步骤 |
| Output Instructions | TypeScript 类型定义 | 作业格式要求 |
| MessagesPlaceholder | 路由参数 | 日记本（可以翻看之前的） |

---

## 6. 【反直觉点】

### 误区1：Prompt 越长越好 ❌

**为什么错？**
- 过长的 Prompt 会稀释关键信息
- 增加 Token 消耗（成本和延迟）
- LLM 可能"迷失"在大量文本中，忽略关键指令

**为什么人们容易这样错？**
人们认为提供更多信息总是更好，就像写作文要"内容丰富"。但 LLM 不是人类阅读者，它对 Prompt 的注意力分布是不均匀的。

**正确理解：**

```python
# ❌ 过长的 Prompt（信息冗余）
bad_prompt = """
你好！我是一个用户，我想问你一个问题。这个问题可能有点复杂，
但我相信你一定能够回答。在回答之前，我想先介绍一下背景...
（省略 500 字背景）
...总之，请告诉我 Python 如何读取文件？
另外，回答时请尽量详细，但也不要太长，要有条理，
最好有代码示例，但代码不要太复杂...
"""

# ✅ 简洁有效的 Prompt
good_prompt = """
任务：写出 Python 读取文件的代码
要求：
1. 使用 with 语句
2. 处理编码问题
3. 包含异常处理
"""
```

**经验法则：** 精简 > 详细，每个词都应该有用

---

### 误区2：System Prompt 写一次就够了 ❌

**为什么错？**
- System Prompt 需要根据任务迭代优化
- 不同场景需要不同的角色设定
- LLM 更新后，之前的 Prompt 可能失效

**为什么人们容易这样错？**
System Prompt 看起来像是"配置文件"，设置一次就不用管了。但实际上它更像是"对话开场白"，需要根据用户反馈不断调整。

**正确理解：**

```python
# ❌ 固定不变的 System Prompt
system_prompt = "你是一个助手"  # 太泛，效果差

# ✅ 针对任务优化的 System Prompt
system_prompts = {
    "code_review": """
你是资深 Python 代码审查专家。
职责：
1. 检查代码规范（PEP8）
2. 发现潜在 bug
3. 提出优化建议
格式：每个问题用 [问题][建议][示例] 三段式回答
    """,

    "translation": """
你是专业的中英翻译。
规则：
1. 保持原文语气和风格
2. 专业术语保留英文
3. 不要添加解释
    """,
}
```

**经验法则：** 定期评估 System Prompt 效果，A/B 测试不同版本

---

### 误区3：Few-shot 示例越多越好 ❌

**为什么错？**
- 太多示例会占用 Context Window
- 边缘案例示例可能误导 LLM
- 示例质量比数量更重要

**为什么人们容易这样错？**
机器学习中"数据越多越好"是常识，但 Few-shot 不是传统训练，它是通过上下文学习（In-context Learning），受限于模型的注意力机制。

**正确理解：**

```python
# ❌ 示例过多（20个示例）
examples = [...]  # 20个示例，占用大量 Token

# ✅ 精选代表性示例（3-5个）
examples = [
    {"input": "非常开心", "output": "positive"},      # 强正面
    {"input": "有点不高兴", "output": "negative"},   # 弱负面
    {"input": "还可以吧", "output": "neutral"},      # 中性
    {"input": "我恨这个！", "output": "negative"},   # 强负面
]
# 4个示例覆盖主要情况，效果更好

# ✅ 使用相似度选择示例（动态 Few-shot）
from langchain_core.example_selectors import SemanticSimilarityExampleSelector

selector = SemanticSimilarityExampleSelector.from_examples(
    examples=all_examples,
    embeddings=OpenAIEmbeddings(),
    k=3  # 只选择最相关的 3 个
)
```

**经验法则：** 3-5 个高质量示例 > 20 个普通示例

---

## 7. 【实战代码】

```python
"""
示例：构建一个完整的 Prompt 工程系统
演示 LangChain 中 Prompt 的核心用法
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    PromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import List, Dict

# ===== 1. 基础 ChatPromptTemplate =====
print("=== 1. 基础 ChatPromptTemplate ===")

# 创建聊天提示模板
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{role}，擅长{skill}。请用{style}的风格回答。"),
    ("human", "{question}")
])

# 格式化消息
messages = chat_prompt.format_messages(
    role="Python专家",
    skill="代码优化",
    style="简洁专业",
    question="什么是列表推导式？"
)

print(f"System: {messages[0].content[:50]}...")
print(f"Human: {messages[1].content}")

# ===== 2. Few-shot 提示模板 =====
print("\n=== 2. Few-shot 提示模板 ===")

# 定义示例
sentiment_examples = [
    {"text": "这个产品太棒了！", "sentiment": "positive", "confidence": "0.95"},
    {"text": "质量太差，退货！", "sentiment": "negative", "confidence": "0.90"},
    {"text": "一般般，没什么特别的", "sentiment": "neutral", "confidence": "0.75"},
]

# 示例格式
example_template = PromptTemplate(
    input_variables=["text", "sentiment", "confidence"],
    template="文本：{text}\n情感：{sentiment}\n置信度：{confidence}"
)

# Few-shot 模板
few_shot_prompt = FewShotPromptTemplate(
    examples=sentiment_examples,
    example_prompt=example_template,
    prefix="你是情感分析专家。请分析以下文本的情感倾向。\n\n示例：",
    suffix="\n现在分析这个文本：\n文本：{input_text}\n情感：",
    input_variables=["input_text"]
)

# 生成完整 Prompt
full_prompt = few_shot_prompt.format(input_text="这个电影还不错，值得一看")
print(full_prompt[:200] + "...")

# ===== 3. 带历史消息的对话模板 =====
print("\n=== 3. 带历史消息的对话模板 ===")

# 创建支持历史消息的模板
conversation_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个编程助手。记住用户之前的问题，保持对话连贯。"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# 模拟对话历史
history = [
    HumanMessage(content="Python 中如何定义函数？"),
    AIMessage(content="使用 def 关键字定义函数：\ndef my_func():\n    pass"),
    HumanMessage(content="如何给函数添加参数？"),
    AIMessage(content="在括号中添加参数名：\ndef my_func(arg1, arg2):\n    pass"),
]

# 格式化带历史的消息
messages_with_history = conversation_prompt.format_messages(
    history=history,
    input="如何设置默认参数值？"
)

print(f"消息数量: {len(messages_with_history)}")
print(f"最后一条: {messages_with_history[-1].content}")

# ===== 4. Chain of Thought 模板 =====
print("\n=== 4. Chain of Thought 模板 ===")

cot_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个逻辑推理专家。
对于每个问题，请按以下格式回答：

【理解问题】简述问题要求
【分析步骤】列出解题步骤
【逐步推理】
- 步骤1：...
- 步骤2：...
【最终答案】给出结论
"""),
    ("human", "{question}")
])

# 格式化 CoT 消息
cot_messages = cot_prompt.format_messages(
    question="一个书店有 200 本书，第一周卖出 25%，第二周又进货 50 本，现在有多少本？"
)

print(f"CoT Prompt 长度: {len(cot_messages[0].content)} 字符")

# ===== 5. 结构化输出指令 =====
print("\n=== 5. 结构化输出指令 ===")

structured_prompt = ChatPromptTemplate.from_messages([
    ("system", """你是代码分析助手。请按以下 JSON 格式输出分析结果：

# ```json
# {{
#     "language": "编程语言名称",
#     "purpose": "代码功能描述",
#     "complexity": "low/medium/high",
#     "issues": ["问题1", "问题2"],
#     "suggestions": ["建议1", "建议2"]
# }}
# ```

# 只输出 JSON，不要其他内容。"""),
#     ("human", "分析这段代码：\n```python\n{code}\n```")
# ])

# 格式化
code_analysis_prompt = structured_prompt.format_messages(
    code="def add(a,b):return a+b"
)

print(f"结构化输出指令已创建")

# ===== 6. 模板组合与复用 =====
print("\n=== 6. 模板组合与复用 ===")

# 创建可复用的子模板
role_template = PromptTemplate.from_template("你是{role}。")
task_template = PromptTemplate.from_template("任务：{task}")
format_template = PromptTemplate.from_template("输出格式：{format}")

# 组合模板
combined_prompt = PromptTemplate.from_template("""
{role_section}

{task_section}

{format_section}

输入：{input}
输出：
""")

# 填充子模板
final_prompt = combined_prompt.format(
    role_section=role_template.format(role="翻译专家"),
    task_section=task_template.format(task="将英文翻译成中文"),
    format_section=format_template.format(format="只输出翻译结果"),
    input="Hello, World!"
)

print(final_prompt)

# ===== 7. 实际 LangChain 链式调用示例 =====
print("\n=== 7. LangChain 风格的 Prompt 使用 ===")

# 展示如何在 LCEL 中使用 Prompt
# 注意：这里只是展示结构，不实际调用 LLM

class MockLLM:
    """模拟 LLM 用于演示"""
    def invoke(self, messages):
        return f"[Mock Response for: {messages[-1].content[:30]}...]"

mock_llm = MockLLM()

# 创建链
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是{role}"),
    ("human", "{question}")
])

# 手动模拟链式调用
formatted_messages = prompt.format_messages(role="助手", question="你好")
response = mock_llm.invoke(formatted_messages)
print(f"链式调用结果: {response}")

# ===== 8. Prompt 变量验证 =====
print("\n=== 8. Prompt 变量验证 ===")

# 创建带验证的模板
validated_prompt = PromptTemplate(
    input_variables=["name", "age"],
    template="姓名：{name}，年龄：{age}",
    validate_template=True
)

# 查看所需变量
print(f"必需变量: {validated_prompt.input_variables}")

# 使用 partial 预填充
partial_prompt = validated_prompt.partial(age="25")
print(f"预填充后的变量: {partial_prompt.input_variables}")

# 最终格式化
final = partial_prompt.format(name="Alice")
print(f"最终结果: {final}")

print("\n=== 完成！===")
```

**运行输出示例：**
```
=== 1. 基础 ChatPromptTemplate ===
System: 你是一个Python专家，擅长代码优化。请用简洁专业的风格回答。...
Human: 什么是列表推导式？

=== 2. Few-shot 提示模板 ===
你是情感分析专家。请分析以下文本的情感倾向。

示例：
文本：这个产品太棒了！
情感：positive
置信度：0.95
文本：质量太差，退货！...

=== 3. 带历史消息的对话模板 ===
消息数量: 6
最后一条: 如何设置默认参数值？

=== 4. Chain of Thought 模板 ===
CoT Prompt 长度: 156 字符

=== 5. 结构化输出指令 ===
结构化输出指令已创建

=== 6. 模板组合与复用 ===
你是翻译专家。

任务：将英文翻译成中文

输出格式：只输出翻译结果

输入：Hello, World!
输出：

=== 7. LangChain 风格的 Prompt 使用 ===
链式调用结果: [Mock Response for: 你好...]

=== 8. Prompt 变量验证 ===
必需变量: ['name', 'age']
预填充后的变量: ['name']
最终结果: 姓名：Alice，年龄：25

=== 完成！===
```

---

## 8. 【面试必问】

### 问题："什么是 Prompt 工程？为什么它很重要？"

**普通回答（❌ 不出彩）：**
"Prompt 工程就是写给 AI 的提示词，写得好 AI 回答就好，写得不好就回答不好。"

**出彩回答（✅ 推荐）：**

> **Prompt 工程有三个层次的含义：**
>
> 1. **接口设计层**：Prompt 是人与 LLM 交互的"API"。就像 REST API 需要精确的参数定义，Prompt 需要精确的指令、上下文和格式要求。糟糕的 Prompt 就像糟糕的 API 设计，导致不可预测的输出。
>
> 2. **知识注入层**：LLM 是通用的"基础模型"，通过 Prompt 我们可以临时注入领域知识、角色设定、行为约束，把通用模型变成专用助手。这就是 System Prompt 和 Few-shot Learning 的价值。
>
> 3. **推理引导层**：通过 Chain of Thought 等技术，我们可以引导 LLM 的推理过程，让它"展示工作"，从而提高复杂任务的准确率。
>
> **在 LangChain 中的重要性**：LangChain 的核心是"链式调用"，而 Prompt 是链的第一环。PromptTemplate、ChatPromptTemplate、FewShotPromptTemplate 都是为了将 Prompt 工程系统化、可复用化。一个好的 Prompt 设计可以让整个 Chain 的效果提升数倍。
>
> **我在实际项目中的经验**：通过 A/B 测试不同的 System Prompt，同一个 LLM 在客服场景的满意度可以从 70% 提升到 90%+。这说明 Prompt 工程是 LLM 应用开发的核心竞争力。

**为什么这个回答出彩？**
1. ✅ 分层回答，有深度
2. ✅ 联系了实际框架（LangChain）
3. ✅ 提到了具体技术（CoT、Few-shot）
4. ✅ 有量化的实际经验

---

### 问题："Few-shot 和 Fine-tuning 有什么区别？什么时候用哪个？"

**普通回答（❌ 不出彩）：**
"Few-shot 是在 Prompt 里加例子，Fine-tuning 是训练模型。Few-shot 简单，Fine-tuning 效果好。"

**出彩回答（✅ 推荐）：**

> **核心区别在于学习方式：**
>
> | 维度 | Few-shot | Fine-tuning |
> |------|----------|-------------|
> | 学习时机 | 推理时（In-context） | 训练时（更新权重） |
> | 数据量 | 3-10 个示例 | 数百到数万条 |
> | 成本 | 每次调用消耗 Token | 一次训练成本高，后续免费 |
> | 灵活性 | 随时调整 | 需要重新训练 |
> | 效果上限 | 受限于模型能力 | 可超越基础模型 |
>
> **选择策略：**
>
> 1. **先尝试 Few-shot**：如果任务可以用几个示例说清楚，Few-shot 是最快最便宜的方案。
>
> 2. **Few-shot 不够时考虑 Fine-tuning**：
>    - 任务太复杂，示例说不清楚
>    - 需要特定领域知识（如医疗、法律术语）
>    - 对延迟敏感（Fine-tuning 后不需要长 Prompt）
>
> 3. **也可以结合使用**：Fine-tuned 模型 + Few-shot 可以获得最好效果
>
> **在 LangChain 中的实践**：LangChain 提供了 FewShotPromptTemplate 和动态示例选择器（SemanticSimilarityExampleSelector），可以根据输入自动选择最相关的示例，这种"智能 Few-shot"在很多场景下效果接近 Fine-tuning。

---

## 9. 【化骨绵掌】

### 卡片1：Prompt 是什么？ 🎯

**一句话：** Prompt 是给 LLM 的输入文本，包含指令、上下文和用户输入。

**举例：**
```python
prompt = """
[指令] 翻译成中文
[上下文] 保持正式语气
[输入] Hello, World!
"""
```

**应用：** LangChain 的所有 LLM 调用都从 Prompt 开始。

---

### 卡片2：System Prompt 的作用 🎭

**一句话：** System Prompt 设定 LLM 的角色和行为规则，在整个对话中持续生效。

**举例：**
```python
from langchain_core.messages import SystemMessage

system = SystemMessage(content="你是 Python 专家，只回答编程问题")
```

**应用：** LangChain ChatModel 的第一条消息通常是 SystemMessage。

---

### 卡片3：PromptTemplate 模板 📝

**一句话：** 使用 `{variable}` 占位符创建可复用的 Prompt 模板。

**举例：**
```python
from langchain_core.prompts import PromptTemplate

template = PromptTemplate.from_template("翻译成{lang}：{text}")
prompt = template.format(lang="中文", text="Hello")
```

**应用：** LangChain 链式调用的标准起点。

---

### 卡片4：ChatPromptTemplate 对话模板 💬

**一句话：** 专门为聊天场景设计，支持多角色消息序列。

**举例：**
```python
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是{role}"),
    ("human", "{question}")
])
```

**应用：** 所有 ChatModel 场景的首选模板。

---

### 卡片5：Few-shot Learning 少样本 📚

**一句话：** 在 Prompt 中提供示例，让 LLM 学习任务模式。

**举例：**
```python
examples = [
    {"input": "开心", "output": "positive"},
    {"input": "难过", "output": "negative"},
]
# LLM 看到示例后就知道如何分类
```

**应用：** LangChain 的 FewShotPromptTemplate 自动管理示例。

---

### 卡片6：Chain of Thought 思维链 🧠

**一句话：** 让 LLM 逐步展示推理过程，提高复杂任务准确率。

**举例：**
```python
prompt = """
问题：100 - 30 + 15 = ?
让我们一步步计算：
1. 100 - 30 = 70
2. 70 + 15 = 85
答案：85
"""
```

**应用：** 复杂推理任务的标准技巧。

---

### 卡片7：MessagesPlaceholder 历史消息 📜

**一句话：** 在模板中预留位置，动态插入对话历史。

**举例：**
```python
from langchain_core.prompts import MessagesPlaceholder

template = ChatPromptTemplate.from_messages([
    ("system", "你是助手"),
    MessagesPlaceholder("history"),  # 历史消息插入点
    ("human", "{input}")
])
```

**应用：** LangChain 对话 Memory 的核心机制。

---

### 卡片8：Output Instructions 输出指令 📋

**一句话：** 在 Prompt 中指定输出格式，确保结构化输出。

**举例：**
```python
prompt = """
分析并输出 JSON：
{"sentiment": "positive/negative", "score": 0.0-1.0}
"""
```

**应用：** 配合 LangChain 的 OutputParser 实现可靠的结构化输出。

---

### 卡片9：动态示例选择 🎯

**一句话：** 根据用户输入动态选择最相关的 Few-shot 示例。

**举例：**
```python
from langchain_core.example_selectors import SemanticSimilarityExampleSelector

selector = SemanticSimilarityExampleSelector.from_examples(
    examples=all_examples,
    embeddings=embeddings,
    k=3  # 选择最相关的 3 个
)
```

**应用：** LangChain 的智能示例选择器。

---

### 卡片10：Prompt 在 LangChain 中的位置 ⭐

**一句话：** Prompt 是 LCEL 链的第一环，连接用户输入和 LLM。

**举例：**
```python
# LCEL 链式调用
chain = prompt | llm | output_parser

# Prompt 处理流程
用户输入 → PromptTemplate → 格式化消息 → LLM → 响应
```

**应用：** 理解 Prompt 就理解了 LangChain 的一半。

---

## 10. 【一句话总结】

**Prompt 工程是设计 LLM 输入的技术，通过 System Prompt 设定角色、Few-shot 提供示例、CoT 引导推理，LangChain 的 PromptTemplate 将这门技术系统化为可复用的模板组件。**

---

## 📚 学习检查清单

- [ ] 能够使用 ChatPromptTemplate 创建对话模板
- [ ] 理解 System Prompt 的作用和最佳实践
- [ ] 会使用 Few-shot 提供示例
- [ ] 理解 Chain of Thought 的原理
- [ ] 能够使用 MessagesPlaceholder 管理历史消息
- [ ] 知道如何在 Prompt 中指定输出格式

## 🔗 下一步学习

- **Token 与上下文窗口**：理解 Prompt 长度限制
- **Output Parser**：解析 LLM 的结构化输出
- **LCEL 表达式语言**：Prompt 与其他组件的链式组合
