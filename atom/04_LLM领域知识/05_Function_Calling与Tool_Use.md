# Function Calling 与 Tool Use

> 原子化知识点 | LLM领域知识 | LangChain 源码学习前置知识

---

## 1. 【30字核心】

**Function Calling 让 LLM 能够调用外部函数，Tool Use 是 LangChain 中工具调用的抽象，是 Agent 系统的核心能力。**

---

## 2. 【第一性原理】

### 什么是第一性原理？

**第一性原理**：回到事物最基本的真理，从源头思考问题

### Function Calling 与 Tool Use 的第一性原理 🎯

#### 1. 最基础的定义

**Function Calling = LLM 决定调用什么函数 + 传什么参数（但不执行）**

仅此而已！没有更基础的了。

```python
# LLM 不能直接执行代码，它只能"建议"调用什么
user_input = "北京今天天气怎么样？"

# LLM 的输出（Function Calling）
llm_response = {
    "function_call": {
        "name": "get_weather",        # 建议调用哪个函数
        "arguments": {
            "city": "北京",           # 建议传什么参数
            "date": "2024-01-15"
        }
    }
}

# 实际执行函数的是你的代码，不是 LLM
result = get_weather(**llm_response["function_call"]["arguments"])
```

#### 2. 为什么需要 Function Calling？

**核心问题：LLM 只能生成文本，不能直接与外部世界交互**

```python
# LLM 的局限性
# ❌ 不能查询数据库
# ❌ 不能调用 API
# ❌ 不能执行代码
# ❌ 不能访问实时信息

# Function Calling 解决方案
# ✅ LLM 决定"需要什么操作"
# ✅ 你的代码"执行操作"
# ✅ 结果返回给 LLM
# ✅ LLM 基于结果生成回答

# 完整流程
# 1. 用户："北京天气怎么样？"
# 2. LLM：决定调用 get_weather(city="北京")
# 3. 代码：执行 get_weather，返回 "晴天，25度"
# 4. LLM：基于结果回答 "北京今天天气晴朗，气温25度"
```

#### 3. Function Calling 的三层价值

##### 价值1：连接 LLM 与外部世界

```python
# LLM 可以"使用"任何工具
tools = [
    get_weather,        # 天气 API
    search_web,         # 搜索引擎
    query_database,     # 数据库查询
    send_email,         # 发送邮件
    control_smart_home, # 智能家居控制
]

# LLM 根据用户需求选择合适的工具
```

##### 价值2：结构化输出保证

```python
# Function Calling 强制 LLM 输出结构化数据
# 比起让 LLM "自由发挥"，这样更可靠

# 定义函数签名
def get_weather(city: str, date: str) -> dict:
    """获取天气信息"""
    pass

# LLM 必须按这个格式输出参数
# 不会输出 "我不知道" 或其他无效格式
```

##### 价值3：Agent 智能体的基础

```python
# Agent = LLM + 工具 + 执行循环
while not done:
    # 1. LLM 决定下一步行动
    action = llm.decide(observation)

    # 2. 执行工具
    result = execute_tool(action)

    # 3. 更新观察
    observation = result
```

#### 4. 从第一性原理推导 LangChain 应用

**推理链：**

```
1. LLM 只能生成文本，不能执行操作
   ↓
2. 需要一种机制让 LLM "表达"想执行的操作
   ↓
3. Function Calling：LLM 输出函数名和参数
   ↓
4. 应用代码执行实际函数
   ↓
5. 需要标准化的工具定义格式
   ↓
6. LangChain Tool：统一的工具抽象
   ↓
7. 需要自动化的执行循环
   ↓
8. LangChain Agent：工具调用 + 执行循环
```

#### 5. 一句话总结第一性原理

**Function Calling 是 LLM 与外部世界的桥梁，LLM 负责"决策"（调用什么、传什么参数），代码负责"执行"，两者配合实现真正的智能助手。**

---

## 3. 【核心概念（全面覆盖）】

### 核心概念1：Tool 定义 🔧

**Tool 是 LangChain 中可被 LLM 调用的函数封装**

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# 方式1：使用 @tool 装饰器（最简单）
@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息

    Args:
        city: 城市名称，如"北京"、"上海"
    """
    # 模拟天气 API
    return f"{city}今天晴天，气温25度"

# 方式2：使用 Pydantic 定义参数结构（推荐）
class WeatherInput(BaseModel):
    """天气查询参数"""
    city: str = Field(description="城市名称")
    date: str = Field(default="today", description="日期，默认今天")

@tool(args_schema=WeatherInput)
def get_weather_v2(city: str, date: str = "today") -> str:
    """获取指定城市和日期的天气"""
    return f"{city}在{date}天气晴朗"

# 方式3：继承 BaseTool（最灵活）
from langchain_core.tools import BaseTool
from typing import Optional, Type

class SearchTool(BaseTool):
    """自定义搜索工具"""
    name: str = "web_search"
    description: str = "搜索互联网获取信息"
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        """同步执行"""
        return f"搜索结果：关于 {query} 的信息..."

    async def _arun(self, query: str) -> str:
        """异步执行"""
        return await async_search(query)
```

**Tool 的核心属性：**

| 属性 | 作用 | 示例 |
|------|------|------|
| `name` | 工具名称（LLM 用来选择） | "get_weather" |
| `description` | 工具描述（LLM 用来理解用途） | "获取天气信息" |
| `args_schema` | 参数定义（Pydantic Model） | `WeatherInput` |
| `return_direct` | 是否直接返回结果 | `False` |

**在 LangChain 源码中的应用：**

```python
# langchain_core/tools/base.py
class BaseTool(ABC, BaseModel):
    """工具基类"""
    name: str
    description: str
    args_schema: Optional[Type[BaseModel]] = None

    @abstractmethod
    def _run(self, *args, **kwargs) -> Any:
        """同步执行工具"""
        pass

    async def _arun(self, *args, **kwargs) -> Any:
        """异步执行工具"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._run, *args, **kwargs
        )
```

---

### 核心概念2：bind_tools() 绑定工具 🔗

**bind_tools() 将工具信息传递给 LLM，让它知道有哪些工具可用**

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# 创建 LLM
llm = ChatOpenAI(model="gpt-4")

# 定义工具
@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    return str(eval(expression))

@tool
def get_weather(city: str) -> str:
    """获取天气"""
    return f"{city}今天晴天"

# 绑定工具到 LLM
llm_with_tools = llm.bind_tools([calculator, get_weather])

# 调用 LLM
response = llm_with_tools.invoke([
    HumanMessage(content="北京天气怎么样？")
])

# 检查是否有工具调用
if response.tool_calls:
    for tool_call in response.tool_calls:
        print(f"工具: {tool_call['name']}")
        print(f"参数: {tool_call['args']}")
```

**tool_calls 结构：**

```python
# response.tool_calls 是一个列表
[
    {
        "id": "call_abc123",           # 调用 ID
        "name": "get_weather",         # 工具名
        "args": {"city": "北京"}       # 参数
    }
]
```

---

### 核心概念3：ToolMessage 工具结果 📨

**ToolMessage 用于将工具执行结果返回给 LLM**

```python
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# 完整的工具调用流程
messages = [
    HumanMessage(content="北京和上海的天气怎么样？")
]

# 1. LLM 决定调用工具
response = llm_with_tools.invoke(messages)
messages.append(response)

# 2. 执行工具并创建 ToolMessage
for tool_call in response.tool_calls:
    # 执行工具
    if tool_call["name"] == "get_weather":
        result = get_weather(tool_call["args"]["city"])

    # 创建 ToolMessage
    tool_message = ToolMessage(
        content=result,
        tool_call_id=tool_call["id"]  # 必须匹配
    )
    messages.append(tool_message)

# 3. LLM 基于工具结果生成最终回答
final_response = llm_with_tools.invoke(messages)
print(final_response.content)
```

**消息流程示意：**

```
1. HumanMessage: "北京天气怎么样？"
   ↓
2. AIMessage: tool_calls=[{name: "get_weather", args: {city: "北京"}}]
   ↓
3. ToolMessage: content="北京晴天25度", tool_call_id="call_123"
   ↓
4. AIMessage: content="北京今天天气晴朗，气温25度，非常适合出行"
```

---

### 核心概念4：Agent 执行循环 🔄

**Agent 自动化工具调用和结果处理的循环**

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

# 创建 Agent Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手，可以使用工具回答问题。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")  # 工具调用历史
])

# 创建 Agent
agent = create_tool_calling_agent(llm, tools, prompt)

# 创建 AgentExecutor（执行器）
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # 显示执行过程
    max_iterations=10,  # 最大迭代次数
)

# 执行
result = executor.invoke({"input": "北京天气怎么样？然后计算 25 + 10"})
print(result["output"])
```

**Agent 执行流程：**

```
while 未完成:
    1. LLM 观察当前状态
    2. LLM 决定下一步行动（调用工具 or 返回结果）
    3. if 调用工具:
           执行工具
           将结果加入历史
       else:
           返回最终答案
           break
```

---

### 扩展概念5：结构化输出 Structured Output 📋

**使用 with_structured_output() 强制 LLM 输出特定结构**

```python
from pydantic import BaseModel, Field
from typing import List, Optional

# 定义输出结构
class MovieRecommendation(BaseModel):
    """电影推荐结果"""
    title: str = Field(description="电影名称")
    year: int = Field(description="上映年份")
    genre: str = Field(description="类型")
    reason: str = Field(description="推荐理由")

class MovieList(BaseModel):
    """电影推荐列表"""
    movies: List[MovieRecommendation]
    total: int

# 绑定结构化输出
structured_llm = llm.with_structured_output(MovieList)

# 调用
result = structured_llm.invoke("推荐3部科幻电影")

# result 是 MovieList 对象
print(f"共 {result.total} 部电影")
for movie in result.movies:
    print(f"- {movie.title} ({movie.year}): {movie.reason}")
```

**结构化输出 vs 工具调用：**

| 特性 | 结构化输出 | 工具调用 |
|------|----------|---------|
| 目的 | 强制输出格式 | 执行外部操作 |
| 是否执行代码 | 否 | 是 |
| 返回类型 | Pydantic Model | 工具执行结果 |
| 适用场景 | 数据提取、分类 | API 调用、操作执行 |

---

## 4. 【最小可用】

掌握以下内容，就能在 LangChain 中使用工具调用：

### 4.1 定义工具

```python
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """搜索信息"""
    return f"关于 {query} 的搜索结果"
```

### 4.2 绑定工具到 LLM

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
llm_with_tools = llm.bind_tools([search])
```

### 4.3 处理工具调用

```python
response = llm_with_tools.invoke("搜索 Python 教程")

if response.tool_calls:
    for call in response.tool_calls:
        result = search(call["args"]["query"])
        print(f"工具结果: {result}")
```

### 4.4 使用 AgentExecutor

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是助手"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

agent = create_tool_calling_agent(llm, [search], prompt)
executor = AgentExecutor(agent=agent, tools=[search])

result = executor.invoke({"input": "搜索 Python 教程"})
```

**这些知识足以：**
- 创建自定义工具
- 让 LLM 调用外部 API
- 构建简单的 Agent

---

## 5. 【1个类比】（双轨制）

### 类比1：Tool 定义

#### 🎨 前端视角：API 接口定义 / TypeScript 类型

Tool 就像定义一个 API 接口，告诉 LLM "有什么能力可以用"。

```typescript
// TypeScript 接口定义
interface WeatherAPI {
  // 函数签名
  getWeather(city: string): Promise<WeatherResult>;

  // 参数类型
  // 返回类型
}

// OpenAPI 规范
{
  "paths": {
    "/weather": {
      "get": {
        "summary": "获取天气",
        "parameters": [
          {"name": "city", "type": "string"}
        ]
      }
    }
  }
}
```

```python
# LangChain Tool 定义
@tool
def get_weather(city: str) -> str:
    """获取天气信息

    Args:
        city: 城市名称
    """
    return f"{city}天气晴朗"
```

**关键相似点：**
- 都是函数签名的声明
- 都包含参数类型和描述
- 都用于让调用方知道如何使用

#### 🧒 小朋友视角：工具箱里的工具

Tool 就像工具箱里的各种工具：

```
工具箱里有：
- 锤子：用来敲钉子（name: hammer, use: 敲东西）
- 螺丝刀：用来拧螺丝（name: screwdriver, use: 拧螺丝）
- 尺子：用来量长度（name: ruler, use: 测量）

每个工具都有：
- 名字（name）
- 用途说明（description）
- 使用方法（args）
```

**生活例子：**
```
你告诉机器人厨师你有这些厨具：
- 平底锅：用来煎东西
- 烤箱：用来烤东西
- 搅拌机：用来搅拌

机器人厨师就知道：
- 做煎蛋要用平底锅
- 做蛋糕要用烤箱和搅拌机
```

---

### 类比2：bind_tools() 绑定

#### 🎨 前端视角：依赖注入 / 插件注册

bind_tools() 就像给系统注册可用的插件。

```javascript
// 插件注册模式
const app = createApp();

// 注册插件
app.use(RouterPlugin);
app.use(StorePlugin);
app.use(I18nPlugin);

// 现在 app 知道有哪些插件可用
```

```python
# LangChain bind_tools
llm = ChatOpenAI()

# 绑定工具
llm_with_tools = llm.bind_tools([
    search_tool,
    calculator_tool,
    weather_tool
])

# 现在 LLM 知道有哪些工具可用
```

#### 🧒 小朋友视角：告诉助手你有什么

bind_tools() 就像告诉你的小助手，你家里有什么工具：

```
你对助手说：
"我家里有这些东西可以用：
 - 电话：可以打电话给别人
 - 电脑：可以上网查资料
 - 计算器：可以算数学题"

助手记住了，以后你问问题时：
- 问天气 → 助手说"我用电脑帮你查"
- 问数学 → 助手说"我用计算器帮你算"
- 问妈妈电话 → 助手说"我帮你打电话问"
```

---

### 类比3：Agent 执行循环

#### 🎨 前端视角：Redux 中间件 / 状态机

Agent 执行循环就像一个状态机，不断响应事件。

```javascript
// Redux 中间件模式
const agentMiddleware = store => next => action => {
  // 观察当前状态
  const state = store.getState();

  // 决定下一步
  if (needsToolCall(state, action)) {
    // 执行工具
    const result = executeTool(action.tool);
    // 更新状态
    store.dispatch({ type: 'TOOL_RESULT', payload: result });
  } else {
    // 返回最终结果
    return next(action);
  }
};

// 状态机
while (state !== 'DONE') {
  const action = llm.decide(state);
  state = executeAction(action);
}
```

```python
# LangChain Agent 循环
while not done:
    action = agent.decide(observation)
    if action.type == "tool_call":
        result = execute_tool(action)
        observation = result
    else:
        return action.output
```

#### 🧒 小朋友视角：做任务的步骤

Agent 就像一个会自己想办法完成任务的助手：

```
任务：帮我查北京天气然后告诉我要不要带伞

助手的思考过程：
1. "我需要查天气" → 使用天气工具
2. "天气是：下雨" → 记住这个信息
3. "下雨需要带伞" → 得出结论
4. "告诉主人：要带伞" → 完成任务

这个循环：
思考 → 行动 → 观察 → 思考 → ...直到完成
```

---

### 类比4：ToolMessage 返回结果

#### 🎨 前端视角：API Response / 回调函数

ToolMessage 就像 API 调用后的响应。

```javascript
// API 调用流程
const request = { type: 'GET_WEATHER', params: { city: '北京' } };
const response = await fetch('/api/weather', request);
const result = await response.json();

// 使用结果
console.log(`天气：${result.weather}`);
```

```python
# LangChain 工具调用流程
tool_call = {"name": "get_weather", "args": {"city": "北京"}}
result = get_weather(**tool_call["args"])  # 执行工具

# 创建 ToolMessage 返回结果
tool_message = ToolMessage(
    content=result,
    tool_call_id=tool_call["id"]
)
```

#### 🧒 小朋友视角：问问题得答案

ToolMessage 就像你问别人问题，得到答案：

```
你问爸爸："今天天气怎么样？"（工具调用）
爸爸说："今天晴天，25度"（ToolMessage）
你用这个答案决定："那我不带伞了"（LLM 最终回答）
```

---

### 类比总结表

| Function Calling 概念 | 前端类比 | 小朋友类比 |
|----------------------|---------|-----------|
| Tool 定义 | API/TypeScript 接口 | 工具箱里的工具 |
| bind_tools() | 插件注册 | 告诉助手你有什么 |
| tool_calls | API 请求 | 助手决定用什么工具 |
| ToolMessage | API 响应 | 工具的使用结果 |
| Agent | 状态机/中间件 | 会思考的助手 |
| 执行循环 | Event Loop | 不断尝试直到完成 |

---

## 6. 【反直觉点】

### 误区1：LLM 直接执行工具 ❌

**为什么错？**
- LLM 只是"建议"调用什么工具
- 实际执行是你的代码做的
- LLM 完全不知道工具的实际实现

**为什么人们容易这样错？**
因为看起来像是 LLM 在"使用"工具，但实际上 LLM 只是生成了调用指令。

**正确理解：**

```python
# ❌ 错误理解：LLM 执行工具
# "LLM 调用了 get_weather 函数"

# ✅ 正确理解：LLM 只是输出调用指令
response = llm_with_tools.invoke("北京天气")
# response 包含：{"tool_calls": [{"name": "get_weather", "args": {...}}]}

# 你的代码执行工具
for call in response.tool_calls:
    if call["name"] == "get_weather":
        result = get_weather(**call["args"])  # 这里才真正执行

# LLM 再基于结果生成回答
```

**经验法则：** LLM 是"决策者"，你的代码是"执行者"

---

### 误区2：工具描述不重要 ❌

**为什么错？**
- LLM 完全依赖 description 来理解工具用途
- 描述不清会导致 LLM 选错工具
- 描述是 LLM 唯一了解工具的途径

**为什么人们容易这样错？**
程序员习惯看代码理解功能，但 LLM 只能看到 description。

**正确理解：**

```python
# ❌ 糟糕的描述
@tool
def search(q: str) -> str:
    """搜索"""  # 搜索什么？怎么搜索？LLM 不知道
    pass

# ✅ 好的描述
@tool
def search(query: str) -> str:
    """在互联网上搜索信息。

    当用户询问最新新闻、实时信息或需要查找资料时使用此工具。
    不要用于已知的常识性问题。

    Args:
        query: 搜索关键词，应该简洁明确

    Returns:
        搜索结果的摘要
    """
    pass
```

**经验法则：** 描述写给 LLM 看，要像写给新员工的操作手册

---

### 误区3：Agent 一定比 Chain 好 ❌

**为什么错？**
- Agent 有不确定性（可能死循环、调用错误工具）
- Agent 更难调试和控制
- 简单任务用 Chain 更可靠

**为什么人们容易这样错？**
Agent 看起来更"智能"，但智能也意味着不可预测。

**正确理解：**

```python
# 场景1：固定流程 → 用 Chain
# "翻译 + 摘要" 总是这两步
chain = translate_prompt | llm | summary_prompt | llm

# 场景2：需要动态决策 → 用 Agent
# "回答问题可能需要搜索、可能需要计算、可能直接回答"
agent = create_tool_calling_agent(llm, tools, prompt)

# 选择标准
# Chain：流程固定、可预测、易调试
# Agent：流程不定、更灵活、难调试
```

| 场景 | 推荐 |
|------|-----|
| 翻译服务 | Chain |
| 数据处理流水线 | Chain |
| 智能客服 | Agent |
| 研究助手 | Agent |

**经验法则：** 能用 Chain 解决就不用 Agent

---

## 7. 【实战代码】

```python
"""
示例：Function Calling 与 Tool Use 完整演示
展示 LangChain 中工具调用的核心用法
"""

from typing import Optional, List
from dataclasses import dataclass
from pydantic import BaseModel, Field

# ===== 1. 模拟 LLM 和工具系统 =====
print("=== 1. 工具定义 ===")

@dataclass
class ToolCall:
    """工具调用"""
    id: str
    name: str
    args: dict

@dataclass
class ToolResult:
    """工具结果"""
    tool_call_id: str
    content: str

class Tool:
    """工具基类"""
    def __init__(self, name: str, description: str, func):
        self.name = name
        self.description = description
        self.func = func

    def invoke(self, **kwargs) -> str:
        return self.func(**kwargs)

    def get_schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "查询内容"}
                }
            }
        }

# 定义工具
def search_web(query: str) -> str:
    """搜索互联网"""
    return f"搜索结果：关于「{query}」的信息..."

def get_weather(city: str) -> str:
    """获取天气"""
    weather_data = {
        "北京": "晴天，25°C",
        "上海": "多云，22°C",
        "广州": "小雨，28°C"
    }
    return weather_data.get(city, f"{city}：天气数据暂无")

def calculator(expression: str) -> str:
    """计算器"""
    try:
        result = eval(expression)
        return f"计算结果：{expression} = {result}"
    except:
        return "计算错误"

# 创建工具实例
tools = [
    Tool("search_web", "搜索互联网获取信息", search_web),
    Tool("get_weather", "获取城市天气", get_weather),
    Tool("calculator", "计算数学表达式", calculator),
]

print("已定义工具：")
for tool in tools:
    print(f"  - {tool.name}: {tool.description}")

# ===== 2. 模拟 LLM 决策 =====
print("\n=== 2. 模拟 LLM 决策 ===")

class MockLLM:
    """模拟 LLM"""

    def __init__(self, tools: List[Tool]):
        self.tools = {t.name: t for t in tools}

    def decide(self, message: str) -> Optional[ToolCall]:
        """决定是否调用工具"""
        message_lower = message.lower()

        # 简单规则匹配（实际 LLM 会更智能）
        if "天气" in message:
            city = "北京"  # 简化处理
            for c in ["北京", "上海", "广州"]:
                if c in message:
                    city = c
                    break
            return ToolCall(
                id="call_001",
                name="get_weather",
                args={"city": city}
            )

        if "搜索" in message or "查找" in message:
            return ToolCall(
                id="call_002",
                name="search_web",
                args={"query": message.replace("搜索", "").replace("查找", "").strip()}
            )

        if any(op in message for op in ["+", "-", "*", "/", "计算"]):
            # 提取数学表达式
            import re
            expr = re.search(r'[\d\+\-\*\/\(\)\s]+', message)
            if expr:
                return ToolCall(
                    id="call_003",
                    name="calculator",
                    args={"expression": expr.group().strip()}
                )

        return None  # 不需要工具

    def generate_response(self, message: str, tool_results: List[ToolResult] = None) -> str:
        """生成最终回答"""
        if tool_results:
            context = "\n".join([r.content for r in tool_results])
            return f"根据查询结果：\n{context}\n\n总结：以上就是您需要的信息。"
        else:
            return f"我直接回答：{message} - 这是一个常识性问题。"

# 测试决策
llm = MockLLM(tools)

test_messages = [
    "北京天气怎么样？",
    "搜索 Python 教程",
    "计算 25 + 17",
    "你好",  # 不需要工具
]

for msg in test_messages:
    tool_call = llm.decide(msg)
    if tool_call:
        print(f"'{msg}' → 调用工具: {tool_call.name}({tool_call.args})")
    else:
        print(f"'{msg}' → 不需要工具")

# ===== 3. 完整执行流程 =====
print("\n=== 3. 完整执行流程 ===")

def execute_with_tools(llm: MockLLM, message: str) -> str:
    """完整的工具调用执行流程"""
    print(f"用户输入: {message}")

    # 1. LLM 决策
    tool_call = llm.decide(message)

    if tool_call is None:
        # 不需要工具，直接回答
        print("  → 不需要工具")
        return llm.generate_response(message)

    print(f"  → 决定调用: {tool_call.name}")

    # 2. 执行工具
    tool = llm.tools[tool_call.name]
    result = tool.invoke(**tool_call.args)
    print(f"  → 工具结果: {result}")

    # 3. 创建 ToolResult
    tool_result = ToolResult(
        tool_call_id=tool_call.id,
        content=result
    )

    # 4. 基于结果生成回答
    response = llm.generate_response(message, [tool_result])
    return response

# 测试完整流程
print("\n--- 测试1: 天气查询 ---")
print(execute_with_tools(llm, "上海天气怎么样？"))

print("\n--- 测试2: 计算 ---")
print(execute_with_tools(llm, "帮我计算 100 - 37"))

print("\n--- 测试3: 直接回答 ---")
print(execute_with_tools(llm, "你是谁？"))

# ===== 4. 多工具调用 =====
print("\n=== 4. 多工具调用 ===")

def execute_multiple_tools(llm: MockLLM, messages: List[str]) -> str:
    """处理多个需要工具的问题"""
    all_results = []

    for msg in messages:
        tool_call = llm.decide(msg)
        if tool_call:
            tool = llm.tools[tool_call.name]
            result = tool.invoke(**tool_call.args)
            all_results.append(ToolResult(
                tool_call_id=tool_call.id,
                content=f"问题「{msg}」的答案: {result}"
            ))

    return llm.generate_response("综合查询", all_results)

# 测试多工具
queries = ["北京天气", "计算 50 + 50"]
print(f"多个问题: {queries}")
print(execute_multiple_tools(llm, queries))

# ===== 5. Agent 执行循环模拟 =====
print("\n=== 5. Agent 执行循环 ===")

class SimpleAgent:
    """简单的 Agent 实现"""

    def __init__(self, llm: MockLLM, max_iterations: int = 5):
        self.llm = llm
        self.max_iterations = max_iterations
        self.history = []

    def run(self, task: str) -> str:
        """执行任务"""
        print(f"任务: {task}")

        for i in range(self.max_iterations):
            print(f"\n--- 迭代 {i+1} ---")

            # 决策
            tool_call = self.llm.decide(task)

            if tool_call is None:
                # 任务完成
                print("  决定：直接回答")
                return self.llm.generate_response(task, self.history)

            print(f"  决定：调用 {tool_call.name}")

            # 执行
            tool = self.llm.tools[tool_call.name]
            result = tool.invoke(**tool_call.args)
            print(f"  结果：{result}")

            # 记录历史
            self.history.append(ToolResult(
                tool_call_id=tool_call.id,
                content=result
            ))

            # 简化：一次工具调用后就完成
            break

        return self.llm.generate_response(task, self.history)

# 测试 Agent
agent = SimpleAgent(llm)
result = agent.run("查询广州天气")
print(f"\n最终答案: {result}")

# ===== 6. 结构化输出模拟 =====
print("\n=== 6. 结构化输出 ===")

class WeatherResponse(BaseModel):
    """天气响应结构"""
    city: str = Field(description="城市名称")
    temperature: int = Field(description="温度（摄氏度）")
    condition: str = Field(description="天气状况")
    suggestion: str = Field(description="出行建议")

def parse_weather_to_structured(weather_str: str, city: str) -> WeatherResponse:
    """将天气字符串解析为结构化数据"""
    # 简单解析
    if "晴" in weather_str:
        condition = "晴天"
        suggestion = "适合户外活动"
    elif "雨" in weather_str:
        condition = "下雨"
        suggestion = "记得带伞"
    else:
        condition = "多云"
        suggestion = "天气适中"

    # 提取温度
    import re
    temp_match = re.search(r'(\d+)', weather_str)
    temp = int(temp_match.group(1)) if temp_match else 20

    return WeatherResponse(
        city=city,
        temperature=temp,
        condition=condition,
        suggestion=suggestion
    )

# 测试结构化输出
weather_str = get_weather("北京")
structured = parse_weather_to_structured(weather_str, "北京")
print(f"结构化天气数据:")
print(f"  城市: {structured.city}")
print(f"  温度: {structured.temperature}°C")
print(f"  状况: {structured.condition}")
print(f"  建议: {structured.suggestion}")

# ===== 7. 工具选择策略 =====
print("\n=== 7. 工具选择策略 ===")

def select_best_tool(query: str, tools: List[Tool]) -> Optional[Tool]:
    """选择最合适的工具"""
    # 简单的关键词匹配策略
    keywords_map = {
        "get_weather": ["天气", "温度", "下雨", "晴天"],
        "search_web": ["搜索", "查找", "了解", "什么是"],
        "calculator": ["计算", "加", "减", "乘", "除", "+", "-", "*", "/"],
    }

    scores = {}
    for tool in tools:
        keywords = keywords_map.get(tool.name, [])
        score = sum(1 for kw in keywords if kw in query)
        scores[tool.name] = score

    # 选择得分最高的
    if max(scores.values()) > 0:
        best_name = max(scores, key=scores.get)
        return next(t for t in tools if t.name == best_name)

    return None

# 测试工具选择
test_queries = [
    "今天温度多少？",
    "什么是机器学习？",
    "5 乘以 8 等于多少？",
    "你好呀",
]

for query in test_queries:
    tool = select_best_tool(query, tools)
    if tool:
        print(f"'{query}' → 选择工具: {tool.name}")
    else:
        print(f"'{query}' → 不需要工具")

print("\n=== 完成！===")
```

**运行输出示例：**
```
=== 1. 工具定义 ===
已定义工具：
  - search_web: 搜索互联网获取信息
  - get_weather: 获取城市天气
  - calculator: 计算数学表达式

=== 2. 模拟 LLM 决策 ===
'北京天气怎么样？' → 调用工具: get_weather({'city': '北京'})
'搜索 Python 教程' → 调用工具: search_web({'query': 'Python 教程'})
'计算 25 + 17' → 调用工具: calculator({'expression': '25 + 17'})
'你好' → 不需要工具

=== 3. 完整执行流程 ===

--- 测试1: 天气查询 ---
用户输入: 上海天气怎么样？
  → 决定调用: get_weather
  → 工具结果: 多云，22°C
根据查询结果：
多云，22°C

总结：以上就是您需要的信息。

--- 测试2: 计算 ---
用户输入: 帮我计算 100 - 37
  → 决定调用: calculator
  → 工具结果: 计算结果：100 - 37 = 63
...

=== 5. Agent 执行循环 ===
任务: 查询广州天气

--- 迭代 1 ---
  决定：调用 get_weather
  结果：小雨，28°C

最终答案: 根据查询结果：
小雨，28°C

总结：以上就是您需要的信息。

=== 6. 结构化输出 ===
结构化天气数据:
  城市: 北京
  温度: 25°C
  状况: 晴天
  建议: 适合户外活动

=== 完成！===
```

---

## 8. 【面试必问】

### 问题："什么是 Function Calling？它和直接让 LLM 输出 JSON 有什么区别？"

**普通回答（❌ 不出彩）：**
"Function Calling 就是让 LLM 调用函数。比直接输出 JSON 更可靠。"

**出彩回答（✅ 推荐）：**

> **Function Calling 是 LLM 表达"我想执行某个操作"的标准化方式：**
>
> **核心区别：**
>
> | 维度 | 直接输出 JSON | Function Calling |
> |------|--------------|-----------------|
> | 格式保证 | 依赖 Prompt，可能失败 | API 层面保证格式 |
> | 参数验证 | 需要手动解析验证 | 自动类型检查 |
> | 多工具选择 | 需要复杂 Prompt | 原生支持 |
> | 调用方式 | 从文本中提取 | 结构化的 tool_calls |
>
> **Function Calling 的工作原理：**
> 1. 开发者定义工具的 schema（名称、描述、参数类型）
> 2. 将 schema 传给 LLM（通过 bind_tools）
> 3. LLM 输出结构化的工具调用指令
> 4. 开发者代码执行实际函数
> 5. 结果通过 ToolMessage 返回给 LLM
>
> **关键洞察**：LLM 不执行代码，它只是"建议"调用什么。实际执行权在开发者手中，这是安全的关键。
>
> **在 LangChain 中**：
> - `@tool` 装饰器定义工具
> - `llm.bind_tools()` 绑定工具
> - `response.tool_calls` 获取调用指令
> - `ToolMessage` 返回执行结果
>
> **实际应用**：我在项目中用 Function Calling 实现了一个智能客服，它可以查询订单、修改地址、发起退款。通过严格的工具定义，避免了 LLM 执行危险操作。

**为什么这个回答出彩？**
1. ✅ 清晰对比两种方式的区别
2. ✅ 解释了安全性考虑
3. ✅ 联系了 LangChain 具体实现
4. ✅ 有实际项目经验

---

### 问题："如何设计一个安全的工具调用系统？"

**普通回答（❌ 不出彩）：**
"限制工具的权限，做好输入验证。"

**出彩回答（✅ 推荐）：**

> **安全的工具调用系统需要多层防护：**
>
> **1. 工具设计层**
> ```python
> # 最小权限原则
> @tool
> def query_order(order_id: str) -> str:
>     """只读查询，不能修改"""
>     pass
>
> # 敏感操作需要确认
> @tool
> def delete_order(order_id: str, confirm: bool = False) -> str:
>     """删除订单，需要明确确认"""
>     if not confirm:
>         return "请设置 confirm=True 确认删除"
>     pass
> ```
>
> **2. 参数验证层**
> ```python
> class OrderInput(BaseModel):
>     order_id: str = Field(pattern=r'^ORD-\d{8}$')  # 格式验证
>
> @tool(args_schema=OrderInput)
> def query_order(order_id: str):
>     pass
> ```
>
> **3. 执行控制层**
> ```python
> # 限制迭代次数
> executor = AgentExecutor(
>     agent=agent,
>     tools=tools,
>     max_iterations=5,  # 防止死循环
>     max_execution_time=60,  # 超时限制
> )
>
> # 工具白名单
> allowed_tools = ["query_order", "get_weather"]
> for call in response.tool_calls:
>     if call["name"] not in allowed_tools:
>         raise SecurityError("未授权的工具调用")
> ```
>
> **4. 审计日志层**
> - 记录所有工具调用
> - 记录参数和结果
> - 异常调用告警
>
> **设计原则**：
> - 默认拒绝，显式允许
> - 读写分离
> - 敏感操作需要二次确认
> - 完整的审计追踪

---

## 9. 【化骨绵掌】

### 卡片1：Function Calling 是什么？ 🎯

**一句话：** Function Calling 让 LLM 输出"调用什么函数、传什么参数"的指令。

**举例：**
```python
# LLM 输出
{"tool_calls": [{"name": "get_weather", "args": {"city": "北京"}}]}

# 你的代码执行
result = get_weather(city="北京")
```

**应用：** LLM 与外部世界交互的标准方式。

---

### 卡片2：Tool 定义 🔧

**一句话：** Tool 是可被 LLM 调用的函数封装，包含名称、描述和参数定义。

**举例：**
```python
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """搜索互联网"""
    return f"搜索结果：{query}"
```

**应用：** @tool 装饰器是最简单的工具定义方式。

---

### 卡片3：bind_tools() 绑定 🔗

**一句话：** bind_tools() 将工具信息传给 LLM，让它知道有哪些工具可用。

**举例：**
```python
llm = ChatOpenAI()
llm_with_tools = llm.bind_tools([search, calculator])
```

**应用：** 绑定后 LLM 才能"看到"这些工具。

---

### 卡片4：tool_calls 调用指令 📋

**一句话：** LLM 响应中的 tool_calls 包含工具名和参数。

**举例：**
```python
response = llm_with_tools.invoke("搜索 Python")
for call in response.tool_calls:
    print(call["name"], call["args"])
```

**应用：** 遍历 tool_calls 来执行工具。

---

### 卡片5：ToolMessage 结果返回 📨

**一句话：** ToolMessage 用于将工具执行结果返回给 LLM。

**举例：**
```python
from langchain_core.messages import ToolMessage

tool_message = ToolMessage(
    content="搜索结果...",
    tool_call_id=call["id"]  # 必须匹配
)
```

**应用：** LLM 需要 ToolMessage 才能基于结果继续回答。

---

### 卡片6：Agent 执行循环 🔄

**一句话：** Agent 自动化：LLM 决策 → 执行工具 → 观察结果 → 继续决策。

**举例：**
```python
executor = AgentExecutor(agent=agent, tools=tools)
result = executor.invoke({"input": "北京天气"})
```

**应用：** 复杂任务自动分解执行。

---

### 卡片7：结构化输出 📊

**一句话：** with_structured_output() 强制 LLM 输出特定 Pydantic 结构。

**举例：**
```python
class Result(BaseModel):
    answer: str
    confidence: float

structured_llm = llm.with_structured_output(Result)
```

**应用：** 数据提取、分类等需要固定格式的场景。

---

### 卡片8：工具描述很重要 📝

**一句话：** LLM 完全依赖 description 来理解工具用途和选择工具。

**举例：**
```python
# ❌ 糟糕
"""搜索"""

# ✅ 优秀
"""在互联网搜索信息。
用于查找最新新闻、实时数据等。
Args: query - 搜索关键词
"""
```

**应用：** 描述写得好，工具调用才准确。

---

### 卡片9：LLM 不执行代码 ⚠️

**一句话：** LLM 只是"建议"调用什么，实际执行的是你的代码。

**举例：**
```python
# LLM 输出：{"name": "delete_all"}
# 你决定是否真的执行！

if is_safe(tool_call):
    execute(tool_call)
else:
    reject(tool_call)
```

**应用：** 安全的关键：执行权在你手中。

---

### 卡片10：Function Calling 在 LangChain 中的位置 ⭐

**一句话：** Function Calling 是 Agent 系统的核心，连接 LLM 决策和实际操作。

**举例：**
```python
# 完整链条
用户输入 → LLM 决策 → 工具调用 → 执行结果 → LLM 总结 → 最终回答
```

**应用：** 理解 Function Calling 就理解了 Agent 的核心。

---

## 10. 【一句话总结】

**Function Calling 让 LLM 能够表达"调用什么工具、传什么参数"的意图，Tool 是 LangChain 中工具的标准封装，两者配合让 LLM 从"只能说"变成"能做事"，是构建 Agent 系统的核心能力。**

---

## 📚 学习检查清单

- [ ] 理解 Function Calling 的工作原理
- [ ] 能够使用 @tool 装饰器定义工具
- [ ] 会使用 bind_tools() 绑定工具
- [ ] 理解 tool_calls 和 ToolMessage 的关系
- [ ] 能够使用 AgentExecutor 执行 Agent
- [ ] 了解工具调用的安全考虑

## 🔗 下一步学习

- **Agent 执行引擎**：LangChain Agent 的深入实现
- **ReAct 模式**：推理 + 行动的 Agent 设计
- **Callback 回调系统**：监控工具调用过程

---

**版本：** v1.0
**最后更新：** 2025-12-12
