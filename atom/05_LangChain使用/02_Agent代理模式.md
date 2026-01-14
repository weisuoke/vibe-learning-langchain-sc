# Agent 代理模式

> 原子化知识点 | LangChain 使用 | LangChain 源码学习核心知识

---

## 1. 【30字核心】

**Agent 是具有自主决策能力的 LLM 应用，通过 ReAct 循环"思考-行动-观察"来动态选择工具完成复杂任务。**

---

## 2. 【第一性原理】

### 什么是第一性原理？

**第一性原理**：回到事物最基本的真理，从源头思考问题

### Agent 代理模式的第一性原理 🎯

#### 1. 最基础的定义

**Agent = LLM + 工具 + 决策循环**

仅此而已！没有更基础的了。

```python
# Agent 的本质
while not 任务完成:
    思考 = LLM.分析当前状态()
    行动 = LLM.决定下一步(思考)
    结果 = 执行工具(行动)
    观察 = 更新状态(结果)
```

#### 2. 为什么需要 Agent？

**核心问题：Chain 是确定性的，但很多任务需要动态决策**

```python
# Chain 的局限
chain = prompt | llm | parser
# 流程固定：永远是 prompt → llm → parser

# 实际需求：
# "帮我查询北京天气，如果下雨就推荐室内活动"
# 需要：
# 1. 决定是否查询天气 → 可能需要
# 2. 根据天气结果决定下一步 → 动态的
# 3. 可能需要搜索活动 → 不确定

# Agent 的解决方案：让 LLM 自己决定！
```

#### 3. Agent 的三层价值

##### 价值1：动态决策 - LLM 决定做什么

```python
# LLM 根据任务动态选择
# 任务："计算 25 * 37"
# LLM 决定：需要使用计算器工具

# 任务："北京天气怎么样"
# LLM 决定：需要使用天气 API 工具

# 任务："你好"
# LLM 决定：不需要工具，直接回答
```

##### 价值2：工具使用 - 扩展 LLM 能力

```python
# LLM 本身不能：
# ❌ 查询实时数据
# ❌ 执行代码
# ❌ 访问数据库
# ❌ 调用 API

# 通过工具，Agent 可以：
# ✅ 搜索互联网
# ✅ 查询数据库
# ✅ 发送邮件
# ✅ 操作文件
```

##### 价值3：复杂任务分解 - 自动规划

```python
# 复杂任务："研究 Python 和 JavaScript 的区别，写一篇对比文章"
# Agent 自动分解：
# 1. 搜索 Python 特点
# 2. 搜索 JavaScript 特点
# 3. 对比分析
# 4. 生成文章
```

#### 4. 从第一性原理推导 Agent

**推理链：**

```
1. LLM 只能生成文本
   ↓
2. 需要与外部世界交互
   ↓
3. 引入工具(Tool)的概念
   ↓
4. 谁来决定用哪个工具？
   ↓
5. 让 LLM 自己决定 (Function Calling)
   ↓
6. 决策可能需要多轮
   ↓
7. 引入循环：决策 → 执行 → 观察 → 决策
   ↓
8. 这就是 Agent！
```

#### 5. 一句话总结第一性原理

**Agent 是让 LLM 从"被动执行"变成"主动决策"的机制，通过工具调用循环实现复杂任务的自动完成。**

---

## 3. 【核心概念（全面覆盖）】

### 核心概念1：ReAct 模式 🔄

**ReAct (Reasoning + Acting) 是 Agent 的核心思维模式：思考→行动→观察**

```python
# ReAct 循环
"""
Thought: 我需要先查询北京的天气
Action: get_weather
Action Input: {"city": "北京"}
Observation: 北京今天下雨，气温15度

Thought: 天气是下雨，我应该推荐室内活动
Action: search
Action Input: {"query": "北京室内活动推荐"}
Observation: [博物馆、电影院、购物中心...]

Thought: 我已经有足够信息回答用户了
Final Answer: 北京今天下雨(15度)，推荐去博物馆或看电影。
"""
```

**ReAct 的三个阶段：**

```python
# 1. Thought（思考）- LLM 分析当前状态
thought = "我需要查询天气信息来回答这个问题"

# 2. Action（行动）- 决定执行什么工具
action = AgentAction(
    tool="get_weather",
    tool_input={"city": "北京"}
)

# 3. Observation（观察）- 获取工具执行结果
observation = "北京今天晴天，25度"
```

**在 LangChain 源码中的应用：**

```python
# langchain/agents/react/agent.py
REACT_TEMPLATE = """Answer the following questions as best you can.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question
Thought: reasoning about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Observation)
Thought: I now know the final answer
Final Answer: the final answer
"""
```

---

### 核心概念2：AgentExecutor 执行引擎 ⚙️

**AgentExecutor 是 Agent 的执行器，负责循环调用 Agent 和执行工具**

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 创建工具
@tool
def get_weather(city: str) -> str:
    """获取城市天气"""
    return f"{city}今天晴天，25度"

@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    return str(eval(expression))

tools = [get_weather, calculator]

# 创建 Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手，可以使用工具回答问题。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")  # 存放中间步骤
])

# 创建 Agent
llm = ChatOpenAI(model="gpt-4")
agent = create_tool_calling_agent(llm, tools, prompt)

# 创建 AgentExecutor
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,  # 显示执行过程
    max_iterations=10,  # 最大迭代次数
    max_execution_time=60,  # 最大执行时间(秒)
    handle_parsing_errors=True,  # 处理解析错误
)

# 执行
result = executor.invoke({"input": "北京天气怎么样？然后计算 25 + 37"})
print(result["output"])
```

**AgentExecutor 的核心循环：**

```python
# langchain/agents/agent.py (简化版)
class AgentExecutor:
    def invoke(self, input):
        intermediate_steps = []

        for i in range(self.max_iterations):
            # 1. Agent 决策
            output = self.agent.invoke({
                "input": input,
                "intermediate_steps": intermediate_steps
            })

            # 2. 检查是否完成
            if isinstance(output, AgentFinish):
                return {"output": output.return_values["output"]}

            # 3. 执行工具
            action = output
            tool_result = self._execute_tool(action)

            # 4. 记录步骤
            intermediate_steps.append((action, tool_result))

        raise MaxIterationsError("达到最大迭代次数")
```

---

### 核心概念3：Tool 工具系统 🔧

**Tool 是 Agent 可以调用的外部能力，通过 @tool 装饰器或 BaseTool 类定义**

```python
from langchain_core.tools import tool, BaseTool
from pydantic import BaseModel, Field
from typing import Type

# 方式1：@tool 装饰器（最简单）
@tool
def search(query: str) -> str:
    """搜索互联网获取信息

    Args:
        query: 搜索关键词
    """
    return f"搜索结果：关于 {query} 的信息..."

# 方式2：带 Pydantic Schema（推荐）
class CalculatorInput(BaseModel):
    """计算器输入"""
    expression: str = Field(description="数学表达式，如 '2 + 3 * 4'")

@tool(args_schema=CalculatorInput)
def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        return str(eval(expression))
    except:
        return "计算错误"

# 方式3：继承 BaseTool（最灵活）
class CustomSearchTool(BaseTool):
    """自定义搜索工具"""
    name: str = "custom_search"
    description: str = "搜索互联网获取最新信息"
    args_schema: Type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        """同步执行"""
        return self._search(query)

    async def _arun(self, query: str) -> str:
        """异步执行"""
        return await self._async_search(query)

    def _search(self, query: str) -> str:
        # 实际搜索逻辑
        return f"搜索 '{query}' 的结果"
```

**工具的关键属性：**

| 属性 | 作用 | 重要性 |
|-----|------|-------|
| `name` | 工具名称，LLM 用来选择 | 必须 |
| `description` | 工具描述，LLM 用来理解用途 | 非常重要 |
| `args_schema` | 参数定义，确保参数正确 | 推荐 |
| `return_direct` | 是否直接返回结果 | 可选 |

---

### 核心概念4：intermediate_steps 中间步骤 📋

**intermediate_steps 记录 Agent 的执行历史，帮助 LLM 了解之前做了什么**

```python
# intermediate_steps 的结构
intermediate_steps = [
    (
        AgentAction(tool="get_weather", tool_input={"city": "北京"}),
        "北京今天晴天，25度"  # 工具返回结果
    ),
    (
        AgentAction(tool="search", tool_input={"query": "北京景点"}),
        "故宫、长城、颐和园..."
    ),
]

# 传递给 Agent
agent.invoke({
    "input": "北京天气怎么样，推荐几个景点",
    "intermediate_steps": intermediate_steps
})

# Agent 可以看到之前的操作和结果，避免重复
```

**在 Prompt 中的体现（agent_scratchpad）：**

```
之前的操作记录：

Action: get_weather
Action Input: {"city": "北京"}
Observation: 北京今天晴天，25度

Action: search
Action Input: {"query": "北京景点"}
Observation: 故宫、长城、颐和园...

现在根据以上信息，继续回答用户问题...
```

---

### 扩展概念5：AgentAction 和 AgentFinish 🎬

**AgentAction 表示要执行的工具调用，AgentFinish 表示任务完成**

```python
from langchain_core.agents import AgentAction, AgentFinish

# AgentAction：需要执行工具
action = AgentAction(
    tool="get_weather",
    tool_input={"city": "北京"},
    log="Thought: 我需要查询北京天气\nAction: get_weather"
)

# AgentFinish：任务完成，返回最终答案
finish = AgentFinish(
    return_values={"output": "北京今天晴天，25度，适合出游"},
    log="Thought: 我已经有了所有信息\nFinal Answer: ..."
)

# Agent 的返回类型
def agent_decision(input, steps) -> Union[AgentAction, AgentFinish]:
    if 需要工具:
        return AgentAction(...)
    else:
        return AgentFinish(...)
```

---

### 扩展概念6：多种 Agent 类型 🎭

**LangChain 提供多种 Agent 创建方式**

```python
from langchain.agents import (
    create_tool_calling_agent,  # 推荐：使用 Function Calling
    create_react_agent,          # ReAct 格式
    create_structured_chat_agent # 结构化聊天
)

# 1. Tool Calling Agent（推荐）
# 使用 LLM 的原生 Function Calling 功能
agent = create_tool_calling_agent(llm, tools, prompt)

# 2. ReAct Agent
# 使用 ReAct 格式的 Prompt
from langchain import hub
react_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(llm, tools, react_prompt)

# 3. Structured Chat Agent
# 支持多轮对话
agent = create_structured_chat_agent(llm, tools, prompt)
```

---

## 4. 【最小可用】

掌握以下内容，就能在 LangChain 中使用 Agent：

### 4.1 创建工具

```python
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """搜索互联网"""
    return f"搜索结果：{query}"

@tool
def calculator(expression: str) -> str:
    """计算数学表达式"""
    return str(eval(expression))

tools = [search, calculator]
```

### 4.2 创建 Agent

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个有帮助的助手"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

llm = ChatOpenAI(model="gpt-4")
agent = create_tool_calling_agent(llm, tools, prompt)
```

### 4.3 执行 Agent

```python
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10
)

result = executor.invoke({"input": "搜索 Python 教程"})
print(result["output"])
```

### 4.4 错误处理

```python
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,  # 处理解析错误
    max_iterations=10,           # 限制迭代次数
    max_execution_time=60        # 限制执行时间
)
```

**这些知识足以：**
- 创建自定义工具
- 构建能够动态决策的 Agent
- 处理常见错误和异常
- 控制 Agent 的执行范围

---

## 5. 【1个类比】（双轨制）

### 类比1：Agent 是游戏 NPC

#### 🎨 前端视角：状态机 / useReducer

Agent 就像一个复杂的状态机，根据当前状态和事件决定下一步动作。

```javascript
// useReducer 状态机
function reducer(state, action) {
  switch (action.type) {
    case 'NEED_WEATHER':
      return { ...state, nextAction: 'call_weather_api' };
    case 'WEATHER_RECEIVED':
      return { ...state, weather: action.payload, nextAction: 'decide' };
    case 'TASK_COMPLETE':
      return { ...state, done: true, answer: action.payload };
    default:
      return state;
  }
}

// 状态机循环
while (!state.done) {
  const action = llm.decide(state);
  state = reducer(state, action);
}
```

```python
# LangChain Agent
while not done:
    action = agent.decide(state)
    if action.type == "tool_call":
        result = execute_tool(action)
        state.update(result)
    else:
        return action.output
```

**关键相似点：**
- 都是根据状态决定下一步
- 都有循环执行直到完成
- 都支持多种动作类型

#### 🧒 小朋友视角：游戏里的 NPC

Agent 就像游戏里会思考的 NPC（非玩家角色）：

```
你问 NPC："怎么去城堡？"

NPC 思考：
1. "玩家要去城堡，我需要查查地图" → 使用地图工具
2. "地图显示要过一座桥" → 记住这个信息
3. "桥可能需要钥匙" → 使用物品检查工具
4. "玩家有钥匙" → 可以过桥
5. "我知道怎么回答了" → 告诉玩家路线

不同于普通 NPC 只会说固定台词，
Agent NPC 会根据情况动态思考和回答！
```

---

### 类比2：ReAct 是思考过程

#### 🎨 前端视角：调试器 / DevTools

ReAct 就像在代码中打断点调试，一步步看执行过程。

```javascript
// 调试过程
console.log("Thought: 需要获取用户数据");
// Action: 调用 API
const user = await fetchUser(id);
console.log("Observation:", user);

console.log("Thought: 需要检查权限");
// Action: 检查权限
const hasPermission = checkPermission(user);
console.log("Observation:", hasPermission);

console.log("Thought: 可以返回结果了");
// Final Answer
return { user, hasPermission };
```

#### 🧒 小朋友视角：做数学应用题

ReAct 就像老师教你做应用题的步骤：

```
题目：小明有 5 个苹果，又买了 3 个，吃了 2 个，还剩几个？

思考（Thought）：我需要先算买了之后有多少
行动（Action）：计算 5 + 3
观察（Observation）：= 8 个

思考（Thought）：然后算吃掉之后还剩多少
行动（Action）：计算 8 - 2
观察（Observation）：= 6 个

思考（Thought）：我知道答案了
最终答案：还剩 6 个苹果
```

---

### 类比3：AgentExecutor 是游戏主循环

#### 🎨 前端视角：Event Loop / 游戏循环

AgentExecutor 就像浏览器的事件循环或游戏的主循环。

```javascript
// 浏览器事件循环
while (true) {
  const event = getNextEvent();
  handleEvent(event);
  render();
}

// 游戏主循环
while (gameRunning) {
  processInput();
  updateGameState();
  render();
}
```

```python
# AgentExecutor 循环
while not done:
    action = agent.decide()
    result = execute(action)
    update_state(result)
```

#### 🧒 小朋友视角：玩游戏的过程

AgentExecutor 就像你玩游戏的过程：

```
游戏开始！

循环：
1. 看看现在在哪（观察状态）
2. 决定往哪走（思考决策）
3. 走过去（执行动作）
4. 看看发生了什么（获取结果）
5. 还没到终点？回到第 1 步

游戏结束：到达终点！
```

---

### 类比总结表

| LangChain 概念 | 前端类比 | 小朋友类比 |
|---------------|---------|-----------|
| Agent | 状态机 / useReducer | 游戏 NPC |
| ReAct | 调试器断点 | 做应用题的步骤 |
| AgentExecutor | Event Loop | 游戏主循环 |
| Tool | API 接口 | NPC 的技能 |
| intermediate_steps | 执行历史 | 之前做过的事 |
| AgentAction | dispatch action | 决定用什么技能 |
| AgentFinish | Promise.resolve | 任务完成！ |
| max_iterations | 超时机制 | 最多尝试几次 |

---

## 6. 【反直觉点】

### 误区1：Agent 比 Chain 更好 ❌

**为什么错？**
- Agent 有不确定性，每次执行可能不同
- Agent 更难调试，因为路径是动态的
- Agent 成本更高，可能多次调用 LLM
- 简单任务用 Agent 是过度设计

**为什么人们容易这样错？**
Agent 看起来更"智能"，但智能也意味着不可预测。

**正确理解：**

```python
# 选择标准

# 用 Chain：
# - 流程固定（翻译、摘要、格式转换）
# - 需要可预测性（生产环境）
# - 成本敏感（固定 LLM 调用次数）
chain = prompt | llm | parser  # 永远 3 次调用

# 用 Agent：
# - 流程不确定（智能助手、研究任务）
# - 需要工具调用（搜索、数据库、API）
# - 任务复杂度未知
agent = AgentExecutor(...)  # 可能 1 次，可能 10 次
```

| 场景 | 推荐 | 原因 |
|-----|-----|------|
| 翻译服务 | Chain | 流程固定 |
| 数据提取 | Chain | 可预测 |
| 智能客服 | Agent | 需要动态决策 |
| 研究助手 | Agent | 任务复杂 |

---

### 误区2：Agent 可以无限执行 ❌

**为什么错？**
- 不设限制，Agent 可能陷入死循环
- 每次迭代都会调用 LLM，成本累积
- 某些任务可能无法完成

**为什么人们容易这样错？**
以为 Agent 会"聪明地"完成任务，但 LLM 也可能犯错或困惑。

**正确理解：**

```python
# 必须设置限制！

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    # 限制1：最大迭代次数
    max_iterations=10,  # 最多 10 轮

    # 限制2：最大执行时间
    max_execution_time=60,  # 最多 60 秒

    # 限制3：错误处理
    handle_parsing_errors=True,

    # 可选：提前结束条件
    early_stopping_method="generate"  # 强制生成答案
)

# 生产环境建议
# max_iterations: 5-10
# max_execution_time: 30-60 秒
```

---

### 误区3：Agent 会自动选择最优路径 ❌

**为什么错？**
- LLM 可能选错工具
- 可能多次重复同样的操作
- 可能走弯路
- 需要好的 Prompt 和工具设计

**为什么人们容易这样错？**
以为 LLM 是"智能"的，会做出最优决策。

**正确理解：**

```python
# Agent 的决策质量取决于：

# 1. 工具描述的清晰度
@tool
def search(query: str) -> str:
    """搜索互联网获取最新信息。

    当用户询问实时信息、新闻或需要查找资料时使用。
    不要用于已知的常识性问题。

    Args:
        query: 搜索关键词，应该简洁明确
    """
    pass

# 2. Prompt 的指导
prompt = """你是一个高效的助手。
在回答问题前：
1. 先判断是否需要工具
2. 选择最合适的工具
3. 避免重复同样的操作
"""

# 3. 工具数量要适中
# 太多工具 → LLM 容易选错
# 太少工具 → 功能受限
# 建议：5-10 个精选工具
```

---

## 7. 【实战代码】

```python
"""
示例：Agent 代理模式完整演示
展示 LangChain 中 Agent 的核心用法
"""

from typing import List, Dict, Any, Union, Optional
from dataclasses import dataclass
from enum import Enum

# ===== 1. 定义核心数据结构 =====
print("=== 1. 核心数据结构 ===")

@dataclass
class AgentAction:
    """Agent 的动作：调用工具"""
    tool: str
    tool_input: Dict[str, Any]
    log: str = ""

@dataclass
class AgentFinish:
    """Agent 完成：返回最终答案"""
    return_values: Dict[str, Any]
    log: str = ""

# ===== 2. 定义工具系统 =====
print("\n=== 2. 工具系统 ===")

class Tool:
    """工具基类"""

    def __init__(self, name: str, description: str, func):
        self.name = name
        self.description = description
        self.func = func

    def invoke(self, **kwargs) -> str:
        return self.func(**kwargs)

# 定义工具函数
def get_weather(city: str) -> str:
    """获取城市天气"""
    weather_data = {
        "北京": "晴天，25°C",
        "上海": "多云，22°C",
        "广州": "小雨，28°C"
    }
    return weather_data.get(city, f"{city}：暂无数据")

def calculator(expression: str) -> str:
    """计算数学表达式"""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {e}"

def search(query: str) -> str:
    """搜索信息"""
    return f"搜索结果：关于「{query}」的相关信息..."

# 创建工具实例
tools = [
    Tool("get_weather", "获取城市天气，参数：city", get_weather),
    Tool("calculator", "计算数学表达式，参数：expression", calculator),
    Tool("search", "搜索互联网信息，参数：query", search),
]

print("可用工具:")
for tool in tools:
    print(f"  - {tool.name}: {tool.description}")

# ===== 3. 模拟 LLM 决策 =====
print("\n=== 3. 模拟 LLM 决策 ===")

class MockAgent:
    """模拟 Agent 决策"""

    def __init__(self, tools: List[Tool]):
        self.tools = {t.name: t for t in tools}

    def decide(self, input_text: str, intermediate_steps: List = None) -> Union[AgentAction, AgentFinish]:
        """根据输入和历史步骤决策"""
        intermediate_steps = intermediate_steps or []

        # 检查是否已经有足够信息
        if intermediate_steps:
            # 简化：如果已经执行过工具，就返回结果
            last_action, last_result = intermediate_steps[-1]
            return AgentFinish(
                return_values={"output": f"根据查询结果：{last_result}，这就是答案。"},
                log="Thought: 我已经有足够信息了\nFinal Answer: ..."
            )

        # 分析输入，决定使用哪个工具
        input_lower = input_text.lower()

        if "天气" in input_text:
            # 提取城市名
            city = "北京"  # 默认
            for c in ["北京", "上海", "广州"]:
                if c in input_text:
                    city = c
                    break
            return AgentAction(
                tool="get_weather",
                tool_input={"city": city},
                log=f"Thought: 用户询问天气，我需要查询{city}的天气\nAction: get_weather"
            )

        if any(op in input_text for op in ["+", "-", "*", "/", "计算", "算"]):
            # 提取数学表达式
            import re
            expr = re.search(r'[\d\+\-\*\/\(\)\s\.]+', input_text)
            if expr:
                return AgentAction(
                    tool="calculator",
                    tool_input={"expression": expr.group().strip()},
                    log="Thought: 用户需要计算\nAction: calculator"
                )

        if "搜索" in input_text or "查找" in input_text or "什么是" in input_text:
            query = input_text.replace("搜索", "").replace("查找", "").replace("什么是", "").strip()
            return AgentAction(
                tool="search",
                tool_input={"query": query or input_text},
                log="Thought: 用户需要搜索信息\nAction: search"
            )

        # 不需要工具，直接回答
        return AgentFinish(
            return_values={"output": f"你好！你说的是：{input_text}"},
            log="Thought: 这个问题我可以直接回答\nFinal Answer: ..."
        )

agent = MockAgent(tools)

# 测试决策
test_inputs = [
    "北京天气怎么样？",
    "计算 25 + 37",
    "搜索 Python 教程",
    "你好"
]

for input_text in test_inputs:
    result = agent.decide(input_text)
    if isinstance(result, AgentAction):
        print(f"'{input_text}' → 调用工具: {result.tool}({result.tool_input})")
    else:
        print(f"'{input_text}' → 直接回答: {result.return_values['output'][:30]}...")

# ===== 4. AgentExecutor 执行器 =====
print("\n=== 4. AgentExecutor 执行器 ===")

class AgentExecutor:
    """Agent 执行器"""

    def __init__(
        self,
        agent: MockAgent,
        tools: List[Tool],
        max_iterations: int = 10,
        verbose: bool = True
    ):
        self.agent = agent
        self.tools = {t.name: t for t in tools}
        self.max_iterations = max_iterations
        self.verbose = verbose

    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """执行 Agent"""
        input_text = input_dict.get("input", "")
        intermediate_steps = []

        if self.verbose:
            print(f"\n输入: {input_text}")
            print("-" * 40)

        for i in range(self.max_iterations):
            if self.verbose:
                print(f"\n迭代 {i + 1}:")

            # 1. Agent 决策
            result = self.agent.decide(input_text, intermediate_steps)

            # 2. 如果是 AgentFinish，返回结果
            if isinstance(result, AgentFinish):
                if self.verbose:
                    print(f"  完成: {result.return_values['output'][:50]}...")
                return result.return_values

            # 3. 执行工具
            action = result
            if self.verbose:
                print(f"  思考: 需要调用 {action.tool}")
                print(f"  参数: {action.tool_input}")

            tool = self.tools.get(action.tool)
            if tool:
                observation = tool.invoke(**action.tool_input)
            else:
                observation = f"未知工具: {action.tool}"

            if self.verbose:
                print(f"  观察: {observation}")

            # 4. 记录步骤
            intermediate_steps.append((action, observation))

        # 达到最大迭代
        return {"output": "达到最大迭代次数，无法完成任务"}

# 创建执行器
executor = AgentExecutor(agent, tools, max_iterations=5, verbose=True)

# 测试执行
print("\n" + "=" * 50)
print("测试1: 天气查询")
result = executor.invoke({"input": "上海天气怎么样？"})
print(f"最终答案: {result['output']}")

print("\n" + "=" * 50)
print("测试2: 数学计算")
result = executor.invoke({"input": "帮我算一下 100 - 37"})
print(f"最终答案: {result['output']}")

# ===== 5. ReAct 格式模拟 =====
print("\n=== 5. ReAct 格式 ===")

def simulate_react(question: str, tools: List[Tool]):
    """模拟 ReAct 执行过程"""
    print(f"Question: {question}")
    print()

    # 模拟 ReAct 循环
    thoughts = [
        ("我需要先分析用户的问题", None, None),
        ("这是一个关于天气的问题，我应该使用天气工具", "get_weather", {"city": "北京"}),
        ("我现在知道答案了", None, None),
    ]

    for i, (thought, action, action_input) in enumerate(thoughts):
        print(f"Thought: {thought}")
        if action:
            print(f"Action: {action}")
            print(f"Action Input: {action_input}")
            # 执行工具
            tool = next((t for t in tools if t.name == action), None)
            if tool:
                result = tool.invoke(**action_input)
                print(f"Observation: {result}")
            print()
        else:
            if i == len(thoughts) - 1:
                print("Final Answer: 北京今天晴天，气温25度，适合出行。")

simulate_react("北京天气怎么样？", tools)

# ===== 6. 多工具协作 =====
print("\n=== 6. 多工具协作 ===")

class MultiStepAgent(MockAgent):
    """支持多步骤的 Agent"""

    def decide(self, input_text: str, intermediate_steps: List = None) -> Union[AgentAction, AgentFinish]:
        intermediate_steps = intermediate_steps or []
        step_count = len(intermediate_steps)

        # 复杂任务：先查天气，再搜索活动
        if "天气" in input_text and "活动" in input_text:
            if step_count == 0:
                # 第一步：查天气
                return AgentAction(
                    tool="get_weather",
                    tool_input={"city": "北京"},
                    log="先查询天气"
                )
            elif step_count == 1:
                # 第二步：搜索活动
                weather = intermediate_steps[0][1]
                if "雨" in weather:
                    query = "室内活动"
                else:
                    query = "户外活动"
                return AgentAction(
                    tool="search",
                    tool_input={"query": f"北京{query}推荐"},
                    log="根据天气搜索活动"
                )
            else:
                # 完成
                weather = intermediate_steps[0][1]
                activities = intermediate_steps[1][1]
                return AgentFinish(
                    return_values={
                        "output": f"北京{weather}。推荐活动：{activities}"
                    }
                )

        # 其他情况使用父类逻辑
        return super().decide(input_text, intermediate_steps)

# 测试多步骤
multi_agent = MultiStepAgent(tools)
multi_executor = AgentExecutor(multi_agent, tools, max_iterations=5, verbose=True)

print("\n" + "=" * 50)
print("测试: 多工具协作")
result = multi_executor.invoke({"input": "北京天气怎么样？推荐什么活动？"})
print(f"最终答案: {result['output']}")

# ===== 7. 错误处理 =====
print("\n=== 7. 错误处理 ===")

class SafeAgentExecutor(AgentExecutor):
    """带错误处理的执行器"""

    def invoke(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return super().invoke(input_dict)
        except Exception as e:
            print(f"执行出错: {e}")
            return {"output": f"抱歉，执行过程中出现错误：{str(e)}"}

# 测试错误处理
safe_executor = SafeAgentExecutor(agent, tools, verbose=False)
result = safe_executor.invoke({"input": "执行一个复杂任务"})
print(f"结果: {result['output']}")

print("\n=== 完成！===")
```

**运行输出示例：**
```
=== 1. 核心数据结构 ===

=== 2. 工具系统 ===
可用工具:
  - get_weather: 获取城市天气，参数：city
  - calculator: 计算数学表达式，参数：expression
  - search: 搜索互联网信息，参数：query

=== 3. 模拟 LLM 决策 ===
'北京天气怎么样？' → 调用工具: get_weather({'city': '北京'})
'计算 25 + 37' → 调用工具: calculator({'expression': '25 + 37'})
'搜索 Python 教程' → 调用工具: search({'query': 'Python 教程'})
'你好' → 直接回答: 你好！你说的是：你好...

=== 4. AgentExecutor 执行器 ===

==================================================
测试1: 天气查询

输入: 上海天气怎么样？
----------------------------------------

迭代 1:
  思考: 需要调用 get_weather
  参数: {'city': '上海'}
  观察: 多云，22°C

迭代 2:
  完成: 根据查询结果：多云，22°C，这就是答案。...
最终答案: 根据查询结果：多云，22°C，这就是答案。

=== 5. ReAct 格式 ===
Question: 北京天气怎么样？

Thought: 我需要先分析用户的问题
Thought: 这是一个关于天气的问题，我应该使用天气工具
Action: get_weather
Action Input: {'city': '北京'}
Observation: 晴天，25°C

Thought: 我现在知道答案了
Final Answer: 北京今天晴天，气温25度，适合出行。

=== 6. 多工具协作 ===
测试: 多工具协作
...
最终答案: 北京晴天，25°C。推荐活动：搜索结果：关于「北京户外活动推荐」的相关信息...

=== 完成！===
```

---

## 8. 【面试必问】

### 问题1："什么是 Agent？它和 Chain 有什么区别？"

**普通回答（❌ 不出彩）：**
"Agent 可以调用工具，Chain 是固定流程。"

**出彩回答（✅ 推荐）：**

> **Agent 是具有自主决策能力的 LLM 应用：**
>
> **核心区别：**
>
> | 维度 | Chain | Agent |
> |-----|-------|-------|
> | 执行路径 | 固定 | 动态 |
> | 决策者 | 开发者 | LLM |
> | 可预测性 | 高 | 低 |
> | 调试难度 | 低 | 高 |
> | LLM 调用次数 | 固定 | 不确定 |
>
> **Agent 的工作原理：**
> 1. LLM 分析任务，决定下一步
> 2. 如果需要，调用工具获取信息
> 3. 根据结果继续决策
> 4. 循环直到任务完成
>
> **ReAct 模式：**
> ```
> Thought → Action → Observation → Thought → ... → Final Answer
> ```
>
> **选择标准：**
> - 流程固定、可预测 → Chain
> - 需要动态决策、工具调用 → Agent
>
> **实际经验**：在智能客服项目中，我用 Agent 处理复杂查询（需要查订单、查物流），用 Chain 处理简单问答。Agent 的灵活性带来的代价是不确定性，需要设置 max_iterations 防止死循环。

**为什么这个回答出彩？**
1. ✅ 清晰的对比表格
2. ✅ 解释了工作原理
3. ✅ 提到了 ReAct 模式
4. ✅ 有实际项目经验

---

### 问题2："如何设计一个可靠的 Agent 系统？"

**普通回答（❌ 不出彩）：**
"设置好工具，限制迭代次数。"

**出彩回答（✅ 推荐）：**

> **设计可靠 Agent 需要多层考虑：**
>
> **1. 工具设计**
> ```python
> # 清晰的描述（最重要！）
> @tool
> def search(query: str) -> str:
>     """在互联网上搜索信息。
>
>     使用场景：查找实时信息、新闻、资料
>     不要用于：常识性问题、历史事实
>
>     Args:
>         query: 简洁明确的搜索关键词
>     """
> ```
>
> **2. 执行控制**
> ```python
> executor = AgentExecutor(
>     agent=agent,
>     tools=tools,
>     max_iterations=10,      # 防止死循环
>     max_execution_time=60,  # 超时保护
>     handle_parsing_errors=True,  # 容错
> )
> ```
>
> **3. 监控与日志**
> ```python
> # 使用 Callback 记录每一步
> class MonitorCallback(BaseCallbackHandler):
>     def on_agent_action(self, action, **kwargs):
>         log.info(f"调用工具: {action.tool}")
> ```
>
> **4. 兜底策略**
> - 设置默认回复
> - 检测循环调用
> - 异常情况人工介入
>
> **5. 成本控制**
> - 工具数量：5-10 个
> - 迭代次数：生产环境 5-10
> - 监控 token 消耗

---

## 9. 【化骨绵掌】

### 卡片1：Agent 是什么？ 🎯

**一句话：** Agent = LLM + 工具 + 决策循环，让 LLM 能够自主决策和执行任务。

**举例：**
```python
# Agent 循环
while 任务未完成:
    决策 = LLM.思考()
    结果 = 执行工具(决策)
    更新状态(结果)
```

**应用：** 智能助手、自动化任务、研究助手。

---

### 卡片2：ReAct 模式 🔄

**一句话：** ReAct = Reasoning + Acting，思考→行动→观察的循环。

**举例：**
```
Thought: 我需要查天气
Action: get_weather
Observation: 晴天25度
Thought: 我可以回答了
Final Answer: 今天晴天
```

**应用：** 大多数 Agent 的核心思维模式。

---

### 卡片3：AgentExecutor 执行器 ⚙️

**一句话：** AgentExecutor 负责循环调用 Agent 和执行工具，直到任务完成。

**举例：**
```python
executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=10
)
result = executor.invoke({"input": "..."})
```

**应用：** 执行和控制 Agent 的核心组件。

---

### 卡片4：Tool 工具定义 🔧

**一句话：** Tool 是 Agent 可调用的能力，通过名称和描述让 LLM 理解用途。

**举例：**
```python
@tool
def search(query: str) -> str:
    """搜索互联网获取信息"""
    return do_search(query)
```

**应用：** 扩展 LLM 能力：搜索、计算、API 调用。

---

### 卡片5：intermediate_steps 中间步骤 📋

**一句话：** 记录 Agent 执行历史，让 LLM 知道之前做了什么。

**举例：**
```python
steps = [
    (AgentAction("search", {"query": "..."}), "结果1"),
    (AgentAction("calc", {"expr": "..."}), "结果2"),
]
```

**应用：** 避免重复操作，保持上下文连贯。

---

### 卡片6：AgentAction vs AgentFinish 🎬

**一句话：** AgentAction 表示要调用工具，AgentFinish 表示任务完成。

**举例：**
```python
# 需要工具
AgentAction(tool="search", tool_input={...})

# 任务完成
AgentFinish(return_values={"output": "答案"})
```

**应用：** Agent 决策的两种结果类型。

---

### 卡片7：max_iterations 限制 ⏱️

**一句话：** 必须设置最大迭代次数，防止 Agent 死循环或成本失控。

**举例：**
```python
executor = AgentExecutor(
    max_iterations=10,      # 最多10轮
    max_execution_time=60   # 最多60秒
)
```

**应用：** 生产环境必备的安全措施。

---

### 卡片8：工具描述很重要 📝

**一句话：** LLM 完全依赖 description 来理解和选择工具。

**举例：**
```python
# ❌ 糟糕
"""搜索"""

# ✅ 优秀
"""搜索互联网获取最新信息。
使用场景：实时新闻、最新资料
不要用于：常识性问题"""
```

**应用：** 好的描述 = 准确的工具选择。

---

### 卡片9：Agent vs Chain 选择 ⚖️

**一句话：** 固定流程用 Chain，需要决策用 Agent。

**举例：**
```python
# Chain：固定3步
chain = prompt | llm | parser

# Agent：动态决策
# 可能1步，可能10步
```

**应用：** 能用 Chain 解决就不用 Agent。

---

### 卡片10：Agent 在 LangChain 源码中的位置 ⭐

**一句话：** Agent 基于 Runnable 协议，核心是 AgentExecutor 的循环执行。

**举例：**
```python
# langchain/agents/agent.py
class AgentExecutor(Runnable):
    def invoke(self, input):
        while not done:
            action = self.agent.invoke(...)
            result = self._execute_tool(action)
```

**应用：** 理解 Agent 执行流程是定制 Agent 的基础。

---

## 10. 【一句话总结】

**Agent 是让 LLM 从被动执行变为主动决策的机制，通过 ReAct 循环（思考→行动→观察）动态选择工具完成任务，是构建智能助手和自动化系统的核心模式。**

---

## 📚 学习检查清单

- [ ] 理解 Agent 与 Chain 的本质区别
- [ ] 掌握 ReAct 模式的思考→行动→观察循环
- [ ] 会使用 @tool 装饰器定义工具
- [ ] 能够创建和配置 AgentExecutor
- [ ] 知道设置 max_iterations 的重要性
- [ ] 理解工具描述对 Agent 决策的影响

## 🔗 下一步学习

- **Memory 记忆系统**：让 Agent 记住对话历史
- **Callback 回调系统**：监控 Agent 执行过程
- **自定义 Agent**：深入 Agent 源码实现

---

**版本：** v1.0
**最后更新：** 2025-01-14
