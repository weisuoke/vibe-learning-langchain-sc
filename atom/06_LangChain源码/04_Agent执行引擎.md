# Agent 执行引擎

> 原子化知识点 | LangChain 源码 | Agent 运行时引擎

---

## 1. 【30字核心】

**Agent 执行引擎是 LangChain 的智能代理运行时，通过"思考-行动-观察"循环让 LLM 自主调用工具完成复杂任务。**

---

## 2. 【第一性原理】

### 什么是第一性原理？

**第一性原理**：回到事物最基本的真理，从源头思考问题

### Agent 执行引擎的第一性原理 🎯

#### 1. 最基础的定义

**Agent = LLM + 工具 + 循环执行**

仅此而已！没有更基础的了。

- **LLM**：思考和决策
- **工具**：执行具体操作
- **循环执行**：重复"思考-行动-观察"直到任务完成

```python
# Agent 的本质
while not done:
    action = llm.think(observation)  # 思考：决定做什么
    observation = tool.execute(action)  # 行动：执行工具
    done = check_if_done(observation)  # 观察：检查是否完成
```

#### 2. 为什么需要 Agent？

**核心问题：如何让 LLM 完成需要多步骤、多工具的复杂任务？**

```python
# 没有 Agent 的困境
user_query = "查询北京天气，如果下雨就提醒我带伞"

# 需要手动编排多个步骤
weather = get_weather("北京")  # 步骤1：调用天气API
if "雨" in weather:
    send_notification("记得带伞")  # 步骤2：发送提醒

# 问题：
# 1. 需要手动编写逻辑
# 2. 无法处理复杂的条件分支
# 3. 难以应对用户的各种问题
```

```python
# 有了 Agent
agent = create_agent(llm, tools=[get_weather, send_notification])

# Agent 自动决定调用什么工具
result = agent.invoke("查询北京天气，如果下雨就提醒我带伞")

# Agent 的思考过程：
# 1. 我需要先查询天气 → 调用 get_weather("北京")
# 2. 天气是"小雨"，需要提醒 → 调用 send_notification("记得带伞")
# 3. 任务完成，返回结果
```

#### 3. Agent 的三层价值

##### 价值1：自主决策 - LLM 决定调用什么工具

```python
# 用户问题多种多样
agent.invoke("北京天气怎么样？")      # → 调用天气工具
agent.invoke("搜索 Python 教程")      # → 调用搜索工具
agent.invoke("计算 123 * 456")        # → 调用计算器工具
agent.invoke("先搜索再总结")          # → 调用多个工具
```

##### 价值2：多步推理 - 自动分解复杂任务

```python
# 复杂任务自动分解
agent.invoke("帮我规划一次北京两日游")
# Agent 自动：
# 1. 搜索北京景点
# 2. 查询天气
# 3. 搜索酒店
# 4. 规划行程
# 5. 生成最终方案
```

##### 价值3：错误恢复 - 失败后自动重试

```python
# 工具调用失败时自动处理
# Agent: 调用天气 API
# 结果: 网络错误
# Agent: 换一个方式，搜索天气信息
# 结果: 搜索成功
```

#### 4. 从第一性原理推导 Agent 架构

**推理链：**

```
1. LLM 单次调用无法完成复杂任务
   ↓
2. 需要多次调用 LLM，每次决定下一步
   ↓
3. 定义"思考-行动-观察"循环
   ↓
4. LLM 输出结构化的 Action（工具名+参数）
   ↓
5. 执行 Action，获得 Observation
   ↓
6. 将 Observation 反馈给 LLM
   ↓
7. 重复直到 LLM 决定完成
   ↓
8. AgentExecutor 封装这个循环
```

#### 5. 一句话总结第一性原理

**Agent 是"LLM + 工具 + 循环"的组合，通过反复"思考-行动-观察"让 LLM 自主完成复杂任务。**

---

## 3. 【核心概念（全面覆盖）】

### 核心概念1：AgentExecutor 执行器 🏗️

**AgentExecutor 是 Agent 的运行时引擎，负责执行循环**

```python
from typing import List, Dict, Any, Optional, Union
from langchain_core.agents import AgentAction, AgentFinish

class AgentExecutor:
    """Agent 执行器：运行 Agent 循环

    核心职责：
    1. 管理"思考-行动-观察"循环
    2. 执行工具调用
    3. 处理错误和超时
    4. 追踪执行历史
    """

    agent: Any                    # Agent（决策者）
    tools: List[BaseTool]         # 可用工具列表
    max_iterations: int = 15      # 最大迭代次数
    max_execution_time: float = None  # 最大执行时间
    early_stopping_method: str = "force"  # 提前停止策略
    handle_parsing_errors: bool = True  # 是否处理解析错误

    def invoke(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """执行 Agent 循环"""
        # 初始化
        intermediate_steps = []  # 中间步骤记录
        iterations = 0

        while True:
            # 1. Agent 思考：决定下一步行动
            output = self.agent.plan(
                intermediate_steps=intermediate_steps,
                **input
            )

            # 2. 检查是否完成
            if isinstance(output, AgentFinish):
                return {"output": output.return_values["output"]}

            # 3. 执行工具
            action: AgentAction = output
            tool_output = self._execute_tool(action)

            # 4. 记录中间步骤
            intermediate_steps.append((action, tool_output))

            # 5. 检查迭代限制
            iterations += 1
            if iterations >= self.max_iterations:
                return self._handle_max_iterations(intermediate_steps)

    def _execute_tool(self, action: AgentAction) -> str:
        """执行单个工具"""
        tool = self._get_tool(action.tool)
        try:
            return tool.run(action.tool_input)
        except Exception as e:
            return f"Error: {str(e)}"

    def _get_tool(self, tool_name: str) -> BaseTool:
        """根据名称获取工具"""
        for tool in self.tools:
            if tool.name == tool_name:
                return tool
        raise ValueError(f"Unknown tool: {tool_name}")
```

**执行流程图：**

```
输入 → Agent.plan() → AgentAction? → 执行工具 → 观察结果 → 循环
                   ↘ AgentFinish → 返回结果
```

---

### 核心概念2：AgentAction 和 AgentFinish 📐

**Agent 的输出是 AgentAction（继续）或 AgentFinish（完成）**

```python
from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class AgentAction:
    """Agent 决定执行的动作

    表示 Agent 想要调用某个工具
    """
    tool: str              # 工具名称
    tool_input: Any        # 工具输入参数
    log: str               # 思考过程日志

@dataclass
class AgentFinish:
    """Agent 决定完成任务

    表示 Agent 认为任务已完成
    """
    return_values: Dict[str, Any]  # 返回值
    log: str                        # 思考过程日志

# 使用示例
# Agent 决定调用工具
action = AgentAction(
    tool="search",
    tool_input="Python tutorial",
    log="I need to search for Python tutorials"
)

# Agent 决定完成
finish = AgentFinish(
    return_values={"output": "Here is the answer..."},
    log="I have enough information to answer"
)
```

---

### 核心概念3：ReAct 模式 🔄

**ReAct (Reasoning + Acting) 是最常用的 Agent 设计模式**

```python
# ReAct 的核心思想：思考 → 行动 → 观察 → 思考 → ...

REACT_PROMPT = """
Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
"""

# ReAct Agent 的输出解析
class ReActOutputParser:
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        # 解析 LLM 输出
        if "Final Answer:" in text:
            return AgentFinish(
                return_values={"output": text.split("Final Answer:")[-1].strip()},
                log=text
            )
        else:
            # 解析 Action 和 Action Input
            action_match = re.search(r"Action: (.*)", text)
            input_match = re.search(r"Action Input: (.*)", text)
            return AgentAction(
                tool=action_match.group(1).strip(),
                tool_input=input_match.group(1).strip(),
                log=text
            )
```

**ReAct 执行示例：**

```
Question: What is the weather in Beijing and should I bring an umbrella?

Thought: I need to check the weather in Beijing first.
Action: get_weather
Action Input: Beijing
Observation: Beijing weather: Light rain, 18°C

Thought: It's raining in Beijing, I should recommend bringing an umbrella.
Final Answer: The weather in Beijing is light rain at 18°C. Yes, you should bring an umbrella.
```

---

### 核心概念4：Tool 工具定义 🔧

**Tool 是 Agent 可以调用的外部能力**

```python
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field

# 方式1：使用 @tool 装饰器
@tool
def search(query: str) -> str:
    """Search for information on the web.

    Args:
        query: The search query string
    """
    return f"Search results for: {query}"

@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression.

    Args:
        expression: The math expression to evaluate
    """
    return str(eval(expression))

# 方式2：继承 BaseTool
class WeatherTool(BaseTool):
    name: str = "get_weather"
    description: str = "Get the current weather for a city"

    def _run(self, city: str) -> str:
        """同步执行"""
        return f"Weather in {city}: Sunny, 25°C"

    async def _arun(self, city: str) -> str:
        """异步执行"""
        return self._run(city)

# 方式3：使用 StructuredTool（带 schema）
class SearchInput(BaseModel):
    query: str = Field(description="Search query")
    max_results: int = Field(default=10, description="Maximum results")

from langchain_core.tools import StructuredTool

structured_search = StructuredTool.from_function(
    func=lambda query, max_results: f"Results for {query}",
    name="search",
    description="Search the web",
    args_schema=SearchInput,
)
```

**Tool 的核心属性：**

| 属性 | 说明 |
|-----|------|
| name | 工具名称（Agent 用于调用） |
| description | 工具描述（LLM 用于理解） |
| args_schema | 参数 schema（Pydantic 模型） |
| return_direct | 是否直接返回（跳过后续思考） |

---

### 核心概念5：create_react_agent 新版 API 🆕

**LangChain 新版使用 create_react_agent 创建 Agent**

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain import hub

# 1. 准备组件
llm = ChatOpenAI(model="gpt-4")
tools = [search, calculator, get_weather]

# 2. 获取 prompt（从 LangChain Hub）
prompt = hub.pull("hwchase17/react")

# 3. 创建 Agent
agent = create_react_agent(llm, tools, prompt)

# 4. 创建执行器
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=10,
)

# 5. 执行
result = agent_executor.invoke({"input": "What is 25 * 4?"})
print(result["output"])
```

**新版 API 的优势：**
- 更模块化的设计
- 更好的类型提示
- 更灵活的 prompt 定制

---

### 核心概念6：Tool Calling Agent（推荐）🌟

**基于 LLM 原生 Tool Calling 的 Agent，更可靠**

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate

# 1. 定义 prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# 2. 创建 Agent（使用 LLM 的 tool calling）
agent = create_tool_calling_agent(llm, tools, prompt)

# 3. 创建执行器
agent_executor = AgentExecutor(agent=agent, tools=tools)

# 4. 执行
result = agent_executor.invoke({"input": "What's the weather in Beijing?"})
```

**Tool Calling vs ReAct：**

| 特性 | Tool Calling | ReAct |
|-----|-------------|-------|
| 可靠性 | 高（LLM 原生支持） | 中（依赖文本解析） |
| 支持模型 | OpenAI, Anthropic 等 | 所有模型 |
| 错误率 | 低 | 较高 |
| 推荐场景 | 生产环境 | 学习/实验 |

---

### 核心概念7：Agent 中间步骤 📝

**intermediate_steps 记录 Agent 的执行历史**

```python
# intermediate_steps 的结构
intermediate_steps: List[Tuple[AgentAction, str]] = [
    (
        AgentAction(tool="search", tool_input="Python", log="..."),
        "Search results: Python is a programming language..."
    ),
    (
        AgentAction(tool="calculator", tool_input="2+2", log="..."),
        "4"
    ),
]

# Agent 使用 intermediate_steps 进行下一步决策
class Agent:
    def plan(
        self,
        intermediate_steps: List[Tuple[AgentAction, str]],
        **kwargs
    ) -> Union[AgentAction, AgentFinish]:
        # 将历史转换为文本
        scratchpad = self._format_scratchpad(intermediate_steps)

        # 让 LLM 基于历史进行决策
        response = self.llm.invoke(
            self.prompt.format(
                input=kwargs["input"],
                agent_scratchpad=scratchpad,
            )
        )

        return self.output_parser.parse(response)
```

---

## 4. 【最小可用】

掌握以下内容，就能使用 Agent：

### 4.1 定义工具

```python
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

@tool
def calculator(expression: str) -> str:
    """Calculate a math expression."""
    return str(eval(expression))

tools = [search, calculator]
```

### 4.2 创建 Agent

```python
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)
```

### 4.3 执行 Agent

```python
result = agent_executor.invoke({"input": "What is 25 * 4?"})
print(result["output"])  # "25 * 4 = 100"
```

**这些知识足以：**
- 创建能调用工具的 Agent
- 处理需要多步骤的复杂任务
- 在生产环境中使用 Agent

---

## 5. 【1个类比】（双轨制）

### 类比1：Agent 执行循环

#### 🎨 前端视角：Redux + Saga

```typescript
// Redux Saga：监听 action，执行副作用，dispatch 新 action
function* watchUserRequest() {
  while (true) {
    const action = yield take('USER_REQUEST');  // 等待 action
    const result = yield call(api.fetch, action.payload);  // 执行副作用
    yield put({ type: 'USER_SUCCESS', result });  // dispatch 结果
  }
}
```

```python
# Agent：思考 action，执行工具，观察结果
while not done:
    action = agent.think(observation)  # 思考
    observation = tool.execute(action)  # 执行
    done = check_if_done(observation)  # 检查
```

#### 🧒 小朋友视角：侦探破案

```
Agent 就像一个侦探在破案：

侦探接到任务："找出谁偷了蛋糕"

循环开始：
1. 思考：我应该先问问厨房的人 → 决定行动
2. 行动：询问厨师 → 执行
3. 观察：厨师说看到小明进过厨房 → 得到线索

继续循环：
4. 思考：我应该去问小明 → 决定行动
5. 行动：询问小明 → 执行
6. 观察：小明承认了 → 破案！

侦探（Agent）不断 思考→行动→观察，直到破案（完成任务）
```

---

### 类比2：Tool 工具

#### 🎨 前端视角：API 接口

```typescript
// 前端调用各种 API
const weather = await fetch('/api/weather?city=Beijing');
const search = await fetch('/api/search?q=Python');
const calc = await fetch('/api/calc?expr=2+2');
```

#### 🧒 小朋友视角：百宝箱

```
Tool 就像多啦A梦的百宝袋：

Agent（大雄）遇到问题时：
- 需要知道天气 → 拿出"天气预报机"
- 需要搜索信息 → 拿出"搜索眼镜"
- 需要计算数学 → 拿出"计算器"

每个道具（Tool）都有：
- 名字：天气预报机
- 说明：可以查询任何城市的天气
- 使用方法：说出城市名字
```

---

### 类比总结表

| Agent 概念 | 前端类比 | 小朋友类比 |
|-----------|---------|-----------|
| Agent | Redux + Saga | 侦探 |
| AgentExecutor | Saga middleware | 侦探的工作流程 |
| AgentAction | dispatch action | 决定下一步 |
| AgentFinish | 完成状态 | 破案 |
| Tool | API 接口 | 百宝袋里的道具 |
| intermediate_steps | action 历史 | 调查笔记 |
| ReAct | 状态机 | 思考-行动-观察 |

---

## 6. 【反直觉点】

### 误区1：Agent 每次都能成功 ❌

**为什么错？**
- Agent 可能陷入循环
- 工具调用可能失败
- LLM 可能产生错误的 action

**正确理解：**
```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=10,          # 限制迭代次数
    handle_parsing_errors=True, # 处理解析错误
    early_stopping_method="generate",  # 提前停止策略
)
```

---

### 误区2：Tool 描述不重要 ❌

**为什么错？**
- LLM 根据描述决定是否调用工具
- 描述不清晰会导致错误调用

**正确理解：**
```python
# ❌ 差的描述
@tool
def search(query: str) -> str:
    """Search."""
    ...

# ✅ 好的描述
@tool
def search(query: str) -> str:
    """Search the web for current information.

    Use this tool when you need to find recent news, facts,
    or information that may not be in your training data.

    Args:
        query: The search query, be specific for better results
    """
    ...
```

---

### 误区3：ReAct 和 Tool Calling 效果相同 ❌

**为什么错？**
- Tool Calling 使用 LLM 原生能力，更可靠
- ReAct 依赖文本解析，容易出错

**正确理解：**
```python
# 生产环境推荐 Tool Calling
agent = create_tool_calling_agent(llm, tools, prompt)

# ReAct 适合学习和不支持 tool calling 的模型
agent = create_react_agent(llm, tools, prompt)
```

---

## 7. 【实战代码】

```python
"""
示例：实现简化版 Agent 执行引擎
演示 Agent 的核心工作原理
"""

from typing import List, Tuple, Union, Any, Dict
from dataclasses import dataclass
import re

# ===== 1. 数据结构 =====
print("=== 1. 定义数据结构 ===")

@dataclass
class AgentAction:
    tool: str
    tool_input: str
    log: str

@dataclass
class AgentFinish:
    return_values: Dict[str, Any]
    log: str

# ===== 2. 工具定义 =====
print("\n=== 2. 定义工具 ===")

class Tool:
    def __init__(self, name: str, description: str, func):
        self.name = name
        self.description = description
        self.func = func

    def run(self, input: str) -> str:
        return self.func(input)

# 定义工具
def search_func(query: str) -> str:
    return f"Search results for '{query}': Python is a programming language."

def calculator_func(expr: str) -> str:
    try:
        return str(eval(expr))
    except:
        return "Error: Invalid expression"

def weather_func(city: str) -> str:
    return f"Weather in {city}: Sunny, 25°C"

tools = [
    Tool("search", "Search for information", search_func),
    Tool("calculator", "Calculate math expressions", calculator_func),
    Tool("weather", "Get weather for a city", weather_func),
]

# ===== 3. 模拟 Agent（简化版 ReAct）=====
print("\n=== 3. 定义 Agent ===")

class SimpleReActAgent:
    def __init__(self, tools: List[Tool]):
        self.tools = tools
        self.tool_names = [t.name for t in tools]

    def plan(
        self,
        input: str,
        intermediate_steps: List[Tuple[AgentAction, str]]
    ) -> Union[AgentAction, AgentFinish]:
        """模拟 LLM 思考过程"""

        # 构建 scratchpad
        scratchpad = ""
        for action, observation in intermediate_steps:
            scratchpad += f"\nAction: {action.tool}\n"
            scratchpad += f"Action Input: {action.tool_input}\n"
            scratchpad += f"Observation: {observation}\n"

        # 模拟 LLM 决策（实际应该调用 LLM）
        if "weather" in input.lower() and "weather" not in scratchpad:
            city = "Beijing" if "beijing" in input.lower() else "Unknown"
            return AgentAction(
                tool="weather",
                tool_input=city,
                log=f"I should check the weather for {city}"
            )
        elif "calculate" in input.lower() or any(c in input for c in "+-*/"):
            # 提取数学表达式
            expr = re.search(r'[\d\+\-\*\/\s\(\)]+', input)
            if expr:
                return AgentAction(
                    tool="calculator",
                    tool_input=expr.group().strip(),
                    log="I need to calculate this expression"
                )
        elif "search" in input.lower() and "search" not in scratchpad:
            query = input.replace("search", "").strip()
            return AgentAction(
                tool="search",
                tool_input=query,
                log=f"I should search for: {query}"
            )

        # 如果已经有足够信息，完成任务
        if intermediate_steps:
            last_observation = intermediate_steps[-1][1]
            return AgentFinish(
                return_values={"output": f"Based on my research: {last_observation}"},
                log="I have enough information to answer"
            )

        # 默认完成
        return AgentFinish(
            return_values={"output": f"I cannot help with: {input}"},
            log="I don't know how to handle this"
        )

# ===== 4. AgentExecutor =====
print("\n=== 4. 定义 AgentExecutor ===")

class AgentExecutor:
    def __init__(
        self,
        agent: SimpleReActAgent,
        tools: List[Tool],
        max_iterations: int = 5,
        verbose: bool = True
    ):
        self.agent = agent
        self.tools = {t.name: t for t in tools}
        self.max_iterations = max_iterations
        self.verbose = verbose

    def invoke(self, input: Dict[str, str]) -> Dict[str, Any]:
        """执行 Agent 循环"""
        query = input["input"]
        intermediate_steps = []
        iterations = 0

        if self.verbose:
            print(f"\n> Query: {query}")

        while iterations < self.max_iterations:
            # 1. Agent 决策
            output = self.agent.plan(query, intermediate_steps)

            # 2. 检查是否完成
            if isinstance(output, AgentFinish):
                if self.verbose:
                    print(f"\n> Final Answer: {output.return_values['output']}")
                return output.return_values

            # 3. 执行工具
            action = output
            if self.verbose:
                print(f"\n> Thought: {action.log}")
                print(f"> Action: {action.tool}")
                print(f"> Action Input: {action.tool_input}")

            tool = self.tools.get(action.tool)
            if tool:
                observation = tool.run(action.tool_input)
            else:
                observation = f"Error: Unknown tool {action.tool}"

            if self.verbose:
                print(f"> Observation: {observation}")

            # 4. 记录步骤
            intermediate_steps.append((action, observation))
            iterations += 1

        return {"output": "Max iterations reached"}

# ===== 5. 执行示例 =====
print("\n=== 5. 执行示例 ===")

agent = SimpleReActAgent(tools)
executor = AgentExecutor(agent, tools, verbose=True)

# 测试1：天气查询
print("\n" + "="*50)
result = executor.invoke({"input": "What's the weather in Beijing?"})

# 测试2：计算
print("\n" + "="*50)
result = executor.invoke({"input": "Calculate 25 * 4"})

# 测试3：搜索
print("\n" + "="*50)
result = executor.invoke({"input": "Search for Python programming"})

print("\n=== 完成 ===")
```

---

## 8. 【面试必问】

### 问题："LangChain 的 Agent 是如何工作的？"

**普通回答（❌ 不出彩）：**
"Agent 可以调用工具，通过循环执行任务。"

**出彩回答（✅ 推荐）：**

> **Agent 的工作原理有三个层面：**
>
> 1. **核心循环**：思考 → 行动 → 观察 → 重复
>    - 思考：LLM 决定下一步做什么
>    - 行动：执行选定的工具
>    - 观察：将结果反馈给 LLM
>
> 2. **关键组件**：
>    - AgentExecutor：执行循环的引擎
>    - Agent：决策者（通常是 LLM + Prompt）
>    - Tool：可调用的外部能力
>
> 3. **两种主流模式**：
>    - ReAct：通过 Prompt 格式化输出
>    - Tool Calling：使用 LLM 原生的工具调用能力（推荐）
>
> **生产建议**：使用 `create_tool_calling_agent`，更可靠。

---

## 9. 【化骨绵掌】

### 卡片1：Agent 是什么 🎯

**一句话：** Agent 是能自主调用工具的 LLM 应用。

**公式：** Agent = LLM + 工具 + 循环执行

**应用：** 复杂任务自动分解执行。

---

### 卡片2：AgentExecutor 📐

**一句话：** AgentExecutor 是 Agent 的运行时引擎。

**职责：**
- 管理执行循环
- 执行工具调用
- 处理错误和超时

**应用：** `executor.invoke({"input": "..."})`

---

### 卡片3：Tool 工具 🔧

**一句话：** Tool 是 Agent 可调用的外部能力。

**定义方式：**
```python
@tool
def search(query: str) -> str:
    """Search the web."""
    return results
```

**应用：** 搜索、计算、API 调用等。

---

### 卡片4：ReAct 模式 🔄

**一句话：** ReAct 是"思考-行动-观察"的循环模式。

**格式：**
```
Thought: 我应该搜索
Action: search
Action Input: Python
Observation: Python is...
```

**应用：** 经典 Agent 设计模式。

---

### 卡片5：Tool Calling 🌟

**一句话：** Tool Calling 使用 LLM 原生工具调用能力。

**优势：**
- 更可靠
- 错误率低
- 生产推荐

**应用：** `create_tool_calling_agent(llm, tools, prompt)`

---

### 卡片6：AgentAction vs AgentFinish ⚡

**一句话：** Agent 输出 Action（继续）或 Finish（完成）。

```python
AgentAction(tool="search", tool_input="query")  # 继续执行
AgentFinish(return_values={"output": "答案"})   # 任务完成
```

**应用：** 控制 Agent 循环。

---

### 卡片7：intermediate_steps 📝

**一句话：** 记录 Agent 的执行历史。

**结构：**
```python
[(AgentAction, observation), ...]
```

**应用：** LLM 基于历史决策下一步。

---

### 卡片8：max_iterations 限制 🛑

**一句话：** 防止 Agent 无限循环。

**设置：**
```python
AgentExecutor(agent, tools, max_iterations=10)
```

**应用：** 生产环境必须设置。

---

### 卡片9：Tool 描述的重要性 📋

**一句话：** Tool 描述决定 LLM 是否正确调用。

**好的描述：**
- 清晰说明用途
- 说明何时使用
- 参数说明完整

**应用：** 提高 Agent 准确率。

---

### 卡片10：创建 Agent 最佳实践 ⭐

**推荐代码：**
```python
from langchain.agents import create_tool_calling_agent, AgentExecutor

agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, max_iterations=10)
result = executor.invoke({"input": query})
```

**应用：** 生产环境标准写法。

---

## 10. 【一句话总结】

**Agent 执行引擎通过"思考-行动-观察"循环让 LLM 自主调用工具，AgentExecutor 管理执行过程，推荐使用 Tool Calling 模式构建可靠的智能代理。**

---

## 📚 学习检查清单

- [ ] 理解 Agent 的"思考-行动-观察"循环
- [ ] 会使用 @tool 定义工具
- [ ] 会创建 AgentExecutor
- [ ] 理解 AgentAction 和 AgentFinish
- [ ] 了解 ReAct 和 Tool Calling 的区别
- [ ] 会设置 max_iterations 等安全参数
- [ ] 能编写清晰的 Tool 描述

## 🔗 下一步学习

- **序列化与加载机制**：保存和加载 Agent
- **Callback 回调系统**：追踪 Agent 执行过程
- **LangGraph**：更复杂的 Agent 编排

---

**版本：** v1.0
**最后更新：** 2025-12-12
