# OOP 面向对象编程

> 原子化知识点 | Python基础 | LangChain 源码学习前置知识

---

## 1. 【30字核心】

**OOP 是通过类和对象组织代码的编程范式，封装、继承、多态是 LangChain 源码架构的基石。**

---

## 2. 【第一性原理】

### 什么是第一性原理？

**第一性原理**：回到事物最基本的真理，从源头思考问题

### OOP 的第一性原理 🎯

#### 1. 最基础的定义

**OOP = 数据 + 行为 的封装单元**

仅此而已！没有更基础的了。

- **数据**：对象的状态（属性）
- **行为**：对象能做的事（方法）
- **封装单元**：把数据和行为绑定在一起，形成独立的"对象"

#### 2. 为什么需要 OOP？

**核心问题：如何组织越来越复杂的代码？**

```python
# 过程式编程：数据和函数分离
user_name = "Alice"
user_age = 25
user_messages = []

def add_message(messages, content):
    messages.append({"content": content, "time": time.time()})

def get_user_info(name, age):
    return f"{name}, {age}岁"

# 问题：
# 1. user_name 和 add_message 有什么关系？看不出来
# 2. 如果有100个用户怎么办？100组变量？
# 3. 如何保证 user_age 不会被设成负数？
```

```python
# OOP：数据和行为封装在一起
class User:
    def __init__(self, name: str, age: int):
        self.name = name
        self._age = age  # 受保护的属性
        self.messages = []

    @property
    def age(self) -> int:
        return self._age

    @age.setter
    def age(self, value: int):
        if value < 0:
            raise ValueError("年龄不能为负数")
        self._age = value

    def add_message(self, content: str):
        self.messages.append({"content": content, "time": time.time()})

    def get_info(self) -> str:
        return f"{self.name}, {self.age}岁"

# 优势：
# 1. 数据和行为的关系一目了然
# 2. 创建100个用户？users = [User(...) for _ in range(100)]
# 3. age 的合法性由 setter 保证
```

#### 3. OOP 的三层价值

##### 价值1：封装 - 隐藏复杂性

```python
# 不需要知道内部如何实现，只需要知道怎么用
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
response = llm.invoke("Hello")  # 内部的 API 调用、重试、Token 计算都被封装了
```

##### 价值2：继承 - 复用与扩展

```python
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

# BaseMessage 定义了所有消息的通用结构
# HumanMessage、AIMessage 继承并扩展
class CustomMessage(BaseMessage):
    """自定义消息类型"""
    type: str = "custom"
    priority: int = 0
```

##### 价值3：多态 - 统一接口，不同实现

```python
from langchain_core.runnables import Runnable

# 所有实现 Runnable 协议的对象都可以用 invoke() 调用
def process(runnable: Runnable, input_data):
    return runnable.invoke(input_data)

# 可以传入 ChatModel、Chain、Retriever... 都行！
process(chat_model, "Hello")
process(chain, {"query": "Hello"})
process(retriever, "search query")
```

#### 4. 从第一性原理推导 LangChain 源码应用

**推理链：**

```
1. LLM 应用需要处理多种组件（模型、提示、解析器、检索器...）
   ↓
2. 每种组件都有数据（配置）和行为（执行）
   ↓
3. 需要一种方式统一组织这些组件
   ↓
4. OOP 的类/对象模型完美匹配这个需求
   ↓
5. 定义抽象基类（如 Runnable）作为统一接口
   ↓
6. 具体组件继承基类，实现具体行为
   ↓
7. 通过多态，所有组件可以用相同方式调用
   ↓
8. LCEL 的管道操作符 `|` 就是基于 Runnable 的 OOP 设计
```

#### 5. 一句话总结第一性原理

**OOP 是将"数据+行为"封装为对象的编程范式，通过继承实现代码复用，通过多态实现统一接口，是 LangChain 组件化架构的基础。**

---

## 3. 【核心概念（全面覆盖）】

### 核心概念1：类与对象 🏗️

**类是对象的蓝图，对象是类的实例**

```python
from typing import Optional, List
from datetime import datetime

class ChatMessage:
    """聊天消息类 - 类似 LangChain 的 BaseMessage"""

    # 类属性：所有实例共享
    message_count: int = 0

    def __init__(self, role: str, content: str, name: Optional[str] = None):
        """初始化方法：创建对象时自动调用"""
        # 实例属性：每个实例独有
        self.role = role
        self.content = content
        self.name = name
        self.timestamp = datetime.now()

        # 修改类属性
        ChatMessage.message_count += 1

    def to_dict(self) -> dict:
        """实例方法：操作实例数据"""
        return {
            "role": self.role,
            "content": self.content,
            "name": self.name,
            "timestamp": self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ChatMessage":
        """类方法：操作类本身，常用于工厂模式"""
        return cls(
            role=data["role"],
            content=data["content"],
            name=data.get("name")
        )

    @staticmethod
    def validate_role(role: str) -> bool:
        """静态方法：与类相关但不需要访问类或实例"""
        return role in ["system", "user", "assistant"]

# 创建对象（实例化）
msg1 = ChatMessage(role="user", content="Hello!")
msg2 = ChatMessage(role="assistant", content="Hi there!")

print(f"消息1: {msg1.to_dict()}")
print(f"消息总数: {ChatMessage.message_count}")  # 2

# 从字典创建（工厂方法）
msg3 = ChatMessage.from_dict({"role": "system", "content": "You are helpful"})
```

**类 vs 对象 对比：**

| 概念 | 类 (Class) | 对象 (Object/Instance) |
|------|-----------|----------------------|
| 定义 | 蓝图/模板 | 具体的实例 |
| 内存 | 只有一份 | 每个实例独立 |
| 属性 | 类属性（共享） | 实例属性（独有） |
| 创建 | `class MyClass:` | `obj = MyClass()` |
| LangChain 例子 | `BaseMessage` 类 | `HumanMessage("Hi")` 实例 |

**在 LangChain 源码中的应用：**

```python
# langchain_core/messages/base.py 简化版
class BaseMessage:
    """所有消息的基类"""
    content: str
    type: str

    def __init__(self, content: str, **kwargs):
        self.content = content

# 具体消息类型
class HumanMessage(BaseMessage):
    type: str = "human"

class AIMessage(BaseMessage):
    type: str = "ai"
```

---

### 核心概念2：继承与多态 📐

**继承是 "is-a" 关系，多态是 "同一接口，不同实现"**

```python
from abc import ABC, abstractmethod
from typing import Any, Optional

# ===== 继承：定义基类和子类 =====

class BaseRunnable(ABC):
    """可运行组件的抽象基类 - 类似 LangChain 的 Runnable"""

    name: str = "base"

    @abstractmethod
    def invoke(self, input: Any) -> Any:
        """抽象方法：子类必须实现"""
        pass

    def batch(self, inputs: list) -> list:
        """非抽象方法：子类可以继承或重写"""
        return [self.invoke(x) for x in inputs]

class TextProcessor(BaseRunnable):
    """文本处理器 - 继承 BaseRunnable"""

    name: str = "text_processor"

    def __init__(self, uppercase: bool = False):
        self.uppercase = uppercase

    def invoke(self, input: str) -> str:
        """实现抽象方法"""
        result = input.strip()
        if self.uppercase:
            result = result.upper()
        return result

class NumberDoubler(BaseRunnable):
    """数字加倍器 - 另一个子类"""

    name: str = "number_doubler"

    def invoke(self, input: int) -> int:
        return input * 2

    def batch(self, inputs: list) -> list:
        """重写父类方法：优化批处理"""
        return [x * 2 for x in inputs]  # 更高效的实现

# ===== 多态：同一接口，不同行为 =====

def run_all(runnables: list[BaseRunnable], inputs: list) -> list:
    """多态的威力：不关心具体类型，只关心接口"""
    results = []
    for runnable, inp in zip(runnables, inputs):
        results.append(runnable.invoke(inp))  # 调用同一个方法
    return results

# 使用
processor = TextProcessor(uppercase=True)
doubler = NumberDoubler()

print(processor.invoke("  hello  "))  # "HELLO"
print(doubler.invoke(5))              # 10

# 多态调用
runnables = [processor, doubler, processor]
inputs = ["world", 3, "python"]
print(run_all(runnables, inputs))  # ["WORLD", 6, "PYTHON"]
```

**继承类型：**

| 类型 | 说明 | 示例 |
|------|------|------|
| 单继承 | 一个子类继承一个父类 | `class Dog(Animal)` |
| 多继承 | 一个子类继承多个父类 | `class Dog(Animal, Pet)` |
| 多层继承 | A → B → C | `BaseMessage → HumanMessage → CustomHuman` |

**方法解析顺序 (MRO)：**

```python
class A:
    def greet(self):
        return "A"

class B(A):
    def greet(self):
        return "B"

class C(A):
    def greet(self):
        return "C"

class D(B, C):  # 多继承
    pass

# MRO: D → B → C → A → object
print(D.__mro__)
# (<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>)

d = D()
print(d.greet())  # "B" - 按 MRO 顺序找到第一个实现
```

**在 LangChain 源码中的应用：**

```python
# langchain_core/runnables/base.py 简化版
class Runnable(ABC, Generic[Input, Output]):
    """LangChain 最核心的抽象基类"""

    @abstractmethod
    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        pass

    def batch(self, inputs: List[Input]) -> List[Output]:
        return [self.invoke(x) for x in inputs]

    def __or__(self, other: "Runnable") -> "RunnableSequence":
        """重载 | 操作符，实现 LCEL 管道"""
        return RunnableSequence(first=self, last=other)

# 具体实现
class ChatOpenAI(Runnable):
    def invoke(self, input, config=None):
        # 调用 OpenAI API
        pass

class PromptTemplate(Runnable):
    def invoke(self, input, config=None):
        # 格式化模板
        pass
```

---

### 核心概念3：封装与抽象 🔧

**封装是隐藏实现细节，抽象是定义接口**

```python
from abc import ABC, abstractmethod
from typing import Optional

class BaseLLM(ABC):
    """LLM 抽象基类 - 定义接口，隐藏实现"""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self._api_key = api_key  # 私有属性：以 _ 开头
        self._model = model
        self._call_count = 0

    # ===== 封装：控制属性访问 =====

    @property
    def model(self) -> str:
        """只读属性"""
        return self._model

    @property
    def call_count(self) -> int:
        """只读统计"""
        return self._call_count

    # ===== 抽象：定义必须实现的接口 =====

    @abstractmethod
    def _call_api(self, prompt: str) -> str:
        """抽象方法：子类必须实现具体的 API 调用"""
        pass

    # ===== 模板方法：定义算法骨架 =====

    def invoke(self, prompt: str) -> str:
        """公开方法：封装了完整的调用流程"""
        # 1. 预处理
        processed_prompt = self._preprocess(prompt)

        # 2. 调用 API（由子类实现）
        response = self._call_api(processed_prompt)

        # 3. 后处理
        result = self._postprocess(response)

        # 4. 更新统计
        self._call_count += 1

        return result

    def _preprocess(self, prompt: str) -> str:
        """可被子类重写的钩子方法"""
        return prompt.strip()

    def _postprocess(self, response: str) -> str:
        """可被子类重写的钩子方法"""
        return response

class OpenAILLM(BaseLLM):
    """OpenAI 实现"""

    def _call_api(self, prompt: str) -> str:
        # 实际会调用 OpenAI API
        return f"OpenAI response to: {prompt}"

class AnthropicLLM(BaseLLM):
    """Anthropic 实现"""

    def _call_api(self, prompt: str) -> str:
        # 实际会调用 Anthropic API
        return f"Claude response to: {prompt}"

    def _preprocess(self, prompt: str) -> str:
        """重写预处理：添加系统提示"""
        return f"Human: {prompt}\n\nAssistant:"

# 使用：用户不需要知道内部实现
openai_llm = OpenAILLM(api_key="sk-xxx")
anthropic_llm = AnthropicLLM(api_key="sk-ant-xxx")

print(openai_llm.invoke("Hello"))
print(anthropic_llm.invoke("Hello"))
print(f"OpenAI 调用次数: {openai_llm.call_count}")
```

**封装级别：**

| 命名约定 | 含义 | 访问性 |
|---------|------|-------|
| `name` | 公开 | 任何地方都可以访问 |
| `_name` | 受保护 | 约定：仅内部和子类使用 |
| `__name` | 私有 | 名称改写，外部难以访问 |
| `__name__` | 魔术方法 | Python 特殊方法 |

```python
class Example:
    def __init__(self):
        self.public = "公开"
        self._protected = "受保护"
        self.__private = "私有"

obj = Example()
print(obj.public)      # ✅ "公开"
print(obj._protected)  # ⚠️ "受保护" - 可访问但不建议
# print(obj.__private) # ❌ AttributeError
print(obj._Example__private)  # ⚠️ "私有" - 名称改写后可访问
```

**在 LangChain 源码中的应用：**

```python
# langchain_core/language_models/chat_models.py 简化版
class BaseChatModel(ABC):
    """聊天模型抽象基类"""

    @abstractmethod
    def _generate(self, messages: List[BaseMessage]) -> ChatResult:
        """抽象方法：子类实现具体生成逻辑"""
        pass

    def invoke(self, input: LanguageModelInput) -> BaseMessage:
        """公开接口：封装了消息转换、生成、结果处理"""
        messages = self._convert_input(input)
        result = self._generate(messages)
        return result.generations[0].message

    def _convert_input(self, input: LanguageModelInput) -> List[BaseMessage]:
        """内部方法：输入转换"""
        # 处理字符串、消息列表等不同输入格式
        pass
```

---

### 扩展概念4：魔术方法 (Dunder Methods) ✨

```python
class Vector:
    """向量类 - 展示魔术方法"""

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __repr__(self) -> str:
        """开发者友好的字符串表示"""
        return f"Vector({self.x}, {self.y})"

    def __str__(self) -> str:
        """用户友好的字符串表示"""
        return f"({self.x}, {self.y})"

    def __eq__(self, other: "Vector") -> bool:
        """相等比较"""
        return self.x == other.x and self.y == other.y

    def __add__(self, other: "Vector") -> "Vector":
        """加法运算符重载"""
        return Vector(self.x + other.x, self.y + other.y)

    def __len__(self) -> int:
        """长度"""
        return 2

    def __getitem__(self, index: int) -> float:
        """索引访问"""
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        raise IndexError("Vector index out of range")

v1 = Vector(1, 2)
v2 = Vector(3, 4)

print(repr(v1))      # Vector(1, 2)
print(str(v1))       # (1, 2)
print(v1 == v2)      # False
print(v1 + v2)       # (4, 6)
print(len(v1))       # 2
print(v1[0])         # 1
```

**常用魔术方法速查：**

| 魔术方法 | 触发场景 | 示例 |
|---------|---------|------|
| `__init__` | 初始化对象 | `obj = MyClass()` |
| `__str__` | `str(obj)`, `print(obj)` | 用户友好输出 |
| `__repr__` | `repr(obj)`, 交互式输出 | 开发者调试 |
| `__eq__` | `obj1 == obj2` | 相等比较 |
| `__hash__` | `hash(obj)`, 字典键 | 哈希值 |
| `__len__` | `len(obj)` | 长度 |
| `__getitem__` | `obj[key]` | 索引/键访问 |
| `__setitem__` | `obj[key] = value` | 索引/键赋值 |
| `__iter__` | `for x in obj` | 迭代 |
| `__call__` | `obj()` | 对象当函数调用 |
| `__or__` | `obj1 \| obj2` | LCEL 管道！ |

**在 LangChain 源码中的应用：**

```python
# Runnable 的 __or__ 实现 LCEL 管道
class Runnable:
    def __or__(self, other: "Runnable") -> "RunnableSequence":
        return RunnableSequence(first=self, last=other)

    def __ror__(self, other) -> "RunnableSequence":
        # 处理左操作数不是 Runnable 的情况
        return RunnableSequence(first=coerce_to_runnable(other), last=self)

# 使用
chain = prompt | llm | parser  # 等价于多次 __or__ 调用
```

---

### 扩展概念5：@property 装饰器 🎯

```python
class Temperature:
    """温度类 - 展示 @property 的用法"""

    def __init__(self, celsius: float = 0):
        self._celsius = celsius  # 内部存储摄氏度

    @property
    def celsius(self) -> float:
        """只读属性：摄氏度"""
        return self._celsius

    @celsius.setter
    def celsius(self, value: float):
        """可写属性：带验证"""
        if value < -273.15:
            raise ValueError("温度不能低于绝对零度")
        self._celsius = value

    @property
    def fahrenheit(self) -> float:
        """计算属性：华氏度"""
        return self._celsius * 9/5 + 32

    @fahrenheit.setter
    def fahrenheit(self, value: float):
        """通过华氏度设置"""
        self.celsius = (value - 32) * 5/9  # 会触发 celsius setter 的验证

    @property
    def kelvin(self) -> float:
        """只读计算属性：开尔文"""
        return self._celsius + 273.15

# 使用
temp = Temperature(25)
print(f"摄氏: {temp.celsius}°C")      # 25°C
print(f"华氏: {temp.fahrenheit}°F")   # 77°F
print(f"开尔文: {temp.kelvin}K")      # 298.15K

temp.fahrenheit = 100  # 通过华氏度设置
print(f"摄氏: {temp.celsius}°C")      # 37.78°C

try:
    temp.celsius = -300  # 触发验证
except ValueError as e:
    print(f"错误: {e}")
```

**@property vs 普通属性：**

| 特性 | 普通属性 | @property |
|------|---------|-----------|
| 访问方式 | `obj.attr` | `obj.attr` (相同！) |
| 赋值验证 | ❌ | ✅ 通过 setter |
| 计算属性 | ❌ | ✅ getter 可以计算 |
| 只读属性 | ❌ | ✅ 不定义 setter |
| 延迟计算 | ❌ | ✅ 每次访问时计算 |

**在 LangChain 源码中的应用：**

```python
# langchain_core/runnables/base.py
class RunnableSequence(Runnable):
    first: Runnable
    last: Runnable

    @property
    def input_schema(self) -> Type[BaseModel]:
        """输入 schema 由第一个 Runnable 决定"""
        return self.first.input_schema

    @property
    def output_schema(self) -> Type[BaseModel]:
        """输出 schema 由最后一个 Runnable 决定"""
        return self.last.output_schema
```

---

## 4. 【最小可用】

掌握以下内容，就能开始进行 LangChain 源码阅读：

### 4.1 定义类和创建对象

```python
class Message:
    """最基本的类定义"""

    def __init__(self, content: str, role: str = "user"):
        self.content = content
        self.role = role

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}

# 创建对象
msg = Message("Hello", role="assistant")
print(msg.to_dict())  # {'role': 'assistant', 'content': 'Hello'}
```

### 4.2 继承和方法重写

```python
class BaseMessage:
    type: str = "base"

    def __init__(self, content: str):
        self.content = content

class HumanMessage(BaseMessage):
    type: str = "human"  # 重写类属性

class AIMessage(BaseMessage):
    type: str = "ai"

    def __init__(self, content: str, model: str = "gpt-4"):
        super().__init__(content)  # 调用父类初始化
        self.model = model  # 添加新属性
```

### 4.3 抽象基类 ABC

```python
from abc import ABC, abstractmethod

class Runnable(ABC):
    """抽象基类：定义接口"""

    @abstractmethod
    def invoke(self, input):
        """子类必须实现"""
        pass

class MyRunnable(Runnable):
    def invoke(self, input):
        return f"处理: {input}"

# runnable = Runnable()  # ❌ TypeError: 不能实例化抽象类
runnable = MyRunnable()  # ✅
```

### 4.4 @property 属性访问

```python
class Config:
    def __init__(self, temperature: float = 0.7):
        self._temperature = temperature

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float):
        if not 0 <= value <= 2:
            raise ValueError("temperature 必须在 0-2 之间")
        self._temperature = value

config = Config()
config.temperature = 0.5  # ✅
# config.temperature = 3  # ❌ ValueError
```

### 4.5 魔术方法 `__str__` 和 `__repr__`

```python
class Message:
    def __init__(self, content: str):
        self.content = content

    def __str__(self) -> str:
        return self.content

    def __repr__(self) -> str:
        return f"Message(content={self.content!r})"

msg = Message("Hello")
print(msg)       # Hello (调用 __str__)
print(repr(msg)) # Message(content='Hello') (调用 __repr__)
```

**这些知识足以：**
- 阅读 LangChain 源码中的类定义
- 理解 Runnable、BaseMessage、BaseChatModel 的继承体系
- 创建自定义的 LangChain 组件
- 理解 LCEL 管道的实现原理

---

## 5. 【1个类比】（双轨制）

### 类比1：类与对象

#### 🎨 前端视角：React Component

类就像 React 组件定义，对象就像组件实例。

```typescript
// React: 组件定义（类）
class UserCard extends React.Component {
  constructor(props) {
    super(props);
    this.state = { likes: 0 };
  }

  render() {
    return <div>{this.props.name}</div>;
  }
}

// 使用（创建实例）
<UserCard name="Alice" />
<UserCard name="Bob" />
```

```python
# Python: 类定义
class UserCard:
    def __init__(self, name: str):
        self.name = name
        self.likes = 0

    def render(self) -> str:
        return f"<div>{self.name}</div>"

# 使用（创建实例）
card1 = UserCard("Alice")
card2 = UserCard("Bob")
```

#### 🧒 小朋友视角：乐高说明书

- **类 = 乐高说明书**：告诉你怎么拼一个东西
- **对象 = 拼好的乐高**：按说明书拼出来的成品

**生活例子：**
```
你有一本"恐龙"乐高说明书（类）
按照说明书，你可以拼出：
- 一只红色恐龙（对象1）
- 一只蓝色恐龙（对象2）
- 一只绿色恐龙（对象3）

每只恐龙都是独立的，但它们都是按同一本说明书拼的！
```

---

### 类比2：继承

#### 🎨 前端视角：组件继承 / extends

```typescript
// 基础按钮
class Button extends React.Component {
  render() {
    return <button className="btn">{this.props.text}</button>;
  }
}

// 主要按钮继承基础按钮
class PrimaryButton extends Button {
  render() {
    return <button className="btn btn-primary">{this.props.text}</button>;
  }
}

// 危险按钮也继承基础按钮
class DangerButton extends Button {
  render() {
    return <button className="btn btn-danger">{this.props.text}</button>;
  }
}
```

```python
# Python 继承
class Button:
    def render(self) -> str:
        return '<button class="btn">Click</button>'

class PrimaryButton(Button):
    def render(self) -> str:
        return '<button class="btn btn-primary">Click</button>'

class DangerButton(Button):
    def render(self) -> str:
        return '<button class="btn btn-danger">Click</button>'
```

#### 🧒 小朋友视角：新版说明书

继承就像基于旧说明书做一本新说明书。

**生活例子：**
```
基础说明书：普通恐龙
  - 有头、有身体、有尾巴、有四条腿

新版说明书1：飞龙（继承自普通恐龙）
  - 有头、有身体、有尾巴、有四条腿 ← 从基础说明书继承
  - 有翅膀 ← 新增的部分

新版说明书2：霸王龙（继承自普通恐龙）
  - 有头、有身体、有尾巴 ← 从基础说明书继承
  - 只有两条腿 ← 修改的部分（重写）
  - 超级大嘴巴 ← 新增的部分
```

---

### 类比3：多态

#### 🎨 前端视角：接口 / 鸭子类型

```typescript
// TypeScript 接口
interface Clickable {
  onClick(): void;
}

class Button implements Clickable {
  onClick() { console.log("Button clicked"); }
}

class Link implements Clickable {
  onClick() { console.log("Link clicked"); }
}

// 多态：不管是 Button 还是 Link，都能 onClick
function handleClick(element: Clickable) {
  element.onClick();
}
```

```python
# Python 多态（鸭子类型）
class Button:
    def click(self):
        print("Button clicked")

class Link:
    def click(self):
        print("Link clicked")

# 多态：不管是什么类型，只要有 click 方法就行
def handle_click(element):
    element.click()

handle_click(Button())  # Button clicked
handle_click(Link())    # Link clicked
```

#### 🧒 小朋友视角：不同玩具都能发声

**生活例子：**
```
你有一个"让玩具发出声音"的游戏：

- 按下机器人 → "嘀嘀嘀"
- 按下小狗 → "汪汪汪"
- 按下小猫 → "喵喵喵"
- 按下汽车 → "嘟嘟嘟"

虽然它们是不同的玩具（不同的类），
但它们都能做同一件事：发出声音（同一个接口）！
这就是多态！
```

---

### 类比4：封装

#### 🎨 前端视角：private / 模块封装

```typescript
class ApiClient {
  private apiKey: string;  // 私有属性

  constructor(apiKey: string) {
    this.apiKey = apiKey;
  }

  public async fetch(url: string) {  // 公开方法
    return fetch(url, {
      headers: { Authorization: `Bearer ${this.apiKey}` }
    });
  }
}

// 外部无法访问 apiKey
const client = new ApiClient("secret");
// client.apiKey  // ❌ Property 'apiKey' is private
```

```python
class ApiClient:
    def __init__(self, api_key: str):
        self._api_key = api_key  # 受保护属性

    def fetch(self, url: str):  # 公开方法
        # 使用 _api_key 但不暴露它
        pass

# Python 约定：_ 开头表示不应从外部访问
client = ApiClient("secret")
# client._api_key  # ⚠️ 可以访问但不应该
```

#### 🧒 小朋友视角：玩具内部零件藏起来

**生活例子：**
```
遥控汽车：
- 你能看到的（公开）：前进、后退、左转、右转按钮
- 藏起来的（封装）：电池、电机、电路板

你只需要按按钮就能控制汽车，
不需要知道电机是怎么转的！

如果电路板暴露在外面：
- 容易坏 ❌
- 太复杂看不懂 ❌
- 可能会弄坏它 ❌

所以要封装起来！
```

---

### 类比5：抽象类 ABC

#### 🎨 前端视角：Interface / 抽象基类

```typescript
// TypeScript 接口定义契约
interface Runnable {
  invoke(input: any): any;  // 必须实现
}

// 实现接口
class MyProcessor implements Runnable {
  invoke(input: any) {
    return `Processed: ${input}`;
  }
}
```

```python
from abc import ABC, abstractmethod

# Python 抽象基类定义契约
class Runnable(ABC):
    @abstractmethod
    def invoke(self, input):  # 必须实现
        pass

# 实现抽象类
class MyProcessor(Runnable):
    def invoke(self, input):
        return f"Processed: {input}"
```

#### 🧒 小朋友视角：考试题目模板

**生活例子：**
```
老师发了一张"自我介绍"的模板（抽象类）：

┌─────────────────────────┐
│ 自我介绍                  │
│                         │
│ 1. 我的名字是：________    │  ← 必须填写
│ 2. 我今年____岁          │  ← 必须填写
│ 3. 我的爱好是：________   │  ← 必须填写
└─────────────────────────┘

小明的答案（具体实现）：
1. 我的名字是：小明
2. 我今年 8 岁
3. 我的爱好是：踢足球

小红的答案（另一个实现）：
1. 我的名字是：小红
2. 我今年 9 岁
3. 我的爱好是：画画

模板规定了要填什么（抽象方法），
每个人填的内容不同（具体实现）！
```

---

### 类比总结表

| OOP 概念 | 前端类比 | 小朋友类比 |
|---------|---------|-----------|
| 类 (Class) | React Component 定义 | 乐高说明书 |
| 对象 (Object) | Component 实例 | 拼好的乐高 |
| 继承 (Inheritance) | extends / 组件继承 | 新版说明书基于旧版 |
| 方法 (Method) | 组件方法 | 玩具能做的动作 |
| 属性 (Attribute) | state / props | 玩具的颜色、大小 |
| 封装 (Encapsulation) | private / 模块化 | 玩具内部零件藏起来 |
| 多态 (Polymorphism) | 接口 / 鸭子类型 | 不同玩具都能发声 |
| 抽象类 (ABC) | Interface | 考试题目模板 |
| `__init__` | constructor | 组装玩具的第一步 |
| `@property` | getter/setter | 只读说明书 |
| `__or__` | pipe 操作符 | 积木拼接器 |

---

## 6. 【反直觉点】

### 误区1：继承就是为了代码复用 ❌

**为什么错？**
- 继承的本质是 **"is-a" 关系**，不是代码复用工具
- 如果只是为了复用代码，应该用 **组合 (Composition)**
- 滥用继承会导致脆弱的基类问题、紧耦合

**为什么人们容易这样错？**
因为继承确实能复用代码，而且教科书经常用"代码复用"来解释继承的好处。但这混淆了手段和目的。

**正确理解：**

```python
# ❌ 错误：为了复用代码而继承
class Logger:
    def log(self, msg):
        print(f"[LOG] {msg}")

class UserService(Logger):  # UserService "is-a" Logger? 不对！
    def create_user(self, name):
        self.log(f"Creating user: {name}")
        # ...

# ✅ 正确：用组合而不是继承
class UserService:
    def __init__(self, logger: Logger):
        self.logger = logger  # "has-a" 关系

    def create_user(self, name):
        self.logger.log(f"Creating user: {name}")
        # ...

# LangChain 中的例子
class ChatOpenAI(BaseChatModel):  # ✅ ChatOpenAI "is-a" ChatModel
    pass

class Chain:
    def __init__(self, llm: BaseChatModel):  # ✅ 组合：Chain "has-a" LLM
        self.llm = llm
```

**判断标准：**
- 用继承：子类真的"是一种"父类吗？
- 用组合：是"有一个"的关系吗？

---

### 误区2：Python 没有真正的私有，所以封装没意义 ❌

**为什么错？**
- Python 的 `_` 约定是 **社区契约**，比强制私有更灵活
- `_` 前缀表示 "请不要直接使用"，IDE 和 linter 会警告
- Python 哲学："我们都是成年人"，信任而非限制

**为什么人们容易这样错？**
来自 Java/C++ 背景的开发者习惯了 private 关键字的强制限制，觉得 Python 的约定"太弱了"。

**正确理解：**

```python
class Config:
    def __init__(self):
        self._internal_state = {}  # 约定：内部使用
        self.__very_private = 42   # 名称改写：更强的暗示

    @property
    def state(self):
        return self._internal_state.copy()  # 返回副本，保护原数据

# 约定的好处
config = Config()

# IDE 会显示 _internal_state 是内部的
# Pylint 会警告直接访问 _internal_state
# 但在调试时你仍然可以访问它！这很有用

# Python 哲学
# "We are all consenting adults here"
# 相信开发者会遵守约定，而不是用技术手段强制限制
```

**LangChain 源码中的实践：**

```python
# langchain_core/runnables/base.py
class Runnable:
    def invoke(self, input):       # 公开 API
        return self._call(input)   # 内部实现

    def _call(self, input):        # 约定：子类重写这个
        raise NotImplementedError
```

---

### 误区3：多重继承会导致混乱，应该避免 ❌

**为什么错？**
- Python 的 **Mixin 模式** 是一种安全的多重继承使用方式
- Mixin 是只提供方法、不提供状态的类
- 正确使用多重继承可以实现灵活的功能组合

**为什么人们容易这样错？**
C++ 的多重继承确实容易造成"菱形继承"问题。但 Python 的 MRO（方法解析顺序）算法很好地解决了这个问题。

**正确理解：**

```python
# Mixin 模式：安全的多重继承

class SerializableMixin:
    """Mixin: 提供序列化能力"""
    def to_json(self) -> str:
        import json
        return json.dumps(self.__dict__)

class LoggableMixin:
    """Mixin: 提供日志能力"""
    def log(self, msg: str):
        print(f"[{self.__class__.__name__}] {msg}")

class ValidatableMixin:
    """Mixin: 提供验证能力"""
    def validate(self) -> bool:
        # 子类应该重写这个方法
        return True

# 组合多个 Mixin
class User(SerializableMixin, LoggableMixin, ValidatableMixin):
    def __init__(self, name: str, age: int):
        self.name = name
        self.age = age

    def validate(self) -> bool:
        return len(self.name) > 0 and self.age >= 0

user = User("Alice", 25)
print(user.to_json())  # {"name": "Alice", "age": 25}
user.log("User created")  # [User] User created
print(user.validate())  # True
```

**LangChain 源码中的 Mixin 使用：**

```python
# langchain_core/runnables/base.py
class RunnableSerializable(Serializable, Runnable):
    """组合序列化能力和可运行能力"""
    pass

# 很多 LangChain 类都使用 Mixin 组合功能
class BaseChatModel(
    BaseLanguageModel,
    RunnableSerializable,
):
    pass
```

---

## 7. 【实战代码】

```python
"""
示例：构建 LangChain 风格的消息和 Runnable 系统
演示 OOP 在 LangChain 源码中的核心用法
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, TypeVar, Generic
from dataclasses import dataclass
from datetime import datetime

# ===== 1. 消息系统（继承 + 多态） =====
print("=== 1. 消息系统 ===")

class BaseMessage(ABC):
    """消息抽象基类 - 类似 langchain_core.messages.BaseMessage"""

    def __init__(self, content: str, **kwargs):
        self.content = content
        self.additional_kwargs: Dict[str, Any] = kwargs
        self.timestamp = datetime.now()

    @property
    @abstractmethod
    def type(self) -> str:
        """消息类型，子类必须实现"""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """序列化为字典"""
        return {
            "type": self.type,
            "content": self.content,
            "additional_kwargs": self.additional_kwargs,
        }

    def __str__(self) -> str:
        return f"{self.type}: {self.content[:50]}..."

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(content={self.content!r})"

class HumanMessage(BaseMessage):
    """用户消息"""

    @property
    def type(self) -> str:
        return "human"

class AIMessage(BaseMessage):
    """AI 消息"""

    def __init__(self, content: str, model: str = "unknown", **kwargs):
        super().__init__(content, **kwargs)
        self.model = model

    @property
    def type(self) -> str:
        return "ai"

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["model"] = self.model
        return d

class SystemMessage(BaseMessage):
    """系统消息"""

    @property
    def type(self) -> str:
        return "system"

# 多态演示
messages: List[BaseMessage] = [
    SystemMessage("You are a helpful assistant"),
    HumanMessage("Hello!"),
    AIMessage("Hi! How can I help you?", model="gpt-4"),
]

for msg in messages:
    print(f"  {msg.type}: {msg.content}")

# ===== 2. Runnable 协议（抽象类 + 泛型） =====
print("\n=== 2. Runnable 协议 ===")

Input = TypeVar("Input")
Output = TypeVar("Output")

class Runnable(ABC, Generic[Input, Output]):
    """可运行组件的抽象基类 - 类似 langchain_core.runnables.Runnable"""

    @property
    def name(self) -> str:
        """组件名称"""
        return self.__class__.__name__

    @abstractmethod
    def invoke(self, input: Input) -> Output:
        """同步调用"""
        pass

    def batch(self, inputs: List[Input]) -> List[Output]:
        """批量调用（默认实现：循环调用 invoke）"""
        return [self.invoke(x) for x in inputs]

    def __or__(self, other: "Runnable") -> "RunnableSequence":
        """重载 | 操作符，实现 LCEL 管道"""
        return RunnableSequence(first=self, last=other)

    def __repr__(self) -> str:
        return f"{self.name}()"

class RunnableSequence(Runnable[Input, Output]):
    """Runnable 序列 - 管道的实现"""

    def __init__(self, first: Runnable, last: Runnable):
        self.first = first
        self.last = last

    @property
    def name(self) -> str:
        return f"{self.first.name} | {self.last.name}"

    def invoke(self, input: Input) -> Output:
        """串联执行"""
        intermediate = self.first.invoke(input)
        return self.last.invoke(intermediate)

# ===== 3. 具体 Runnable 实现 =====
print("\n=== 3. 具体实现 ===")

class PromptTemplate(Runnable[Dict[str, Any], str]):
    """提示模板 - 类似 langchain_core.prompts.PromptTemplate"""

    def __init__(self, template: str):
        self.template = template
        self._input_variables = self._extract_variables()

    def _extract_variables(self) -> List[str]:
        """提取模板变量"""
        import re
        return re.findall(r'\{(\w+)\}', self.template)

    @property
    def input_variables(self) -> List[str]:
        return self._input_variables

    def invoke(self, input: Dict[str, Any]) -> str:
        """格式化模板"""
        return self.template.format(**input)

class FakeLLM(Runnable[str, str]):
    """模拟 LLM - 用于演示"""

    def __init__(self, response_prefix: str = "LLM says:"):
        self.response_prefix = response_prefix
        self._call_count = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    def invoke(self, input: str) -> str:
        """模拟 LLM 调用"""
        self._call_count += 1
        return f"{self.response_prefix} {input[::-1]}"  # 反转输入作为"响应"

class OutputParser(Runnable[str, Dict[str, Any]]):
    """输出解析器 - 类似 langchain_core.output_parsers"""

    def invoke(self, input: str) -> Dict[str, Any]:
        """解析输出"""
        return {
            "raw": input,
            "length": len(input),
            "word_count": len(input.split()),
        }

# 演示
prompt = PromptTemplate("Hello {name}, you are learning {topic}!")
print(f"模板变量: {prompt.input_variables}")
print(f"格式化: {prompt.invoke({'name': 'Alice', 'topic': 'OOP'})}")

# ===== 4. LCEL 管道（操作符重载） =====
print("\n=== 4. LCEL 管道 ===")

# 创建管道
chain = prompt | FakeLLM() | OutputParser()
print(f"管道: {chain.name}")

# 执行管道
result = chain.invoke({"name": "Bob", "topic": "LangChain"})
print(f"结果: {result}")

# ===== 5. Mixin 模式 =====
print("\n=== 5. Mixin 模式 ===")

class SerializableMixin:
    """序列化 Mixin"""

    def to_json(self) -> str:
        import json
        return json.dumps(self._get_serializable_fields())

    def _get_serializable_fields(self) -> Dict[str, Any]:
        """子类可重写以自定义序列化字段"""
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_')}

class CallbackMixin:
    """回调 Mixin"""

    def __init__(self):
        self._callbacks: List[callable] = []

    def add_callback(self, callback: callable):
        self._callbacks.append(callback)

    def _trigger_callbacks(self, event: str, data: Any):
        for callback in self._callbacks:
            callback(event, data)

class EnhancedLLM(FakeLLM, SerializableMixin, CallbackMixin):
    """增强版 LLM - 组合多个 Mixin"""

    def __init__(self, response_prefix: str = "Enhanced LLM:"):
        FakeLLM.__init__(self, response_prefix)
        CallbackMixin.__init__(self)

    def invoke(self, input: str) -> str:
        self._trigger_callbacks("before_invoke", input)
        result = super().invoke(input)
        self._trigger_callbacks("after_invoke", result)
        return result

    def _get_serializable_fields(self) -> Dict[str, Any]:
        return {"response_prefix": self.response_prefix, "call_count": self.call_count}

# 演示 Mixin
llm = EnhancedLLM()
llm.add_callback(lambda event, data: print(f"  [Callback] {event}: {data[:30]}..."))

print(f"调用前: {llm.to_json()}")
result = llm.invoke("Hello World")
print(f"调用后: {llm.to_json()}")

# ===== 6. @property 高级用法 =====
print("\n=== 6. @property 高级用法 ===")

class LLMConfig:
    """LLM 配置类 - 展示 @property"""

    def __init__(self, temperature: float = 0.7, max_tokens: int = 1000):
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._model = "gpt-4"

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float):
        if not 0 <= value <= 2:
            raise ValueError(f"temperature must be between 0 and 2, got {value}")
        self._temperature = value

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @max_tokens.setter
    def max_tokens(self, value: int):
        if value <= 0:
            raise ValueError(f"max_tokens must be positive, got {value}")
        self._max_tokens = value

    @property
    def model(self) -> str:
        """只读属性"""
        return self._model

    @property
    def config_summary(self) -> str:
        """计算属性"""
        return f"{self._model}(temp={self._temperature}, max={self._max_tokens})"

config = LLMConfig()
print(f"配置摘要: {config.config_summary}")

config.temperature = 0.5
print(f"修改后: {config.config_summary}")

try:
    config.temperature = 3.0  # 触发验证
except ValueError as e:
    print(f"验证错误: {e}")

try:
    config.model = "gpt-3.5"  # 只读属性
except AttributeError as e:
    print(f"只读错误: {e}")

print("\n=== 完成 ===")
```

**运行输出示例：**
```
=== 1. 消息系统 ===
  system: You are a helpful assistant
  human: Hello!
  ai: Hi! How can I help you?

=== 2. Runnable 协议 ===

=== 3. 具体实现 ===
模板变量: ['name', 'topic']
格式化: Hello Alice, you are learning OOP!

=== 4. LCEL 管道 ===
管道: PromptTemplate | FakeLLM | OutputParser
结果: {'raw': 'LLM says: !niahCgnaL gninrael era uoy ,boB olleH', 'length': 47, 'word_count': 7}

=== 5. Mixin 模式 ===
调用前: {"response_prefix": "Enhanced LLM:", "call_count": 0}
  [Callback] before_invoke: Hello World...
  [Callback] after_invoke: Enhanced LLM: dlroW olleH...
调用后: {"response_prefix": "Enhanced LLM:", "call_count": 1}

=== 6. @property 高级用法 ===
配置摘要: gpt-4(temp=0.7, max=1000)
修改后: gpt-4(temp=0.5, max=1000)
验证错误: temperature must be between 0 and 2, got 3.0
只读错误: property 'model' of 'LLMConfig' object has no setter

=== 完成 ===
```

---

## 8. 【面试必问】

### 问题："Python 中的继承和组合怎么选择？"

**普通回答（❌ 不出彩）：**
"继承是 is-a 关系，组合是 has-a 关系。能用组合就用组合，因为组合更灵活。"

**出彩回答（✅ 推荐）：**

> **继承和组合的选择有三个层面：**
>
> 1. **语义层面**：
>    - 继承表示 **"is-a"** 关系：Dog is an Animal
>    - 组合表示 **"has-a"** 关系：Car has an Engine
>    - 问自己：子类真的是父类的一种特殊类型吗？
>
> 2. **实践层面**：
>    - 继承的问题：脆弱的基类、紧耦合、难以单独测试
>    - 组合的优势：松耦合、易测试、可以运行时替换
>    - Python 社区的共识：**优先组合，必要时继承**
>
> 3. **LangChain 中的实际例子**：
>    - **用继承**：`HumanMessage(BaseMessage)` - AI 消息"是一种"消息
>    - **用组合**：`Chain` 包含 `LLM` 实例 - Chain"有一个"LLM，不是"是一种"LLM
>    - **Mixin 模式**：`BaseChatModel(BaseLanguageModel, RunnableSerializable)` - 组合多个能力
>
> **我的选择原则**：
> - 如果是定义类型层次（如消息类型），用继承
> - 如果是组合功能（如 Chain 使用 LLM），用组合
> - 如果需要共享行为但不是类型关系，用 Mixin

**为什么这个回答出彩？**
1. ✅ 从多个层面分析（语义、实践、源码）
2. ✅ 用 LangChain 真实代码举例
3. ✅ 给出了清晰的选择原则
4. ✅ 提到了 Mixin 作为第三种选择

---

### 问题："Python 的 @property 有什么用？"

**普通回答（❌ 不出彩）：**
"@property 可以把方法变成属性，实现 getter 和 setter。"

**出彩回答（✅ 推荐）：**

> **@property 有四个核心用途：**
>
> 1. **数据验证**：在赋值时自动检查
>    ```python
>    @temperature.setter
>    def temperature(self, value):
>        if not 0 <= value <= 2:
>            raise ValueError("...")
>    ```
>
> 2. **计算属性**：每次访问时动态计算
>    ```python
>    @property
>    def token_count(self):
>        return len(self.content.split())
>    ```
>
> 3. **只读属性**：只定义 getter，不定义 setter
>    ```python
>    @property
>    def model_name(self):
>        return self._model_name
>    ```
>
> 4. **接口兼容**：把属性访问变成方法调用，但保持原有 API
>
> **与直接属性的区别**：
> - 外部使用方式完全相同：`obj.value`
> - 但内部可以有复杂逻辑：验证、计算、日志...
>
> **在 LangChain 中的应用**：
> - `Runnable.input_schema` - 计算输入 schema
> - `RunnableSequence.first` - 只读访问管道第一步
> - `BaseChatModel.model_name` - 只读模型名称

---

## 9. 【化骨绵掌】

### 卡片1：类与对象基础 🎯

**一句话：** 类是蓝图，对象是根据蓝图创建的实例。

**举例：**
```python
class Dog:
    def __init__(self, name):
        self.name = name

    def bark(self):
        return f"{self.name}: 汪汪！"

dog1 = Dog("小白")  # 对象1
dog2 = Dog("小黑")  # 对象2
```

**应用：** LangChain 中 `HumanMessage("Hello")` 就是创建一个消息对象。

---

### 卡片2：`__init__` 初始化方法 📐

**一句话：** `__init__` 是对象创建时自动调用的初始化方法。

**举例：**
```python
class Message:
    def __init__(self, content: str, role: str = "user"):
        self.content = content  # 必填参数
        self.role = role        # 带默认值的参数
        self.created_at = datetime.now()  # 自动生成

msg = Message("Hello")  # 自动调用 __init__
```

**应用：** LangChain 的 `ChatOpenAI(model="gpt-4", temperature=0.7)` 在 `__init__` 中初始化配置。

---

### 卡片3：实例属性 vs 类属性 🔧

**一句话：** 实例属性每个对象独有，类属性所有对象共享。

**举例：**
```python
class Counter:
    total = 0  # 类属性：共享

    def __init__(self):
        Counter.total += 1
        self.id = Counter.total  # 实例属性：独有

c1 = Counter()  # c1.id = 1, Counter.total = 1
c2 = Counter()  # c2.id = 2, Counter.total = 2
```

**应用：** LangChain 用类属性定义默认配置，实例属性存储具体值。

---

### 卡片4：继承基础 🏗️

**一句话：** 子类继承父类的属性和方法，表示 "is-a" 关系。

**举例：**
```python
class Animal:
    def speak(self):
        return "..."

class Dog(Animal):  # Dog is an Animal
    def speak(self):  # 重写父类方法
        return "汪汪！"

class Cat(Animal):  # Cat is an Animal
    def speak(self):
        return "喵喵！"
```

**应用：** `HumanMessage(BaseMessage)` - 用户消息是一种消息。

---

### 卡片5：方法重写与 super() ⚡

**一句话：** 子类可以重写父类方法，用 `super()` 调用父类实现。

**举例：**
```python
class Parent:
    def greet(self):
        return "Hello from Parent"

class Child(Parent):
    def greet(self):
        parent_greeting = super().greet()  # 调用父类方法
        return f"{parent_greeting}, and Child!"
```

**应用：** LangChain 子类常用 `super().__init__()` 初始化父类属性。

---

### 卡片6：@property 装饰器 🎨

**一句话：** `@property` 把方法变成属性，支持验证和只读。

**举例：**
```python
class Config:
    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        if not 0 <= value <= 2:
            raise ValueError("Invalid temperature")
        self._temperature = value
```

**应用：** LangChain 用 `@property` 实现 `Runnable.input_schema` 等只读属性。

---

### 卡片7：魔术方法 `__str__` `__repr__` 📝

**一句话：** `__str__` 给用户看，`__repr__` 给开发者看。

**举例：**
```python
class Point:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __str__(self):
        return f"({self.x}, {self.y})"  # 用户友好

    def __repr__(self):
        return f"Point(x={self.x}, y={self.y})"  # 调试用

p = Point(3, 4)
print(p)       # (3, 4)
print(repr(p)) # Point(x=3, y=4)
```

**应用：** LangChain 消息类定义 `__repr__` 方便调试。

---

### 卡片8：抽象基类 ABC 🔒

**一句话：** 抽象类定义接口，子类必须实现抽象方法。

**举例：**
```python
from abc import ABC, abstractmethod

class Runnable(ABC):
    @abstractmethod
    def invoke(self, input):
        """子类必须实现"""
        pass

# Runnable()  # ❌ 不能实例化

class MyRunnable(Runnable):
    def invoke(self, input):  # ✅ 必须实现
        return f"Processed: {input}"
```

**应用：** LangChain 的 `Runnable`、`BaseMessage`、`BaseChatModel` 都是抽象类。

---

### 卡片9：多重继承与 Mixin 🔄

**一句话：** Mixin 是只提供方法的类，可以安全地多重继承。

**举例：**
```python
class JSONMixin:
    def to_json(self):
        import json
        return json.dumps(self.__dict__)

class LogMixin:
    def log(self, msg):
        print(f"[{self.__class__.__name__}] {msg}")

class User(JSONMixin, LogMixin):
    def __init__(self, name):
        self.name = name

user = User("Alice")
user.log("Created")  # [User] Created
print(user.to_json())  # {"name": "Alice"}
```

**应用：** LangChain 用 `RunnableSerializable` Mixin 添加序列化能力。

---

### 卡片10：LangChain 源码中的 OOP 实践 ⭐

**一句话：** LangChain 用 OOP 构建可组合的组件系统。

**核心设计：**
```python
# 1. 抽象基类定义接口
class Runnable(ABC):
    @abstractmethod
    def invoke(self, input): pass

# 2. 具体实现继承抽象类
class ChatOpenAI(Runnable): ...
class PromptTemplate(Runnable): ...

# 3. 操作符重载实现 LCEL
class Runnable:
    def __or__(self, other):
        return RunnableSequence(self, other)

# 4. 使用
chain = prompt | llm | parser  # OOP 的威力！
```

**应用：** 理解这个模式，就能读懂 LangChain 90% 的源码结构。

---

## 10. 【一句话总结】

**OOP 是通过类和对象组织代码的编程范式，通过封装隐藏复杂性、继承复用代码、多态统一接口，是 LangChain 构建可组合组件系统（Runnable、Message、Model）的核心基础。**

---

## 📚 学习检查清单

- [ ] 能够定义类和创建对象
- [ ] 理解 `__init__` 初始化方法的作用
- [ ] 区分实例属性和类属性
- [ ] 会使用继承和方法重写
- [ ] 理解 `super()` 的用法
- [ ] 会使用 `@property` 装饰器
- [ ] 理解魔术方法 `__str__`、`__repr__` 的区别
- [ ] 会使用抽象基类 ABC 定义接口
- [ ] 理解 Mixin 模式
- [ ] 能够阅读 LangChain 源码中的类定义

## 🔗 下一步学习

- **模块与包系统**：理解 Python 代码组织方式
- **异常处理机制**：理解错误处理和自定义异常
- **Pydantic 数据验证**：LangChain 数据模型的基础

---

**版本：** v1.0
**最后更新：** 2025-12-12
