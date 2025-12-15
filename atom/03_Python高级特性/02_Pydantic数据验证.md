# Pydantic 数据验证

> 原子化知识点 | Python高级特性 | LangChain 源码学习前置知识

---

## 1. 【30字核心】

**Pydantic 是 Python 的数据验证库，通过类型注解自动校验数据，是 LangChain 所有数据结构的基石。**

---

## 2. 【第一性原理】

### 什么是第一性原理？

**第一性原理**：回到事物最基本的真理，从源头思考问题

### Pydantic 的第一性原理 🎯

#### 1. 最基础的定义

**Pydantic = 类型注解 + 自动验证 + 数据转换**

仅此而已！没有更基础的了。

- **类型注解**：告诉 Python 这个字段应该是什么类型
- **自动验证**：在赋值时自动检查是否符合类型要求
- **数据转换**：尝试将输入数据转换成目标类型

#### 2. 为什么需要 Pydantic？

**核心问题：Python 是动态类型语言，运行时不会自动检查类型**

```python
# Python 原生行为：不检查类型，直接运行
def greet(name: str) -> str:
    return f"Hello, {name}"

greet(123)  # 不报错！输出 "Hello, 123"
greet(None)  # 不报错！输出 "Hello, None"
```

这在构建 LLM 应用时会导致严重问题：
- LLM 输出可能是任意格式
- API 调用参数可能类型错误
- 配置文件可能格式不对
- 用户输入可能不符合预期

#### 3. Pydantic 的三层价值

##### 价值1：数据安全门卫

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

# Pydantic 会自动验证
User(name="Alice", age="25")  # ✅ "25" 自动转为 int
User(name="Bob", age="abc")   # ❌ ValidationError!
```

##### 价值2：自动类型转换

```python
class Config(BaseModel):
    debug: bool
    port: int

# 智能转换
Config(debug="true", port="8080")  # ✅ 自动转换
Config(debug=1, port=8080.0)       # ✅ 自动转换
```

##### 价值3：结构化数据的统一接口

```python
class LLMResponse(BaseModel):
    content: str
    tokens_used: int

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_json(cls, json_str: str):
        return cls.model_validate_json(json_str)
```

#### 4. 从第一性原理推导 LangChain 源码应用

**推理链：**

```
1. LLM 的输入输出都是非结构化数据（文本）
   ↓
2. 需要将非结构化数据转换为结构化数据
   ↓
3. 结构化数据需要类型安全保证
   ↓
4. Python 原生不提供运行时类型检查
   ↓
5. Pydantic 提供运行时类型验证 + 数据转换
   ↓
6. LangChain 使用 Pydantic 作为所有数据模型的基础
   ↓
7. Runnable、Chain、Agent 的配置都是 BaseModel 子类
```

#### 5. 一句话总结第一性原理

**Pydantic 是 Python 动态类型世界的"类型守护者"，在运行时强制执行类型安全，是 LangChain 处理 LLM 输入输出的基础设施。**

---

## 3. 【核心概念（全面覆盖）】

### 核心概念1：BaseModel 基类 🏗️

**BaseModel 是 Pydantic 的核心，所有数据模型都继承它**

```python
from pydantic import BaseModel
from typing import Optional, List

class Message(BaseModel):
    """聊天消息模型 - LangChain 中最基础的数据结构"""
    role: str                      # 必填字段
    content: str                   # 必填字段
    name: Optional[str] = None     # 可选字段，默认 None
    metadata: dict = {}            # 带默认值的字段

# 创建实例的多种方式
msg1 = Message(role="user", content="Hello")
msg2 = Message(**{"role": "assistant", "content": "Hi"})
msg3 = Message.model_validate({"role": "system", "content": "You are helpful"})
```

**BaseModel 自动提供的能力：**

| 方法/属性 | 作用 | 示例 |
|----------|------|------|
| `model_dump()` | 转换为字典 | `msg.model_dump()` |
| `model_dump_json()` | 转换为 JSON 字符串 | `msg.model_dump_json()` |
| `model_validate()` | 从字典创建实例 | `Message.model_validate(data)` |
| `model_validate_json()` | 从 JSON 创建实例 | `Message.model_validate_json(json_str)` |
| `model_fields` | 获取字段定义 | `Message.model_fields` |
| `model_copy()` | 浅拷贝 | `msg.model_copy(update={"content": "new"})` |

**在 LangChain 源码中的应用：**

```python
# langchain_core/messages/base.py
class BaseMessage(BaseModel):
    """LangChain 消息基类"""
    content: Union[str, List[Union[str, Dict]]]
    additional_kwargs: dict = Field(default_factory=dict)
    response_metadata: dict = Field(default_factory=dict)
    type: str
    name: Optional[str] = None
    id: Optional[str] = None
```

---

### 核心概念2：Field 字段配置 📐

**Field() 用于配置字段的详细行为**

```python
from pydantic import BaseModel, Field
from typing import List

class LLMConfig(BaseModel):
    """LLM 配置模型 - 展示 Field 的各种用法"""

    # 带描述的字段（用于文档和 JSON Schema）
    model_name: str = Field(
        default="gpt-4",
        description="模型名称"
    )

    # 带验证约束的字段
    temperature: float = Field(
        default=0.7,
        ge=0.0,      # greater than or equal (>=)
        le=2.0,      # less than or equal (<=)
        description="采样温度，0-2之间"
    )

    # 带别名的字段（JSON 中用别名）
    max_tokens: int = Field(
        default=1000,
        alias="maxTokens",
        gt=0         # greater than (>)
    )

    # 动态默认值
    stop_sequences: List[str] = Field(
        default_factory=list,  # 每次创建新列表
        description="停止序列"
    )

    # 私有字段（不参与序列化）
    _api_key: str = ""

# 使用别名创建
config = LLMConfig(maxTokens=2000)  # 使用别名
print(config.max_tokens)  # 2000
```

**Field 常用参数速查：**

| 参数 | 作用 | 示例 |
|------|------|------|
| `default` | 默认值 | `Field(default=0)` |
| `default_factory` | 动态默认值工厂 | `Field(default_factory=list)` |
| `alias` | JSON 别名 | `Field(alias="userName")` |
| `description` | 字段描述 | `Field(description="用户名")` |
| `gt/ge/lt/le` | 数值约束 | `Field(ge=0, le=100)` |
| `min_length/max_length` | 字符串/列表长度 | `Field(min_length=1)` |
| `pattern` | 正则匹配 | `Field(pattern=r"^\d+$")` |
| `exclude` | 排除序列化 | `Field(exclude=True)` |

**在 LangChain 源码中的应用：**

```python
# langchain_core/runnables/config.py
class RunnableConfig(TypedDict, total=False):
    """Runnable 配置"""
    tags: List[str]
    metadata: Dict[str, Any]
    callbacks: Callbacks
    max_concurrency: Optional[int]
    recursion_limit: int
```

---

### 核心概念3：验证器 Validators 🔧

**验证器允许自定义验证逻辑**

```python
from pydantic import BaseModel, field_validator, model_validator
from typing import Optional

class PromptTemplate(BaseModel):
    """提示模板 - 展示验证器用法"""
    template: str
    input_variables: list[str]

    # 字段验证器：验证单个字段
    @field_validator('template')
    @classmethod
    def template_must_have_variables(cls, v: str) -> str:
        """确保模板包含变量占位符"""
        if '{' not in v:
            raise ValueError('模板必须包含至少一个变量占位符 {variable}')
        return v

    @field_validator('input_variables')
    @classmethod
    def variables_not_empty(cls, v: list) -> list:
        """确保变量列表不为空"""
        if not v:
            raise ValueError('input_variables 不能为空')
        return v

    # 模型验证器：验证整个模型
    @model_validator(mode='after')
    def check_variables_in_template(self) -> 'PromptTemplate':
        """确保所有声明的变量都在模板中"""
        for var in self.input_variables:
            if f'{{{var}}}' not in self.template:
                raise ValueError(f'变量 {var} 未在模板中使用')
        return self

# 正确使用
prompt = PromptTemplate(
    template="Hello {name}, you are {age} years old",
    input_variables=["name", "age"]
)

# 错误使用 - 触发验证
try:
    bad_prompt = PromptTemplate(
        template="Hello {name}",
        input_variables=["name", "missing"]  # missing 不在模板中
    )
except ValueError as e:
    print(f"验证失败: {e}")
```

**验证器类型对比：**

| 验证器 | 时机 | 用途 |
|-------|------|------|
| `@field_validator` | 字段赋值前/后 | 验证/转换单个字段 |
| `@model_validator(mode='before')` | 所有字段处理前 | 预处理原始数据 |
| `@model_validator(mode='after')` | 所有字段处理后 | 跨字段验证 |

**在 LangChain 源码中的应用：**

```python
# langchain_core/prompts/base.py 简化版
class BasePromptTemplate(BaseModel):
    input_variables: List[str]

    @model_validator(mode='before')
    @classmethod
    def validate_input_variables(cls, values):
        # 从模板自动提取变量
        if 'input_variables' not in values:
            values['input_variables'] = extract_variables(values.get('template', ''))
        return values
```

---

### 扩展概念4：模型配置 model_config 📋

```python
from pydantic import BaseModel, ConfigDict

class StrictConfig(BaseModel):
    """严格模式配置示例"""
    model_config = ConfigDict(
        strict=True,           # 严格类型检查，不自动转换
        frozen=True,           # 不可变（类似 dataclass frozen）
        extra='forbid',        # 禁止额外字段
        validate_assignment=True,  # 赋值时也验证
        str_strip_whitespace=True, # 自动去除字符串首尾空白
    )

    name: str
    value: int

# 严格模式
try:
    StrictConfig(name="test", value="123")  # ❌ 不会自动转换
except Exception as e:
    print(f"严格模式: {e}")

# 禁止额外字段
try:
    StrictConfig(name="test", value=123, extra_field="x")  # ❌
except Exception as e:
    print(f"禁止额外字段: {e}")
```

**常用配置选项：**

| 配置项 | 作用 | 默认值 |
|-------|------|-------|
| `strict` | 严格类型检查 | `False` |
| `frozen` | 不可变实例 | `False` |
| `extra` | 额外字段处理 | `'ignore'` |
| `validate_assignment` | 赋值验证 | `False` |
| `populate_by_name` | 允许字段名和别名 | `False` |
| `use_enum_values` | 枚举使用值而非枚举对象 | `False` |

---

### 扩展概念5：嵌套模型与泛型 🔄

```python
from pydantic import BaseModel
from typing import Generic, TypeVar, List, Optional

# 泛型类型变量
T = TypeVar('T')

class Response(BaseModel, Generic[T]):
    """泛型响应模型 - LangChain 中常见模式"""
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None

class User(BaseModel):
    name: str
    email: str

class Message(BaseModel):
    role: str
    content: str

# 使用泛型
user_response: Response[User] = Response(
    success=True,
    data=User(name="Alice", email="alice@example.com")
)

message_response: Response[List[Message]] = Response(
    success=True,
    data=[
        Message(role="user", content="Hello"),
        Message(role="assistant", content="Hi!")
    ]
)
```

**在 LangChain 源码中的应用：**

```python
# langchain_core/runnables/base.py 简化版
Input = TypeVar("Input")
Output = TypeVar("Output")

class Runnable(Generic[Input, Output], ABC):
    """LangChain 最核心的泛型抽象"""

    @abstractmethod
    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        pass
```

---

## 4. 【最小可用】

掌握以下内容，就能开始进行 LangChain 源码阅读：

### 4.1 定义数据模型

```python
from pydantic import BaseModel, Field
from typing import Optional, List

class ChatMessage(BaseModel):
    role: str = Field(description="消息角色: user/assistant/system")
    content: str = Field(description="消息内容")
    name: Optional[str] = None
```

### 4.2 创建和验证实例

```python
# 从关键字参数创建
msg = ChatMessage(role="user", content="Hello")

# 从字典创建
data = {"role": "assistant", "content": "Hi!"}
msg = ChatMessage.model_validate(data)

# 从 JSON 创建
json_str = '{"role": "system", "content": "You are helpful"}'
msg = ChatMessage.model_validate_json(json_str)
```

### 4.3 序列化

```python
# 转字典
msg.model_dump()  # {'role': 'user', 'content': 'Hello', 'name': None}

# 转 JSON
msg.model_dump_json()  # '{"role":"user","content":"Hello","name":null}'

# 排除空值
msg.model_dump(exclude_none=True)  # {'role': 'user', 'content': 'Hello'}
```

### 4.4 字段验证

```python
from pydantic import field_validator

class Temperature(BaseModel):
    value: float

    @field_validator('value')
    @classmethod
    def check_range(cls, v):
        if not 0 <= v <= 2:
            raise ValueError('temperature must be between 0 and 2')
        return v
```

### 4.5 继承与扩展

```python
class BaseMessage(BaseModel):
    content: str

class HumanMessage(BaseMessage):
    """用户消息"""
    type: str = "human"

class AIMessage(BaseMessage):
    """AI 消息"""
    type: str = "ai"
    response_metadata: dict = Field(default_factory=dict)
```

**这些知识足以：**
- 阅读 LangChain 源码中的所有数据模型定义
- 理解 Runnable、Chain、Agent 的配置结构
- 自定义 LLM 输出解析器
- 创建类型安全的 LangChain 应用

---

## 5. 【1个类比】（双轨制）

### 类比1：BaseModel 数据模型

#### 🎨 前端视角：TypeScript Interface + Class

Pydantic BaseModel 就像 TypeScript 中的接口定义加上类的验证能力。

```typescript
// TypeScript: 只有类型检查，编译时
interface User {
  name: string;
  age: number;
  email?: string;
}

// 编译时检查类型
const user: User = { name: "Alice", age: 25 };
```

```python
# Pydantic: 类型检查 + 运行时验证 + 自动转换
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int
    email: str | None = None

# 运行时验证和转换
user = User(name="Alice", age="25")  # "25" 自动转为 int
```

**关键区别：** TypeScript 只在编译时检查，Pydantic 在运行时检查

#### 🧒 小朋友视角：智能分类垃圾桶

想象一个超级智能的分类垃圾桶：

- **普通垃圾桶**：什么都能扔进去，不会检查
- **智能分类垃圾桶（Pydantic）**：
  - 有标签说明应该放什么（类型注解）
  - 扔东西进去时会自动检查是不是对的类型
  - 如果放错了会"报警"（ValidationError）
  - 还能自动把一些东西变成正确的类型（比如把字符串 "25" 变成数字 25）

**生活例子：**
```
你有一个标着"只能放玩具"的箱子：
- 放进一个积木 ✅ 没问题
- 放进一本书 ❌ 箱子会"叫"说这不是玩具！
- 放进一个玩具车的图片 🔄 箱子会自动把它变成真的玩具车
```

---

### 类比2：Field 字段配置

#### 🎨 前端视角：Zod Schema 或 Yup 验证

```typescript
// Zod: 前端表单验证库
import { z } from 'zod';

const userSchema = z.object({
  name: z.string().min(1, "名字不能为空"),
  age: z.number().min(0).max(150),
  email: z.string().email().optional(),
});
```

```python
# Pydantic Field: 类似的声明式验证
from pydantic import BaseModel, Field

class User(BaseModel):
    name: str = Field(min_length=1, description="名字不能为空")
    age: int = Field(ge=0, le=150)
    email: str | None = Field(default=None, pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
```

#### 🧒 小朋友视角：游戏规则卡

Field 就像游戏的规则卡，告诉你每个东西应该是什么样的：

- **年龄**：必须是 0-150 之间的数字（不能是负数，也不能超过 150岁）
- **名字**：必须有字，不能是空的
- **描述**：告诉别人这个东西是干什么用的

**生活例子：**
```
班级登记表的规则：
- 姓名：必须填写，不能空着
- 年龄：只能写 6-12 岁
- 座位号：只能写 1-50 的数字
- 如果你写错了，老师会让你重写！
```

---

### 类比3：Validators 验证器

#### 🎨 前端视角：React Hook Form 的 validate 函数

```typescript
// React Hook Form: 自定义验证
<input
  {...register("password", {
    validate: {
      hasNumber: (v) => /\d/.test(v) || "密码必须包含数字",
      hasLetter: (v) => /[a-zA-Z]/.test(v) || "密码必须包含字母",
      minLength: (v) => v.length >= 8 || "密码至少8位",
    }
  })}
/>
```

```python
# Pydantic validator: 类似的自定义验证
from pydantic import BaseModel, field_validator

class User(BaseModel):
    password: str

    @field_validator('password')
    @classmethod
    def validate_password(cls, v):
        if not any(c.isdigit() for c in v):
            raise ValueError('密码必须包含数字')
        if not any(c.isalpha() for c in v):
            raise ValueError('密码必须包含字母')
        if len(v) < 8:
            raise ValueError('密码至少8位')
        return v
```

#### 🧒 小朋友视角：门卫检查清单

验证器就像学校门口的门卫叔叔，有一个检查清单：

- ✅ 是不是穿了校服？
- ✅ 有没有戴校牌？
- ✅ 书包里有没有危险物品？

只有全部通过检查，才能进学校！

**生活例子：**
```
进入游乐园的检查：
1. 检查门票是不是真的
2. 检查身高够不够（有些游戏要求120cm以上）
3. 检查年龄（有些游戏只能大人玩）
全部通过才能进去玩！
```

---

### 类比总结表

| Pydantic 概念 | 前端类比 | 小朋友类比 |
|--------------|---------|-----------|
| BaseModel | TypeScript Interface + Class | 智能分类垃圾桶 |
| Field | Zod/Yup Schema 定义 | 游戏规则卡 |
| field_validator | Hook Form validate | 门卫检查清单 |
| model_validator | Form 整体验证 | 全班作业互相检查 |
| model_dump() | JSON.stringify() | 把玩具打包装箱 |
| model_validate() | JSON.parse() + 验证 | 检查快递包裹内容 |
| ConfigDict | ESLint/Prettier 配置 | 班级纪律手册 |
| 泛型 Generic[T] | TypeScript Generic<T> | 万能收纳盒 |

---

## 6. 【反直觉点】

### 误区1：Pydantic 只是类型提示的增强 ❌

**为什么错？**
- Pydantic 不仅仅检查类型，还会**自动转换数据**
- 它是运行时验证，而 Python 类型提示只是静态检查的标注
- Pydantic 还提供序列化、反序列化、JSON Schema 生成等功能

**为什么人们容易这样错？**
因为 Pydantic 使用了类型注解语法，看起来和普通类型提示很像。但类型提示在 Python 中只是"建议"，运行时完全不生效。

**正确理解：**

```python
# Python 类型提示：运行时不检查
def greet(name: str) -> str:
    return f"Hello, {name}"

greet(123)  # ✅ 运行正常，输出 "Hello, 123"

# Pydantic：运行时验证 + 转换
from pydantic import BaseModel

class Greeting(BaseModel):
    name: str

g = Greeting(name=123)  # ✅ 自动转换为 "123"
print(g.name)  # "123" (字符串)

class StrictGreeting(BaseModel):
    model_config = {"strict": True}
    name: str

StrictGreeting(name=123)  # ❌ ValidationError: 严格模式不转换
```

---

### 误区2：model_dump() 和 dict() 是一样的 ❌

**为什么错？**
- `model_dump()` 是 Pydantic v2 的方法，功能更强大
- `dict()` 是 Python 内置，在 Pydantic v2 中已废弃
- `model_dump()` 支持排除字段、别名处理、序列化模式等高级功能

**为什么人们容易这样错？**
在 Pydantic v1 中确实使用 `.dict()` 方法，很多旧教程和代码还在使用这个写法。

**正确理解：**

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    name: str
    password: str = Field(exclude=True)  # 排除敏感字段
    age: int | None = None

user = User(name="Alice", password="secret123", age=None)

# Pydantic v2 正确写法
print(user.model_dump())
# {'name': 'Alice', 'age': None}  # password 被排除

print(user.model_dump(exclude_none=True))
# {'name': 'Alice'}  # 排除 None 值

print(user.model_dump(mode='json'))
# JSON 兼容模式，datetime 等会转为字符串

# ❌ 废弃写法（v1）
# print(user.dict())  # DeprecationWarning
```

---

### 误区3：default 和 default_factory 可以互换 ❌

**为什么错？**
- `default` 是静态默认值，所有实例共享同一个对象
- `default_factory` 是工厂函数，每次创建新对象
- 对于可变对象（list、dict），必须使用 `default_factory`

**为什么人们容易这样错？**
在普通 Python 类中，很多人习惯直接写 `items = []`，没意识到这是共享对象的陷阱。

**正确理解：**

```python
from pydantic import BaseModel, Field

# ❌ 错误：所有实例共享同一个 list
class BadConfig(BaseModel):
    items: list = []  # Pydantic 会警告这个写法

# ✅ 正确：每次创建新 list
class GoodConfig(BaseModel):
    items: list = Field(default_factory=list)

# 演示问题
config1 = GoodConfig()
config2 = GoodConfig()

config1.items.append("a")
print(config1.items)  # ['a']
print(config2.items)  # []  ✅ 互不影响

# 同样适用于 dict
class Settings(BaseModel):
    metadata: dict = Field(default_factory=dict)
    callbacks: list = Field(default_factory=list)
```

---

## 7. 【实战代码】

```python
"""
示例：构建一个 LLM 请求/响应模型系统
演示 Pydantic 在 LangChain 风格应用中的核心用法
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List, Union, Literal
from datetime import datetime
from enum import Enum

# ===== 1. 基础消息模型 =====
print("=== 1. 基础消息模型 ===")

class MessageRole(str, Enum):
    """消息角色枚举"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class ChatMessage(BaseModel):
    """聊天消息 - 类似 LangChain 的 BaseMessage"""
    role: MessageRole
    content: str
    name: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)

    @field_validator('content')
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('消息内容不能为空')
        return v.strip()

# 创建消息
msg = ChatMessage(role="user", content="  Hello, AI!  ")
print(f"消息: {msg.model_dump(exclude={'timestamp'})}")
# 输出: 消息: {'role': <MessageRole.USER: 'user'>, 'content': 'Hello, AI!', 'name': None}

# ===== 2. LLM 配置模型 =====
print("\n=== 2. LLM 配置模型 ===")

class LLMConfig(BaseModel):
    """LLM 配置 - 带详细验证"""
    model: str = Field(default="gpt-4", description="模型名称")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, gt=0, le=128000)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stop: List[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def check_temperature_top_p(self) -> 'LLMConfig':
        """temperature 和 top_p 不应同时修改"""
        if self.temperature != 0.7 and self.top_p != 1.0:
            raise ValueError('不建议同时修改 temperature 和 top_p')
        return self

config = LLMConfig(model="gpt-4-turbo", temperature=0.5)
print(f"配置: {config.model_dump()}")

# ===== 3. 请求/响应模型 =====
print("\n=== 3. 请求/响应模型 ===")

class ChatRequest(BaseModel):
    """聊天请求"""
    messages: List[ChatMessage]
    config: LLMConfig = Field(default_factory=LLMConfig)
    stream: bool = False

    @field_validator('messages')
    @classmethod
    def at_least_one_message(cls, v: List) -> List:
        if not v:
            raise ValueError('至少需要一条消息')
        return v

class TokenUsage(BaseModel):
    """Token 使用统计"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @model_validator(mode='after')
    def calculate_total(self) -> 'TokenUsage':
        if self.total_tokens == 0:
            object.__setattr__(self, 'total_tokens',
                              self.prompt_tokens + self.completion_tokens)
        return self

class ChatResponse(BaseModel):
    """聊天响应"""
    id: str
    message: ChatMessage
    usage: TokenUsage
    model: str
    created: datetime = Field(default_factory=datetime.now)

    @classmethod
    def from_llm_output(cls, raw_output: dict) -> 'ChatResponse':
        """从原始 LLM 输出创建响应"""
        return cls(
            id=raw_output.get('id', 'unknown'),
            message=ChatMessage(
                role="assistant",
                content=raw_output['choices'][0]['message']['content']
            ),
            usage=TokenUsage(**raw_output.get('usage', {})),
            model=raw_output.get('model', 'unknown')
        )

# 模拟 LLM 响应
raw = {
    "id": "chatcmpl-123",
    "model": "gpt-4",
    "choices": [{"message": {"role": "assistant", "content": "Hello! How can I help?"}}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 8}
}
response = ChatResponse.from_llm_output(raw)
print(f"响应: {response.message.content}")
print(f"Token: {response.usage.model_dump()}")

# ===== 4. 工具/函数调用模型 =====
print("\n=== 4. 工具/函数调用模型 ===")

class ToolParameter(BaseModel):
    """工具参数定义"""
    name: str
    type: Literal["string", "number", "boolean", "array", "object"]
    description: str
    required: bool = True
    enum: Optional[List[str]] = None

class Tool(BaseModel):
    """工具定义 - 类似 LangChain 的 Tool"""
    name: str = Field(pattern=r'^[a-z_][a-z0-9_]*$')  # 只允许小写和下划线
    description: str
    parameters: List[ToolParameter] = Field(default_factory=list)

    def to_openai_schema(self) -> dict:
        """转换为 OpenAI 函数调用格式"""
        properties = {}
        required = []
        for param in self.parameters:
            properties[param.name] = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                properties[param.name]["enum"] = param.enum
            if param.required:
                required.append(param.name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }

# 定义工具
search_tool = Tool(
    name="web_search",
    description="搜索互联网获取信息",
    parameters=[
        ToolParameter(name="query", type="string", description="搜索关键词"),
        ToolParameter(name="max_results", type="number", description="最大结果数", required=False)
    ]
)
print(f"OpenAI Schema: {search_tool.to_openai_schema()}")

# ===== 5. 序列化和反序列化 =====
print("\n=== 5. 序列化和反序列化 ===")

# 完整请求构建
request = ChatRequest(
    messages=[
        ChatMessage(role="system", content="你是一个有帮助的助手"),
        ChatMessage(role="user", content="今天天气怎么样？")
    ],
    config=LLMConfig(temperature=0.3),
    stream=False
)

# 序列化为 JSON（可以直接发送给 API）
json_str = request.model_dump_json(indent=2, exclude={'messages': {'__all__': {'timestamp'}}})
print(f"JSON 请求:\n{json_str}")

# 从 JSON 反序列化
restored = ChatRequest.model_validate_json(json_str)
print(f"\n恢复的请求包含 {len(restored.messages)} 条消息")
```

**运行输出示例：**
```
=== 1. 基础消息模型 ===
消息: {'role': <MessageRole.USER: 'user'>, 'content': 'Hello, AI!', 'name': None}

=== 2. LLM 配置模型 ===
配置: {'model': 'gpt-4-turbo', 'temperature': 0.5, 'max_tokens': 1000, 'top_p': 1.0, 'stop': []}

=== 3. 请求/响应模型 ===
响应: Hello! How can I help?
Token: {'prompt_tokens': 10, 'completion_tokens': 8, 'total_tokens': 18}

=== 4. 工具/函数调用模型 ===
OpenAI Schema: {'type': 'function', 'function': {'name': 'web_search', ...}}

=== 5. 序列化和反序列化 ===
JSON 请求:
{
  "messages": [...],
  "config": {"model": "gpt-4", "temperature": 0.3, ...},
  "stream": false
}

恢复的请求包含 2 条消息
```

---

## 8. 【面试必问】

### 问题："Pydantic 的作用是什么？它和 Python 类型提示有什么区别？"

**普通回答（❌ 不出彩）：**
"Pydantic 是一个数据验证库，可以验证数据类型，比 Python 类型提示更强大。"

**出彩回答（✅ 推荐）：**

> **Pydantic 有三层作用：**
>
> 1. **运行时类型验证**：Python 的类型提示只是"注释"，运行时不会检查。而 Pydantic 在实例化时真正验证类型，发现错误立即抛出 ValidationError。
>
> 2. **智能数据转换**：Pydantic 不仅验证，还会尝试类型转换。比如把字符串 "123" 自动转为整数 123，把 "true" 转为布尔值 True。
>
> 3. **序列化基础设施**：提供 `model_dump()`、`model_dump_json()`、`model_validate()` 等方法，是 JSON 序列化的标准方案。
>
> **和类型提示的核心区别**：类型提示是静态的、给 IDE 和 mypy 看的；Pydantic 是动态的、运行时生效的。一个是"建议"，一个是"强制"。
>
> **在 LangChain 中的应用**：LangChain 的所有核心数据结构（Message、Runnable、Config）都继承自 Pydantic BaseModel。这使得 LLM 的非结构化输出可以被可靠地转换为类型安全的 Python 对象。

**为什么这个回答出彩？**
1. ✅ 分层回答，结构清晰
2. ✅ 明确了"静态 vs 运行时"的核心区别
3. ✅ 联系了实际应用（LangChain）
4. ✅ 展示了对 LLM 应用开发的理解

---

### 问题："Field 和 field_validator 分别什么时候用？"

**普通回答（❌ 不出彩）：**
"Field 用来定义字段，validator 用来验证字段。"

**出彩回答（✅ 推荐）：**

> **Field 适用于：**
> - 声明式的简单约束（数值范围、字符串长度、正则匹配）
> - 设置默认值和默认工厂
> - 添加元数据（description、alias）
> - 控制序列化行为（exclude、include）
>
> **field_validator 适用于：**
> - 复杂的自定义逻辑（如：密码强度检查）
> - 需要访问其他字段的验证
> - 数据清洗和转换（如：去除空白、格式化）
> - 依赖外部资源的验证（如：检查数据库唯一性）
>
> **选择原则**：能用 Field 声明式解决的，优先用 Field，因为更简洁、性能更好、生成的 JSON Schema 更准确。需要命令式逻辑时才用 validator。
>
> ```python
> # Field: 声明式
> age: int = Field(ge=0, le=150)
>
> # validator: 命令式（需要复杂逻辑）
> @field_validator('email')
> def normalize_email(cls, v):
>     return v.lower().strip()
> ```

---

## 9. 【化骨绵掌】

### 卡片1：Pydantic 是什么？ 🎯

**一句话：** Pydantic 是 Python 的运行时数据验证库，通过类型注解定义数据模型。

**举例：**
```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

user = User(name="Alice", age="25")  # "25" 自动转为 int
```

**应用：** LangChain 的所有数据结构（Message、Config、Tool）都基于 Pydantic。

---

### 卡片2：BaseModel 三板斧 📐

**一句话：** BaseModel 提供三个核心能力：创建、序列化、反序列化。

**举例：**
```python
# 创建
user = User(name="Bob", age=30)

# 序列化
data = user.model_dump()          # → dict
json_str = user.model_dump_json() # → JSON string

# 反序列化
user2 = User.model_validate(data)
user3 = User.model_validate_json(json_str)
```

**应用：** LangChain 消息的 JSON 序列化和反序列化。

---

### 卡片3：Field 配置字段 🔧

**一句话：** Field() 用于配置字段的默认值、约束条件、元数据。

**举例：**
```python
from pydantic import Field

class Config(BaseModel):
    temp: float = Field(default=0.7, ge=0, le=2, description="温度")
    tags: list = Field(default_factory=list)
```

**应用：** LangChain 的 RunnableConfig 使用 Field 定义各种可选参数。

---

### 卡片4：数值约束 gt/ge/lt/le 📊

**一句话：** 用简单的参数约束数值范围，无需写验证器。

**举例：**
```python
class Score(BaseModel):
    value: int = Field(ge=0, le=100)  # 0-100
    # gt: greater than (>)
    # ge: greater than or equal (>=)
    # lt: less than (<)
    # le: less than or equal (<=)
```

**应用：** LLM temperature 必须在 0-2 之间，用 `Field(ge=0, le=2)` 约束。

---

### 卡片5：field_validator 字段验证器 ✅

**一句话：** 自定义单个字段的验证逻辑，可以转换或拒绝数据。

**举例：**
```python
from pydantic import field_validator

class Email(BaseModel):
    address: str

    @field_validator('address')
    @classmethod
    def must_contain_at(cls, v):
        if '@' not in v:
            raise ValueError('必须包含 @')
        return v.lower()  # 返回转换后的值
```

**应用：** LangChain PromptTemplate 验证模板变量是否正确。

---

### 卡片6：model_validator 模型验证器 🔄

**一句话：** 验证整个模型，可以在字段处理前（before）或后（after）执行。

**举例：**
```python
from pydantic import model_validator

class DateRange(BaseModel):
    start: date
    end: date

    @model_validator(mode='after')
    def end_after_start(self):
        if self.end < self.start:
            raise ValueError('结束日期必须在开始日期之后')
        return self
```

**应用：** 验证 LLM 配置的参数组合是否合理。

---

### 卡片7：Optional 和默认值 🎨

**一句话：** Optional[T] 表示可以是 T 或 None，配合默认值使用。

**举例：**
```python
from typing import Optional

class Message(BaseModel):
    content: str                      # 必填
    name: Optional[str] = None        # 可选，默认 None
    metadata: dict = {}               # 可选，默认空字典（注意！）
    tags: list = Field(default_factory=list)  # 正确的可变默认值
```

**应用：** LangChain 消息的 name、metadata 等可选字段。

---

### 卡片8：枚举类型 Enum 🎭

**一句话：** 用 Enum 限制字段只能是预定义的几个值。

**举例：**
```python
from enum import Enum

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class Message(BaseModel):
    role: Role
    content: str

msg = Message(role="user", content="Hi")  # 自动转为 Role.USER
```

**应用：** LangChain 消息角色限制为 system/user/assistant。

---

### 卡片9：嵌套模型 🏗️

**一句话：** Pydantic 模型可以嵌套，自动递归验证。

**举例：**
```python
class Address(BaseModel):
    city: str
    street: str

class User(BaseModel):
    name: str
    address: Address  # 嵌套模型

user = User(
    name="Alice",
    address={"city": "NYC", "street": "5th Ave"}  # 自动转换
)
```

**应用：** LangChain 的 ChatRequest 包含嵌套的 Message 列表和 Config。

---

### 卡片10：在 LangChain 源码中的实际应用 ⭐

**一句话：** LangChain 核心类都继承 BaseModel，实现类型安全的组件系统。

**举例：**
```python
# langchain_core/messages/base.py
class BaseMessage(BaseModel):
    content: Union[str, List]
    additional_kwargs: dict = Field(default_factory=dict)
    type: str
    name: Optional[str] = None

# langchain_core/runnables/base.py
class RunnableConfig(TypedDict):
    tags: List[str]
    metadata: Dict[str, Any]
    callbacks: Callbacks
```

**应用：** 理解这个模式后，就能读懂 LangChain 源码中 90% 的数据结构定义。

---

## 10. 【一句话总结】

**Pydantic 是 Python 的运行时数据验证框架，通过 BaseModel 和类型注解实现类型安全、自动转换和序列化，是 LangChain 所有数据模型的基础设施。**

---

## 📚 学习检查清单

- [ ] 能够定义一个 BaseModel 子类
- [ ] 理解 Field 的常用参数（default、ge/le、description）
- [ ] 会使用 field_validator 自定义验证
- [ ] 理解 model_dump() 和 model_validate() 的用法
- [ ] 知道 default 和 default_factory 的区别
- [ ] 能够阅读 LangChain 源码中的 Pydantic 模型定义

## 🔗 下一步学习

- **类型提示与泛型**：理解 TypeVar、Generic 如何与 Pydantic 配合
- **异步编程 async/await**：LangChain 的异步接口实现
- **Runnable 协议**：LangChain 核心抽象，大量使用 Pydantic

---

**版本：** v1.0
**最后更新：** 2025-12-12
