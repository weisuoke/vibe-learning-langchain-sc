# 抽象类与接口 (ABC模块)

> 原子化知识点 | 软件工程基础 | LangChain 源码学习前置知识

---

## 1. 【30字核心】

**Python ABC 模块提供抽象基类机制，通过定义抽象方法强制子类实现特定接口，是 LangChain 所有 Base 类的实现基础。**

---

## 2. 【第一性原理】

### 什么是第一性原理？

**第一性原理**：回到事物最基本的真理，从源头思考问题

### 抽象类与接口的第一性原理 🎯

#### 1. 最基础的定义

**抽象类 = 不能实例化的类 + 必须被子类实现的方法**

仅此而已！没有更基础的了。

- **抽象类**：定义"做什么"的契约，不关心"怎么做"
- **抽象方法**：只有声明没有实现，子类必须实现
- **接口**：纯抽象类，所有方法都是抽象的

#### 2. 为什么需要抽象类？

**核心问题：如何确保所有子类都实现特定的方法？**

```python
# 没有抽象类：无法强制约束
class BaseChatModel:
    def invoke(self, input: str) -> str:
        # 子类应该覆盖这个方法，但没有强制！
        raise NotImplementedError("子类必须实现")

class ChatOpenAI(BaseChatModel):
    pass  # 忘记实现 invoke 了，Python 不会报错！

# 运行时才发现错误
model = ChatOpenAI()
model.invoke("Hello")  # NotImplementedError: 子类必须实现

# 问题：
# 1. 编码时没有任何提示
# 2. 必须运行到那一行才报错
# 3. IDE 无法提供代码补全
# 4. 代码审查时容易漏掉
```

```python
# 使用 ABC：编译时强制约束
from abc import ABC, abstractmethod

class BaseChatModel(ABC):
    @abstractmethod
    def invoke(self, input: str) -> str:
        """子类必须实现这个方法"""
        pass

class ChatOpenAI(BaseChatModel):
    pass  # 忘记实现 invoke

# 实例化时就会报错！
model = ChatOpenAI()
# TypeError: Can't instantiate abstract class ChatOpenAI
#            with abstract method invoke

# 优势：
# 1. 实例化时就报错，不用等到调用
# 2. IDE 会提示缺少实现
# 3. 更早发现问题
# 4. 形成明确的接口契约
```

#### 3. 抽象类的三层价值

##### 价值1：接口契约 - 定义"必须做什么"

```python
from abc import ABC, abstractmethod

class Runnable(ABC):
    """定义可运行组件的契约"""

    @abstractmethod
    def invoke(self, input):
        """必须实现：同步调用"""
        pass

    @abstractmethod
    def batch(self, inputs):
        """必须实现：批量调用"""
        pass

    @abstractmethod
    def stream(self, input):
        """必须实现：流式调用"""
        pass

# 任何 Runnable 子类都必须实现这三个方法
# 使用者可以放心调用，不用担心"没有这个方法"
```

##### 价值2：多态基础 - 统一的类型

```python
from typing import List

def run_all(runnables: List[Runnable], input: str):
    """接受任何 Runnable 实现"""
    for r in runnables:
        print(r.invoke(input))  # 多态调用

# 不管传入什么具体类型，只要是 Runnable 就行
run_all([ChatOpenAI(), ChatAnthropic(), LocalLLM()], "Hello")
```

##### 价值3：部分实现 - 模板方法模式

```python
class BaseChatModel(ABC):
    """可以提供部分实现"""

    def invoke(self, input: str) -> str:
        """模板方法：固定流程"""
        validated = self._validate(input)
        result = self._generate(validated)  # 子类实现
        return self._format(result)

    def _validate(self, input: str) -> str:
        """固定实现：验证输入"""
        return input.strip()

    def _format(self, result: str) -> str:
        """固定实现：格式化输出"""
        return result

    @abstractmethod
    def _generate(self, input: str) -> str:
        """抽象方法：子类必须实现"""
        pass
```

#### 4. 从第一性原理推导 LangChain 源码应用

**推理链：**

```
1. LangChain 需要支持多种 LLM（OpenAI、Anthropic、本地模型...）
   ↓
2. 每种 LLM 的调用方式不同，但都需要"可调用"
   ↓
3. 需要统一的接口，让使用者不关心具体实现
   ↓
4. 使用抽象类定义接口契约
   ↓
5. BaseChatModel 定义所有聊天模型必须实现的方法
   ↓
6. 使用 @abstractmethod 标记必须实现的方法
   ↓
7. ChatOpenAI、ChatAnthropic 等继承并实现
   ↓
8. 用户代码可以用 BaseChatModel 类型接收任何实现
   ↓
9. LCEL 管道可以组合任何 Runnable 实现
```

#### 5. 一句话总结第一性原理

**抽象类通过定义"必须实现的方法"形成接口契约，让 LangChain 能够用统一的方式调用不同的 LLM 实现，是实现多态和可扩展架构的基础。**

---

## 3. 【核心概念（全面覆盖）】

### 核心概念1：ABC 和 abstractmethod 🏗️

**ABC 是 Abstract Base Class 的缩写，abstractmethod 标记抽象方法**

```python
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Optional

class BaseDocumentLoader(ABC):
    """
    文档加载器抽象基类 - 类似 LangChain 的 BaseLoader

    ABC 的作用：
    1. 标记这个类不能直接实例化
    2. 包含的 abstractmethod 必须被子类实现
    """

    # ===== 抽象方法：子类必须实现 =====

    @abstractmethod
    def load(self) -> List[Dict[str, Any]]:
        """
        加载文档

        这是一个抽象方法，子类必须实现。
        不实现就不能实例化子类。
        """
        pass

    @abstractmethod
    def lazy_load(self):
        """
        惰性加载文档

        返回生成器，按需加载
        """
        pass

    # ===== 具体方法：提供默认实现 =====

    def load_and_split(self, splitter=None) -> List[Dict[str, Any]]:
        """
        加载并分割文档

        这是一个具体方法，有默认实现。
        子类可以覆盖，也可以不覆盖。
        """
        docs = self.load()
        if splitter:
            return splitter.split(docs)
        return docs

# 尝试实例化抽象类
try:
    loader = BaseDocumentLoader()
except TypeError as e:
    print(f"错误: {e}")
    # 错误: Can't instantiate abstract class BaseDocumentLoader
    #       with abstract methods lazy_load, load

# ===== 正确的实现 =====

class TextFileLoader(BaseDocumentLoader):
    """文本文件加载器 - 具体实现类"""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[Dict[str, Any]]:
        """实现抽象方法"""
        with open(self.file_path, 'r') as f:
            content = f.read()
        return [{"content": content, "source": self.file_path}]

    def lazy_load(self):
        """实现抽象方法"""
        with open(self.file_path, 'r') as f:
            for line in f:
                yield {"content": line.strip(), "source": self.file_path}

# 现在可以实例化了
loader = TextFileLoader("example.txt")
```

**ABC 的关键特征：**

| 特征 | 说明 | 示例 |
|------|------|------|
| 不能实例化 | 只能被继承 | `ABC()` 会报错 |
| 强制实现 | 子类必须实现所有 abstractmethod | 否则子类也不能实例化 |
| 可以有具体方法 | 不是所有方法都必须抽象 | `load_and_split` 有默认实现 |
| 支持多重继承 | 可以继承多个 ABC | `class A(ABC1, ABC2)` |

**在 LangChain 源码中的应用：**

```python
# langchain_core/language_models/base.py 简化版
from abc import ABC, abstractmethod

class BaseLanguageModel(ABC):
    """所有语言模型的抽象基类"""

    @abstractmethod
    def generate_prompt(self, prompts, stop=None):
        """生成响应 - 必须实现"""
        pass

    @abstractmethod
    def predict(self, text: str, stop=None) -> str:
        """预测 - 必须实现"""
        pass

    @property
    @abstractmethod
    def _llm_type(self) -> str:
        """模型类型 - 必须实现"""
        pass
```

---

### 核心概念2：抽象属性（abstractproperty） 📐

**使用 @property 和 @abstractmethod 组合定义抽象属性**

```python
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseModel(ABC):
    """
    模型基类 - 展示抽象属性

    抽象属性强制子类定义某些属性
    """

    # ===== 抽象属性：子类必须实现 =====

    @property
    @abstractmethod
    def model_name(self) -> str:
        """模型名称 - 必须实现"""
        pass

    @property
    @abstractmethod
    def model_type(self) -> str:
        """模型类型 - 必须实现"""
        pass

    # ===== 抽象属性 setter =====

    @property
    @abstractmethod
    def temperature(self) -> float:
        """温度参数"""
        pass

    @temperature.setter
    @abstractmethod
    def temperature(self, value: float):
        """设置温度"""
        pass

    # ===== 具体属性 =====

    @property
    def model_info(self) -> Dict[str, Any]:
        """模型信息 - 有默认实现"""
        return {
            "name": self.model_name,
            "type": self.model_type,
        }

class ChatOpenAI(BaseModel):
    """OpenAI 聊天模型"""

    def __init__(self, model: str = "gpt-4", temp: float = 0.7):
        self._model_name = model
        self._temperature = temp

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_type(self) -> str:
        return "chat"

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float):
        if not 0 <= value <= 2:
            raise ValueError("温度必须在 0-2 之间")
        self._temperature = value

# 使用
model = ChatOpenAI()
print(f"模型: {model.model_name}")     # gpt-4
print(f"类型: {model.model_type}")     # chat
print(f"温度: {model.temperature}")    # 0.7

model.temperature = 0.5  # 使用 setter
print(f"新温度: {model.temperature}")  # 0.5

print(f"信息: {model.model_info}")     # {'name': 'gpt-4', 'type': 'chat'}
```

**抽象属性 vs 抽象方法：**

| 特性 | 抽象属性 | 抽象方法 |
|------|---------|---------|
| 定义方式 | `@property + @abstractmethod` | `@abstractmethod` |
| 访问方式 | `obj.prop` | `obj.method()` |
| 适用场景 | 固定的配置信息 | 需要执行的操作 |
| 示例 | `model_name`, `temperature` | `invoke()`, `generate()` |

**在 LangChain 源码中的应用：**

```python
# langchain_core/language_models/chat_models.py 简化版
class BaseChatModel(ABC):
    """聊天模型基类"""

    @property
    @abstractmethod
    def _llm_type(self) -> str:
        """返回 LLM 类型标识"""
        pass

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """返回标识参数（有默认实现）"""
        return {}
```

---

### 核心概念3：接口与协议（Interface & Protocol） 🔧

**Python 支持两种方式定义接口：ABC（名义类型）和 Protocol（结构类型）**

```python
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

# ===== 方式1：ABC（名义类型） =====
# 子类必须显式继承才算实现了接口

class Runnable(ABC):
    """Runnable 接口（ABC 版本）"""

    @abstractmethod
    def invoke(self, input: str) -> str:
        pass

class MyRunnable(Runnable):  # 必须继承
    def invoke(self, input: str) -> str:
        return f"Processed: {input}"

# ===== 方式2：Protocol（结构类型） =====
# 只要有相同的方法签名就算实现了接口（鸭子类型）

@runtime_checkable  # 允许用 isinstance 检查
class RunnableProtocol(Protocol):
    """Runnable 协议（Protocol 版本）"""

    def invoke(self, input: str) -> str:
        ...  # 用 ... 而不是 pass

class AnotherRunnable:  # 不需要继承！
    def invoke(self, input: str) -> str:
        return f"Also processed: {input}"

# 检查是否符合协议
obj = AnotherRunnable()
print(isinstance(obj, RunnableProtocol))  # True（结构匹配）

# ===== 对比 =====

class NotRunnable:
    def call(self, input: str) -> str:  # 方法名不对
        return input

not_runnable = NotRunnable()
print(isinstance(not_runnable, RunnableProtocol))  # False

# ===== 完整的 Protocol 示例 =====

@runtime_checkable
class Serializable(Protocol):
    """可序列化协议"""

    def to_dict(self) -> dict:
        """转换为字典"""
        ...

    @classmethod
    def from_dict(cls, data: dict) -> 'Serializable':
        """从字典创建"""
        ...

class Config:
    """配置类 - 自动符合 Serializable 协议"""

    def __init__(self, name: str, value: int):
        self.name = name
        self.value = value

    def to_dict(self) -> dict:
        return {"name": self.name, "value": self.value}

    @classmethod
    def from_dict(cls, data: dict) -> 'Config':
        return cls(data["name"], data["value"])

# Config 没有继承任何东西，但符合 Serializable 协议
config = Config("test", 42)
print(isinstance(config, Serializable))  # True
```

**ABC vs Protocol：**

| 特性 | ABC（名义类型） | Protocol（结构类型） |
|------|---------------|-------------------|
| 继承要求 | 必须显式继承 | 不需要继承 |
| 类型检查 | isinstance 可用 | 需要 @runtime_checkable |
| 适用场景 | 明确的继承层次 | 鸭子类型、第三方类 |
| LangChain 使用 | BaseChatModel, BaseRetriever | 部分工具类型 |

**在 LangChain 源码中的应用：**

```python
# langchain_core/runnables/base.py
# LangChain 主要使用 ABC，因为需要明确的继承层次

from abc import ABC, abstractmethod

class Runnable(ABC):
    """所有可运行组件的基类"""

    @abstractmethod
    def invoke(self, input, config=None):
        pass

    @abstractmethod
    def batch(self, inputs, config=None):
        pass

    @abstractmethod
    def stream(self, input, config=None):
        pass

# 但也支持 Protocol 风格的检查
def is_runnable(obj) -> bool:
    """检查对象是否可运行"""
    return hasattr(obj, 'invoke') and callable(obj.invoke)
```

---

### 扩展概念4：抽象类的继承层次 🏛️

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

# ===== 多层继承结构 =====

class BaseComponent(ABC):
    """最顶层的抽象基类"""

    @abstractmethod
    def get_name(self) -> str:
        pass

class Runnable(BaseComponent):
    """可运行组件 - 中间层抽象类"""

    @abstractmethod
    def invoke(self, input: Any) -> Any:
        pass

    @abstractmethod
    def batch(self, inputs: List[Any]) -> List[Any]:
        pass

    # 提供默认实现
    def get_name(self) -> str:
        return self.__class__.__name__

class BaseLanguageModel(Runnable):
    """语言模型基类 - 更具体的抽象类"""

    @abstractmethod
    def _generate(self, prompts: List[str]) -> str:
        pass

    # 实现父类的抽象方法
    def invoke(self, input: str) -> str:
        return self._generate([input])

    def batch(self, inputs: List[str]) -> List[str]:
        return [self.invoke(i) for i in inputs]

class BaseChatModel(BaseLanguageModel):
    """聊天模型基类 - 最具体的抽象类"""

    @abstractmethod
    def _generate_chat(self, messages: List[Dict]) -> str:
        pass

    def _generate(self, prompts: List[str]) -> str:
        # 将 prompt 转换为 message 格式
        messages = [{"role": "user", "content": p} for p in prompts]
        return self._generate_chat(messages)

# ===== 具体实现类 =====

class ChatOpenAI(BaseChatModel):
    """OpenAI 聊天模型 - 具体实现"""

    def __init__(self, model: str = "gpt-4"):
        self.model = model

    def _generate_chat(self, messages: List[Dict]) -> str:
        # 实现具体的 API 调用
        return f"[{self.model}] Response to: {messages[-1]['content']}"

# 继承层次：
# BaseComponent (抽象)
#     ↓
# Runnable (抽象)
#     ↓
# BaseLanguageModel (抽象)
#     ↓
# BaseChatModel (抽象)
#     ↓
# ChatOpenAI (具体)

# 使用
model = ChatOpenAI()
print(model.get_name())      # ChatOpenAI（继承自 Runnable）
print(model.invoke("Hello")) # [gpt-4] Response to: Hello
```

---

### 扩展概念5：Mixin 与多重继承 🔀

```python
from abc import ABC, abstractmethod
from typing import Any, Dict

# ===== Mixin 类：提供可复用的功能 =====

class SerializableMixin:
    """可序列化 Mixin"""

    def to_dict(self) -> Dict[str, Any]:
        return {"class": self.__class__.__name__, "data": self.__dict__}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        obj = cls.__new__(cls)
        obj.__dict__.update(data.get("data", {}))
        return obj

class LoggableMixin:
    """可日志 Mixin"""

    def log(self, message: str):
        print(f"[{self.__class__.__name__}] {message}")

class CacheableMixin:
    """可缓存 Mixin"""

    _cache: Dict[str, Any] = {}

    def get_cached(self, key: str) -> Any:
        return self._cache.get(key)

    def set_cached(self, key: str, value: Any):
        self._cache[key] = value

# ===== 抽象基类 =====

class BaseRetriever(ABC):
    """检索器抽象基类"""

    @abstractmethod
    def retrieve(self, query: str):
        pass

# ===== 组合 Mixin 和抽象类 =====

class SmartRetriever(BaseRetriever, SerializableMixin, LoggableMixin, CacheableMixin):
    """
    智能检索器 - 组合多个 Mixin

    继承顺序很重要（MRO）
    """

    def __init__(self, name: str):
        self.name = name

    def retrieve(self, query: str):
        # 使用缓存（CacheableMixin）
        cached = self.get_cached(query)
        if cached:
            self.log(f"缓存命中: {query}")  # LoggableMixin
            return cached

        # 执行检索
        self.log(f"执行检索: {query}")
        result = f"Results for: {query}"

        # 存入缓存
        self.set_cached(query, result)
        return result

# 使用
retriever = SmartRetriever("MyRetriever")
print(retriever.retrieve("Python"))  # 执行检索
print(retriever.retrieve("Python"))  # 缓存命中

# 序列化
data = retriever.to_dict()
print(data)  # {'class': 'SmartRetriever', 'data': {'name': 'MyRetriever'}}
```

---

## 4. 【最小可用】

掌握以下内容，就能理解 LangChain 源码中的抽象类设计：

### 4.1 定义抽象基类

```python
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """抽象基类"""

    @abstractmethod
    def invoke(self, input: str) -> str:
        """子类必须实现"""
        pass
```

### 4.2 实现抽象类

```python
class ChatOpenAI(BaseModel):
    """具体实现类"""

    def invoke(self, input: str) -> str:
        return f"Response: {input}"

# 可以实例化
model = ChatOpenAI()
print(model.invoke("Hello"))  # Response: Hello
```

### 4.3 抽象属性

```python
class BaseModel(ABC):

    @property
    @abstractmethod
    def model_name(self) -> str:
        """子类必须定义这个属性"""
        pass

class ChatOpenAI(BaseModel):

    @property
    def model_name(self) -> str:
        return "gpt-4"
```

### 4.4 检查实例类型

```python
model = ChatOpenAI()

# isinstance 检查
print(isinstance(model, BaseModel))  # True
print(isinstance(model, ChatOpenAI)) # True

# 类型标注
def run_model(model: BaseModel):
    return model.invoke("test")
```

**这些知识足以：**
- 理解 LangChain 中 `BaseChatModel`、`BaseRetriever` 等基类的设计
- 实现自定义的 LangChain 组件
- 理解为什么可以用 `BaseModel` 类型接收不同的 LLM 实现
- 阅读 LangChain 源码时识别抽象类和接口

---

## 5. 【1个类比】（双轨制）

### 类比1：抽象类 = 合同/契约

#### 🎨 前端视角：TypeScript Interface

抽象类就像 TypeScript 的 interface，定义必须实现的方法。

```typescript
// TypeScript interface = Python ABC
interface ChatModel {
  invoke(input: string): string;
  stream(input: string): AsyncGenerator<string>;
  readonly modelName: string;  // 只读属性
}

// 实现接口
class ChatOpenAI implements ChatModel {
  readonly modelName = "gpt-4";

  invoke(input: string): string {
    return `Response: ${input}`;
  }

  async *stream(input: string) {
    yield "Hello";
    yield " World";
  }
}

// 如果漏了方法，TypeScript 会报错
class BadModel implements ChatModel {
  // Error: Property 'invoke' is missing
}
```

```python
# Python 对应：ABC
from abc import ABC, abstractmethod

class ChatModel(ABC):
    @abstractmethod
    def invoke(self, input: str) -> str:
        pass

    @abstractmethod
    def stream(self, input: str):
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

class ChatOpenAI(ChatModel):
    @property
    def model_name(self) -> str:
        return "gpt-4"

    def invoke(self, input: str) -> str:
        return f"Response: {input}"

    def stream(self, input: str):
        yield "Hello"
        yield " World"
```

#### 🧒 小朋友视角：工作合同

抽象类就像工作合同，规定你必须做什么。

**生活例子：**
```
快递员工作合同（抽象类）：

┌─────────────────────────────────────┐
│           快递员合同                 │
│                                     │
│ 必须做的事情（抽象方法）：            │
│ □ 取件：去仓库取包裹                 │
│ □ 送件：把包裹送到客户手上            │
│ □ 签收：让客户签字确认               │
│                                     │
│ 如果不做这些，就不是快递员！          │
└─────────────────────────────────────┘

小明签了合同成为快递员：
✓ 取件：骑电动车去取
✓ 送件：电话联系客户送上门
✓ 签收：用手机 APP 让客户签字

小红签了合同成为快递员：
✓ 取件：开货车去取（方式不同，但做了）
✓ 送件：放在快递柜（方式不同，但做了）
✓ 签收：短信确认

合同规定"做什么"，不管"怎么做"！
```

---

### 类比2：抽象方法 vs 具体方法

#### 🎨 前端视角：Required vs Optional

```typescript
// TypeScript 中的必选和可选
interface User {
  id: string;         // 必选（抽象方法）
  name: string;       // 必选
  email?: string;     // 可选（具体方法，有默认行为）
  avatar?: string;    // 可选
}

// 必选的必须提供
const user: User = {
  id: "1",
  name: "Tom",
  // email 和 avatar 可以不提供
};
```

```python
# Python 对应
from abc import ABC, abstractmethod

class User(ABC):
    @abstractmethod
    def get_id(self) -> str:
        """必须实现"""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """必须实现"""
        pass

    def get_email(self) -> str:
        """有默认实现，可以不覆盖"""
        return ""

    def get_avatar(self) -> str:
        """有默认实现，可以不覆盖"""
        return "default.png"
```

#### 🧒 小朋友视角：考试必答题和选做题

```
考试试卷（抽象类）：

┌─────────────────────────────────────┐
│           数学考试                   │
│                                     │
│ 必答题（抽象方法）：                  │
│ 1. 计算 1+1=? （10分）★必须做        │
│ 2. 计算 2×3=? （10分）★必须做        │
│                                     │
│ 选做题（具体方法，有默认分）：         │
│ 3. 证明勾股定理（附加10分）           │
│    （不做也行，默认得0分）            │
└─────────────────────────────────────┘

小明的答卷：
✓ 第1题：2 （必须答）
✓ 第2题：6 （必须答）
✗ 第3题：没做（选做题，可以不做）

如果必答题不做 = 试卷无效（不能实例化）
```

---

### 类比3：继承层次 = 职业分类

#### 🎨 前端视角：React 组件层次

```typescript
// React 组件的继承层次
abstract class Component {
  abstract render(): ReactNode;
}

abstract class PureComponent extends Component {
  shouldComponentUpdate(nextProps, nextState): boolean {
    // 默认浅比较实现
  }
}

class MyComponent extends PureComponent {
  render() {
    return <div>Hello</div>;
  }
}
```

```python
# Python 对应：LangChain 的模型层次
class Runnable(ABC):           # 最顶层
    @abstractmethod
    def invoke(self): pass

class BaseLanguageModel(Runnable):  # 中间层
    @abstractmethod
    def generate(self): pass

class BaseChatModel(BaseLanguageModel):  # 再具体
    @abstractmethod
    def _generate_chat(self): pass

class ChatOpenAI(BaseChatModel):  # 具体实现
    def _generate_chat(self): ...
```

#### 🧒 小朋友视角：职业分类

```
职业分类（继承层次）：

                    人（最顶层）
                      │
              ┌───────┴───────┐
              │               │
          工作者            学生
              │
      ┌───────┼───────┐
      │       │       │
    医生    教师    程序员
      │
  ┌───┴───┐
  │       │
内科医生 外科医生

每一层都有自己的"必须会的技能"：
- 人：呼吸、吃饭（基本）
- 工作者：按时上班、领工资（更具体）
- 医生：看病、开药（再具体）
- 内科医生：内科检查、内科治疗（最具体）

越往下，要求越具体！
```

---

### 类比4：Protocol = 鸭子类型

#### 🎨 前端视角：TypeScript 鸭子类型

```typescript
// TypeScript 的结构类型（鸭子类型）
interface Quackable {
  quack(): void;
}

// 不需要显式 implements
class Duck {
  quack() {
    console.log("Quack!");
  }
}

class Robot {
  quack() {
    console.log("Beep boop quack!");
  }
}

// 只要有 quack 方法就行
function makeQuack(thing: Quackable) {
  thing.quack();
}

makeQuack(new Duck());   // OK
makeQuack(new Robot());  // OK
```

```python
# Python 对应：Protocol
from typing import Protocol, runtime_checkable

@runtime_checkable
class Quackable(Protocol):
    def quack(self) -> None:
        ...

class Duck:
    def quack(self):
        print("Quack!")

class Robot:
    def quack(self):
        print("Beep boop quack!")

# 检查是否符合协议
print(isinstance(Duck(), Quackable))   # True
print(isinstance(Robot(), Quackable))  # True
```

#### 🧒 小朋友视角：看起来像就是

```
鸭子类型（Protocol）：

"如果它走路像鸭子，叫声像鸭子，那它就是鸭子！"

┌─────────────────────────────────────┐
│              鸭子协议                │
│                                     │
│   能做的事：                         │
│   - 会嘎嘎叫                        │
│   - 会游泳                          │
│   - 会走路                          │
└─────────────────────────────────────┘

真鸭子：✓ 嘎嘎叫 ✓ 游泳 ✓ 走路 → 是鸭子！
玩具鸭：✓ 嘎嘎叫 ✓ 游泳 ✓ 走路 → 是鸭子！
机器鸭：✓ 嘎嘎叫 ✓ 游泳 ✓ 走路 → 是鸭子！

不管你是什么，只要会这些就行！
（不需要"继承"鸭子基因）
```

---

### 类比总结表

| ABC 概念 | 前端类比 | 小朋友类比 |
|---------|---------|-----------|
| 抽象类（ABC） | TypeScript interface | 工作合同 |
| 抽象方法 | required 属性 | 考试必答题 |
| 具体方法 | optional 属性 + 默认值 | 考试选做题 |
| 继承层次 | React 组件层次 | 职业分类 |
| Protocol | 鸭子类型 / 结构类型 | "像鸭子就是鸭子" |
| @abstractmethod | 接口方法声明 | ★必填项 |
| 不能实例化 | 抽象类不能 new | 合同本身不能工作 |

---

## 6. 【反直觉点】

### 误区1：抽象类不能有任何实现 ❌

**为什么错？**
- 抽象类**可以有**具体方法（非抽象方法）
- 只有标记了 `@abstractmethod` 的方法才必须被子类实现
- 抽象类常用于模板方法模式：部分固定实现 + 部分抽象

**为什么人们容易这样错？**
因为"抽象"这个词容易让人理解为"全部都是抽象的"，但实际上抽象类是"至少有一个抽象方法的类"。

**正确理解：**

```python
from abc import ABC, abstractmethod

class BaseChatModel(ABC):
    """抽象类可以有具体实现！"""

    # 具体方法：有完整实现
    def invoke(self, input: str) -> str:
        """模板方法：固定流程"""
        validated = self._validate(input)     # 具体实现
        result = self._generate(validated)     # 抽象，子类实现
        return self._format(result)            # 具体实现

    def _validate(self, input: str) -> str:
        """具体方法"""
        return input.strip()

    def _format(self, result: str) -> str:
        """具体方法"""
        return result

    # 抽象方法：没有实现，子类必须实现
    @abstractmethod
    def _generate(self, input: str) -> str:
        """这是唯一必须子类实现的"""
        pass

# 子类只需要实现抽象方法
class ChatOpenAI(BaseChatModel):
    def _generate(self, input: str) -> str:
        return f"OpenAI: {input}"

# invoke, _validate, _format 都是继承来的！
model = ChatOpenAI()
print(model.invoke("  Hello  "))  # OpenAI: Hello
```

---

### 误区2：不实现抽象方法会立即报错 ❌

**为什么错？**
- Python 不会在**定义时**报错
- 只有在**实例化时**才会报错
- 这意味着错误可能被推迟发现

**为什么人们容易这样错？**
习惯了编译型语言（Java、TypeScript）在定义时就报错的行为。Python 是动态类型语言，很多检查在运行时才发生。

**正确理解：**

```python
from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def invoke(self, input: str) -> str:
        pass

# 定义时不会报错！
class IncompleteModel(BaseModel):
    pass  # 忘记实现 invoke，Python 不报错

# 只有实例化时才报错
try:
    model = IncompleteModel()  # TypeError!
except TypeError as e:
    print(f"实例化时报错: {e}")

# 甚至可以继续添加子类...
class StillIncomplete(IncompleteModel):
    pass  # 还是不实现，定义时仍然不报错

# 直到实例化
try:
    model = StillIncomplete()
except TypeError as e:
    print(f"还是报错: {e}")

# 解决方案：使用类型检查工具（mypy, pyright）
# 可以在开发时就发现问题
```

---

### 误区3：ABC 和 Protocol 是一样的 ❌

**为什么错？**
- **ABC（名义类型）**：必须显式继承才算实现
- **Protocol（结构类型）**：只要有相同的方法就算实现（鸭子类型）
- 适用场景完全不同

**为什么人们容易这样错？**
两者都可以定义"接口"，表面功能相似。但设计理念和使用方式完全不同。

**正确理解：**

```python
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

# ===== ABC：必须继承 =====
class RunnableABC(ABC):
    @abstractmethod
    def invoke(self, input: str) -> str:
        pass

class Model1(RunnableABC):  # 显式继承
    def invoke(self, input: str) -> str:
        return input

class Model2:  # 没有继承
    def invoke(self, input: str) -> str:
        return input

print(isinstance(Model1(), RunnableABC))  # True（继承了）
print(isinstance(Model2(), RunnableABC))  # False（没继承！）

# ===== Protocol：看方法 =====
@runtime_checkable
class RunnableProtocol(Protocol):
    def invoke(self, input: str) -> str:
        ...

print(isinstance(Model1(), RunnableProtocol))  # True（有 invoke 方法）
print(isinstance(Model2(), RunnableProtocol))  # True（有 invoke 方法）

# 选择标准：
# - ABC：需要明确的继承层次，框架内部类
# - Protocol：第三方类、鸭子类型场景
```

---

## 7. 【实战代码】

```python
"""
示例：构建 LangChain 风格的抽象类层次
演示 ABC 模块在 LLM 应用框架中的核心用法
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Generator, Protocol, runtime_checkable
from dataclasses import dataclass
from datetime import datetime

# ===== 1. 基础类型定义 =====
print("=== 1. 基础类型定义 ===")

@dataclass
class Message:
    """消息类型"""
    role: str
    content: str

@dataclass
class ChatResult:
    """聊天结果"""
    message: Message
    model: str
    usage: Dict[str, int]

# ===== 2. 顶层抽象类：Runnable =====
print("\n=== 2. Runnable 抽象类 ===")

class Runnable(ABC):
    """
    可运行组件抽象基类 - 类似 LangChain 的 Runnable

    定义所有可运行组件必须实现的接口
    """

    # ===== 抽象方法：子类必须实现 =====

    @abstractmethod
    def invoke(self, input: Any, config: Optional[Dict] = None) -> Any:
        """同步调用"""
        pass

    @abstractmethod
    def batch(self, inputs: List[Any], config: Optional[Dict] = None) -> List[Any]:
        """批量调用"""
        pass

    @abstractmethod
    def stream(self, input: Any, config: Optional[Dict] = None) -> Generator[Any, None, None]:
        """流式调用"""
        pass

    # ===== 抽象属性 =====

    @property
    @abstractmethod
    def input_type(self) -> type:
        """输入类型"""
        pass

    @property
    @abstractmethod
    def output_type(self) -> type:
        """输出类型"""
        pass

    # ===== 具体方法：提供默认实现 =====

    def get_name(self) -> str:
        """获取名称"""
        return self.__class__.__name__

    def bind(self, **kwargs) -> 'Runnable':
        """绑定参数（简化版）"""
        return BoundRunnable(self, kwargs)

# ===== 3. 绑定参数的包装类 =====

class BoundRunnable(Runnable):
    """绑定了参数的 Runnable"""

    def __init__(self, runnable: Runnable, bound_kwargs: Dict):
        self._runnable = runnable
        self._bound_kwargs = bound_kwargs

    @property
    def input_type(self) -> type:
        return self._runnable.input_type

    @property
    def output_type(self) -> type:
        return self._runnable.output_type

    def invoke(self, input: Any, config: Optional[Dict] = None) -> Any:
        merged_config = {**(config or {}), **self._bound_kwargs}
        return self._runnable.invoke(input, merged_config)

    def batch(self, inputs: List[Any], config: Optional[Dict] = None) -> List[Any]:
        merged_config = {**(config or {}), **self._bound_kwargs}
        return self._runnable.batch(inputs, merged_config)

    def stream(self, input: Any, config: Optional[Dict] = None) -> Generator[Any, None, None]:
        merged_config = {**(config or {}), **self._bound_kwargs}
        yield from self._runnable.stream(input, merged_config)

# ===== 4. 中间层抽象类：BaseLanguageModel =====
print("\n=== 4. BaseLanguageModel 抽象类 ===")

class BaseLanguageModel(Runnable):
    """
    语言模型抽象基类 - 类似 LangChain 的 BaseLanguageModel

    在 Runnable 基础上添加语言模型特有的抽象
    """

    def __init__(self, model_name: str = "base", temperature: float = 0.7):
        self._model_name = model_name
        self._temperature = temperature

    # ===== 实现 Runnable 的抽象属性 =====

    @property
    def input_type(self) -> type:
        return str

    @property
    def output_type(self) -> type:
        return str

    # ===== 新增抽象属性 =====

    @property
    @abstractmethod
    def _llm_type(self) -> str:
        """LLM 类型标识"""
        pass

    # ===== 新增抽象方法 =====

    @abstractmethod
    def _generate(self, prompts: List[str], **kwargs) -> List[str]:
        """内部生成方法"""
        pass

    # ===== 实现 Runnable 的抽象方法 =====

    def invoke(self, input: str, config: Optional[Dict] = None) -> str:
        """同步调用"""
        results = self._generate([input], **(config or {}))
        return results[0]

    def batch(self, inputs: List[str], config: Optional[Dict] = None) -> List[str]:
        """批量调用"""
        return self._generate(inputs, **(config or {}))

    def stream(self, input: str, config: Optional[Dict] = None) -> Generator[str, None, None]:
        """流式调用（默认实现）"""
        result = self.invoke(input, config)
        for char in result:
            yield char

    # ===== 具体属性 =====

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def temperature(self) -> float:
        return self._temperature

# ===== 5. 更具体的抽象类：BaseChatModel =====
print("\n=== 5. BaseChatModel 抽象类 ===")

class BaseChatModel(BaseLanguageModel):
    """
    聊天模型抽象基类 - 类似 LangChain 的 BaseChatModel

    处理消息格式的语言模型
    """

    # ===== 新增抽象方法 =====

    @abstractmethod
    def _generate_chat(self, messages: List[Message], **kwargs) -> ChatResult:
        """聊天生成方法"""
        pass

    # ===== 实现父类的抽象方法 =====

    def _generate(self, prompts: List[str], **kwargs) -> List[str]:
        """将 prompt 转换为消息格式"""
        results = []
        for prompt in prompts:
            messages = [Message(role="user", content=prompt)]
            result = self._generate_chat(messages, **kwargs)
            results.append(result.message.content)
        return results

    # ===== 新增具体方法 =====

    def chat(self, messages: List[Message], **kwargs) -> Message:
        """聊天接口"""
        result = self._generate_chat(messages, **kwargs)
        return result.message

# ===== 6. 具体实现类 =====
print("\n=== 6. 具体实现类 ===")

class ChatOpenAI(BaseChatModel):
    """OpenAI 聊天模型 - 具体实现"""

    def __init__(self, model: str = "gpt-4", temperature: float = 0.7):
        super().__init__(model_name=model, temperature=temperature)

    @property
    def _llm_type(self) -> str:
        return "openai-chat"

    def _generate_chat(self, messages: List[Message], **kwargs) -> ChatResult:
        """实现聊天生成"""
        # 模拟 API 调用
        last_message = messages[-1].content
        response = f"[{self.model_name}] Response to: {last_message}"

        return ChatResult(
            message=Message(role="assistant", content=response),
            model=self.model_name,
            usage={"input_tokens": len(last_message), "output_tokens": len(response)}
        )

    def stream(self, input: str, config: Optional[Dict] = None) -> Generator[str, None, None]:
        """覆盖流式调用，提供真正的流式实现"""
        messages = [Message(role="user", content=input)]
        response = f"[{self.model_name}] Streaming: {input}"
        for char in response:
            yield char

class ChatAnthropic(BaseChatModel):
    """Anthropic 聊天模型 - 具体实现"""

    def __init__(self, model: str = "claude-3-opus", temperature: float = 0.7):
        super().__init__(model_name=model, temperature=temperature)

    @property
    def _llm_type(self) -> str:
        return "anthropic-chat"

    def _generate_chat(self, messages: List[Message], **kwargs) -> ChatResult:
        last_message = messages[-1].content
        response = f"[{self.model_name}] I'll help you with: {last_message}"

        return ChatResult(
            message=Message(role="assistant", content=response),
            model=self.model_name,
            usage={"input_tokens": len(last_message), "output_tokens": len(response)}
        )

# ===== 7. 使用 Protocol 定义接口 =====
print("\n=== 7. Protocol 接口 ===")

@runtime_checkable
class Serializable(Protocol):
    """可序列化协议"""

    def to_dict(self) -> Dict[str, Any]:
        ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Serializable':
        ...

class Config:
    """配置类 - 符合 Serializable 协议但不继承它"""

    def __init__(self, model: str, temperature: float):
        self.model = model
        self.temperature = temperature

    def to_dict(self) -> Dict[str, Any]:
        return {"model": self.model, "temperature": self.temperature}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        return cls(data["model"], data["temperature"])

# 检查是否符合协议
config = Config("gpt-4", 0.7)
print(f"Config 符合 Serializable 协议: {isinstance(config, Serializable)}")

# ===== 8. Mixin 示例 =====
print("\n=== 8. Mixin 示例 ===")

class CacheableMixin:
    """缓存 Mixin"""

    _cache: Dict[str, Any] = {}

    def get_cached(self, key: str) -> Optional[Any]:
        return self._cache.get(key)

    def set_cached(self, key: str, value: Any):
        self._cache[key] = value

class LoggableMixin:
    """日志 Mixin"""

    def log(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] [{self.__class__.__name__}] {message}")

class SmartChatOpenAI(ChatOpenAI, CacheableMixin, LoggableMixin):
    """带缓存和日志的 OpenAI 模型"""

    def invoke(self, input: str, config: Optional[Dict] = None) -> str:
        # 检查缓存
        cached = self.get_cached(input)
        if cached:
            self.log(f"缓存命中: {input[:20]}...")
            return cached

        # 调用父类
        self.log(f"调用 API: {input[:20]}...")
        result = super().invoke(input, config)

        # 存入缓存
        self.set_cached(input, result)
        return result

# ===== 9. 测试所有功能 =====
print("\n=== 9. 功能测试 ===")

# 测试基本功能
openai = ChatOpenAI()
anthropic = ChatAnthropic()

print(f"\nOpenAI 类型: {openai._llm_type}")
print(f"OpenAI 调用: {openai.invoke('Hello')}")

print(f"\nAnthropic 类型: {anthropic._llm_type}")
print(f"Anthropic 调用: {anthropic.invoke('Hello')}")

# 测试多态
def run_model(model: BaseChatModel, prompt: str) -> str:
    """接受任何 BaseChatModel 实现"""
    return model.invoke(prompt)

print("\n多态调用:")
for model in [openai, anthropic]:
    print(f"  {model.get_name()}: {run_model(model, 'Test')}")

# 测试批量调用
print("\n批量调用:")
results = openai.batch(["Hello", "World", "Test"])
for r in results:
    print(f"  {r}")

# 测试流式调用
print("\n流式调用:")
print("  ", end="")
for chunk in openai.stream("Stream test"):
    print(chunk, end="", flush=True)
print()

# 测试绑定参数
print("\n绑定参数:")
bound_model = openai.bind(temperature=0.5)
print(f"  绑定后调用: {bound_model.invoke('Bound test')}")

# 测试带缓存的模型
print("\n带缓存的模型:")
smart_model = SmartChatOpenAI()
print(f"  第一次: {smart_model.invoke('Cache test')}")
print(f"  第二次: {smart_model.invoke('Cache test')}")  # 应该命中缓存

# 测试聊天接口
print("\n聊天接口:")
messages = [
    Message(role="user", content="What is Python?")
]
response = openai.chat(messages)
print(f"  {response.role}: {response.content}")

# ===== 10. 类型检查示例 =====
print("\n=== 10. 类型检查 ===")

def check_types(obj):
    """检查对象的类型"""
    print(f"\n{obj.__class__.__name__}:")
    print(f"  是 Runnable: {isinstance(obj, Runnable)}")
    print(f"  是 BaseLanguageModel: {isinstance(obj, BaseLanguageModel)}")
    print(f"  是 BaseChatModel: {isinstance(obj, BaseChatModel)}")
    print(f"  是 ChatOpenAI: {isinstance(obj, ChatOpenAI)}")

check_types(openai)
check_types(anthropic)
check_types(smart_model)

print("\n=== 完成 ===")
```

**运行输出示例：**
```
=== 1. 基础类型定义 ===

=== 2. Runnable 抽象类 ===

=== 4. BaseLanguageModel 抽象类 ===

=== 5. BaseChatModel 抽象类 ===

=== 6. 具体实现类 ===

=== 7. Protocol 接口 ===
Config 符合 Serializable 协议: True

=== 8. Mixin 示例 ===

=== 9. 功能测试 ===

OpenAI 类型: openai-chat
OpenAI 调用: [gpt-4] Response to: Hello

Anthropic 类型: anthropic-chat
Anthropic 调用: [claude-3-opus] I'll help you with: Hello

多态调用:
  ChatOpenAI: [gpt-4] Response to: Test
  ChatAnthropic: [claude-3-opus] I'll help you with: Test

批量调用:
  [gpt-4] Response to: Hello
  [gpt-4] Response to: World
  [gpt-4] Response to: Test

流式调用:
  [gpt-4] Streaming: Stream test

绑定参数:
  绑定后调用: [gpt-4] Response to: Bound test

带缓存的模型:
[10:30:45] [SmartChatOpenAI] 调用 API: Cache test...
  第一次: [gpt-4] Response to: Cache test
[10:30:45] [SmartChatOpenAI] 缓存命中: Cache test...
  第二次: [gpt-4] Response to: Cache test

聊天接口:
  assistant: [gpt-4] Response to: What is Python?

=== 10. 类型检查 ===

ChatOpenAI:
  是 Runnable: True
  是 BaseLanguageModel: True
  是 BaseChatModel: True
  是 ChatOpenAI: True

ChatAnthropic:
  是 Runnable: True
  是 BaseLanguageModel: True
  是 BaseChatModel: True
  是 ChatOpenAI: False

SmartChatOpenAI:
  是 Runnable: True
  是 BaseLanguageModel: True
  是 BaseChatModel: True
  是 ChatOpenAI: True

=== 完成 ===
```

---

## 8. 【面试必问】

### 问题："Python 中如何定义接口？ABC 和 Protocol 有什么区别？"

**普通回答（❌ 不出彩）：**
"Python 用 ABC 定义接口，就是在类上加 ABC 继承，方法上加 @abstractmethod。Protocol 也可以定义接口，但不需要继承。"

**出彩回答（✅ 推荐）：**

> **Python 定义接口有两种方式：**
>
> 1. **ABC（抽象基类）**：
>    - 基于**名义类型**：必须显式继承才算实现
>    - 使用 `from abc import ABC, abstractmethod`
>    - 特点：强制约束、明确的继承层次
>    - 适用：框架内部类、需要明确契约的场景
>
>    ```python
>    from abc import ABC, abstractmethod
>
>    class Runnable(ABC):
>        @abstractmethod
>        def invoke(self, input): pass
>
>    class MyRunnable(Runnable):  # 必须继承
>        def invoke(self, input): ...
>    ```
>
> 2. **Protocol（协议）**：
>    - 基于**结构类型**：只要有相同方法就算实现（鸭子类型）
>    - 使用 `from typing import Protocol`
>    - 特点：更灵活、支持第三方类
>    - 适用：鸭子类型、与第三方代码交互
>
>    ```python
>    from typing import Protocol
>
>    class Runnable(Protocol):
>        def invoke(self, input): ...
>
>    class MyClass:  # 不需要继承！
>        def invoke(self, input): ...
>
>    # 自动符合 Runnable 协议
>    ```
>
> **LangChain 的选择**：
> - 核心类（BaseChatModel、BaseRetriever）用 **ABC**
> - 因为需要明确的继承层次和模板方法模式
> - 类型检查工具可以更好地验证
>
> **选择标准**：
> - 需要强制继承 → ABC
> - 鸭子类型 / 第三方类 → Protocol

**为什么这个回答出彩？**
1. ✅ 对比两种方式的本质区别（名义 vs 结构）
2. ✅ 有代码示例
3. ✅ 联系 LangChain 实际应用
4. ✅ 给出选择标准

---

### 问题："为什么 LangChain 要设计这么多抽象基类？"

**普通回答（❌ 不出彩）：**
"为了代码复用和统一接口，这样不同的 LLM 可以用同样的方式调用。"

**出彩回答（✅ 推荐）：**

> **LangChain 的抽象类层次设计有四个核心目的：**
>
> 1. **统一接口**：
>    - 所有 LLM 都实现 `BaseChatModel`
>    - 使用者只需要知道 `invoke()`，不用关心具体是 OpenAI 还是 Anthropic
>    - 这是**依赖倒置原则**的体现
>
> 2. **模板方法模式**：
>    - `BaseChatModel.invoke()` 定义了调用流程（验证→生成→后处理）
>    - 子类只需实现 `_generate()` 差异化部分
>    - 避免每个实现都重复写流程代码
>
> 3. **LCEL 管道组合**：
>    - 所有组件都是 `Runnable`
>    - `chain = prompt | llm | parser` 之所以能工作
>    - 是因为它们都遵循 `Runnable` 接口
>
> 4. **可扩展性**：
>    - 添加新模型只需继承 `BaseChatModel`
>    - 实现 `_generate()` 方法即可
>    - 不需要修改框架代码（开闭原则）
>
> ```python
> # 继承层次
> Runnable (最抽象：invoke/batch/stream)
>     ↓
> BaseLanguageModel (语言模型：_generate)
>     ↓
> BaseChatModel (聊天模型：_generate_chat)
>     ↓
> ChatOpenAI/ChatAnthropic (具体实现)
> ```
>
> **核心价值**：用户代码依赖抽象（`BaseChatModel`），不依赖具体实现（`ChatOpenAI`），实现了真正的"可插拔"。

---

## 9. 【化骨绵掌】

### 卡片1：什么是抽象类 🎯

**一句话：** 不能直接实例化、包含抽象方法的类，子类必须实现所有抽象方法才能实例化。

**举例：**
```python
from abc import ABC, abstractmethod

class Animal(ABC):
    @abstractmethod
    def speak(self): pass

Animal()  # TypeError: 不能实例化抽象类
```

**应用：** LangChain 的 BaseChatModel、BaseRetriever 都是抽象类。

---

### 卡片2：@abstractmethod 装饰器 🏷️

**一句话：** 标记一个方法为抽象方法，子类必须实现，否则子类也不能实例化。

**举例：**
```python
class BaseChatModel(ABC):
    @abstractmethod
    def invoke(self, input: str) -> str:
        """必须实现！"""
        pass
```

**应用：** LangChain 用它标记 `_generate`、`_get_relevant_documents` 等方法。

---

### 卡片3：抽象属性 📐

**一句话：** 用 `@property + @abstractmethod` 定义必须实现的属性。

**举例：**
```python
class BaseModel(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

class ChatOpenAI(BaseModel):
    @property
    def model_name(self) -> str:
        return "gpt-4"
```

**应用：** LangChain 的 `_llm_type` 属性就是抽象属性。

---

### 卡片4：抽象类可以有具体方法 ✨

**一句话：** 抽象类不是"全部抽象"，可以有完整实现的方法。

**举例：**
```python
class BaseChatModel(ABC):
    def invoke(self, input):
        # 具体实现：模板方法
        return self._generate(input)

    @abstractmethod
    def _generate(self, input):
        # 抽象方法：子类实现
        pass
```

**应用：** LangChain 的 `invoke()` 是具体方法，`_generate()` 是抽象方法。

---

### 卡片5：ABC vs Protocol 🔀

**一句话：** ABC 要继承（名义类型），Protocol 只看方法（结构类型）。

**对比：**
```python
# ABC：必须继承
class Cat(Animal):
    def speak(self): return "Meow"

# Protocol：只看方法
class Dog:
    def speak(self): return "Woof"

isinstance(Dog(), SpeakableProtocol)  # True
```

**应用：** LangChain 核心类用 ABC，灵活场景用 Protocol。

---

### 卡片6：继承层次 🏛️

**一句话：** 抽象类可以继承抽象类，形成多层结构，越往下越具体。

**举例：**
```
Runnable (invoke, batch, stream)
    ↓
BaseLanguageModel (_generate)
    ↓
BaseChatModel (_generate_chat)
    ↓
ChatOpenAI (具体实现)
```

**应用：** LangChain 的类层次就是这样设计的。

---

### 卡片7：Mixin 模式 🔌

**一句话：** Mixin 是提供可复用功能的类，通过多重继承"混入"到目标类。

**举例：**
```python
class CacheableMixin:
    def get_cached(self, key): ...

class LoggableMixin:
    def log(self, msg): ...

class SmartModel(ChatOpenAI, CacheableMixin, LoggableMixin):
    # 同时拥有缓存和日志功能
    pass
```

**应用：** 为 LangChain 组件添加额外功能。

---

### 卡片8：实例化时才报错 ⚠️

**一句话：** Python 不在定义时检查，只有实例化时才报 TypeError。

**举例：**
```python
class IncompleteModel(BaseChatModel):
    pass  # 忘记实现，定义时不报错

IncompleteModel()  # TypeError!（实例化时才报错）
```

**应用：** 使用 mypy/pyright 可以在开发时发现问题。

---

### 卡片9：isinstance 检查 🔍

**一句话：** 用 isinstance 检查对象是否是某个抽象类的实例。

**举例：**
```python
model = ChatOpenAI()

isinstance(model, BaseChatModel)  # True
isinstance(model, Runnable)       # True
isinstance(model, ChatOpenAI)     # True
isinstance(model, ChatAnthropic)  # False
```

**应用：** LangChain 内部用于类型检查和分发。

---

### 卡片10：ABC 总结 ⭐

**一句话：** 定义接口契约，强制子类实现，支持多态。

**核心要点：**
1. `from abc import ABC, abstractmethod`
2. `@abstractmethod` 标记必须实现的方法
3. `@property + @abstractmethod` 标记必须实现的属性
4. 抽象类可以有具体方法
5. Protocol 是另一种选择（鸭子类型）

**记住：** 看到 `class XxxBase(ABC)` 就知道这是抽象基类！

---

## 10. 【一句话总结】

**Python ABC 模块通过 abstractmethod 装饰器定义接口契约，强制子类实现特定方法，是 LangChain 中 Runnable、BaseChatModel、BaseRetriever 等所有基类的实现基础，使得不同的 LLM 实现可以通过统一接口调用。**

---

## 📚 学习检查清单

- [ ] 理解 ABC 和 abstractmethod 的作用
- [ ] 能定义包含抽象方法的抽象类
- [ ] 能实现抽象类的子类
- [ ] 理解抽象属性的定义方式
- [ ] 能区分 ABC（名义类型）和 Protocol（结构类型）
- [ ] 理解抽象类的继承层次设计
- [ ] 能使用 Mixin 模式添加功能
- [ ] 能识别 LangChain 源码中的抽象类结构

## 🔗 下一步学习

- **依赖注入原理**：如何将抽象类的实例注入到使用者
- **模板方法模式**：抽象类常用的设计模式
- **Runnable 协议**：LangChain 核心抽象类的详细分析
- **BaseChatModel 实现**：LangChain 聊天模型的源码分析

---

**版本：** v1.0
**最后更新：** 2025-12-12
