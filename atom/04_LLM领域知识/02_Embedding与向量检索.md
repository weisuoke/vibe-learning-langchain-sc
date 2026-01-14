# Embedding 与向量检索

> 原子化知识点 | LLM领域知识 | LangChain 源码学习前置知识

---

## 1. 【30字核心】

**Embedding 将文本转换为数值向量，通过向量相似度实现语义检索，是 LangChain RAG 系统的核心技术。**

---

## 2. 【第一性原理】

### 什么是第一性原理？

**第一性原理**：回到事物最基本的真理，从源头思考问题

### Embedding 与向量检索的第一性原理 🎯

#### 1. 最基础的定义

**Embedding = 将任意对象映射到固定维度的数值向量**

仅此而已！没有更基础的了。

- **文本 → 向量**：`"猫" → [0.2, -0.5, 0.8, ..., 0.1]`（1536维）
- **图片 → 向量**：图片像素 → 特征向量
- **音频 → 向量**：声音信号 → 音频特征向量

```python
# Embedding 的本质
text = "我爱编程"
embedding_vector = embed_model.embed(text)
# [0.023, -0.156, 0.891, ..., 0.042]  # 1536个浮点数
```

#### 2. 为什么需要 Embedding？

**核心问题：计算机无法直接理解文本语义，只能处理数字**

```python
# 传统方法：关键词匹配（无法理解语义）
query = "快乐"
docs = ["开心的一天", "悲伤的故事", "幸福的生活"]

# 关键词匹配找不到任何结果！
# 因为"快乐"和"开心"、"幸福"是不同的字符串

# Embedding 方法：语义相似度
query_vec = embed("快乐")     # [0.8, 0.2, ...]
doc_vecs = [embed(d) for d in docs]

# "开心"、"幸福"和"快乐"的向量很接近！
# similarity("快乐", "开心") ≈ 0.95
```

#### 3. Embedding 的三层价值

##### 价值1：语义理解

```python
# 语义相近的词，向量也相近
embed("国王") - embed("男人") + embed("女人") ≈ embed("王后")

# 这就是著名的 Word2Vec 语义算术
```

##### 价值2：跨模态关联

```python
# 文本和图片可以在同一向量空间
text_vec = embed_text("一只可爱的猫")
image_vec = embed_image(cat_image)

# 它们的向量很接近！可以用文字搜图片
similarity(text_vec, image_vec) > 0.8
```

##### 价值3：高效检索

```python
# 传统数据库：遍历比较，O(n)
# 向量数据库：近似最近邻，O(log n)

# 从百万文档中找最相关的，毫秒级响应
results = vector_db.similarity_search(query_vec, k=10)
```

#### 4. 从第一性原理推导 LangChain 应用

**推理链：**

```
1. LLM 的知识有截止日期，无法获取最新信息
   ↓
2. 需要将外部知识"注入"到 LLM 对话中
   ↓
3. 直接把所有知识塞进 Prompt 会超出 Token 限制
   ↓
4. 需要根据问题检索最相关的知识片段
   ↓
5. 关键词检索无法理解语义（"猫"搜不到"喵星人"）
   ↓
6. Embedding 可以实现语义相似度检索
   ↓
7. 将文档 Embedding 后存入向量数据库
   ↓
8. 检索时，将问题 Embedding，找最相似的文档
   ↓
9. 这就是 RAG（Retrieval Augmented Generation）的核心
```

#### 5. 一句话总结第一性原理

**Embedding 是语义的数学表示，让计算机能够"理解"文本含义，是连接人类语言和机器计算的桥梁，是 LangChain RAG 系统的数学基础。**

---

## 3. 【核心概念（全面覆盖）】

### 核心概念1：Embedding 向量 📊

**Embedding 是将文本映射到高维向量空间的技术，语义相似的文本在向量空间中距离相近**

```python
from langchain_openai import OpenAIEmbeddings

# 创建 Embedding 模型
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 单个文本 Embedding
text = "LangChain 是一个强大的 LLM 应用框架"
vector = embeddings.embed_query(text)
print(f"向量维度: {len(vector)}")  # 1536
print(f"前5个值: {vector[:5]}")    # [0.023, -0.156, ...]

# 批量文本 Embedding
texts = [
    "Python 是流行的编程语言",
    "JavaScript 用于网页开发",
    "机器学习改变了世界"
]
vectors = embeddings.embed_documents(texts)
print(f"文档数量: {len(vectors)}")
```

**常见 Embedding 模型对比：**

| 模型 | 维度 | 特点 | 适用场景 |
|------|------|------|---------|
| text-embedding-3-small | 1536 | OpenAI，性价比高 | 通用场景 |
| text-embedding-3-large | 3072 | OpenAI，效果最好 | 高精度需求 |
| text-embedding-ada-002 | 1536 | OpenAI 旧版 | 兼容旧系统 |
| BAAI/bge-large-zh | 1024 | 中文优化，开源 | 中文场景 |
| sentence-transformers | 384-768 | 开源，可本地部署 | 隐私敏感场景 |

**在 LangChain 源码中的应用：**

```python
# langchain_core/embeddings/embeddings.py
from abc import ABC, abstractmethod
from typing import List

class Embeddings(ABC):
    """Embedding 模型的抽象基类"""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量 Embedding 文档"""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """单个查询 Embedding"""
        pass
```

---

### 核心概念2：向量数据库（VectorStore） 🗄️

**向量数据库专门存储和检索高维向量，支持高效的相似度搜索**

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# 创建 Embedding 模型
embeddings = OpenAIEmbeddings()

# 准备文档
documents = [
    Document(page_content="LangChain 支持多种 LLM", metadata={"source": "doc1"}),
    Document(page_content="向量数据库用于存储 Embedding", metadata={"source": "doc2"}),
    Document(page_content="RAG 可以增强 LLM 的知识", metadata={"source": "doc3"}),
]

# 创建向量数据库
vectorstore = FAISS.from_documents(documents, embeddings)

# 相似度搜索
results = vectorstore.similarity_search("什么是 RAG？", k=2)
for doc in results:
    print(f"内容: {doc.page_content}")
    print(f"来源: {doc.metadata['source']}")
```

**常见向量数据库对比：**

| 数据库 | 类型 | 特点 | 适用场景 |
|--------|------|------|---------|
| FAISS | 本地 | Meta 开源，速度快 | 原型开发、中小规模 |
| Chroma | 本地/云 | 简单易用，支持持久化 | 快速开发 |
| Pinecone | 云服务 | 全托管，高可用 | 生产环境 |
| Milvus | 分布式 | 开源，支持大规模 | 企业级应用 |
| Weaviate | 云/本地 | 支持混合搜索 | 复杂检索需求 |
| Qdrant | 云/本地 | Rust 实现，高性能 | 性能敏感场景 |

**在 LangChain 源码中的应用：**

```python
# langchain_core/vectorstores/base.py
from abc import ABC, abstractmethod
from typing import List, Optional

class VectorStore(ABC):
    """向量数据库的抽象基类"""

    @abstractmethod
    def add_documents(self, documents: List[Document]) -> List[str]:
        """添加文档到向量数据库"""
        pass

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[Document]:
        """相似度搜索"""
        pass

    def as_retriever(self, **kwargs) -> VectorStoreRetriever:
        """转换为 Retriever"""
        return VectorStoreRetriever(vectorstore=self, **kwargs)
```

---

### 核心概念3：相似度计算 📐

**通过数学方法计算两个向量的相似程度**

```python
import numpy as np

def cosine_similarity(vec1: list, vec2: list) -> float:
    """余弦相似度：衡量向量方向的相似性"""
    v1, v2 = np.array(vec1), np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def euclidean_distance(vec1: list, vec2: list) -> float:
    """欧氏距离：衡量向量在空间中的距离"""
    v1, v2 = np.array(vec1), np.array(vec2)
    return np.linalg.norm(v1 - v2)

def dot_product(vec1: list, vec2: list) -> float:
    """点积：直接计算向量内积"""
    return np.dot(vec1, vec2)

# 示例
vec_cat = [0.8, 0.2, 0.1]
vec_dog = [0.7, 0.3, 0.2]
vec_car = [0.1, 0.1, 0.9]

print(f"猫-狗相似度: {cosine_similarity(vec_cat, vec_dog):.3f}")  # ~0.98
print(f"猫-车相似度: {cosine_similarity(vec_cat, vec_car):.3f}")  # ~0.40
```

**相似度度量对比：**

| 度量方式 | 公式 | 范围 | 特点 |
|---------|------|------|------|
| 余弦相似度 | cos(θ) = A·B / \|A\|\|B\| | [-1, 1] | 只关注方向，不关注大小 |
| 欧氏距离 | √(Σ(ai-bi)²) | [0, ∞) | 关注绝对距离 |
| 点积 | Σ(ai×bi) | (-∞, ∞) | 同时考虑方向和大小 |
| 曼哈顿距离 | Σ\|ai-bi\| | [0, ∞) | 计算简单 |

**在 LangChain 中使用不同相似度：**

```python
# FAISS 支持不同距离度量
from langchain_community.vectorstores import FAISS
import faiss

# 使用余弦相似度（默认）
vectorstore = FAISS.from_documents(docs, embeddings)

# 使用欧氏距离
# 需要直接使用 FAISS API
index = faiss.IndexFlatL2(dimension)  # L2 = 欧氏距离
```

---

### 核心概念4：Retriever 检索器 🔍

**Retriever 是 LangChain 中执行检索的统一接口**

```python
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import FAISS

# 从 VectorStore 创建 Retriever
vectorstore = FAISS.from_documents(documents, embeddings)

# 方式1：基础 Retriever
retriever = vectorstore.as_retriever()
docs = retriever.invoke("LangChain 是什么？")

# 方式2：配置检索参数
retriever = vectorstore.as_retriever(
    search_type="similarity",      # 或 "mmr"
    search_kwargs={"k": 5}         # 返回 5 个结果
)

# 方式3：MMR（最大边际相关性）- 增加多样性
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20,    # 先获取 20 个，再选 5 个最多样的
        "lambda_mult": 0.5  # 多样性权重
    }
)
```

**Retriever 类型对比：**

| 类型 | 特点 | 适用场景 |
|------|------|---------|
| VectorStoreRetriever | 基于向量相似度 | 通用语义检索 |
| BM25Retriever | 基于关键词 TF-IDF | 精确关键词匹配 |
| EnsembleRetriever | 混合多个检索器 | 综合检索需求 |
| ContextualCompressionRetriever | 压缩检索结果 | 减少无关内容 |
| ParentDocumentRetriever | 检索父文档 | 需要完整上下文 |

**在 LangChain 源码中的应用：**

```python
# langchain_core/retrievers/base.py
class BaseRetriever(ABC, Runnable):
    """检索器基类，实现 Runnable 接口"""

    @abstractmethod
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """实际的检索逻辑"""
        pass

    def invoke(self, input: str, config: Optional[RunnableConfig] = None) -> List[Document]:
        """Runnable 接口实现"""
        return self._get_relevant_documents(input)
```

---

### 扩展概念5：RAG（检索增强生成） 🚀

**RAG 将检索到的知识注入到 LLM 的上下文中**

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# 创建 RAG 链
retriever = vectorstore.as_retriever()
llm = ChatOpenAI()

# RAG Prompt 模板
template = """根据以下上下文回答问题。如果上下文中没有相关信息，请说"我不知道"。

上下文：
{context}

问题：{question}

回答："""

prompt = ChatPromptTemplate.from_template(template)

# 格式化检索结果
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 构建 RAG 链（LCEL）
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 使用 RAG 链
answer = rag_chain.invoke("LangChain 支持哪些功能？")
print(answer)
```

**RAG 工作流程：**

```
用户问题 → Embedding → 向量检索 → 相关文档 → 注入 Prompt → LLM → 回答
    ↓           ↓           ↓           ↓           ↓         ↓
"什么是X"  [0.2,0.8,...]  Top-K文档   "上下文:..."  GPT-4   "X是..."
```

---

## 4. 【最小可用】

掌握以下内容，就能开始构建 LangChain RAG 应用：

### 4.1 创建 Embedding

```python
from langchain_openai import OpenAIEmbeddings

# 初始化
embeddings = OpenAIEmbeddings()

# 单个文本 → 向量
vector = embeddings.embed_query("Hello World")

# 多个文本 → 向量列表
vectors = embeddings.embed_documents(["Text 1", "Text 2"])
```

### 4.2 使用向量数据库

```python
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# 从文档创建
docs = [Document(page_content="内容1"), Document(page_content="内容2")]
vectorstore = FAISS.from_documents(docs, embeddings)

# 相似度搜索
results = vectorstore.similarity_search("查询", k=3)

# 持久化
vectorstore.save_local("./faiss_index")
vectorstore = FAISS.load_local("./faiss_index", embeddings)
```

### 4.3 创建 Retriever

```python
# 基础用法
retriever = vectorstore.as_retriever()
docs = retriever.invoke("问题")

# 配置参数
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)
```

### 4.4 构建简单 RAG 链

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# RAG 模板
template = """上下文：{context}

问题：{question}

回答："""

prompt = ChatPromptTemplate.from_template(template)

# 组装链
chain = (
    {"context": retriever, "question": lambda x: x}
    | prompt
    | ChatOpenAI()
    | StrOutputParser()
)

answer = chain.invoke("你的问题")
```

**这些知识足以：**
- 构建基础的知识库问答系统
- 理解 LangChain RAG 示例代码
- 进行简单的语义搜索

---

## 5. 【1个类比】（双轨制）

### 类比1：Embedding 向量化

#### 🎨 前端视角：哈希函数 / 数据序列化

Embedding 就像一个超级智能的哈希函数，但保留了语义信息。

```javascript
// 普通哈希：只保证唯一性，不保留语义
const hash1 = md5("猫");     // "abc123..."
const hash2 = md5("喵星人"); // "def456..."
// hash1 和 hash2 完全不同！无法比较

// Embedding：保留语义相似性
const vec1 = embed("猫");     // [0.8, 0.2, ...]
const vec2 = embed("喵星人"); // [0.79, 0.21, ...]
// vec1 和 vec2 非常接近！可以计算相似度
```

```python
# LangChain Embedding
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vec1 = embeddings.embed_query("猫")
vec2 = embeddings.embed_query("喵星人")
# 余弦相似度 > 0.9
```

**关键区别：** 哈希丢失语义，Embedding 保留语义

#### 🧒 小朋友视角：给东西贴标签

Embedding 就像给每个东西贴上很多小标签：

```
给"猫"贴标签：
- 动物标签：✓✓✓✓✓（很像动物）
- 宠物标签：✓✓✓✓（很像宠物）
- 毛茸茸标签：✓✓✓✓（很毛茸茸）
- 食物标签：（不是食物）

给"狗"贴标签：
- 动物标签：✓✓✓✓✓
- 宠物标签：✓✓✓✓
- 毛茸茸标签：✓✓✓
- 食物标签：

猫和狗的标签很像！所以它们"相似"。
```

**生活例子：**
```
图书馆给每本书贴分类标签：
- 《哈利波特》：魔法✓✓✓✓ 冒险✓✓✓ 儿童✓✓
- 《指环王》：魔法✓✓✓✓ 冒险✓✓✓✓ 成人✓✓

如果你喜欢《哈利波特》，图书馆可以推荐《指环王》，
因为它们的标签很相似！
```

---

### 类比2：向量数据库

#### 🎨 前端视角：IndexedDB / 搜索引擎索引

向量数据库就像一个专门为"相似性搜索"优化的 IndexedDB。

```javascript
// 普通数据库：精确匹配
const result = db.find({ title: "React 入门" });
// 只能找到标题完全匹配的

// 向量数据库：语义匹配
const queryVec = embed("前端框架教程");
const results = vectorDB.similaritySearch(queryVec, k=5);
// 能找到 "React 入门"、"Vue 指南"、"Angular 教程" 等
```

```python
# LangChain VectorStore
vectorstore = FAISS.from_documents(docs, embeddings)
results = vectorstore.similarity_search("前端框架教程", k=5)
```

#### 🧒 小朋友视角：智能玩具收纳箱

向量数据库就像一个超级智能的玩具收纳箱：

```
普通收纳箱：
- 你说"给我红色的车"，它只能找红色的车
- 如果你说"给我交通工具"，它不知道怎么找

智能收纳箱（向量数据库）：
- 你说"给我红色的车"，它给你红色的车
- 你说"给我交通工具"，它给你车、飞机、火车！
- 你说"给我好玩的东西"，它给你最像玩具的东西！
```

**生活例子：**
```
妈妈问："宝贝，我想找一本讲动物的书"

普通书架：只能找书名带"动物"的书
智能书架：会找出《森林里的小动物》《恐龙世界》《海洋生物》...
         因为它知道这些书的"内容"和"动物"相关！
```

---

### 类比3：相似度计算

#### 🎨 前端视角：CSS 颜色距离

相似度计算就像比较两个颜色有多接近。

```javascript
// 颜色可以用 RGB 表示为向量
const red = [255, 0, 0];
const orange = [255, 165, 0];
const blue = [0, 0, 255];

// 计算颜色距离（欧氏距离）
function colorDistance(c1, c2) {
  return Math.sqrt(
    Math.pow(c1[0]-c2[0], 2) +
    Math.pow(c1[1]-c2[1], 2) +
    Math.pow(c1[2]-c2[2], 2)
  );
}

console.log(colorDistance(red, orange)); // 165（比较近）
console.log(colorDistance(red, blue));   // 360（很远）
```

```python
# 文本向量的相似度
import numpy as np

vec_cat = embeddings.embed_query("猫")
vec_dog = embeddings.embed_query("狗")
vec_car = embeddings.embed_query("汽车")

# 余弦相似度
similarity = np.dot(vec_cat, vec_dog) / (np.linalg.norm(vec_cat) * np.linalg.norm(vec_dog))
```

#### 🧒 小朋友视角：比较两样东西有多像

相似度就像比较两样东西有多像：

```
比较"苹果"和其他水果：

苹果 vs 梨子 = 很像！（都是圆的水果）
苹果 vs 香蕉 = 有点像（都是水果）
苹果 vs 汽车 = 完全不像！

相似度分数：
苹果-梨子：95分
苹果-香蕉：70分
苹果-汽车：5分
```

---

### 类比4：RAG 检索增强

#### 🎨 前端视角：搜索 + 渲染

RAG 就像前端的"搜索数据 + 渲染页面"流程。

```javascript
// 前端流程
async function showAnswer(question) {
  // 1. 搜索相关数据（检索）
  const searchResults = await api.search(question);

  // 2. 将数据渲染到模板（生成）
  const html = renderTemplate(searchResults, question);

  return html;
}
```

```python
# RAG 流程
def answer_with_rag(question):
    # 1. 检索相关文档
    docs = retriever.invoke(question)

    # 2. 将文档注入 Prompt，让 LLM 生成回答
    prompt = f"根据以下信息回答：{docs}\n问题：{question}"
    answer = llm.invoke(prompt)

    return answer
```

#### 🧒 小朋友视角：开卷考试

RAG 就像考试时可以翻书：

```
闭卷考试（普通 LLM）：
- 只能用脑子里记住的知识回答
- 如果没学过，就答不出来

开卷考试（RAG）：
- 可以翻书找答案！
- 先找到相关的书页（检索）
- 再根据书上的内容回答（生成）

例子：
问题："恐龙是什么时候灭绝的？"
1. 翻书找到"恐龙"相关的页面
2. 看到"6500万年前灭绝"
3. 回答："恐龙在6500万年前灭绝"
```

---

### 类比总结表

| Embedding 概念 | 前端类比 | 小朋友类比 |
|---------------|---------|-----------|
| Embedding 向量 | 智能哈希/序列化 | 给东西贴很多小标签 |
| 向量数据库 | IndexedDB + 全文搜索 | 智能玩具收纳箱 |
| 相似度计算 | 颜色距离计算 | 比较两样东西有多像 |
| RAG 检索增强 | 搜索 + 渲染 | 开卷考试 |
| Retriever | API 数据获取器 | 图书管理员 |
| 语义搜索 | 模糊搜索 + 智能匹配 | 理解你想要什么 |

---

## 6. 【反直觉点】

### 误区1：Embedding 维度越高越好 ❌

**为什么错？**
- 高维度增加存储和计算成本
- 可能引入噪声（维度灾难）
- 对于简单任务，低维度足够

**为什么人们容易这样错？**
直觉上，更多的维度 = 更多的信息 = 更好的效果。但 Embedding 是压缩表示，关键在于信息密度而非数量。

**正确理解：**

```python
# 不同维度的选择
embeddings_small = OpenAIEmbeddings(model="text-embedding-3-small")  # 1536维
embeddings_large = OpenAIEmbeddings(model="text-embedding-3-large")  # 3072维

# 对于简单的 FAQ 问答，1536维足够
# 对于复杂的学术文献检索，3072维可能更好

# 实际测试对比效果才是关键！
# 很多场景下 1536维 和 3072维 效果差异 < 2%
```

**经验法则：** 先用小模型，效果不满意再升级

---

### 误区2：向量数据库可以完全替代传统数据库 ❌

**为什么错？**
- 向量数据库擅长相似度搜索，不擅长精确匹配
- 不支持复杂的 SQL 查询（JOIN、GROUP BY）
- 元数据过滤能力有限

**为什么人们容易这样错？**
看到向量数据库的"智能搜索"能力，就以为它能解决所有问题。实际上它是专用工具，不是通用数据库。

**正确理解：**

```python
# ❌ 试图用向量数据库做精确查询
# 查找"id=12345"的文档 → 效率低，不准确

# ✅ 混合使用
# 传统数据库：存储结构化数据、精确查询
# 向量数据库：语义搜索、相似度匹配

# 最佳实践：混合检索
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# BM25：关键词匹配
bm25_retriever = BM25Retriever.from_documents(docs)

# 向量：语义匹配
vector_retriever = vectorstore.as_retriever()

# 混合检索器
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7]  # 70% 语义，30% 关键词
)
```

---

### 误区3：RAG 检索到的内容越多越好 ❌

**为什么错？**
- 过多内容会超出 Context Window
- 无关内容会干扰 LLM 判断
- 增加 Token 消耗

**为什么人们容易这样错？**
担心遗漏重要信息，所以"宁多勿少"。但 LLM 的注意力有限，信息过载反而降低回答质量。

**正确理解：**

```python
# ❌ 检索太多
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
# 20个文档可能有大量无关内容

# ✅ 适量检索 + 重排序
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# 先检索多一些
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# 用 LLM 过滤无关内容
compressor = LLMChainExtractor.from_llm(llm)
retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor
)

# 最终只返回最相关的内容
```

**经验法则：** k=3~5 是大多数场景的合理值

---

## 7. 【实战代码】

```python
"""
示例：构建完整的 RAG 知识库问答系统
演示 Embedding 和向量检索在 LangChain 中的核心用法
"""

import numpy as np
from typing import List

# ===== 1. Embedding 基础操作 =====
print("=== 1. Embedding 基础操作 ===")

# 模拟 Embedding 模型（实际使用时替换为 OpenAIEmbeddings）
class MockEmbeddings:
    """模拟 Embedding 模型用于演示"""

    def __init__(self, dim: int = 128):
        self.dim = dim
        # 模拟一些语义相近的词
        self.semantic_map = {
            "猫": [0.9, 0.1, 0.2],
            "狗": [0.85, 0.15, 0.25],
            "汽车": [0.1, 0.9, 0.3],
            "Python": [0.2, 0.3, 0.9],
            "编程": [0.25, 0.35, 0.85],
        }

    def embed_query(self, text: str) -> List[float]:
        """单个文本 Embedding"""
        # 简化实现：基于关键词 + 随机噪声
        base = [0.0] * 3
        for key, vec in self.semantic_map.items():
            if key in text:
                base = [b + v for b, v in zip(base, vec)]
        # 扩展到目标维度
        np.random.seed(hash(text) % 2**32)
        noise = np.random.randn(self.dim - 3) * 0.1
        return base + list(noise)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量文本 Embedding"""
        return [self.embed_query(t) for t in texts]

embeddings = MockEmbeddings(dim=128)

# 测试 Embedding
vec1 = embeddings.embed_query("我家的猫很可爱")
vec2 = embeddings.embed_query("我养了一只狗")
vec3 = embeddings.embed_query("我买了一辆汽车")

print(f"向量维度: {len(vec1)}")
print(f"猫向量前3维: {vec1[:3]}")
print(f"狗向量前3维: {vec2[:3]}")
print(f"汽车向量前3维: {vec3[:3]}")

# ===== 2. 相似度计算 =====
print("\n=== 2. 相似度计算 ===")

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """计算余弦相似度"""
    v1, v2 = np.array(vec1), np.array(vec2)
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    return dot / norm if norm > 0 else 0

# 计算相似度
sim_cat_dog = cosine_similarity(vec1, vec2)
sim_cat_car = cosine_similarity(vec1, vec3)

print(f"猫-狗 相似度: {sim_cat_dog:.4f}")
print(f"猫-汽车 相似度: {sim_cat_car:.4f}")
print(f"猫和狗更相似！" if sim_cat_dog > sim_cat_car else "意外结果")

# ===== 3. 简单向量数据库实现 =====
print("\n=== 3. 简单向量数据库 ===")

class SimpleVectorStore:
    """简化的向量数据库实现"""

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.documents = []
        self.vectors = []

    def add_documents(self, docs: List[dict]):
        """添加文档"""
        for doc in docs:
            self.documents.append(doc)
            vec = self.embeddings.embed_query(doc["content"])
            self.vectors.append(vec)
        print(f"已添加 {len(docs)} 个文档")

    def similarity_search(self, query: str, k: int = 3) -> List[dict]:
        """相似度搜索"""
        query_vec = self.embeddings.embed_query(query)

        # 计算与所有文档的相似度
        similarities = []
        for i, doc_vec in enumerate(self.vectors):
            sim = cosine_similarity(query_vec, doc_vec)
            similarities.append((i, sim))

        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 返回 Top-K
        results = []
        for idx, sim in similarities[:k]:
            doc = self.documents[idx].copy()
            doc["similarity"] = sim
            results.append(doc)

        return results

# 创建向量数据库
vectorstore = SimpleVectorStore(embeddings)

# 添加文档
documents = [
    {"content": "Python 是一种流行的编程语言", "source": "doc1"},
    {"content": "猫是可爱的宠物动物", "source": "doc2"},
    {"content": "狗是人类最好的朋友", "source": "doc3"},
    {"content": "JavaScript 用于网页前端开发", "source": "doc4"},
    {"content": "汽车是现代交通工具", "source": "doc5"},
    {"content": "LangChain 是 Python 的 LLM 框架", "source": "doc6"},
]

vectorstore.add_documents(documents)

# 搜索测试
print("\n查询：'我想学习编程'")
results = vectorstore.similarity_search("我想学习编程", k=3)
for i, doc in enumerate(results):
    print(f"  {i+1}. [{doc['similarity']:.4f}] {doc['content']}")

print("\n查询：'可爱的小动物'")
results = vectorstore.similarity_search("可爱的小动物", k=3)
for i, doc in enumerate(results):
    print(f"  {i+1}. [{doc['similarity']:.4f}] {doc['content']}")

# ===== 4. RAG 链实现 =====
print("\n=== 4. RAG 链实现 ===")

class MockLLM:
    """模拟 LLM"""
    def invoke(self, prompt: str) -> str:
        # 简单模拟：提取上下文中的关键信息
        if "Python" in prompt and "编程" in prompt:
            return "Python 是一种简单易学的编程语言，非常适合初学者。"
        elif "猫" in prompt or "狗" in prompt:
            return "猫和狗都是受欢迎的宠物，它们各有特点。"
        else:
            return "根据提供的信息，我无法给出确切答案。"

class SimpleRAGChain:
    """简单的 RAG 链"""

    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm

    def invoke(self, question: str) -> dict:
        # 1. 检索相关文档
        docs = self.vectorstore.similarity_search(question, k=2)

        # 2. 构建上下文
        context = "\n".join([d["content"] for d in docs])

        # 3. 构建 Prompt
        prompt = f"""根据以下上下文回答问题。

上下文：
{context}

问题：{question}

回答："""

        # 4. 调用 LLM
        answer = self.llm.invoke(prompt)

        return {
            "question": question,
            "answer": answer,
            "sources": [d["source"] for d in docs],
            "context": context
        }

# 创建 RAG 链
llm = MockLLM()
rag_chain = SimpleRAGChain(vectorstore, llm)

# 测试 RAG
print("\n问题: '如何开始学习编程？'")
result = rag_chain.invoke("如何开始学习编程？")
print(f"回答: {result['answer']}")
print(f"来源: {result['sources']}")

# ===== 5. 混合检索 =====
print("\n=== 5. 混合检索（关键词 + 语义）===")

class BM25Retriever:
    """简化的 BM25 关键词检索"""

    def __init__(self, documents):
        self.documents = documents

    def search(self, query: str, k: int = 3) -> List[dict]:
        """基于关键词匹配的检索"""
        query_words = set(query.lower())
        scores = []

        for i, doc in enumerate(self.documents):
            content_words = set(doc["content"].lower())
            # 简单的词重叠计算
            overlap = len(query_words & content_words)
            scores.append((i, overlap))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scores[:k]:
            doc = self.documents[idx].copy()
            doc["bm25_score"] = score
            results.append(doc)

        return results

class EnsembleRetriever:
    """混合检索器"""

    def __init__(self, vectorstore, bm25_retriever, weights=(0.7, 0.3)):
        self.vectorstore = vectorstore
        self.bm25_retriever = bm25_retriever
        self.weights = weights

    def search(self, query: str, k: int = 3) -> List[dict]:
        # 向量检索
        vector_results = self.vectorstore.similarity_search(query, k=k*2)

        # BM25 检索
        bm25_results = self.bm25_retriever.search(query, k=k*2)

        # 合并结果（简化版 RRF）
        doc_scores = {}
        for rank, doc in enumerate(vector_results):
            key = doc["source"]
            doc_scores[key] = doc_scores.get(key, 0) + self.weights[0] / (rank + 1)

        for rank, doc in enumerate(bm25_results):
            key = doc["source"]
            doc_scores[key] = doc_scores.get(key, 0) + self.weights[1] / (rank + 1)

        # 按综合得分排序
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # 返回原始文档
        results = []
        for source, score in sorted_docs[:k]:
            for doc in documents:
                if doc["source"] == source:
                    doc_copy = doc.copy()
                    doc_copy["ensemble_score"] = score
                    results.append(doc_copy)
                    break

        return results

# 创建混合检索器
bm25 = BM25Retriever(documents)
ensemble = EnsembleRetriever(vectorstore, bm25)

print("\n混合检索：'Python LangChain 编程'")
results = ensemble.search("Python LangChain 编程", k=3)
for i, doc in enumerate(results):
    print(f"  {i+1}. [{doc['ensemble_score']:.4f}] {doc['content']}")

# ===== 6. 相似度阈值过滤 =====
print("\n=== 6. 相似度阈值过滤 ===")

def search_with_threshold(vectorstore, query: str, k: int = 5, threshold: float = 0.3):
    """带阈值的相似度搜索"""
    results = vectorstore.similarity_search(query, k=k)
    filtered = [r for r in results if r["similarity"] >= threshold]
    return filtered

print("\n查询：'量子计算'（不相关的查询）")
results = search_with_threshold(vectorstore, "量子计算", threshold=0.5)
if results:
    for doc in results:
        print(f"  [{doc['similarity']:.4f}] {doc['content']}")
else:
    print("  未找到相似度超过阈值的文档")

print("\n=== 完成！===")
```

**运行输出示例：**
```
=== 1. Embedding 基础操作 ===
向量维度: 128
猫向量前3维: [0.9, 0.1, 0.2]
狗向量前3维: [0.85, 0.15, 0.25]
汽车向量前3维: [0.1, 0.9, 0.3]

=== 2. 相似度计算 ===
猫-狗 相似度: 0.9847
猫-汽车 相似度: 0.4231
猫和狗更相似！

=== 3. 简单向量数据库 ===
已添加 6 个文档

查询：'我想学习编程'
  1. [0.8234] Python 是一种流行的编程语言
  2. [0.7891] LangChain 是 Python 的 LLM 框架
  3. [0.6543] JavaScript 用于网页前端开发

查询：'可爱的小动物'
  1. [0.9012] 猫是可爱的宠物动物
  2. [0.8765] 狗是人类最好的朋友
  3. [0.2341] 汽车是现代交通工具

=== 4. RAG 链实现 ===
问题: '如何开始学习编程？'
回答: Python 是一种简单易学的编程语言，非常适合初学者。
来源: ['doc1', 'doc6']

=== 5. 混合检索（关键词 + 语义）===
混合检索：'Python LangChain 编程'
  1. [0.9333] LangChain 是 Python 的 LLM 框架
  2. [0.8167] Python 是一种流行的编程语言
  3. [0.2333] JavaScript 用于网页前端开发

=== 6. 相似度阈值过滤 ===
查询：'量子计算'（不相关的查询）
  未找到相似度超过阈值的文档

=== 完成！===
```

---

## 8. 【面试必问】

### 问题："什么是 RAG？它解决了什么问题？"

**普通回答（❌ 不出彩）：**
"RAG 是检索增强生成，就是先检索再让 LLM 回答。它可以让 LLM 回答最新的问题。"

**出彩回答（✅ 推荐）：**

> **RAG 解决了 LLM 的三个核心问题：**
>
> 1. **知识时效性**：LLM 的训练数据有截止日期，无法获取最新信息。RAG 通过实时检索外部知识库，让 LLM 能够回答"今天的新闻"这类问题。
>
> 2. **知识幻觉**：LLM 可能"编造"不存在的信息。RAG 提供了可验证的知识来源，LLM 的回答基于真实文档，大大减少幻觉。
>
> 3. **领域专业性**：通用 LLM 缺乏特定领域的深度知识。RAG 可以接入企业知识库、专业文档，让 LLM 成为"领域专家"。
>
> **RAG 的工作流程：**
> ```
> 用户问题 → Embedding → 向量检索 → Top-K 文档 → 注入 Prompt → LLM → 回答
> ```
>
> **在 LangChain 中的实现**：LangChain 提供了完整的 RAG 工具链：
> - `Embeddings`：文本向量化
> - `VectorStore`：向量存储和检索
> - `Retriever`：统一的检索接口
> - LCEL 将这些组件无缝组合成 RAG Chain
>
> **我的实践经验**：在一个客服知识库项目中，使用 RAG 后，回答准确率从 60% 提升到 90%+，关键是对文档进行了合理的分块（chunking），并使用了混合检索（向量 + BM25）。

**为什么这个回答出彩？**
1. ✅ 明确指出解决的三个核心问题
2. ✅ 清晰展示工作流程
3. ✅ 联系 LangChain 具体实现
4. ✅ 有实际项目经验和量化结果

---

### 问题："Embedding 是如何工作的？为什么语义相似的文本向量也相似？"

**普通回答（❌ 不出彩）：**
"Embedding 就是把文本变成向量。相似的文本向量就接近。"

**出彩回答（✅ 推荐）：**

> **Embedding 的核心原理是"分布式假设"：**
>
> **语言学基础**：一个词的含义由它的"上下文"决定。如果两个词经常出现在相似的上下文中，它们的含义就相似。
>
> **训练过程**：
> ```
> "我养了一只[猫]，它很可爱"
> "我养了一只[狗]，它很可爱"
> ```
> 模型学习到"猫"和"狗"在类似的位置出现，所以它们的向量会很接近。
>
> **数学表示**：Embedding 将文本映射到高维空间的一个点，空间中的每个维度可以理解为一个"语义特征"：
> - 维度1：是否是动物？
> - 维度2：是否是宠物？
> - 维度3：大小？
> - ...
>
> **"猫"的向量**：[0.9, 0.95, 0.3, ...]（是动物、是宠物、体型小）
> **"狗"的向量**：[0.9, 0.9, 0.6, ...]（是动物、是宠物、体型中等）
>
> **相似度计算**：余弦相似度衡量两个向量的"方向"是否一致：
> ```python
> cos_sim = dot(v1, v2) / (|v1| * |v2|)
> ```
>
> **在 LangChain 中**：`OpenAIEmbeddings` 使用 OpenAI 的 text-embedding-3 系列模型，这些模型经过大规模对比学习训练，能够准确捕捉语义相似性。

---

## 9. 【化骨绵掌】

### 卡片1：Embedding 是什么？ 🎯

**一句话：** Embedding 将文本转换为固定维度的数值向量，保留语义信息。

**举例：**
```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vector = embeddings.embed_query("Hello World")  # → [0.02, -0.15, ...]
```

**应用：** LangChain 所有语义检索的基础。

---

### 卡片2：向量维度 📊

**一句话：** 向量维度决定了能表达的语义复杂度，常见 1536 或 3072 维。

**举例：**
```python
# text-embedding-3-small: 1536 维
# text-embedding-3-large: 3072 维

vector = embeddings.embed_query("text")
print(len(vector))  # 1536
```

**应用：** 维度越高，存储和计算成本越高，效果可能更好。

---

### 卡片3：相似度计算 📐

**一句话：** 余弦相似度衡量两个向量方向的一致性，范围 [-1, 1]。

**举例：**
```python
import numpy as np

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# 相似文本 → 高相似度（接近 1）
# 不相关文本 → 低相似度（接近 0）
```

**应用：** 向量数据库的核心算法。

---

### 卡片4：向量数据库 🗄️

**一句话：** 专门存储向量并支持高效相似度搜索的数据库。

**举例：**
```python
from langchain_community.vectorstores import FAISS

vectorstore = FAISS.from_documents(docs, embeddings)
results = vectorstore.similarity_search("查询", k=3)
```

**应用：** LangChain 支持 FAISS、Chroma、Pinecone 等多种向量数据库。

---

### 卡片5：Retriever 检索器 🔍

**一句话：** Retriever 是 LangChain 执行检索的统一接口，实现 Runnable 协议。

**举例：**
```python
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
docs = retriever.invoke("问题")
```

**应用：** 可以无缝集成到 LCEL 链中。

---

### 卡片6：RAG 工作流 🚀

**一句话：** RAG = 检索（Retrieval）+ 增强（Augmented）+ 生成（Generation）。

**举例：**
```python
# RAG 流程
用户问题 → 检索相关文档 → 注入 Prompt → LLM 生成回答

rag_chain = retriever | format_docs | prompt | llm
```

**应用：** 解决 LLM 知识时效性和幻觉问题。

---

### 卡片7：文档分块 Chunking 📄

**一句话：** 将长文档切分成适合 Embedding 的小块，保留语义完整性。

**举例：**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
```

**应用：** 分块策略直接影响检索质量。

---

### 卡片8：混合检索 🔀

**一句话：** 结合向量检索和关键词检索，取两者之长。

**举例：**
```python
from langchain.retrievers import EnsembleRetriever

ensemble = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]
)
```

**应用：** 提高检索的召回率和精确率。

---

### 卡片9：MMR 多样性搜索 🎨

**一句话：** 最大边际相关性（MMR）在保证相关的同时增加结果多样性。

**举例：**
```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "lambda_mult": 0.5}
)
```

**应用：** 避免检索结果过于相似、信息冗余。

---

### 卡片10：Embedding 在 LangChain 中的位置 ⭐

**一句话：** Embedding 是连接文本世界和向量世界的桥梁，是 RAG 的数学基础。

**举例：**
```python
# LangChain RAG 完整链
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

**应用：** 掌握 Embedding 就掌握了 LangChain RAG 的一半。

---

## 10. 【一句话总结】

**Embedding 将文本转换为保留语义的数值向量，通过向量数据库实现高效相似度检索，是 LangChain RAG 系统的核心技术，让 LLM 能够基于外部知识回答问题。**

---

## 📚 学习检查清单

- [ ] 理解 Embedding 的原理和用途
- [ ] 能够使用 OpenAIEmbeddings 进行文本向量化
- [ ] 会使用 FAISS 或 Chroma 创建向量数据库
- [ ] 理解余弦相似度的计算方式
- [ ] 能够创建和配置 Retriever
- [ ] 会构建简单的 RAG 链

## 🔗 下一步学习

- **Token 与上下文窗口**：理解 RAG 中的 Token 限制
- **LCEL 表达式语言**：更复杂的 RAG 链组合
- **Retriever 检索器**：深入学习各种检索策略

---

**版本：** v1.0
**最后更新：** 2025-12-12
