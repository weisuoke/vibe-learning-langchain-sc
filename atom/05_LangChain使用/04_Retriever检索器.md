# Retriever 检索器

> 原子化知识点 | LangChain 使用 | LangChain 源码学习核心知识

---

## 1. 【30字核心】

**Retriever 是 RAG 模式的核心组件，负责从知识库中检索与查询相关的文档，VectorStoreRetriever 是最常用的实现。**

---

## 2. 【第一性原理】

### 什么是第一性原理？

**第一性原理**：回到事物最基本的真理，从源头思考问题

### Retriever 检索器的第一性原理 🎯

#### 1. 最基础的定义

**Retriever = 根据查询找到相关文档的组件**

仅此而已！没有更基础的了。

```python
# Retriever 的本质
def retrieve(query: str) -> List[Document]:
    # 输入：用户问题
    # 输出：相关文档列表
    relevant_docs = search_knowledge_base(query)
    return relevant_docs
```

#### 2. 为什么需要 Retriever？

**核心问题：LLM 的知识有局限**

```python
# LLM 的局限性
# ❌ 知识截止日期（2023年之前）
# ❌ 不知道你的私有数据（公司文档、个人笔记）
# ❌ 不知道实时信息（今天的新闻）

# Retriever 的解决方案
# ✅ 从你的知识库检索相关信息
# ✅ 把信息作为上下文传给 LLM
# ✅ LLM 基于这些信息回答

user_question = "我们公司的退款政策是什么？"
relevant_docs = retriever.invoke(user_question)
# [Document(content="退款政策：30天内可退...")]

# LLM 看到文档后可以准确回答
```

#### 3. Retriever 的三层价值

##### 价值1：扩展知识范围

```python
# 让 LLM 能回答它本来不知道的问题
knowledge_base = [
    "公司成立于2020年",
    "退款政策：30天内全额退款",
    "技术架构使用微服务",
    ...  # 公司内部文档
]

# 用户问：公司什么时候成立的？
# LLM 本身不知道，但 Retriever 找到文档后就能回答
```

##### 价值2：提供准确信息

```python
# 避免 LLM "幻觉"（编造信息）

# ❌ 没有 Retriever
# 用户：我们的API限流是多少？
# LLM：（可能编造）大概是每分钟100次请求吧...

# ✅ 有 Retriever
# Retriever 找到：API限流文档说明...
# LLM：根据文档，API限流是每分钟1000次请求。
```

##### 价值3：实现问答系统

```python
# RAG = Retrieval Augmented Generation
# 检索增强生成

rag_pipeline = (
    检索相关文档
    → 将文档作为上下文
    → LLM 生成回答
)
```

#### 4. 从第一性原理推导 RAG

**推理链：**

```
1. LLM 知识有限
   ↓
2. 需要访问外部知识
   ↓
3. 如何找到相关的知识？
   ↓
4. 需要检索机制
   ↓
5. 文本如何比较相似度？
   ↓
6. 使用 Embedding 向量化
   ↓
7. 向量相似度检索
   ↓
8. 这就是 VectorStoreRetriever
```

#### 5. 一句话总结第一性原理

**Retriever 是连接 LLM 和外部知识的桥梁，通过检索相关文档为 LLM 提供上下文，实现基于知识库的精准问答。**

---

## 3. 【核心概念（全面覆盖）】

### 核心概念1：BaseRetriever 接口 🔍

**BaseRetriever 是所有检索器的抽象基类，定义了统一的检索接口**

```python
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List

# BaseRetriever 的核心接口
class BaseRetriever:
    """检索器基类"""

    def invoke(self, input: str) -> List[Document]:
        """检索相关文档（推荐使用）"""
        return self._get_relevant_documents(input)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """抽象方法：子类必须实现"""
        raise NotImplementedError

    async def ainvoke(self, input: str) -> List[Document]:
        """异步检索"""
        return await self._aget_relevant_documents(input)
```

**自定义 Retriever 示例：**

```python
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List

class KeywordRetriever(BaseRetriever):
    """基于关键词的简单检索器"""

    documents: List[Document]

    def _get_relevant_documents(self, query: str) -> List[Document]:
        """根据关键词匹配检索"""
        results = []
        query_words = query.lower().split()

        for doc in self.documents:
            content_lower = doc.page_content.lower()
            if any(word in content_lower for word in query_words):
                results.append(doc)

        return results[:4]  # 返回前4个

# 使用
docs = [
    Document(page_content="Python 是一种编程语言"),
    Document(page_content="LangChain 是 AI 框架"),
    Document(page_content="机器学习需要数据"),
]

retriever = KeywordRetriever(documents=docs)
results = retriever.invoke("Python 编程")
# [Document(page_content="Python 是一种编程语言")]
```

**在 LangChain 源码中的应用：**

```python
# langchain_core/retrievers.py
class BaseRetriever(RunnableSerializable, ABC):
    """检索器基类，同时是 Runnable"""

    class Config:
        arbitrary_types_allowed = True

    @abstractmethod
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """检索相关文档"""

    # Runnable 接口实现
    def invoke(
        self, input: str, config: Optional[RunnableConfig] = None
    ) -> List[Document]:
        return self._get_relevant_documents(input)
```

---

### 核心概念2：VectorStoreRetriever 向量检索 📊

**VectorStoreRetriever 基于向量相似度检索，是最常用的检索器**

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# 1. 准备文档
documents = [
    Document(page_content="Python 是一种解释型编程语言", metadata={"source": "python.md"}),
    Document(page_content="LangChain 是构建 LLM 应用的框架", metadata={"source": "langchain.md"}),
    Document(page_content="机器学习是人工智能的一个分支", metadata={"source": "ml.md"}),
]

# 2. 创建 Embedding 模型
embeddings = OpenAIEmbeddings()

# 3. 创建 VectorStore
vectorstore = FAISS.from_documents(documents, embeddings)

# 4. 创建 Retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",  # 相似度搜索
    search_kwargs={"k": 4}     # 返回 4 个结果
)

# 5. 检索
results = retriever.invoke("什么是 LangChain？")
for doc in results:
    print(f"内容: {doc.page_content}")
    print(f"来源: {doc.metadata.get('source')}")
```

**search_type 选项：**

| 类型 | 说明 | 适用场景 |
|-----|------|---------|
| `similarity` | 纯相似度排序 | 通用场景 |
| `mmr` | 最大边际相关性（多样性） | 避免结果太相似 |
| `similarity_score_threshold` | 带阈值过滤 | 质量要求高 |

```python
# MMR 检索：平衡相关性和多样性
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 4,
        "fetch_k": 20,       # 先取 20 个
        "lambda_mult": 0.5   # 多样性权重
    }
)

# 带阈值的检索
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "score_threshold": 0.8,  # 只返回相似度 > 0.8 的
        "k": 4
    }
)
```

---

### 核心概念3：RAG Pattern 检索增强生成 🔗

**RAG (Retrieval Augmented Generation) 是检索 + 生成的完整流程**

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# RAG Chain 模板
template = """基于以下上下文回答问题。如果上下文中没有相关信息，请说"我不知道"。

上下文：
{context}

问题：{question}

回答："""

prompt = ChatPromptTemplate.from_template(template)
llm = ChatOpenAI()

# 构建 RAG Chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 使用
answer = rag_chain.invoke("LangChain 是什么？")
print(answer)
```

**RAG 流程图：**

```
用户问题: "LangChain 是什么？"
         ↓
    [Retriever 检索]
         ↓
    相关文档: ["LangChain 是构建 LLM 应用的框架..."]
         ↓
    [构建 Prompt]
    "上下文：LangChain 是...
     问题：LangChain 是什么？"
         ↓
    [LLM 生成]
         ↓
    回答: "LangChain 是一个用于构建 LLM 应用的框架..."
```

---

### 核心概念4：Document 文档对象 📄

**Document 是 LangChain 中表示文档的标准数据结构**

```python
from langchain_core.documents import Document

# 创建文档
doc = Document(
    page_content="这是文档的内容",
    metadata={
        "source": "docs/readme.md",
        "page": 1,
        "author": "张三",
        "date": "2024-01-01"
    }
)

# 访问属性
print(doc.page_content)  # 内容
print(doc.metadata)      # 元数据

# 批量创建
docs = [
    Document(page_content="文档1", metadata={"id": 1}),
    Document(page_content="文档2", metadata={"id": 2}),
]
```

**元数据的用途：**

```python
# 1. 过滤检索结果
results = retriever.invoke("查询", filter={"source": "official"})

# 2. 追踪来源
for doc in results:
    print(f"来源: {doc.metadata.get('source')}")

# 3. 构建引用
answer = f"{response}\n\n参考来源：{doc.metadata.get('source')}"
```

---

### 扩展概念5：EnsembleRetriever 多路召回 🔀

**EnsembleRetriever 组合多个检索器，融合不同检索策略**

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# 创建不同类型的检索器
# 1. 向量检索（语义相似）
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 2. BM25 检索（关键词匹配）
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 4

# 3. 组合检索器
ensemble_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]  # 向量检索权重更高
)

# 使用
results = ensemble_retriever.invoke("Python 编程语言")
# 融合两种检索的结果
```

**多路召回的优势：**

| 检索方式 | 优点 | 缺点 |
|---------|------|------|
| 向量检索 | 语义理解好 | 可能忽略关键词 |
| BM25 | 关键词精确 | 缺乏语义理解 |
| Ensemble | 两者结合 | 计算成本高 |

---

### 扩展概念6：文档加载与分割 📚

**完整的 RAG 流程包括文档加载和分割**

```python
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. 加载文档
loader = TextLoader("docs/readme.md")
documents = loader.load()

# 或者加载 PDF
# loader = PyPDFLoader("docs/manual.pdf")
# documents = loader.load()

# 2. 分割文档（因为文档可能太长）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,    # 每块 1000 字符
    chunk_overlap=200,  # 重叠 200 字符
    separators=["\n\n", "\n", "。", "，", " ", ""]
)

splits = text_splitter.split_documents(documents)
print(f"原始 {len(documents)} 个文档，分割为 {len(splits)} 个块")

# 3. 创建向量存储
vectorstore = FAISS.from_documents(splits, embeddings)

# 4. 创建检索器
retriever = vectorstore.as_retriever()
```

---

## 4. 【最小可用】

掌握以下内容，就能在 LangChain 中使用 Retriever：

### 4.1 创建向量检索器

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

# 准备文档
docs = [
    Document(page_content="Python 是编程语言"),
    Document(page_content="LangChain 是 AI 框架"),
]

# 创建检索器
vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# 检索
results = retriever.invoke("什么是 Python？")
```

### 4.2 构建 RAG Chain

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

template = """根据上下文回答：
{context}

问题：{question}"""

prompt = ChatPromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("LangChain 是什么？")
```

### 4.3 文档加载与分割

```python
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 加载
docs = TextLoader("readme.md").load()

# 分割
splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
splits = splitter.split_documents(docs)
```

### 4.4 自定义检索器

```python
from langchain_core.retrievers import BaseRetriever

class MyRetriever(BaseRetriever):
    def _get_relevant_documents(self, query: str):
        # 自定义检索逻辑
        return search_my_database(query)
```

**这些知识足以：**
- 构建基本的 RAG 应用
- 从文件创建知识库
- 实现问答系统
- 自定义检索逻辑

---

## 5. 【1个类比】（双轨制）

### 类比1：Retriever 是图书管理员

#### 🎨 前端视角：API 数据获取 / SWR

Retriever 就像前端的数据获取层，根据查询条件获取数据。

```javascript
// SWR / React Query
const { data } = useSWR(
  `/api/search?q=${query}`,
  fetcher
);

// 或者自定义 fetcher
async function searchDocs(query) {
  const response = await fetch(`/api/docs?q=${encodeURIComponent(query)}`);
  return response.json();
}
```

```python
# LangChain Retriever
results = retriever.invoke(query)
# 返回相关文档列表
```

**关键相似点：**
- 都是根据查询获取数据
- 都有缓存/索引优化
- 都返回结构化结果

#### 🧒 小朋友视角：图书管理员

Retriever 就像图书馆的管理员：

```
你去图书馆：
"我想找关于恐龙的书"

图书管理员（Retriever）：
1. 听懂你的问题
2. 在书架上搜索
3. 找到相关的书
4. 把书给你

"给你，这 3 本都是讲恐龙的！"
[恐龙百科] [恐龙的秘密] [恐龙大发现]
```

---

### 类比2：VectorStore 是智能书架

#### 🎨 前端视角：搜索索引 / Elasticsearch

VectorStore 就像一个智能搜索索引，支持语义搜索。

```javascript
// Elasticsearch 全文搜索
const results = await client.search({
  index: 'documents',
  body: {
    query: {
      match: {
        content: 'Python 编程'
      }
    }
  }
});

// 向量搜索
const results = await client.search({
  body: {
    knn: {
      field: 'embedding',
      query_vector: [0.1, 0.2, ...],
      k: 10
    }
  }
});
```

#### 🧒 小朋友视角：智能玩具收纳箱

VectorStore 就像一个智能收纳箱：

```
普通收纳箱：
按颜色分类 → 红色区、蓝色区、绿色区
只能找"红色玩具"

智能收纳箱（VectorStore）：
按"感觉"分类 →
  "开心的玩具"放一起
  "刺激的玩具"放一起
  "安静的玩具"放一起

你说"我想玩开心的"
智能收纳箱就能找到气球、积木、玩偶...
（它们颜色不同，但都是"开心的"）
```

---

### 类比3：RAG 是先查资料再回答

#### 🎨 前端视角：SSR 数据预取

RAG 就像服务端渲染时的数据预取。

```javascript
// Next.js getServerSideProps
export async function getServerSideProps(context) {
  // 1. 获取数据（相当于 Retriever）
  const docs = await fetchRelevantDocs(context.query.q);

  // 2. 传给页面组件（相当于传给 LLM）
  return { props: { docs } };
}

// 页面组件使用数据渲染（相当于 LLM 生成回答）
function Page({ docs }) {
  return <Answer docs={docs} />;
}
```

#### 🧒 小朋友视角：考试时可以查资料

RAG 就像开卷考试：

```
闭卷考试（没有 RAG）：
问：恐龙什么时候灭绝的？
你：呃...好像是很久以前...6500年？（可能答错）

开卷考试（有 RAG）：
问：恐龙什么时候灭绝的？
你：等等，让我查查资料...
   [翻书：恐龙在6500万年前灭绝]
你：恐龙在6500万年前灭绝的！（准确答案）
```

---

### 类比总结表

| LangChain 概念 | 前端类比 | 小朋友类比 |
|---------------|---------|-----------|
| Retriever | API fetcher / SWR | 图书管理员 |
| VectorStore | 搜索索引 / Elasticsearch | 智能收纳箱 |
| Embedding | 数据向量化 / 特征提取 | 给东西贴标签 |
| RAG | SSR 数据预取 | 开卷考试 |
| Document | 数据对象 | 一本书 |
| 相似度搜索 | 模糊搜索 | 找类似的东西 |
| chunk_size | 分页大小 | 把书撕成小纸条 |

---

## 6. 【反直觉点】

### 误区1：Retriever 只能用向量检索 ❌

**为什么错？**
- 还有 BM25（关键词检索）
- 还有 SQL（数据库查询）
- 还有 API（外部服务）
- 可以组合多种方式

**为什么人们容易这样错？**
因为向量检索最常见，教程都从这里开始。

**正确理解：**

```python
# 1. 向量检索
vector_retriever = vectorstore.as_retriever()

# 2. BM25 关键词检索
from langchain_community.retrievers import BM25Retriever
bm25_retriever = BM25Retriever.from_documents(docs)

# 3. SQL 检索
from langchain_community.retrievers import SQLDatabaseRetriever
sql_retriever = SQLDatabaseRetriever(db=database)

# 4. 自定义 API 检索
class APIRetriever(BaseRetriever):
    def _get_relevant_documents(self, query):
        response = requests.get(f"/api/search?q={query}")
        return [Document(page_content=d["content"]) for d in response.json()]

# 5. 组合检索
from langchain.retrievers import EnsembleRetriever
ensemble = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]
)
```

---

### 误区2：RAG 就是把文档塞给 LLM ❌

**为什么错？**
- 需要检索质量优化
- 需要考虑上下文窗口限制
- 需要设计好 prompt
- 可能需要重排序

**为什么人们容易这样错？**
RAG 看起来简单，但细节决定效果。

**正确理解：**

```python
# ❌ 简单但效果差
rag_chain = retriever | prompt | llm

# ✅ 考虑更多因素

# 1. 检索质量：使用 MMR 增加多样性
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20}
)

# 2. 重排序：用小模型对结果重新排序
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker

compressor = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
rerank_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)

# 3. Prompt 设计：明确指示
prompt = """你是一个问答助手。请严格基于以下上下文回答问题。
如果上下文中没有相关信息，请说"根据提供的资料无法回答"。
不要编造信息。

上下文：
{context}

问题：{question}

回答："""

# 4. 上下文长度控制
def limit_context(docs, max_tokens=3000):
    total = 0
    result = []
    for doc in docs:
        tokens = len(doc.page_content) // 4  # 估算
        if total + tokens > max_tokens:
            break
        result.append(doc)
        total += tokens
    return result
```

---

### 误区3：检索越多文档越好 ❌

**为什么错？**
- 太多文档会稀释相关性
- 增加噪音和干扰
- 可能超出上下文窗口
- 增加成本和延迟

**为什么人们容易这样错？**
以为"信息越多越好"。

**正确理解：**

```python
# ❌ 检索太多
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
# 20 个文档，很多可能不相关

# ✅ 适量检索 + 质量控制
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 5,              # 最多 5 个
        "score_threshold": 0.7  # 相似度 > 0.7
    }
)

# 实践建议
# - 一般场景：3-5 个文档
# - 复杂问题：5-10 个文档
# - 使用 MMR 增加多样性而非数量
```

---

## 7. 【实战代码】

```python
"""
示例：Retriever 检索器完整演示
展示 LangChain 中 Retriever 的核心用法
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import math

# ===== 1. Document 数据结构 =====
print("=== 1. Document 数据结构 ===")

@dataclass
class Document:
    """文档对象"""
    page_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        return f"Document(content='{self.page_content[:50]}...', metadata={self.metadata})"

# 创建文档
docs = [
    Document(
        page_content="Python 是一种解释型、面向对象的高级编程语言",
        metadata={"source": "python.md", "category": "编程语言"}
    ),
    Document(
        page_content="LangChain 是一个用于构建 LLM 应用的框架",
        metadata={"source": "langchain.md", "category": "AI框架"}
    ),
    Document(
        page_content="机器学习是人工智能的一个重要分支",
        metadata={"source": "ml.md", "category": "AI"}
    ),
    Document(
        page_content="深度学习使用神经网络处理复杂任务",
        metadata={"source": "dl.md", "category": "AI"}
    ),
    Document(
        page_content="向量数据库用于存储和检索向量数据",
        metadata={"source": "vectordb.md", "category": "数据库"}
    ),
]

for doc in docs:
    print(f"  {doc}")

# ===== 2. 简单的 Embedding 实现 =====
print("\n=== 2. 简单 Embedding ===")

class SimpleEmbedding:
    """简单的 Embedding（基于词频）"""

    def __init__(self, vocabulary: List[str] = None):
        self.vocabulary = vocabulary or []

    def fit(self, texts: List[str]):
        """构建词汇表"""
        all_words = set()
        for text in texts:
            words = text.lower().split()
            all_words.update(words)
        self.vocabulary = list(all_words)
        print(f"词汇表大小: {len(self.vocabulary)}")

    def embed(self, text: str) -> List[float]:
        """将文本转换为向量"""
        text_lower = text.lower()
        vector = []
        for word in self.vocabulary:
            # 简单的词频
            count = text_lower.count(word)
            vector.append(count)
        # 归一化
        norm = math.sqrt(sum(x**2 for x in vector)) or 1
        return [x / norm for x in vector]

# 创建 Embedding
embedding = SimpleEmbedding()
embedding.fit([doc.page_content for doc in docs])

# 测试
vec = embedding.embed("Python 编程语言")
print(f"向量维度: {len(vec)}")
print(f"向量（前10维）: {vec[:10]}")

# ===== 3. VectorStore 实现 =====
print("\n=== 3. VectorStore ===")

class SimpleVectorStore:
    """简单的向量存储"""

    def __init__(self, embedding: SimpleEmbedding):
        self.embedding = embedding
        self.documents: List[Document] = []
        self.vectors: List[List[float]] = []

    def add_documents(self, docs: List[Document]):
        """添加文档"""
        for doc in docs:
            self.documents.append(doc)
            vec = self.embedding.embed(doc.page_content)
            self.vectors.append(vec)
        print(f"添加 {len(docs)} 个文档")

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """相似度搜索"""
        query_vec = self.embedding.embed(query)

        # 计算相似度（余弦相似度）
        scores = []
        for i, doc_vec in enumerate(self.vectors):
            similarity = sum(a * b for a, b in zip(query_vec, doc_vec))
            scores.append((i, similarity))

        # 排序取 top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        results = [self.documents[i] for i, _ in scores[:k]]
        return results

    def as_retriever(self, search_kwargs: Dict = None) -> "VectorStoreRetriever":
        """创建检索器"""
        search_kwargs = search_kwargs or {"k": 4}
        return VectorStoreRetriever(vectorstore=self, search_kwargs=search_kwargs)

# 创建向量存储
vectorstore = SimpleVectorStore(embedding)
vectorstore.add_documents(docs)

# 测试搜索
print("\n搜索 'Python 编程':")
results = vectorstore.similarity_search("Python 编程", k=2)
for doc in results:
    print(f"  {doc.page_content[:50]}...")

# ===== 4. BaseRetriever 实现 =====
print("\n=== 4. Retriever 实现 ===")

class BaseRetriever:
    """检索器基类"""

    def invoke(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        raise NotImplementedError

class VectorStoreRetriever(BaseRetriever):
    """向量存储检索器"""

    def __init__(self, vectorstore: SimpleVectorStore, search_kwargs: Dict = None):
        self.vectorstore = vectorstore
        self.search_kwargs = search_kwargs or {"k": 4}

    def _get_relevant_documents(self, query: str) -> List[Document]:
        k = self.search_kwargs.get("k", 4)
        return self.vectorstore.similarity_search(query, k=k)

# 创建检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 测试检索
print("\n检索 '人工智能':")
results = retriever.invoke("人工智能")
for doc in results:
    print(f"  - {doc.page_content[:50]}...")

# ===== 5. 自定义检索器 =====
print("\n=== 5. 自定义检索器 ===")

class KeywordRetriever(BaseRetriever):
    """关键词检索器"""

    def __init__(self, documents: List[Document]):
        self.documents = documents

    def _get_relevant_documents(self, query: str) -> List[Document]:
        query_words = set(query.lower().split())
        results = []

        for doc in self.documents:
            content_lower = doc.page_content.lower()
            # 计算匹配的关键词数量
            matches = sum(1 for word in query_words if word in content_lower)
            if matches > 0:
                results.append((doc, matches))

        # 按匹配数排序
        results.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in results[:4]]

# 测试
keyword_retriever = KeywordRetriever(docs)
print("\n关键词检索 '深度学习 神经网络':")
results = keyword_retriever.invoke("深度学习 神经网络")
for doc in results:
    print(f"  - {doc.page_content[:50]}...")

# ===== 6. Ensemble 组合检索 =====
print("\n=== 6. Ensemble 组合检索 ===")

class EnsembleRetriever(BaseRetriever):
    """组合检索器"""

    def __init__(self, retrievers: List[BaseRetriever], weights: List[float] = None):
        self.retrievers = retrievers
        self.weights = weights or [1.0 / len(retrievers)] * len(retrievers)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # 收集所有结果
        all_docs = {}

        for retriever, weight in zip(self.retrievers, self.weights):
            results = retriever.invoke(query)
            for i, doc in enumerate(results):
                key = doc.page_content
                score = weight * (len(results) - i)  # 位置越靠前分数越高
                if key in all_docs:
                    all_docs[key] = (doc, all_docs[key][1] + score)
                else:
                    all_docs[key] = (doc, score)

        # 按总分排序
        sorted_docs = sorted(all_docs.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in sorted_docs[:4]]

# 创建组合检索器
ensemble = EnsembleRetriever(
    retrievers=[retriever, keyword_retriever],
    weights=[0.7, 0.3]
)

print("\n组合检索 'Python 编程语言':")
results = ensemble.invoke("Python 编程语言")
for doc in results:
    print(f"  - {doc.page_content[:50]}...")

# ===== 7. RAG Chain 实现 =====
print("\n=== 7. RAG Chain ===")

class MockLLM:
    """模拟 LLM"""

    def invoke(self, prompt: str) -> str:
        # 简单的规则响应
        if "Python" in prompt:
            return "根据上下文，Python 是一种解释型、面向对象的高级编程语言。"
        if "LangChain" in prompt:
            return "根据上下文，LangChain 是一个用于构建 LLM 应用的框架。"
        return "根据提供的上下文，我找到了相关信息。"

def format_docs(docs: List[Document]) -> str:
    """格式化文档为字符串"""
    return "\n\n".join(doc.page_content for doc in docs)

class RAGChain:
    """RAG Chain 实现"""

    def __init__(self, retriever: BaseRetriever, llm: MockLLM):
        self.retriever = retriever
        self.llm = llm

    def invoke(self, question: str) -> Dict[str, Any]:
        # 1. 检索
        docs = self.retriever.invoke(question)
        context = format_docs(docs)

        # 2. 构建 prompt
        prompt = f"""基于以下上下文回答问题：

上下文：
{context}

问题：{question}

回答："""

        # 3. 生成回答
        answer = self.llm.invoke(prompt)

        return {
            "question": question,
            "context": context,
            "answer": answer,
            "source_documents": docs
        }

# 创建 RAG Chain
llm = MockLLM()
rag_chain = RAGChain(retriever, llm)

# 测试
print("\n问：什么是 Python？")
result = rag_chain.invoke("什么是 Python？")
print(f"答：{result['answer']}")
print(f"来源文档数：{len(result['source_documents'])}")

# ===== 8. 文档分割 =====
print("\n=== 8. 文档分割 ===")

class TextSplitter:
    """文本分割器"""

    def __init__(self, chunk_size: int = 100, chunk_overlap: int = 20):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """分割文本"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap
        return chunks

    def split_documents(self, docs: List[Document]) -> List[Document]:
        """分割文档"""
        result = []
        for doc in docs:
            chunks = self.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                new_doc = Document(
                    page_content=chunk,
                    metadata={**doc.metadata, "chunk_index": i}
                )
                result.append(new_doc)
        return result

# 测试分割
long_doc = Document(
    page_content="这是一段很长的文本。" * 20,
    metadata={"source": "long.md"}
)

splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
splits = splitter.split_documents([long_doc])
print(f"原始文档: {len(long_doc.page_content)} 字符")
print(f"分割后: {len(splits)} 个块")
for i, split in enumerate(splits[:3]):
    print(f"  块 {i}: {len(split.page_content)} 字符")

# ===== 9. 带元数据过滤 =====
print("\n=== 9. 元数据过滤 ===")

class FilteredRetriever(BaseRetriever):
    """支持元数据过滤的检索器"""

    def __init__(self, base_retriever: BaseRetriever, filter_fn=None):
        self.base_retriever = base_retriever
        self.filter_fn = filter_fn

    def _get_relevant_documents(self, query: str) -> List[Document]:
        docs = self.base_retriever.invoke(query)
        if self.filter_fn:
            docs = [doc for doc in docs if self.filter_fn(doc)]
        return docs

# 只检索 AI 相关的文档
ai_retriever = FilteredRetriever(
    retriever,
    filter_fn=lambda doc: doc.metadata.get("category") == "AI"
)

print("\n只检索 AI 类别的文档:")
results = ai_retriever.invoke("学习")
for doc in results:
    print(f"  - [{doc.metadata.get('category')}] {doc.page_content[:30]}...")

print("\n=== 完成！===")
```

**运行输出示例：**
```
=== 1. Document 数据结构 ===
  Document(content='Python 是一种解释型、面向对象的高级编程语言...', metadata={'source': 'python.md'})
  ...

=== 2. 简单 Embedding ===
词汇表大小: 28
向量维度: 28

=== 3. VectorStore ===
添加 5 个文档

搜索 'Python 编程':
  Python 是一种解释型、面向对象的高级编程语言...

=== 4. Retriever 实现 ===

检索 '人工智能':
  - 机器学习是人工智能的一个重要分支...
  - 深度学习使用神经网络处理复杂任务...

=== 7. RAG Chain ===

问：什么是 Python？
答：根据上下文，Python 是一种解释型、面向对象的高级编程语言。
来源文档数：3

=== 完成！===
```

---

## 8. 【面试必问】

### 问题1："什么是 RAG？为什么需要它？"

**普通回答（❌ 不出彩）：**
"RAG 是检索增强生成，先检索文档再让 LLM 回答。"

**出彩回答（✅ 推荐）：**

> **RAG (Retrieval Augmented Generation) 解决 LLM 的知识局限：**
>
> **为什么需要 RAG：**
> - LLM 知识有截止日期
> - LLM 不知道私有数据
> - LLM 可能产生"幻觉"
>
> **RAG 工作流程：**
> ```
> 1. 用户问题 → 2. 检索相关文档 → 3. 文档作为上下文 → 4. LLM 生成回答
> ```
>
> **核心组件：**
> - **Retriever**：检索相关文档
> - **VectorStore**：存储和索引文档
> - **Embedding**：将文本转换为向量
>
> **关键优化点：**
> - 检索质量：使用 MMR 增加多样性
> - 文档分割：合理的 chunk_size
> - Prompt 设计：明确指示 LLM 基于上下文回答
>
> **实际经验**：在企业知识库项目中，我使用 RAG 让 LLM 能回答公司内部文档的问题。通过 Ensemble Retriever（向量+BM25）提升了 30% 的检索准确率。

**为什么这个回答出彩？**
1. ✅ 解释了 RAG 的必要性
2. ✅ 清晰的流程说明
3. ✅ 提到了优化点
4. ✅ 有实际项目经验

---

### 问题2："如何提升 RAG 的效果？"

**出彩回答（✅ 推荐）：**

> **RAG 优化从三个维度：**
>
> **1. 检索质量**
> ```python
> # 多路召回
> ensemble = EnsembleRetriever([vector, bm25], weights=[0.7, 0.3])
>
> # 重排序
> reranker = CrossEncoderReranker()
>
> # MMR 多样性
> retriever = vectorstore.as_retriever(search_type="mmr")
> ```
>
> **2. 文档处理**
> ```python
> # 合理分块
> splitter = RecursiveCharacterTextSplitter(
>     chunk_size=500,
>     chunk_overlap=50
> )
>
> # 保留元数据
> metadata={"source": "...", "section": "..."}
> ```
>
> **3. Prompt 优化**
> ```python
> prompt = """严格基于上下文回答，不要编造。
> 如果上下文不足，请说明。
> 引用来源。"""
> ```

---

## 9. 【化骨绵掌】

### 卡片1：Retriever 是什么？ 🎯

**一句话：** Retriever 根据查询从知识库中检索相关文档。

**举例：**
```python
docs = retriever.invoke("什么是 Python？")
# 返回相关文档列表
```

**应用：** RAG 应用的核心组件。

---

### 卡片2：VectorStore 向量存储 📊

**一句话：** 存储文档向量，支持相似度搜索。

**举例：**
```python
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever()
```

**应用：** 语义搜索的基础设施。

---

### 卡片3：Embedding 文本向量化 🔢

**一句话：** 将文本转换为数值向量，相似文本的向量距离更近。

**举例：**
```python
embeddings = OpenAIEmbeddings()
vector = embeddings.embed_query("Python 编程")
```

**应用：** 实现语义相似度计算。

---

### 卡片4：RAG Pattern 检索增强生成 🔗

**一句话：** 先检索相关文档，再让 LLM 基于文档生成回答。

**举例：**
```python
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt | llm
)
```

**应用：** 知识库问答系统。

---

### 卡片5：Document 文档对象 📄

**一句话：** LangChain 中表示文档的标准结构，包含内容和元数据。

**举例：**
```python
doc = Document(
    page_content="内容",
    metadata={"source": "file.md"}
)
```

**应用：** 所有文档处理的基础。

---

### 卡片6：search_type 搜索类型 🔍

**一句话：** similarity（相似度）、mmr（多样性）、threshold（阈值过滤）。

**举例：**
```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4}
)
```

**应用：** 根据场景选择合适的搜索策略。

---

### 卡片7：EnsembleRetriever 多路召回 🔀

**一句话：** 组合多个检索器，融合不同检索策略的优势。

**举例：**
```python
ensemble = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]
)
```

**应用：** 提升检索准确率。

---

### 卡片8：TextSplitter 文档分割 ✂️

**一句话：** 将长文档分割成适合检索的小块。

**举例：**
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
```

**应用：** 处理长文档的必要步骤。

---

### 卡片9：自定义 Retriever 🔧

**一句话：** 继承 BaseRetriever，实现 _get_relevant_documents 方法。

**举例：**
```python
class MyRetriever(BaseRetriever):
    def _get_relevant_documents(self, query):
        return search_my_database(query)
```

**应用：** 接入自定义数据源。

---

### 卡片10：Retriever 在 LangChain 源码中的位置 ⭐

**一句话：** Retriever 实现 Runnable 接口，可以直接用在 LCEL Chain 中。

**举例：**
```python
# Retriever 作为 Runnable
chain = {"context": retriever, "q": RunnablePassthrough()} | prompt
```

**应用：** 理解 Retriever 与 LCEL 的无缝集成。

---

## 10. 【一句话总结】

**Retriever 是 RAG 模式的核心组件，负责从知识库检索与查询相关的文档，VectorStoreRetriever 基于向量相似度是最常用的实现，配合 Embedding 和合理的检索策略可以大幅提升 LLM 问答的准确性。**

---

## 📚 学习检查清单

- [ ] 理解 Retriever 在 RAG 中的作用
- [ ] 会创建 VectorStoreRetriever
- [ ] 能够构建完整的 RAG Chain
- [ ] 了解不同 search_type 的区别
- [ ] 知道如何分割长文档
- [ ] 会使用 EnsembleRetriever 多路召回

## 🔗 下一步学习

- **Callback 回调系统**：监控检索过程
- **Agent 与 Retriever**：让 Agent 使用检索工具
- **高级 RAG**：重排序、查询扩展、假设文档

---

**版本：** v1.0
**最后更新：** 2025-01-14
