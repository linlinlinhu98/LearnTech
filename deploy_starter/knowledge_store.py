"""In-memory knowledge store with httpx-based embeddings.

No external dependencies beyond httpx (pre-installed on Bailian).
Supports both standard OpenAI-compatible /embeddings and
Volcengine multimodal embedding endpoints.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import httpx

try:
    from .llm_utils import load_dotenv
except ImportError:  # Direct execution / local tests
    from llm_utils import load_dotenv

load_dotenv()  # make sure .env secrets (DASHSCOPE_API_KEY etc.) are loaded


@dataclass
class LessonChunk:
    """A chunk of lesson content."""
    id: str
    lesson_id: str
    lesson_title: str
    text: str
    course_title: str = ""


# ---- Helpers ----

def _read_flat_config() -> dict:
    config_path = os.path.join(os.path.dirname(__file__), "config.yml")
    result = {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip().strip("\"'")
                if value.isdigit():
                    result[key] = int(value)
                else:
                    result[key] = value
    except Exception:
        pass
    return result


_config = _read_flat_config()


def _cfg(key: str, default: str = "") -> str:
    env_val = os.getenv(key)
    if env_val is not None:
        return env_val
    val = _config.get(key, default)
    return str(val) if val is not None else default


def _strip_openai_suffix(url: str) -> str:
    value = (url or "").strip()
    for suffix in ("/chat/completions", "/completions", "/responses", "/embeddings"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    return value.rstrip("/")


# ---- Embedding Provider ----

class EmbeddingProvider:
    """httpx-based embedding provider with auto-fallback."""

    def __init__(self) -> None:
        self.dim = int(_cfg("EMBEDDING_DIM", "2048"))
        self.model = _cfg("EMBEDDING_MODEL", "text-embedding-v3")
        self.api_key = _cfg("DASHSCOPE_API_KEY")
        self.api_base = _strip_openai_suffix(
            _cfg("DASHSCOPE_API_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        )
        self._use_multimodal = "volces.com" in self.api_base and self.model.startswith("ep-m")

    @property
    def model_id(self) -> str:
        return self.model

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.api_key:
            return self._noop_embed(texts)
        try:
            return self._call_api(texts)
        except Exception:
            try:
                self._use_multimodal = not self._use_multimodal
                return self._call_api(texts)
            except Exception:
                return self._noop_embed(texts)

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        if self._use_multimodal:
            inp: list = [{"type": "text", "text": t} for t in texts]
            url = f"{self.api_base}/embeddings/multimodal"
            payload: dict = {
                "model": self.model,
                "input": inp,
                "encoding_format": "float",
                "dimensions": self.dim,
            }
        else:
            inp = texts
            url = f"{self.api_base}/embeddings"
            payload: dict = {
                "model": self.model,
                "input": inp,
                "encoding_format": "float",
            }

        with httpx.Client(timeout=60.0) as client:
            resp = client.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            if not resp.is_success:
                raise RuntimeError(f"Embedding API HTTP {resp.status_code}: {resp.text[:300]}")
            data = resp.json()

        vectors: list[list[float]] = []
        if "data" in data:
            items = data["data"]
            if isinstance(items, dict):
                emb = items.get("embedding")
                if emb:
                    vectors = [list(map(float, emb))]
            elif isinstance(items, list):
                items_sorted = sorted(items, key=lambda r: r.get("index", 0))
                for item in items_sorted:
                    emb = item.get("embedding")
                    if emb:
                        vectors.append(list(map(float, emb)))

        if not vectors:
            raise RuntimeError(f"Could not parse embedding response: {str(data)[:300]}")

        if vectors and len(vectors[0]) != self.dim:
            self.dim = len(vectors[0])

        if len(vectors) == 1 and len(texts) > 1:
            result = [vectors[0]]
            for text in texts[1:]:
                result.append(self._call_api([text])[0])
            return result

        return vectors

    def _noop_embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            seed = list(digest[:16])
            vec = [(seed[i % 16] / 255.0) for i in range(self.dim)]
            norm = sum(v * v for v in vec) ** 0.5 or 1.0
            out.append([v / norm for v in vec])
        return out


# ---- Knowledge Store ----

class KnowledgeStore:
    """In-memory vector store with cosine similarity search."""

    def __init__(self):
        self._chunks: list[tuple[LessonChunk, list[float]]] = []

    def add_chunks(self, chunks: list[LessonChunk], embeddings: list[list[float]]):
        for chunk, emb in zip(chunks, embeddings):
            self._chunks.append((chunk, emb))

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[tuple[LessonChunk, float]]:
        if not self._chunks:
            return []
        results = []
        for chunk, emb in self._chunks:
            sim = self._cosine_similarity(query_embedding, emb)
            results.append((chunk, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def __len__(self) -> int:
        return len(self._chunks)


# ---- Demo Data ----

DEMO_COURSE = "Python 数据分析入门"

DEMO_CHUNKS: list[dict] = [
    {
        "lesson_id": "L001",
        "lesson_title": "Python 变量与数据类型",
        "text": """Python 中数据类型分为可变类型和不可变类型：
- 不可变类型：int, float, str, tuple — 一旦创建不能修改
- 可变类型：list, dict, set — 可以修改内容
类型转换常用函数：int(), float(), str(), list()
理解可变与不可变的区别很重要：不可变类型修改时创建新对象，可变类型原地修改。""",
    },
    {
        "lesson_id": "L002",
        "lesson_title": "Python 列表与字典",
        "text": """列表 (list) 是有序的可变序列：fruits.append("grape") 添加元素，fruits.pop() 弹出最后一个。
列表推导式是 Python 的特色：squares = [x**2 for x in range(10)]
字典 (dict) 是键值对的无序集合：student = {"name": "张三", "age": 20}
安全获取用 .get("name")，直接访问用 ["name"]（不存在会报 KeyError）。""",
    },
    {
        "lesson_id": "L003",
        "lesson_title": "Python 控制流与函数",
        "text": """条件语句 if/elif/else 用于分支判断。循环有 for（遍历可迭代对象）和 while（条件循环）。
函数定义：def calculate_mean(numbers): return sum(numbers) / len(numbers)
带默认参数：def greet(name, greeting="你好")
Lambda 匿名函数：square = lambda x: x ** 2""",
    },
    {
        "lesson_id": "L004",
        "lesson_title": "NumPy 数组基础",
        "text": """NumPy 是 Python 数据科学的基础库。创建数组：
arr = np.array([1, 2, 3, 4, 5])
zeros = np.zeros((3, 4))
ones = np.ones((2, 3))
np.arange(0, 10, 0.5)  # 0到10，步长0.5
np.linspace(0, 1, 100)  # 0到1，等分100份
数组属性：shape、ndim、dtype、size""",
    },
    {
        "lesson_id": "L005",
        "lesson_title": "NumPy 数组运算与广播",
        "text": """向量化运算比 Python 循环快 100 倍以上：a + b, a * b, np.sqrt(a)
聚合运算：a.mean(), a.std(), a.sum()
广播 (Broadcasting) 规则：
1. 从后向前比较维度
2. 维度相等或其中一个是 1 时可以广播
3. 缺失的维度自动补 1""",
    },
    {
        "lesson_id": "L006",
        "lesson_title": "Pandas 数据结构：Series 与 DataFrame",
        "text": """Series 是一维标签数组，DataFrame 是二维表格数据。
创建：pd.DataFrame({"name": ["张三", "李四"], "age": [25, 30]})
查看：df.head(), df.info(), df.describe()
选择：df["name"]（单列），df[["name","age"]]（多列），df.loc[0]（标签），df.iloc[1]（位置）""",
    },
    {
        "lesson_id": "L007",
        "lesson_title": "Pandas 数据清洗与处理",
        "text": """处理缺失值：df.isnull().sum() 检测，df.dropna() 删除，df.fillna(0) 填充。
数据过滤：df[df["score"] > 90]，多条件：df[(df["age"] > 20) & (df["score"] > 85)]
排序：df.sort_values("score", ascending=False)
分组聚合：df.groupby("department")["salary"].mean()""",
    },
    {
        "lesson_id": "L008",
        "lesson_title": "Matplotlib 数据可视化基础",
        "text": """基本绘图：plt.plot(x, y) 折线图，plt.scatter(x, y) 散点图，plt.bar() 柱状图，plt.hist() 直方图。
子图布局：fig, axes = plt.subplots(2, 2, figsize=(10, 8))
图表选择：趋势→折线图 | 分布→直方图/箱线图 | 比较→柱状图 | 关系→散点图""",
    },
    {
        "lesson_id": "L009",
        "lesson_title": "数据可视化最佳实践",
        "text": """可视化设计原则：
1. 明确目的：每个图表只传达一个核心信息
2. 选择合适的图表类型
3. 简洁优先：删除不必要的装饰
4. 颜色有意义：用颜色编码数据
5. 标注清晰：标题、轴标签、图例缺一不可
配色方案：类别数据用色相区分，连续数据用亮度渐变，红绿色盲友好用蓝橙配色（viridis/cividis）""",
    },
]

# ---- Global singletons ----

knowledge_store = KnowledgeStore()
embedding_provider = EmbeddingProvider()


def seed_demo_data():
    """Seed the knowledge store with demo course content."""
    if len(knowledge_store) > 0:
        return

    chunks = []
    texts = []
    for c in DEMO_CHUNKS:
        chunks.append(LessonChunk(
            id=f"chunk_{c['lesson_id']}",
            lesson_id=c["lesson_id"],
            lesson_title=c["lesson_title"],
            text=c["text"],
            course_title=DEMO_COURSE,
        ))
        texts.append(c["text"])

    try:
        embeddings = embedding_provider.embed(texts)
    except Exception:
        embeddings = embedding_provider._noop_embed(texts)

    knowledge_store.add_chunks(chunks, embeddings)
