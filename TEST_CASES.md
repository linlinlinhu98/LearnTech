# Lumen-Bailian 对话调试案例

Demo 知识库：**Python 数据分析入门**（9 课时：变量、列表字典、控制流、NumPy基础、NumPy广播、Pandas数据结构、Pandas清洗、Matplotlib绘图、可视化实践）

---

## 案例 1：课程内容检索（RAG 核心流程）

**目的**：验证 tool_retrieve → 检索 → 标注引用 的完整链路

```
用户：Python 中可变类型和不可变类型有什么区别？
```

**预期行为**：

1. Agent 调用 `tool_retrieve` 检索 "可变类型和不可变类型"
2. 返回结果应包含 [参考1] 标注（来自 L001 Python变量与数据类型）
3. 回答应包含：int/str/tuple 不可变，list/dict/set 可变，以及代码示例

**检查点**：响应是否包含 `[参考N]` 引用标记？内容是否与 L001 一致？

---

## 案例 2：概念深度讲解

**目的**：验证 tool_explain_concept 的分层讲解结构

```
用户：什么是 NumPy 的广播机制？我刚开始学，请用简单的方式解释。
```

**预期行为**：

1. Agent 先调用 `tool_retrieve` 检索相关知识
2. Agent 调用 `tool_explain_concept(concept="NumPy广播机制", level="beginner")`
3. 回答结构应为：一句话总结 → 生活类比 → 深入细节 → 关键要点

**检查点**：回答是否包含四层结构？类比是否贴近生活？

---

## 案例 3：练习测验生成

**目的**：验证 tool_generate_quiz 功能

```
用户：我刚学完 Pandas 的 DataFrame，帮我出几道题测试一下。
```

**预期行为**：

1. Agent 调用 `tool_retrieve` 检索 Pandas 相关内容
2. Agent 调用 `tool_generate_quiz(topic="Pandas DataFrame", difficulty="medium")`
3. 返回选择题、判断题、简答题各一，含答案解析

**检查点**：是否生成了 3 种题型的题目？是否包含解析？

---

## 案例 4：多工具协同

**目的**：验证单轮对话中多个工具的调用

```
用户：我想复习一下列表和字典的知识，然后给我出两道相关的练习题。
```

**预期行为**：

1. Agent 调用 `tool_retrieve` 检索列表和字典内容
2. Agent 输出检索结果（含引用标注）
3. Agent 调用 `tool_generate_quiz(topic="Python列表和字典")`
4. 返回对应练习题

**检查点**：同一条消息是否先后触发了 search + quiz 两个工具？

---

## 案例 5：超出知识库范围

**目的**：验证知识库无匹配时的降级处理

```
用户：请给我讲讲 Rust 的所有权系统。
```

**预期行为**：

1. Agent 调用 `tool_retrieve`，检索结果为空或无高相关度结果
2. Agent 应诚实告知当前课程资料无法覆盖此问题
3. 不应编造内容

**检查点**：Agent 是否诚实承认知识不足？是否给出了有用建议（换个角度、查其他资料）？

---

## 案例 6：混合中英文提问

**目的**：验证中英混合场景

```
用户：How to use df.groupby() in Pandas? 请用中文解释。
```

**预期行为**：

1. Agent 调用 `tool_retrieve` 检索 groupby 相关内容
2. 检索到 L007（Pandas数据清洗与处理，含 groupby 示例）
3. 用中文回答，代码保持英文

**检查点**：是否能跨语言检索？回答语言是否与用户最后指定的一致？

---

## 案例 7：连续对话（Session 记忆）

**目的**：验证多轮对话中 Session 记忆是否正常

```
第1轮：Python 数据分析常用的库有哪些？
第2轮：你刚才提到的 NumPy，它的数组和 Python 原生的列表有什么不同？
第3轮：给我两个 NumPy 数组运算的例子。
```

**预期行为**：

- 第2轮应能引用第1轮提到的库名
- 第3轮应能承接第2轮的上下文
- 每轮涉及课程内容时应检索知识库

**检查点**：后续轮次是否记住了前文的对话上下文？

---

## 案例 8：知识库管理 API 测试

**目的**：验证 /api/v1/knowledge/* 端点

```bash
# 8a. 查看知识库状态
curl -X POST http://<HOST>:<PORT>/api/v1/knowledge/stats
# 预期：{"code":200, "data":{"total_chunks":9, ...}}

# 8b. 语义搜索
curl -X POST http://<HOST>:<PORT>/api/v1/knowledge/search \
  -H "Content-Type: application/json" \
  -d '{"query":"数据清洗","top_k":3}'
# 预期：返回 L007 相关 chunk，similarity_score 降序

# 8c. 注入新知识
curl -X POST http://<HOST>:<PORT>/api/v1/knowledge/ingest \
  -H "Content-Type: application/json" \
  -d '{"course_title":"测试课程","chunks":[{"lesson_id":"T01","lesson_title":"测试章节","text":"这是一段测试内容，用于验证知识注入功能是否正常。"}]}'
# 预期：chunks_ingested=1, total_chunks 增加1

# 8d. 验证注入后检索
curl -X POST http://<HOST>:<PORT>/api/v1/knowledge/search \
  -H "Content-Type: application/json" \
  -d '{"query":"测试内容","top_k":3}'
# 预期：能检索到刚注入的 T01 chunk
```

---

## 案例 9：会话管理

**目的**：验证 Bailian 标准端点

```bash
# 9a. 创建会话
curl -X POST http://<HOST>:<PORT>/createSession \
  -H "Content-Type: application/json" \
  -d '{}'
# 预期：{"uniqueCode":"<uuid>","code":200,"message":"success","data":null,"host":"<ip>"}

# 9b. 健康检查
curl http://<HOST>:<PORT>/health
# 预期："OK"

# 9c. 根路径
curl http://<HOST>:<PORT>/
# 预期：{"name":"Lumen-Bailian","status":"running"}
```

---

## 调试优先级

| 优先级 | 案例  | 验证的核心能力                   |
| ------ | ----- | -------------------------------- |
| P0     | 案例1 | RAG检索 + 引用标注（Lumen 核心） |
| P0     | 案例3 | Quiz工具调用                     |
| P1     | 案例2 | 分层讲解结构                     |
| P1     | 案例4 | 多工具协同                       |
| P1     | 案例7 | Session 记忆                     |
| P2     | 案例5 | 边界情况处理                     |
| P2     | 案例8 | API 端点                         |
| P2     | 案例6 | 多语言场景                       |

---

## 模块三：多智能体辅导调度

### 案例 10：核心调度器（REST 端点）

**目的**：验证 `/api/v1/tutor/ask` 的 plan→execute→synthesize 全链路 + 预算约束

```bash
curl -X POST http://<HOST>:<PORT>/api/v1/tutor/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Python 中列表和字典有什么区别？","user_id":"u1"}'
```

**预期行为**：

1. 返回 `code=200`，`data` 含 `answer`、`evidence`、`iterations`、`llm_calls`
2. 调度器先规划（plan）调用 `retrieve`，执行后综合（synthesize）回答
3. `llm_calls <= 8`（MAX_DISPATCH_LLM_CALLS），`iterations <= 5`（MAX_DISPATCH_ITERATIONS）
4. 回答引用课程内容时带 `[参考N]` 标注

**检查点**：`data.llm_calls` 是否 ≤ 8？`data.iterations` 是否 ≤ 5？回答是否含引用？

### 案例 11：代码执行子代理

**目的**：验证 `tool_run_code` 沙箱真实执行 + 危险模块封禁

```
用户：帮我算一下 1 加到 100 的和，用 Python 代码。
```

**预期行为**：

1. Agent 调用 `tool_run_code`，传入 `print(sum(range(1,101)))`
2. 返回 `stdout=5050`、`exit_code=0`

```
用户：帮我看看这台机器的当前目录有哪些文件，用 os.listdir()。
```

**预期行为**：`tool_run_code` 返回 `success=false`，stderr 含 `not allowed`（os 被白名单封禁）

**检查点**：算术代码真实执行成功？`import os` 被拦截？

### 案例 12：预算硬约束（本地单测）

**目的**：验证调度器在 LLM 预算耗尽时不超发

```bash
cd lumen-bailian && python tests/test_module3.py
```

**预期行为**：10 个用例全部 `OK`，覆盖：代码沙箱（执行/封禁/超时）、LlmBudget 计数、调度器迭代上限、提前 finalize、预算预留综合器、预算耗尽回退、RAG 检索。

**检查点**：`Ran 10 tests ... OK`

### 案例 13：辅导聊天全链路（五工具）

**目的**：验证协调 Agent 聊天路径下五工具可用

```
第1轮：什么是 NumPy 广播机制？
第2轮：给我出两道关于它的练习题。
第3轮：用代码演示一下广播。
```

**预期行为**：

1. 第1轮调用 `tool_retrieve` + `tool_explain_concept`
2. 第2轮调用 `tool_generate_quiz`
3. 第3轮调用 `tool_run_code`（代码演示）

**检查点**：三个工具是否按需触发？`tool_web_search` 是否在课程无法覆盖时被触发？

> **联网搜索已接入真搜索（Tavily）**：`tool_web_search` 不再是 LLM 内部知识兜底，而是调用
> Tavily 搜索 API（`POST https://api.tavily.com/search`），返回答案 + 来源链接，**不占用 LLM
> 调用预算**。需在 `deploy_starter/config.yml` 填入 `TAVILY_API_KEY`（免费注册
> https://app.tavily.com）。未填 key 时 `search_web` 返回提示文案（不报错、不影响其余子代理）。

---

## 本地 HTTP 服务全链路（无 AgentScope 运行时）

百炼平台用 `main.py`（AgentScope ReActAgent + 运行时引擎，AgentScope v1.0.11）。本地安装的是
AgentScope v2.x，缺少 `agentscope.agent.ReActAgent`，因此 `main.py` 无法在本地启动。为此提供
`local_server.py`：用 FastAPI + 无 AgentScope 依赖的 `TutorDispatcher` / `tutor_core` 镜像同样的
端点，`/process/sync` 自动降级到调度器（同一个五工具 + plan→execute→synthesize）。

### 案例 14：本地起服务 + HTTP 全链路

```bash
cd lumen-bailian/deploy_starter
python local_server.py            # http://127.0.0.1:8080
```

**预期行为**：

```bash
# 14a. 健康检查
curl http://127.0.0.1:8080/health
# 预期："OK"

# 14b. 知识库状态（启动时自动 seed 9 条 demo 数据）
curl -X POST http://127.0.0.1:8080/api/v1/knowledge/stats
# 预期：total_chunks=9, embedding_dim=2048

# 14c. 语义检索
curl -X POST http://127.0.0.1:8080/api/v1/knowledge/search \
  -H "Content-Type: application/json" -d '{"query":"数据清洗","top_k":3}'
# 预期：L007 Pandas 数据清洗与处理 排在首位

# 14d. 知识注入
curl -X POST http://127.0.0.1:8080/api/v1/knowledge/ingest \
  -H "Content-Type: application/json" \
  -d '{"course_title":"测试课","chunks":[{"lesson_id":"T01","lesson_title":"测试章节","text":"用于验证注入的测试内容。"}]}'
# 预期：chunks_ingested=1, total_chunks 增加 1

# 14e. 多智能体调度（REST）
curl -X POST http://127.0.0.1:8080/api/v1/tutor/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Python 中列表和字典有什么区别？","user_id":"u1"}'
# 预期：data.llm_calls <= 8，data.iterations <= 5，回答带 [参考N] 引用

# 14f. 聊天路径（ReActAgent 降级为调度器）
curl -X POST http://127.0.0.1:8080/process/sync \
  -H "Content-Type: application/json" \
  -d '{"session_id":"s1","user_id":"u1","messages":[{"role":"user","content":"什么是 NumPy 广播机制？"}]}'
# 预期：同 14e，走 TutorDispatcher
```

**检查点**：六个端点均返回 HTTP 200；`tutor/ask` 与 `process/sync` 的 `data.llm_calls ≤ 8`、
`data.iterations ≤ 5`、`data.evidence` 非空（实际调用了子代理）。

> **Windows curl 中文编码注意**：Windows Git Bash 的 curl 会把中文按 GBK 编码发出，服务端按
> UTF-8 解析会报 `UnicodeDecodeError`（表现为 500）。真实客户端（浏览器 / Python / Postman）
> 都是 UTF-8，无此问题。curl 测中文时用 `curl -d @body.json`（body.json 存 UTF-8）或改用英文查询。
