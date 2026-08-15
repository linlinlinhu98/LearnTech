# Lumen-Bailian — 个性化学习路径规划师

基于 [Lumen](https://github.com/ahmedEid1/lumen) 改造、面向阿里云百炼（Bailian）平台的「个性化学习路径规划师」。三大模块：

- **模块一**：学习目标定义 + 课程自动生成（6 阶段多智能体流水线）
- **模块二**：课程范围内 RAG 检索 + ACL 权限 + 引用标注
- **模块三**：多智能体辅导调度（`plan → execute → synthesize`，≤5 轮迭代、≤8 次 LLM 调用；5 个子代理：检索 / 联网搜索 / 代码执行 / 出题 / 概念讲解）

## 目录结构

```
lumen-bailian/
├── deploy_starter/
│   ├── main.py             # 百炼平台部署入口（AgentScope ReActAgent + runtime）
│   ├── local_server.py     # 本地调试 HTTP 服务（纯 FastAPI，无需 AgentScope）
│   ├── config.yml          # 非敏感配置（模型标识 / 开关 / 预算）
│   ├── dispatcher.py       # 模块三：多智能体调度器（并发执行子代理）
│   ├── code_runner.py      # 模块三：受限 Python 代码沙箱
│   ├── tutor_core.py       # 5 子代理工具注册 + Tavily 联网搜索
│   ├── knowledge_store.py  # 课程作用域检索 + ACL（含演示数据 seed_demo_data）
│   ├── llm_utils.py        # httpx LLM 调用 + .env 加载 + JSON 解析
│   ├── prompts.py          # 各阶段 Prompt
│   ├── agent_tools.py      # 5 子代理工具函数
│   └── whl/                # 空目录（百炼平台占位）
├── tests/
│   ├── test_module3.py     # 离线单元测试（无需密钥、无需联网）
│   └── test_e2e.py         # 端到端演示（需密钥，走真实 LLM/搜索）
├── .env.example            # 密钥模板（复制为 .env 后填真实值）
├── .gitignore              # 已忽略 .env / __pycache__ / *.log 等
├── requirements.txt        # 本地调试依赖
├── Dockerfile
├── docker-compose.yml
├── README.md
└── TEST_CASES.md           # 测试用例说明
```

## 快速开始（本地调试）

### 1. 环境要求

- Python 3.10+

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置密钥

```bash
cp .env.example .env   # 然后编辑 .env，填入真实密钥
```

`.env`（已被 `.gitignore` 忽略，**切勿提交**）里需要：

| 变量 | 说明 |
|---|---|
| `DASHSCOPE_API_KEY` | LLM / 向量嵌入的密钥。本地调试默认走 Volcengine ARK（见 `deploy_starter/config.yml` 的 `DASHSCOPE_API_URL`）；部署到百炼平台时由平台自动注入，无需配置 |
| `TAVILY_API_KEY` | 可选。真联网搜索用的 Tavily key（免费注册 https://app.tavily.com）。不填则联网搜索返回「未配置」提示 |

> 不想配置密钥也能验证代码：直接跑第 4 步的离线单元测试，全程无需密钥、无需联网。

### 4. 跑离线单元测试（无需密钥）

```bash
python tests/test_module3.py
```

预期输出 `Ran 10 tests ... OK`。覆盖代码沙箱、LLM 预算、调度器（并发执行 / 迭代上限 / finalize）等。

### 5. 启动本地服务

```bash
cd deploy_starter
python local_server.py     # http://127.0.0.1:8080
```

打开 http://127.0.0.1:8080/docs 查看接口；`POST /api/v1/tutor/ask` 是模块三的核心入口。

### 6. 端到端演示（需密钥）

```bash
python tests/test_e2e.py                      # 跑全部演示问题
python tests/test_e2e.py "Rust 所有权是什么"   # 单问一句
```

会真实调用 LLM / 搜索 / 代码沙箱，并打印每个子代理的分工与最终综合回答。

## 联网搜索

模块三的 `web_search` 已接入 **Tavily 真搜索**（`POST https://api.tavily.com/search`），返回 `answer + 来源标题/链接/摘要`，不占用 LLM 调用预算。配置见第 3 步的 `TAVILY_API_KEY`。

## 代码执行沙箱

`code_runner.run_python` 在子进程里覆写 `__builtins__.__import__` 做白名单：允许 `numpy / pandas / matplotlib` 与常用标准库，封禁 `os / subprocess / socket` 等，带超时与输出截断。

## 部署到百炼平台

`deploy_starter/main.py` 是平台部署入口（AgentScope ReActAgent + runtime engine）。平台环境已预装 `agentscope / agentscope-runtime / fastapi / httpx / aiohttp / opentelemetry`，并注入 `DASHSCOPE_*` 环境变量。上传前在 `config.yml` 中启用「百炼平台版」配置段（注释掉本地 ARK 段）。

## 与 Lumen 原版差异

| Lumen 原版 | Lumen-Bailian |
|---|---|
| Groq Llama 3.3 70B | DashScope / Volcengine ARK（httpx OpenAI 兼容） |
| PostgreSQL + pgvector | 内存知识库（生产可替换 pgvector） |
| Cloudflare BGE 嵌入 (384d) | DashScope / Volcengine 嵌入 (2048d) |
| MCP Server（9 工具） | REST API + 5 子代理工具 |

## 参考

- 原版 Lumen：https://github.com/ahmedEid1/lumen
- 百炼平台：https://bailian.console.aliyun.com/
- Tavily：https://app.tavily.com
