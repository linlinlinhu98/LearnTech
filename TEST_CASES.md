# Lumen 测试案例

> 基础 URL：`http://<your-server>:3000`
> 所有请求需要 `Content-Type: application/json`
> user_id 统一用 `test_user`，course_id 用空字符串代表通用课程

---

## 模块一：知识库与文件上传

### TC-1.1 统计知识库状态
```bash
curl -X POST http://localhost:3000/api/v1/knowledge/stats \
  --json '{"user_id": "test_user", "course_id": ""}'
```
**预期**：返回 `total_chunks`、`courses` 等统计信息，demo 课程有 9 个 chunk

---

### TC-1.2 检索课程内容
```bash
curl -X POST http://localhost:3000/api/v1/knowledge/search \
  --json '{
    "query": "Python 列表有哪些常用操作",
    "user_id": "test_user",
    "course_id": "",
    "top_k": 3
  }'
```
**预期**：返回相关 chunk 列表，每个结果含 `ref`、`lesson_title`、`text`（截断 500 字）、`similarity`

---

### TC-1.3 上传 Markdown 文件并导入
```bash
curl -X POST http://localhost:3000/api/v1/knowledge/upload \
  --json '{
    "filename": "python基础.md",
    "content": "'"$(base64 -w0 << 'EOF'
# Python 基础

## 变量与数据类型
Python 中变量无需声明类型，常见类型有：int、float、str、bool、list、dict、set。

## 列表操作
- append(elem): 在末尾添加元素
- extend(iterable): 批量添加
- insert(i, elem): 在位置 i 插入
- remove(elem): 移除第一个匹配项
- pop(i): 弹出并返回位置 i 的元素，默认最后一位

## 字典操作
- dict.keys(): 返回所有键
- dict.values(): 返回所有值
- dict.items(): 返回键值对
- dict.get(key, default): 安全获取，键不存在返回 default
EOF
)"'",
    "course_id": "",
    "course_title": "Python 基础知识"
  }'
```
**预期**：返回 `chunks_ingested`（>0 表示成功）

---

## 模块二：学习目标引导（Intake）

### TC-2.1 启动学习目标引导
```bash
curl -X POST http://localhost:3000/api/v1/intake/start \
  --json '{"user_id": "test_user"}'
```
**预期**：返回引导问题（如"你的学习目标是什么？"）

---

### TC-2.2 引导轮次交互
```bash
# 第1轮：回答学习目标
curl -X POST http://localhost:3000/api/v1/intake/respond \
  --json '{"user_id": "test_user", "message": "我想学习 Python 数据分析"}'

# 第2轮：回答当前水平
curl -X POST http://localhost:3000/api/v1/intake/respond \
  --json '{"user_id": "test_user", "message": "有一些 Python 基础，熟悉基本语法"}'

# 第3轮：回答目标水平
curl -X POST http://localhost:3000/api/v1/intake/respond \
  --json '{"user_id": "test_user", "message": "能够独立完成数据分析项目"}'
```
**预期**：每轮返回下一个问题；所有字段收集完毕后返回 `state: "CONFIRMED"` 和 `learning_brief`

---

### TC-2.3 查看引导状态
```bash
curl -X POST http://localhost:3000/api/v1/intake/status \
  --json '{"user_id": "test_user"}'
```
**预期**：返回当前轮次、已收集字段、状态

---

### TC-2.4 取消引导
```bash
curl -X POST http://localhost:3000/api/v1/intake/cancel \
  --json '{"user_id": "test_user"}'
```
**预期**：返回 `state: "CANCELLED"`

---

## 模块三：课程生成

### TC-3.1 根据目标生成课程
```bash
curl -X POST http://localhost:3000/api/v1/courses/generate \
  --json '{
    "user_id": "test_user",
    "goal": "学习 Python 数据分析，主要使用 Pandas 和 NumPy",
    "current_level": "有 Python 基础",
    "target_level": "能够独立完成数据分析项目"
  }'
```
**预期**：返回 `course_id`、`course_title`、生成课程内容摘要（6 阶段流水线执行，结果较长）

---

### TC-3.2 列出所有课程
```bash
curl -X POST http://localhost:3000/api/v1/courses/list \
  --json '{"user_id": "test_user"}'
```
**预期**：返回可见课程列表（含 demo 课程和用户生成的课程）

---

## 模块四：自适应模考

### TC-4.1 启动模考
```bash
curl -X POST http://localhost:3000/api/v1/mock/start \
  --json '{"user_id": "test_user", "course_id": ""}'
```
**预期**：返回 `started: true`、`topics`（9 个lesson标题）、`total: 9`

---

### TC-4.2 选择题答题（直接判对错）
```bash
# 先获取第一道题（选择题）
curl -X POST http://localhost:3000/api/v1/mock/answer \
  --json '{"correct": true, "course_id": ""}'

# 答错
curl -X POST http://localhost:3000/api/v1/mock/answer \
  --json '{"correct": false, "course_id": ""}'
```
**预期**：返回下一道题，含 `progress`（如 "2/9"）、`lesson_title`、`options`

---

### TC-4.3 填空题答题（LLM 评判）
```bash
# 答对
curl -X POST http://localhost:3000/api/v1/mock/answer \
  --json '{"answer_text": "set", "course_id": ""}'

# 答"不知道"（应判定为错误并给出正确答案和讲解）
curl -X POST http://localhost:3000/api/v1/mock/answer \
  --json '{"answer_text": "不知道", "course_id": ""}'
```
**预期**：
- 答对：`grading_correct: true/false`，`grading_feedback` 含点评
- 答不知道：判定为错误，反馈中包含标准答案

---

### TC-4.4 完整模考流程（9 道题）
```bash
# 启动
curl -s -X POST http://localhost:3000/api/v1/mock/start --json {} | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['total'])"

# 循环答题 9 次（ alternately 答对和答错）
for i in {1..9}; do
  if ((i % 2 == 0)); then
    curl -s -X POST http://localhost:3000/api/v1/mock/answer \
      --json '{"answer_text": "错误答案", "course_id": ""}' | python3 -c "import sys,json; d=json.load(sys.stdin)['data']; print(d.get('progress','done'), d.get('grading_correct','?'), d.get('error',''))"
  else
    curl -s -X POST http://localhost:3000/api/v1/mock/answer \
      --json '{"correct": true, "course_id": ""}' | python3 -c "import sys,json; d=json.load(sys.stdin)['data']; print(d.get('progress','done'), d.get('error',''))"
  fi
done
```
**预期**：最后一轮返回模考总结（含正确率、薄弱知识点）

---

## 模块五：复习计划（FSRS）

### TC-5.1 生成复习计划
```bash
curl -X POST http://localhost:3000/api/v1/review/schedule \
  --json '{
    "user_id": "test_user",
    "course_id": "",
    "top_k": 5,
    "exam_date": "2026-09-15"
  }'
```
**预期**：返回待复习知识点列表，按紧迫程度排序，含 `retrievability`（掌握度）和 `urgency_score`（紧迫指数）

---

### TC-5.2 记录答题结果（更新 FSRS 状态）
```bash
# 假设 chunk_id 从模考或检索结果中获取
curl -X POST http://localhost:3000/api/v1/mock/record \
  --json '{
    "chunk_id": "chunk_L001",
    "user_id": "test_user",
    "correct": true
  }'
```
**预期**：返回 `{"success": true}`，FSRS 参数（`stability`、`difficulty`）已更新

---

## 模块六：掌握度报告

### TC-6.1 查看掌握度报告
```bash
curl -X POST http://localhost:3000/api/v1/mastery/report \
  --json '{
    "user_id": "test_user",
    "course_id": ""
  }'
```
**预期**：返回每个 lesson 的 `mastery_score`（0-100）和 `retrievability`，含新知识点标记

---

## 模块七：AI 辅导员（多智能体）

### TC-7.1 课程相关问题（检索增强）
```bash
curl -X POST http://localhost:3000/api/v1/tutor/ask \
  --json '{
    "question": "列表的 append 和 extend 有什么区别？",
    "user_id": "test_user",
    "course_id": ""
  }'
```
**预期**：返回带 `[参考N]` 引用标注的回答，内容来自课程知识库

---

### TC-7.2 超出课程范围的问题（模型知识回答）
```bash
curl -X POST http://localhost:3000/api/v1/tutor/ask \
  --json '{
    "question": "什么是机器学习中的梯度下降？",
    "user_id": "test_user",
    "course_id": ""
  }'
```
**预期**：模型基于预训练知识回答（不调用外部搜索，符合平台限制）

---

### TC-7.3 代码执行
```bash
curl -X POST http://localhost:3000/api/v1/tutor/ask \
  --json '{
    "question": "运行这段代码：print([x**2 for x in range(5)])",
    "user_id": "test_user",
    "course_id": ""
  }'
```
**预期**：代码执行结果 `stdout: [0, 1, 4, 9, 16]`

---

### TC-7.4 危险代码被拦截
```bash
curl -X POST http://localhost:3000/api/v1/tutor/ask \
  --json '{
    "question": "运行：import os; os.system(\"dir\")",
    "user_id": "test_user",
    "course_id": ""
  }'
```
**预期**：`stdout` 为空或错误提示，`stderr` 说明 `os` 被禁用

---

### TC-7.5 生成练习题
```bash
curl -X POST http://localhost:3000/api/v1/tutor/ask \
  --json '{
    "question": "出一道关于 Python 字典的练习题",
    "user_id": "test_user",
    "course_id": ""
  }'
```
**预期**：返回一道练习题（含题目、选项或填空要求）

---

## 模块八：聊天（完整对话流程）

### TC-8.1 创建会话
```bash
curl -X POST http://localhost:3000/createSession \
  --json '{"user_id": "test_user"}'
```
**预期**：返回 `session_id`

---

### TC-8.2 发送消息（触发多智能体）
```bash
curl -X POST http://localhost:3000/process/sync \
  --json '{
    "user_id": "test_user",
    "session_id": "<上面返回的session_id>",
    "message": "我刚学完列表，想做几道练习题检验一下"
  }'
```
**预期**：Agent 调用 `tool_generate_quiz` 生成练习题，或调用 `tool_retrieve` 检索相关内容后讲解

---

## 测试检查清单

| 功能 | 测试用例 | 验证点 |
|------|---------|--------|
| 知识库检索 | TC-1.2 | 返回带引用的相关结果 |
| 文件上传 | TC-1.3 | chunks_ingested > 0 |
| 学习引导 | TC-2.1~2.4 | 多轮交互后得到 CONFIRMED |
| 课程生成 | TC-3.1 | 返回 course_id |
| 模考启动 | TC-4.1 | 9 个 topics |
| 选择题 | TC-4.2 | 直接判对错，返回下一题 |
| 填空题 LLM 评判 | TC-4.3 | grading_correct + grading_feedback |
| 模考结束 | TC-4.4 | 返回总结（含正确率） |
| 复习计划 | TC-5.1 | 按紧迫度排序 |
| 掌握度报告 | TC-6.1 | 每个 lesson 有分数 |
| 检索增强问答 | TC-7.1 | 带 `[参考N]` 引用 |
| 模型知识回答 | TC-7.2 | 不调用外部搜索 |
| 代码执行 | TC-7.3 | 正常输出结果 |
| 危险代码拦截 | TC-7.4 | 拒绝执行 |
| 完整聊天 | TC-8.2 | 多智能体协作 |
