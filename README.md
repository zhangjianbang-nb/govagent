# govagent — 政务服务智能体框架

零核心依赖的政务服务智能体：**意图路由 → 工具调用 → 政策知识库检索 → 组装回复**，
LLM 可选（任何 OpenAI 兼容端点），未配置时走确定性规划器，离线可用、行为可复现。

## 特性

- **双路径 agent 循环**：配置 LLM 走 ReAct 工具调用循环；不配置走确定性规划器
  （路由 → 检索 → 工具 → 组装），同一工具表两条路径共享
- **内置政务知识库**：社保 / 公积金 / 户籍 / 企业开办 / 居住证 5 领域、15 项服务、
  11 段政策法规原文，支持自定义 JSON 目录接入本地政务数据
- **零依赖 BM25 检索**：中文 bigram + 单字混合词元，纯标准库实现（不依赖 jieba）
- **5 个内置工具**：办事指南 / 材料清单 / 办理时限 / 就近大厅 / 办件进度（mock，可替换真实后端）
- **多种交付形态**：Python API、CLI、FastAPI HTTP 服务（可选依赖）、OpenAI 工具 schema 导出

## 快速开始

```bash
pip install govagent          # 或 pip install -e ".[server,dev]"
```

### Python API（离线模式，无需任何 Key）

```python
from govagent import GovAgent

agent = GovAgent()
reply = agent.run("社保卡丢了怎么补办？")
print(reply.text)          # 办理部门 / 流程 / 材料 / 费用 / 法定依据
print(reply.tools_used)    # ['get_service_guide', 'get_materials']
```

### CLI

```bash
govagent "办营业执照需要什么材料"
govagent --domains                  # 列出支持的领域
govagent --chat                     # 交互模式
govagent --trace "公积金贷款利率"    # 输出工具调用轨迹
```

### HTTP 服务

```bash
pip install "govagent[server]"
uvicorn govagent.server:app --port 8300
curl -X POST localhost:8300/ask -H "Content-Type: application/json" \
     -d '{"query": "居住证怎么办理"}'
```

### 接入 LLM（ReAct 模式）

```python
from govagent import GovAgent
from govagent.llm import LLMClient

llm = LLMClient(base_url="https://api.example.com/v1",
                api_key="sk-...", model="your-model")
agent = GovAgent(llm=llm)      # 自动导出工具 schema，走 function calling
```

LLM 调用失败时自动降级到确定性规划器，服务不中断。

## 自定义知识库

```python
kb = KnowledgeBase(data_dir="./my_policies")   # 同内置格式的 JSON 目录
agent = GovAgent(knowledge_base=kb)
```

JSON 格式见 `govagent/policies/*.json`：每文件一个领域，含
`domain_keywords`（路由词）、`services`（服务条目）、`policy_docs`（政策原文）。

## 架构

```
用户问题
   │
   ▼
IntentRouter ── 领域关键词计分（社保/公积金/户籍/企业开办/居住证）
   │
   ▼
GovAgent
   ├─ LLM 路径:  ReAct 循环（tool_calls ↔ ToolRegistry，max_steps 防失控）
   └─ 规划器路径: KnowledgeBase.search（BM25）→ 服务条目优先 → 工具直调
   │
   ▼
AgentReply(text, intent, tools_used, citations)
```

## 设计决策

- **服务条目优先于政策文档**：问"怎么办"应回办事指南，政策原文只做兜底引用——
  检索层纯按 BM25 分数排序，选择策略放在规划器层
- **进度查询是 mock**：`check_progress` 返回可注入的演示数据，接真实政务系统时
  替换 handler 即可，接口不变
- **政策原文带 `legal_basis`**：每条服务标注法定依据，回复可溯源
- **LLM 故障降级**：任何 LLM 链路异常静默回退规划器，政务场景可用性优先

## 测试

```bash
pip install -e ".[dev]"
pytest tests/ -q          # 21 tests
python examples/demo.py   # 端到端演示
```

## License

MIT
