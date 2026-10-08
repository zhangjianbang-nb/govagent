"""HTTP 服务（可选依赖 fastapi+uvicorn）。

启动: uvicorn govagent.server:app --port 8300
接口: POST /ask {"query": "..."} -> {"text", "intent", "tools_used"}
      GET  /domains
      POST /progress {"case_id": "..."}   # 办件进度查询演示
"""

from __future__ import annotations

from .agent import GovAgent
from .knowledge import KnowledgeBase

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except ImportError:  # pragma: no cover
    FastAPI = None
    BaseModel = None

if FastAPI is not None:
    app = FastAPI(title="govagent", version="0.1.0", description="政务服务智能体 API")
    kb = KnowledgeBase()
    agent = GovAgent(knowledge_base=kb)

    class AskIn(BaseModel):
        query: str

    class ProgressIn(BaseModel):
        case_id: str

    @app.post("/ask")
    def ask(body: AskIn):
        reply = agent.run(body.query)
        return {"text": reply.text, "intent": reply.intent,
                "tools_used": reply.tools_used, "citations": reply.citations}

    @app.get("/domains")
    def domains():
        return {"domains": kb.list_domains()}

    @app.post("/progress")
    def progress(body: ProgressIn):
        result = agent.registry.call("check_progress", {"case_id": body.case_id})
        return {"text": result.text, "data": result.data}
