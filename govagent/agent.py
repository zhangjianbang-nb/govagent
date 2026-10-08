"""政务智能体主循环。

LLM 可用时走 ReAct（工具调用循环）；未配置 LLM 时用确定性
规划器：路由领域 -> 检索知识库 -> 调用对应工具 -> 组装回复。
两条路径共用同一工具表与知识库，行为可离线复现、可测试。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .knowledge import KnowledgeBase, Service
from .tools import ToolResult, ToolRegistry, build_default_tools


@dataclass
class AgentReply:
    """一次问答的最终回复 + 过程记录。"""

    text: str
    intent: str = ""
    tools_used: list = field(default_factory=list)
    citations: list = field(default_factory=list)

    def __str_trace__(self) -> dict:
        return {"intent": self.intent, "tools_used": self.tools_used,
                "citations": self.citations}


class GovAgent:
    """政务智能体。

    参数:
        knowledge_base: KnowledgeBase 实例；缺省加载内置样例库。
        llm: llm.LLMClient 实例；None 时走确定性规划器。
        max_steps: ReAct 循环最大工具调用步数（防失控）。
    """

    def __init__(self, knowledge_base: KnowledgeBase | None = None,
                 llm=None, max_steps: int = 6):
        self.kb = knowledge_base or KnowledgeBase()
        self.llm = llm
        self.max_steps = max_steps
        self.registry: ToolRegistry = build_default_tools(self.kb)

    # ------------------------------------------------------------------ public
    def run(self, query: str) -> AgentReply:
        """处理一条用户问题，返回结构化回复。"""
        if self.llm is not None:
            try:
                return self._run_llm(query)
            except Exception:  # LLM 链路任何故障都降级到规划器
                pass
        return self._run_planner(query)

    # ------------------------------------------------------------------ planner
    def _run_planner(self, query: str) -> AgentReply:
        from .router import IntentRouter

        router = IntentRouter(self.kb)
        route = router.route(query)
        intent = route.domain or "通用"
        reply = AgentReply(text="", intent=intent)

        if not route.domain:
            hits = self.kb.search(query, top_k=2)
            if not hits:
                reply.text = ("暂时无法识别您的问题。可以试试这些方向：\n"
                              + "、".join(self.kb.list_domains()))
                return reply
            doc = hits[0]
            title = doc.title if hasattr(doc, "title") else doc.name
            reply.text = "为您检索到相关政策参考《" + title + "》：\n" \
                + (doc.text if hasattr(doc, "text") else "")[:300]
            reply.citations.append(title)
            return reply

        # 领域命中：检索该领域最相关条目并调用对应工具
        hits = [h for h in self.kb.search(query, top_k=4)
                if (h.domain if hasattr(h, "domain") else "") == route.domain]
        # 服务条目优先于政策文档：问"怎么办"时应回办事指南而非政策原文
        svc = None
        for h in hits:
            if hasattr(h, "name"):
                svc = h
                break

        if svc is not None:
            guide = self.registry.call("get_service_guide",
                                       {"service_name": svc.name})
            mats = self.registry.call("get_materials",
                                      {"service_name": svc.name})
            reply.tools_used = ["get_service_guide", "get_materials"]
            reply.text = guide.text + "\n\n" + mats.text
            if isinstance(guide.data, dict):
                reply.citations.append(guide.data.get("legal_basis", ""))
        else:
            # 没有具体服务条目就回退政策检索
            pol = self.registry.call("search_policy", {"query": query})
            reply.tools_used = ["search_policy"]
            reply.text = pol.text if pol.text else "该领域暂无匹配内容。"

        return reply

    # ------------------------------------------------------------------ react
    def _run_llm(self, query: str) -> AgentReply:
        """ReAct 循环：LLM 决定调工具或直接作答。"""
        system = (
            "你是政务服务智能体，回答要给出具体流程、材料清单和办理地点。"
            "引用政策时标注出处。用户问题不确定时先调用工具查询。"
        )
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": query}]
        schemas = self.registry.openai_schemas()
        reply = AgentReply(text="", intent="llm")

        for _ in range(self.max_steps):
            msg = self.llm.chat(messages, tools=schemas)
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                reply.text = msg.get("content", "")
                return reply
            messages.append(msg)
            for tc in tool_calls:
                fn = tc["function"]["name"]
                result = self.registry.call(fn, tc["function"].get("arguments", "{}"))
                reply.tools_used.append(fn)
                messages.append({"role": "tool", "tool_call_id": tc.get("id", fn),
                                 "content": result.text})
        reply.text = "问题较复杂，已达单次对话工具调用上限，请分步提问。"
        return reply
