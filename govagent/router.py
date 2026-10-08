"""意图路由：把用户问题分派到政务领域。

规则优先（关键词命中打分），可选叠加 LLM 分类（llm.py 配置后）。
路由结果只决定检索范围与默认工具，不阻断兜底逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RouteResult:
    """路由结果：主领域、置信度分数、全部候选。"""

    domain: str
    score: float
    candidates: list


class IntentRouter:
    """基于关键词计数的轻量意图路由器。

    知识库里每个领域带一组 domain_keywords；问题文本命中的
    关键词越多，该领域得分越高。得分并列时取先注册领域。
    """

    def __init__(self, knowledge_base):
        self.kb = knowledge_base

    def route(self, query: str) -> RouteResult:
        scores = dict()
        for domain, info in self.kb.domains.items():
            hit = 0
            for kw in info.keywords:
                if kw and kw in query:
                    hit += 1
            if hit > 0:
                scores[domain] = float(hit)
        if not scores:
            return RouteResult(domain="", score=0.0, candidates=[])
        ranked = sorted(scores.keys(), key=scores.get, reverse=True)
        main = ranked[0]
        cands = [(d, scores[d]) for d in ranked]
        return RouteResult(domain=main, score=scores[main], candidates=cands)

    def route_or_default(self, query: str, default_domain: str = "") -> RouteResult:
        """路由失败时回退到 default_domain（或空串表示全库检索）。"""
        result = self.route(query)
        if result.domain:
            return result
        return RouteResult(domain=default_domain, score=0.0, candidates=[])
