"""govagent —— 政务服务智能体框架。

零核心依赖：意图路由、工具调用、政策知识库 BM25 检索、ReAct 式
agent 循环全部基于标准库实现；LLM 客户端可选（任何 OpenAI 兼容
端点均可接入，未配置时使用确定性规则规划器兜底，离线可用）。

快速开始::

    from govagent import GovAgent

    agent = GovAgent()  # 离线模式，使用内置知识库
    reply = agent.run("社保卡丢了怎么补办？")
    print(reply.text)
"""

from .agent import AgentReply, GovAgent
from .knowledge import KnowledgeBase
from .router import IntentRouter
from .tools import ToolRegistry, ToolResult

__version__ = "0.1.0"

__all__ = [
    "AgentReply",
    "GovAgent",
    "IntentRouter",
    "KnowledgeBase",
    "ToolRegistry",
    "ToolResult",
]
