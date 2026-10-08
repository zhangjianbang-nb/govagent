"""端到端演示：离线规划器 + 全工具链路 + MockLLM ReAct。

运行: python examples/demo.py
"""

from govagent import GovAgent, KnowledgeBase
from govagent.llm import MockLLM


def main() -> int:
    kb = KnowledgeBase()
    agent = GovAgent(knowledge_base=kb)

    print("=" * 60)
    print("1) 办事指南 + 材料清单（社保卡补办）")
    print("=" * 60)
    reply = agent.run("社保卡丢了怎么补办？")
    print(reply.text)
    print("[tools]", reply.tools_used)
    assert "get_service_guide" in reply.tools_used

    print()
    print("=" * 60)
    print("2) 政策检索（公积金贷款）")
    print("=" * 60)
    reply = agent.run("公积金贷款利率是多少？")
    print(reply.text[:200])
    print("[tools]", reply.tools_used)

    print()
    print("=" * 60)
    print("3) 办件进度查询")
    print("=" * 60)
    result = agent.registry.call("check_progress", {"case_id": "DEMO2026"})
    print(result.text)
    assert result.ok

    print()
    print("=" * 60)
    print("4) ReAct 路径（MockLLM 两轮：工具调用 -> 总结）")
    print("=" * 60)
    llm = MockLLM(script=[
        {"tool_calls": [{"id": "t1", "function": {
            "name": "check_progress",
            "arguments": '{"case_id": "LLM77"}'}}]},
        {"content": "您的办件 LLM77 当前处于审核中阶段。"},
    ])
    react_agent = GovAgent(knowledge_base=kb, llm=llm)
    reply = react_agent.run("查下 LLM77 办件")
    print(reply.text)
    print("[tools]", reply.tools_used)
    assert "check_progress" in reply.tools_used

    print()
    print("DEMO_PASS: 规划器路径 / 政策检索 / 工具直调 / ReAct 全部通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
