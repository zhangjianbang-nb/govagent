"""Agent 规划器 + 工具表 + MockLLM ReAct 路径测试。"""

from govagent import GovAgent, KnowledgeBase
from govagent.llm import MockLLM
from govagent.router import IntentRouter


def _agent():
    return GovAgent()


def test_planner_service_query():
    agent = _agent()
    reply = agent.run("社保卡丢了怎么补办？")
    assert reply.intent == "社保"
    assert "get_service_guide" in reply.tools_used
    assert "get_materials" in reply.tools_used
    assert "补办" in reply.text


def test_planner_policy_fallback():
    agent = _agent()
    reply = agent.run("公积金贷款利率是多少？")
    assert reply.intent == "公积金"


def test_planner_unknown_query():
    agent = _agent()
    reply = agent.run("！！！！")
    assert reply.text


def test_router_scores_domains():
    kb = KnowledgeBase()
    router = IntentRouter(kb)
    result = router.route("我想办营业执照")
    assert result.domain == "企业开办"
    assert router.route("天气不错").domain == ""


def test_registry_call_unknown_tool():
    agent = _agent()
    result = agent.registry.call("no_such_tool", {})
    assert not result.ok


def test_registry_bad_arguments():
    agent = _agent()
    result = agent.registry.call("get_materials", "{bad json")
    assert not result.ok


def test_check_progress_mock():
    agent = _agent()
    result = agent.registry.call("check_progress", {"case_id": "AB123"})
    assert result.ok and "办件" in result.text


def test_mock_llm_react_path():
    script = [
        {"tool_calls": [{"id": "call1", "function": {
            "name": "check_progress",
            "arguments": '{"case_id": "X1"}'}}]},
        {"content": "您的办件已受理。"},
    ]
    llm = MockLLM(script=script)
    agent = GovAgent(llm=llm)
    reply = agent.run("帮我查下办件进度 X1")
    assert reply.text == "您的办件已受理。"
    assert "check_progress" in reply.tools_used
    assert len(llm.calls) == 2  # 一次工具轮 + 一次最终回答


def test_llm_failure_degrades_to_planner():
    class Boom:
        def chat(self, *a, **kw):
            raise RuntimeError("network down")

    agent = GovAgent(llm=Boom())
    reply = agent.run("社保卡丢了怎么补办？")
    assert "补办" in reply.text  # 规划器兜底成功
