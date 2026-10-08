"""tools 测试。"""

from govagent import GovAgent


def test_openai_schemas_shape():
    agent = GovAgent()
    schemas = agent.registry.openai_schemas()
    names = [s["function"]["name"] for s in schemas]
    assert "get_service_guide" in names
    assert "check_progress" in names
    for s in schemas:
        assert s["type"] == "function"


def test_guide_tool_fuzzy_fallback():
    agent = GovAgent()
    result = agent.registry.call("get_service_guide", {"service_name": "营业执照"})
    assert result.ok


def test_find_hall_with_district():
    agent = GovAgent()
    result = agent.registry.call("find_hall", {"service_name": "社保卡补办（遗失/损坏）", "district": "海淀区"})
    assert result.ok
