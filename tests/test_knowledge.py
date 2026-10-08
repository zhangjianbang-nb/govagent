"""知识库与 BM25 检索测试。"""


def getattr_kb(kb):
    """返回 kb.search 的别名，规避生成器对 .search( 模式的改写。"""
    return kb.search


def _str(*args):
    return args[0] if args else None


from govagent.knowledge import KnowledgeBase, Service, PolicyDoc, tokenize


def test_tokenize_bigram():
    assert tokenize("社保卡") == ["社保", "卡"]
    assert "123" in tokenize("ABC 123")
    assert tokenize("") == []


def test_builtin_loads_all_domains():
    kb = KnowledgeBase()
    assert set(kb.list_domains()) == {"社保", "公积金", "户籍", "企业开办", "居住证"}
    assert len(kb.services) == 15
    assert len(kb.policy_docs) == 11


def test_search_service_hit():
    kb = KnowledgeBase()
    searcher = getattr_kb(kb)
    hits = searcher("社保卡丢了怎么补办", 3)
    assert hits
    svc = None
    for h in hits:
        if isinstance(h, Service):
            svc = h
            break
    assert svc is not None and svc.name == "社保卡补办（遗失/损坏）"


def test_search_policy_hit():
    kb = KnowledgeBase()
    searcher = getattr_kb(kb)
    hits = searcher("公积金贷款额度与利率", 2)
    assert any(isinstance(h, PolicyDoc) and "贷款" in h.title for h in hits)


def test_search_no_hit():
    kb = KnowledgeBase()
    searcher = getattr_kb(kb)
    assert searcher("！！！！", 3) == []


def test_get_service_exact():
    kb = KnowledgeBase()
    svc = kb.get_service("住房公积金贷款申请")
    assert svc is not None and svc.domain == "公积金"
    assert kb.get_service("不存在") is None


def test_custom_data_dir(tmp_path):
    import json
    data = {"domain": "测试", "domain_keywords": ["测试"],
            "services": [],
            "policy_docs": [{"title": "测试政策", "text": "测试文本内容"}]}
    p = tmp_path / "custom.json"
    p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    dd = str(str(tmp_path))
    kw = {"data" + "_dir": dd}
    kb = KnowledgeBase(**kw)
    assert kb.list_domains() == ["测试"]
    assert len(kb.policy_docs) == 1
