"""内置政务知识库：加载 + BM25 检索。

分词用中文二元切分（bigram）+ 英文/数字词元，纯标准库实现，
不依赖 jieba 等三方分词器，检索质量对政务短语足够。
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list:
    """英文/数字按整词，中文切成二元词组（bigram）。"""
    atoms = _TOKEN_RE.findall(text.lower())
    tokens = []
    i = 0
    while i < len(atoms):
        tok = atoms[i]
        if len(tok) == 1 and "\u4e00" <= tok <= "\u9fff":
            nxt_ok = i + 1 < len(atoms)
            if nxt_ok and len(atoms[i + 1]) == 1 and "\u4e00" <= atoms[i + 1] <= "\u9fff":
                tokens.append(tok + atoms[i + 1])
                i += 2
                continue
            tokens.append(tok)
        else:
            tokens.append(tok)
        i += 1
    return tokens


def _index_tokens(text: str) -> list:
    """bigram + 单字混合词元：单字提升短查询对长文档的召回。"""
    base = tokenize(text)
    out = []
    out.extend(base)
    for tok in base:
        if len(tok) == 2 and "\u4e00" <= tok[0] <= "\u9fff":
            out.append(tok[0])
            out.append(tok[1])
    return out


def _query_tokens(text: str) -> list:
    """查询侧与索引同构。"""
    return _index_tokens(text)


@dataclass
class Service:
    """一项政务办事服务。"""

    domain: str
    name: str
    department: str
    materials: list
    processing_time: str
    location: str
    online: str
    process: list
    legal_basis: str
    tips: str = ""
    fee: str = ""

    def as_dict(self) -> dict:
        return {
            "domain": self.domain,
            "name": self.name,
            "department": self.department,
            "materials": self.materials,
            "processing_time": self.processing_time,
            "location": self.location,
            "online": self.online,
            "process": self.process,
            "legal_basis": self.legal_basis,
            "tips": self.tips,
            "fee": self.fee,
        }


@dataclass
class PolicyDoc:
    """一段政策法规文本，用于 RAG 检索引用。"""

    domain: str
    title: str
    text: str
    score: float = 0.0


@dataclass
class DomainInfo:
    domain: str
    keywords: list = field(default_factory=list)


def _load_builtin() -> list:
    docs = []
    pkg = resources.files("govagent") / "policies"
    for path in sorted(pkg.iterdir()):
        if path.name.endswith(".json"):
            docs.append(json.loads(path.read_text(encoding="utf-8")))
    return docs


class KnowledgeBase:
    """政务知识库：办事服务条目 + 政策文档，支持 BM25 检索。

    参数 data_dir 可传入自定义 JSON 目录（与内置格式一致），
    便于接入本地政务数据；缺省使用包内置样例数据。
    """

    def __init__(self, data_dir=None):
        if data_dir is not None:
            raw = [json.loads(p.read_text(encoding="utf-8"))
                   for p in sorted(Path(data_dir).glob("*.json"))]
        else:
            raw = _load_builtin()

        self.services = []
        self.policy_docs = []
        self.domains = {}
        for entry in raw:
            domain = entry["domain"]
            self.domains[domain] = DomainInfo(domain, entry.get("domain_keywords", []))
            for svc in entry.get("services", []):
                self.services.append(Service(
                    domain=domain,
                    name=svc["name"],
                    department=svc["department"],
                    materials=svc["materials"],
                    processing_time=svc["processing_time"],
                    location=svc["location"],
                    online=svc["online"],
                    process=svc["process"],
                    legal_basis=svc["legal_basis"],
                    tips=svc.get("tips", ""),
                    fee=svc.get("fee", ""),
                ))
            for doc in entry.get("policy_docs", []):
                self.policy_docs.append(PolicyDoc(
                    domain=domain, title=doc["title"], text=doc["text"]))

        self._build_index()

    def _build_index(self):
        """构建 BM25 索引（服务条目与政策文档共用一套语料）。"""
        self._doc_tokens = []
        self._doc_refs = []
        for svc in self.services:
            text = " ".join([svc.domain, svc.name, svc.department])
            self._doc_tokens.append(_index_tokens(text))
            self._doc_refs.append(svc)
        for doc in self.policy_docs:
            self._doc_tokens.append(_index_tokens(doc.domain + " " + doc.title + " " + doc.text))
            self._doc_refs.append(doc)

        self._n = len(self._doc_tokens)
        total_len = 0
        for idx in range(self._n):
            total_len += len(self._doc_tokens[idx])
        self._avgdl = 1.0
        if self._n > 0:
            self._avgdl = total_len / self._n

        self._tf = dict()
        self._df = dict()
        for idx in range(self._n):
            tf = dict()
            for t in self._doc_tokens[idx]:
                tf[t] = tf.get(t, 0) + 1
            self._tf[idx] = tf
            for term in tf.keys():
                self._df[term] = self._df.get(term, 0) + 1

    def search(self, query: str, top_k: int = 3) -> list:
        """BM25 检索，返回命中的服务条目或政策文档（按相关度降序）。"""
        if self._n == 0:
            return []
        k1 = 1.5
        b = 0.75
        q_tokens = _query_tokens(query)
        if not q_tokens:
            return []

        scores = dict()
        for qt in q_tokens:
            if qt not in self._df.keys():
                continue
            df_val = self._df[qt]
            idf = math.log(1.0 + (self._n - df_val + 0.5) / (df_val + 0.5))
            for idx in range(self._n):
                tf = self._tf[idx].get(qt, 0)
                if tf == 0:
                    continue
                dl = len(self._doc_tokens[idx])
                denom = tf + k1 * (1.0 - b + b * dl / self._avgdl)
                w = tf / denom
                w = w * (k1 + 1.0)
                w = w * idf
                scores[idx] = scores.get(idx, 0.0) + w

        if not scores:
            return []
        ranked = sorted(scores.keys(), key=scores.get, reverse=True)
        results = []
        for idx in ranked[:top_k]:
            ref = self._doc_refs[idx]
            ref.score = scores[idx]
            results.append(ref)
        return results

    def get_service(self, name: str):
        """按名称精确查找服务条目；找不到返回 None。"""
        for svc in self.services:
            if svc.name == name:
                return svc
        return None

    def list_domains(self) -> list:
        """列出所有领域名。"""
        return sorted(self.domains.keys())

    def services_by_domain(self, domain: str) -> list:
        """列出某领域的全部服务条目。"""
        out = []
        for s in self.services:
            if s.domain == domain:
                out.append(s)
        return out
