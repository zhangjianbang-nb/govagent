"""工具注册表：装饰器注册 + 按名调用，零依赖。

内置 5 个政务工具（办事指南/材料清单/办理时限/就近大厅/进度查询）
+ 政策检索工具。进度查询使用可注入的 mock 后端，接入真实系统时
替换 handler 即可。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from .knowledge import KnowledgeBase


@dataclass
class ToolResult:
    """工具执行结果：给用户看的内容 + 给 LLM 看的结构化数据。"""

    tool: str
    text: str
    data: dict
    ok: bool = True


class ToolRegistry:
    """工具表：name -> (schema, handler)。"""

    def __init__(self):
        self._tools = dict()

    def register(self, name: str, description: str, parameters: dict):
        """装饰器：把函数注册为工具。parameters 是 JSON schema 片段。"""
        def deco(fn: Callable):
            self._tools[name] = {"name": name, "description": description,
                                 "parameters": parameters, "handler": fn}
            return fn
        return deco

    def names(self) -> list:
        return sorted(self._tools.keys())

    def get(self, name: str):
        entry = self._tools.get(name)
        return entry["handler"] if entry else None

    def openai_schemas(self) -> list:
        """导出 OpenAI function-calling 工具 schema 列表。"""
        out = []
        for name in sorted(self._tools.keys()):
            e = self._tools[name]
            out.append({"type": "function", "function": {
                "name": e["name"],
                "description": e["description"],
                "parameters": e["parameters"],
            }})
        return out

    def call(self, name: str, arguments: str | dict) -> ToolResult:
        """按名执行工具；arguments 是 JSON 字符串或已解析 dict。"""
        entry = self._tools.get(name)
        if entry is None:
            return ToolResult(tool=name, text="未知工具: " + name, data={}, ok=False)
        if isinstance(arguments, str):
            try:
                args = json.loads(arguments) if arguments.strip() else dict()
            except json.JSONDecodeError as exc:
                return ToolResult(tool=name, text="参数不是合法 JSON: " + str(exc), data={}, ok=False)
        else:
            args = dict(arguments)
        try:
            return entry["handler"](**args)
        except TypeError as exc:
            return ToolResult(tool=name, text="参数错误: " + str(exc), data={}, ok=False)


def build_default_tools(kb: KnowledgeBase) -> ToolRegistry:
    """构建默认工具集，绑定到知识库 kb。"""
    reg = ToolRegistry()

    @reg.register(
        "get_service_guide",
        "查询某项政务服务的办理指南：部门、流程、线上入口",
        {"type": "object", "properties": {"service_name": {"type": "string"}},
         "required": ["service_name"]},
    )
    def guide(service_name: str) -> ToolResult:
        svc = kb.get_service(service_name)
        if svc is None:
            hits = kb.search(service_name, top_k=1)
            if hits and hasattr(hits[0], "name"):
                svc = hits[0]
        if svc is None:
            return ToolResult("get_service_guide", "未找到该服务: " + service_name, {})
        lines = []
        lines.append("办理部门: " + svc.department)
        lines.append("办理流程:")
        for i, step in enumerate(svc.process, 1):
            lines.append("  %d. %s" % (i, step))
        lines.append("线上办理: " + svc.online)
        lines.append("法定依据: " + svc.legal_basis)
        if svc.tips:
            lines.append("温馨提示: " + svc.tips)
        return ToolResult("get_service_guide", "\n".join(lines), svc.as_dict())

    @reg.register(
        "get_materials",
        "查询某项服务需要提交的申请材料清单",
        {"type": "object", "properties": {"service_name": {"type": "string"}},
         "required": ["service_name"]},
    )
    def materials(service_name: str) -> ToolResult:
        svc = kb.get_service(service_name)
        if svc is None:
            hits = kb.search(service_name, top_k=1)
            if hits and hasattr(hits[0], "name"):
                svc = hits[0]
        if svc is None:
            return ToolResult("get_materials", "未找到该服务: " + service_name, {})
        lines = ["办理【" + svc.name + "】需要提交的材料:"]
        for i, mat in enumerate(svc.materials, 1):
            lines.append("  %d. %s" % (i, mat))
        if svc.fee:
            lines.append("费用: " + svc.fee)
        return ToolResult("get_materials", "\n".join(lines),
                          {"service": svc.name, "materials": svc.materials, "fee": svc.fee})

    @reg.register(
        "get_processing_time",
        "查询某项服务的办理时限",
        {"type": "object", "properties": {"service_name": {"type": "string"}},
         "required": ["service_name"]},
    )
    def proc_time(service_name: str) -> ToolResult:
        svc = kb.get_service(service_name)
        if svc is None:
            hits = kb.search(service_name, top_k=1)
            if hits and hasattr(hits[0], "name"):
                svc = hits[0]
        if svc is None:
            return ToolResult("get_processing_time", "未找到该服务: " + service_name, {})
        return ToolResult("get_processing_time",
                          "【" + svc.name + "】办理时限: " + svc.processing_time,
                          {"service": svc.name, "processing_time": svc.processing_time})

    @reg.register(
        "find_hall",
        "查询就近办理大厅（窗口地址与线上渠道）",
        {"type": "object",
         "properties": {"service_name": {"type": "string"}, "district": {"type": "string"}},
         "required": ["service_name"]},
    )
    def find_hall(service_name: str, district: str = "") -> ToolResult:
        svc = kb.get_service(service_name)
        if svc is None:
            hits = kb.search(service_name, top_k=1)
            if hits and hasattr(hits[0], "name"):
                svc = hits[0]
        if svc is None:
            return ToolResult("find_hall", "未找到该服务: " + service_name, {})
        suffix = ("（" + district + "）" if district else "（各区政务服务大厅均可）")
        text = "【" + svc.name + "】办理地点" + suffix + ": " + svc.location
        return ToolResult("find_hall", text, {"location": svc.location, "online": svc.online})

    @reg.register(
        "check_progress",
        "查询办件进度（演示用 mock 后端，接真实系统时替换）",
        {"type": "object", "properties": {"case_id": {"type": "string"}},
         "required": ["case_id"]},
    )
    def check_progress(case_id: str) -> ToolResult:
        stage_map = {"0": "已受理", "1": "审核中", "2": "待补正材料", "3": "办结"}
        stage = stage_map.get(case_id[-1], "审核中")
        text = "办件 " + case_id + " 当前状态: " + stage
        return ToolResult("check_progress", text, {"case_id": case_id, "stage": stage})

    @reg.register(
        "search_policy",
        "按关键词检索政策法规原文片段（RAG）",
        {"type": "object", "properties": {"query": {"type": "string"}},
         "required": ["query"]},
    )
    def search_policy(query: str) -> ToolResult:
        hits = kb.search(query, top_k=2)
        if not hits:
            return ToolResult("search_policy", "政策库中未检索到相关内容。", {})
        out_lines = []
        data = []
        for h in hits:
            title = h.title if hasattr(h, "title") else h.name
            text = (h.text if hasattr(h, "text") else "")
            out_lines.append("《" + title + "》 " + text)
            data.append({"title": title, "excerpt": text})
        return ToolResult("search_policy", "\n".join(out_lines), {"hits": data})

    return reg
