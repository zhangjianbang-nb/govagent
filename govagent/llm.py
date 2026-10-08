"""可选 LLM 客户端：任何 OpenAI 兼容 /chat/completions 端点。

零 SDK 依赖——直接用 urllib 发请求。未配置时 GovAgent 自动
回退到确定性规划器，不影响离线使用。
"""

from __future__ import annotations

import json
import urllib.request


class LLMClient:
    """极简 OpenAI 兼容客户端（chat.completions + tool_calls）。"""

    def __init__(self, base_url: str, api_key: str = "EMPTY",
                 model: str = "default", temperature: float = 0.3):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

    def chat(self, messages: list, tools: list | None = None) -> dict:
        """调用 /chat/completions，返回 assistant message dict。"""
        payload = {"model": self.model, "messages": messages,
                   "temperature": self.temperature}
        if tools:
            payload["tools"] = tools
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + self.api_key})
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]


class MockLLM(LLMClient):
    """测试用伪 LLM：按预设脚本依次返回消息，不发网络请求。

    用法: MockLLM(script=[{"tool_calls": [...]}, {"content": "..."}])
    """

    def __init__(self, script: list):
        self.script = list(script)
        self.calls = []
        super().__init__("http://mock.local", model="mock")

    def chat(self, messages: list, tools: list | None = None) -> dict:
        self.calls.append(messages)
        return self.script.pop(0)
