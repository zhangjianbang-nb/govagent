"""命令行入口：python -m govagent 或 govagent 命令。

用法:
    govagent "社保卡丢了怎么补办"
    govagent --chat                 # 交互模式
    govagent --domains              # 列出支持的领域
"""

from __future__ import annotations

import argparse
import sys

from .agent import GovAgent
from .knowledge import KnowledgeBase


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="govagent", description="政务服务智能体")
    parser.add_argument("query", nargs="*", help="要咨询的问题")
    parser.add_argument("--chat", action="store_true", help="进入交互模式")
    parser.add_argument("--domains", action="store_true", help="列出支持的办事领域")
    parser.add_argument("--data-dir", default=None, help="自定义知识库 JSON 目录")
    parser.add_argument("--trace", action="store_true", help="输出工具调用轨迹")
    args = parser.parse_args(argv)

    kb = KnowledgeBase(data_dir=args.data_dir)
    agent = GovAgent(knowledge_base=kb)

    if args.domains:
        for d in kb.list_domains():
            print(d)
        return 0

    if args.chat or not args.query:
        return _chat(agent, trace=args.trace)

    reply = agent.run(" ".join(args.query))
    print(reply.text)
    if args.trace:
        print("\n-- 意图: %s | 工具: %s" % (reply.intent, ", ".join(reply.tools_used)), file=sys.stderr)
    return 0


def _chat(agent: GovAgent, trace: bool = False) -> int:
    print("政务服务智能体已启动（离线规划器模式），输入问题咨询，exit 退出。")
    while True:
        try:
            line = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if line in {"exit", "quit", "q"}:
            return 0
        if not line:
            continue
        reply = agent.run(line)
        print("助手> " + reply.text)
        if trace:
            print("[意图: %s | 工具: %s]" % (reply.intent, ", ".join(reply.tools_used)), file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
