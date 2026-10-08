"""CLI 冒烟测试。"""

from govagent.cli import main


def test_cli_query(capsys):
    rc = main(["社保卡丢了怎么补办"])
    assert rc == 0


def test_cli_domains(capsys):
    rc = main(["--domains"])
    assert rc == 0
