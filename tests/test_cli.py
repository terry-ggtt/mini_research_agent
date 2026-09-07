from types import SimpleNamespace

from langchain_core.messages import AIMessage

from mini_research_agent import cli


class StubGraph:
    def __init__(self, results):
        self.results = list(results)
        self.inputs = []
        self.configs = []

    async def ainvoke(self, graph_input, config=None):
        self.inputs.append(graph_input)
        self.configs.append(config)
        result = self.results.pop(0)
        if callable(result):
            return result(graph_input)
        return result


def install_application_stub(monkeypatch, graph):
    async def create_application(**_kwargs):
        return SimpleNamespace(graph=graph)

    monkeypatch.setattr(
        cli,
        "create_application",
        create_application,
    )


def test_cli_prints_final_report_for_clear_request(monkeypatch, capsys):
    graph = StubGraph(
        [
            {
                "research_brief": "旧金山咖啡店研究简报",
                "final_report": "旧金山咖啡店最终报告",
                "notes": ["压缩后的研究资料"],
                "raw_notes": ["原始来源"],
            }
        ]
    )
    install_application_stub(monkeypatch, graph)
    monkeypatch.setattr(
        "builtins.input",
        lambda _prompt: "研究旧金山咖啡店",
    )

    cli.main()

    output = capsys.readouterr().out
    assert "最终报告：" in output
    assert "旧金山咖啡店最终报告" in output
    assert "原始来源" not in output
    assert graph.inputs[0]["messages"][-1].content == "研究旧金山咖啡店"
    assert graph.configs[0]["configurable"]["thread_id"] == (
        "research-session-1"
    )


def test_cli_reuses_thread_across_clarification(monkeypatch, capsys):
    graph = StubGraph(
        [
            {
                "messages": [
                    AIMessage(content="你希望研究哪个城市？"),
                ]
            },
            {
                "research_brief": "研究旧金山咖啡店",
                "final_report": "最终研究报告",
                "notes": [],
                "raw_notes": [],
            },
        ]
    )
    answers = iter(["研究咖啡店", "旧金山"])
    install_application_stub(monkeypatch, graph)
    monkeypatch.setattr(
        "builtins.input",
        lambda _prompt: next(answers),
    )

    cli.main()

    output = capsys.readouterr().out
    assert "需要补充信息：" in output
    assert "你希望研究哪个城市？" in output
    assert "最终报告：" in output
    assert "最终研究报告" in output

    assert [
        invocation["messages"][0].content
        for invocation in graph.inputs
    ] == ["研究咖啡店", "旧金山"]
    assert graph.configs[0] == graph.configs[1]


def test_cli_exits_when_input_is_empty(monkeypatch, capsys):
    graph = StubGraph([])
    install_application_stub(monkeypatch, graph)
    monkeypatch.setattr(
        "builtins.input",
        lambda _prompt: "   ",
    )

    cli.main()

    assert "未提供有效内容" in capsys.readouterr().out
    assert graph.inputs == []
