from langchain_core.messages import AIMessage

from mini_research_agent import cli


class StubGraph:
    def __init__(self, results):
        self.results = list(results)
        self.inputs = []

    def invoke(self, graph_input):
        self.inputs.append(graph_input)
        result = self.results.pop(0)
        if callable(result):
            return result(graph_input)
        return result


def test_cli_prints_brief_for_clear_request(monkeypatch, capsys):
    graph = StubGraph(
        [{"research_brief": "旧金山咖啡店研究简报"}]
    )
    monkeypatch.setattr(cli, "create_scope_graph", lambda: graph)
    monkeypatch.setattr("builtins.input", lambda _prompt: "研究旧金山咖啡店")

    cli.main()

    output = capsys.readouterr().out
    assert "研究简报" in output
    assert "旧金山咖啡店研究简报" in output
    assert graph.inputs[0]["messages"][-1].content == "研究旧金山咖啡店"


def test_cli_preserves_history_across_clarification(monkeypatch, capsys):
    def clarification_result(graph_input):
        return {
            "messages": [
                *graph_input["messages"],
                AIMessage(content="你希望研究哪个城市？"),
            ]
        }

    graph = StubGraph(
        [
            clarification_result,
            {"research_brief": "研究旧金山咖啡店"},
        ]
    )
    answers = iter(["研究咖啡店", "旧金山"])
    monkeypatch.setattr(cli, "create_scope_graph", lambda: graph)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    cli.main()

    output = capsys.readouterr().out
    second_messages = graph.inputs[1]["messages"]
    assert "需要澄清" in output
    assert "你希望研究哪个城市？" in output
    assert "研究旧金山咖啡店" in output
    assert [message.content for message in second_messages] == [
        "研究咖啡店",
        "你希望研究哪个城市？",
        "旧金山",
    ]


def test_cli_exits_when_input_is_empty(monkeypatch, capsys):
    graph = StubGraph([])
    monkeypatch.setattr(cli, "create_scope_graph", lambda: graph)
    monkeypatch.setattr("builtins.input", lambda _prompt: "   ")

    cli.main()

    assert "未提供有效内容" in capsys.readouterr().out
    assert graph.inputs == []
