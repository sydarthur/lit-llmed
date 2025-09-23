import json
from pathlib import Path

from src.core.models import Article
from src.enrich.abstract_summarizer import AbstractSummarizer
from src.enrich.settings import EnrichmentPaths


class DummyLLM:
    def __init__(self, response: str):
        self.response = response

    def chat(self, prompt: str):  # pragma: no cover - simple stub
        return self.response


def make_article(**kwargs):
    data = {
        "title": "Sample",
        "doi": "10.1234/abc",
        "journal_name": "Journal",
        "abstract": "This is an abstract about supply chain optimization.",
    }
    data.update(kwargs)
    return Article.model_validate(data)


def make_paths(tmp_path: Path) -> EnrichmentPaths:
    paths = EnrichmentPaths(input_root=tmp_path / "in", output_root=tmp_path / "out")
    paths.ensure()
    return paths


def test_build_prompt_includes_metadata(tmp_path):
    summarizer = AbstractSummarizer(paths=make_paths(tmp_path), llm=DummyLLM("{}"))
    article = make_article()
    prompt = summarizer._build_prompt(article)
    assert "Sample" in prompt
    assert "10.1234/abc" in prompt
    assert "supply chain" in prompt


def test_parse_response_strips_extra_text():
    payload = {"summary": "ok", "primary_topic": "topic", "methodology": "method", "key_findings": [], "tags": [], "title": "Sample", "doi": None, "journal": None, "published_date": None}
    raw = "Some preface\n" + json.dumps(payload) + "\nThanks!"
    parsed = AbstractSummarizer._parse_response(raw)
    assert parsed["summary"] == "ok"


def test_summarise_article_returns_none_without_abstract(tmp_path):
    summarizer = AbstractSummarizer(paths=make_paths(tmp_path), llm=DummyLLM("{}"))
    article = make_article(abstract=None)
    assert summarizer.summarise_article(article) is None


def test_summarise_article_parses_json(tmp_path):
    response = json.dumps(
        {
            "title": "Sample",
            "summary": "Summary",
            "primary_topic": "Topic",
            "methodology": "Method",
            "key_findings": ["Finding"],
            "tags": ["tag"],
            "journal": "Journal",
            "published_date": "2024-01-01",
            "doi": "10.1234/abc",
        }
    )
    summarizer = AbstractSummarizer(paths=make_paths(tmp_path), llm=DummyLLM(response))
    article = make_article()
    result = summarizer.summarise_article(article)
    assert result["summary"] == "Summary"
    assert result["key_findings"] == ["Finding"]
