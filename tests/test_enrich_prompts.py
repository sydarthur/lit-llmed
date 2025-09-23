from src.enrich.prompts import get_paper_template, DEFAULT_TEMPLATE


def test_get_paper_template_known_type():
    template = get_paper_template("editorial")
    assert "prompt" in template
    assert "output" in template


def test_get_paper_template_default():
    template = get_paper_template("unknown")
    assert template == DEFAULT_TEMPLATE
