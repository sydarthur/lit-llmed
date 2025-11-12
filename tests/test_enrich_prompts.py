from src.features.pdf_enrich.prompts import DEFAULT_TEMPLATE, get_paper_template, get_abstract_template


def test_get_paper_template_known_type():
    template = get_paper_template("editorial")
    assert "prompt" in template
    assert "output" in template


def test_get_paper_template_default():
    template = get_paper_template("unknown")
    assert template == DEFAULT_TEMPLATE


def test_get_abstract_template_structure():
    template = get_abstract_template()
    prompt = template["prompt"]
    assert "Return ONLY valid JSON" in prompt
    assert "{abstract}" in prompt
