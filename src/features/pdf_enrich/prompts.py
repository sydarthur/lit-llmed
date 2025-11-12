"""Prompt templates for transforming academic texts into structured notes."""

from __future__ import annotations

from typing import Dict


PAPER_TEMPLATES: Dict[str, Dict[str, str]] = {
    "editorial": {
        "prompt": """You are analyzing an editorial or opinion paper. Create structured notes for Obsidian.\n\nCreate notes with these sections:\n\n## Summary\n- Brief overview of the editorial's main argument or position\n\n## Key Arguments  \n- Main points the author makes\n- Supporting evidence or examples\n\n## Editorial Position\n- What stance does the author take?\n- What recommendations are made?\n\n## Context\n- What issue or debate is being addressed?\n- Background information provided\n\n## Tags\n- Add relevant tags using #tag format\n\nText to analyze:\n{text}\n\nGenerate the structured notes:""",
        "output": "# {title}\n\n**Source:** {source}\n**Type:** Editorial\n**Date Processed:** {date}\n\n{content}",
    },
    "theory": {
        "prompt": """You are analyzing a theoretical paper. Create structured notes for Obsidian.\n\nCreate notes with these sections:\n\n## Summary\n- Brief overview of the theoretical contribution\n\n## Theory/Framework\n- Main theoretical framework presented\n- Key concepts and definitions\n\n## Propositions/Hypotheses\n- Theoretical propositions made\n- Relationships between concepts\n\n## Literature Integration\n- How this builds on existing theory\n- Key citations and connections\n\n## Implications\n- Theoretical implications\n- Future research directions\n\n## Tags\n- Add relevant tags using #tag format\n\nText to analyze:\n{text}\n\nGenerate the structured notes:""",
        "output": "# {title}\n\n**Source:** {source}\n**Type:** Theory Paper\n**Date Processed:** {date}\n\n{content}",
    },
    "method": {
        "prompt": """You are analyzing a methodology paper. Create structured notes for Obsidian.\n\nCreate notes with these sections:\n\n## Summary\n- Brief overview of the methodological contribution\n\n## Method/Approach\n- New method or approach presented\n- Technical details and procedures\n\n## Validation\n- How the method was tested or validated\n- Performance metrics or results\n\n## Advantages/Limitations\n- Benefits of this approach\n- Limitations or constraints\n\n## Applications\n- Where this method can be used\n- Example applications shown\n\n## Tags\n- Add relevant tags using #tag format\n\nText to analyze:\n{text}\n\nGenerate the structured notes:""",
        "output": "# {title}\n\n**Source:** {source}\n**Type:** Methodology Paper\n**Date Processed:** {date}\n\n{content}",
    },
    "topic": {
        "prompt": """You are analyzing a research paper on a specific topic. Create structured notes for Obsidian.\n\nCreate notes with these sections:\n\n## Summary\n- Brief overview of the research\n\n## Research Question/Problem\n- What problem is being addressed?\n- Research questions or objectives\n\n## Key Findings\n- Main results or findings\n- Important data or statistics\n\n## Methods Used\n- Research approach and methods\n- Data sources\n\n## Conclusions\n- Main conclusions drawn\n- Practical implications\n\n## Future Work\n- Suggested future research\n- Open questions\n\n## Tags\n- Add relevant tags using #tag format\n\nText to analyze:\n{text}\n\nGenerate the structured notes:""",
        "output": "# {title}\n\n**Source:** {source}\n**Type:** Research Paper\n**Date Processed:** {date}\n\n{content}",
    },
}

DEFAULT_TEMPLATE = PAPER_TEMPLATES["topic"]

ABSTRACT_TEMPLATE: Dict[str, str] = {
    "prompt": """Convert this article title and abstract into structured JSON. Return ONLY valid JSON:\n{{\n  "summary": str,\n  "primary_topic": str,\n  "methodology": str,\n  "key_findings": [str, ...],\n  "tags": [str, ...]\n}}\n\nTitle: {title}\n\nAbstract:\n{abstract}\n\nJSON only:""",
}


def get_paper_template(paper_type: str) -> Dict[str, str]:
    """Return the prompt/output template pair for the given paper type."""

    return PAPER_TEMPLATES.get(paper_type, DEFAULT_TEMPLATE)


def get_abstract_template() -> Dict[str, str]:
    """Return the template for abstract summarisation.""" 

    return ABSTRACT_TEMPLATE
