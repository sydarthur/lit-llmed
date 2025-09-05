"""Configuration for different paper types and their prompts"""

PAPER_TYPES = {
    "editorial": {
        "prompt_template": """You are analyzing an editorial or opinion paper. Create structured notes for Obsidian.

Create notes with these sections:

## Summary
- Brief overview of the editorial's main argument or position

## Key Arguments  
- Main points the author makes
- Supporting evidence or examples

## Editorial Position
- What stance does the author take?
- What recommendations are made?

## Context
- What issue or debate is being addressed?
- Background information provided

## Tags
- Add relevant tags using #tag format

Text to analyze:
{text}

Generate the structured notes:""",
        
        "output_template": "# {title}\n\n**Source:** {source}\n**Type:** Editorial\n**Date Processed:** {date}\n\n{content}"
    },
    
    "theory": {
        "prompt_template": """You are analyzing a theoretical paper. Create structured notes for Obsidian.

Create notes with these sections:

## Summary
- Brief overview of the theoretical contribution

## Theory/Framework
- Main theoretical framework presented
- Key concepts and definitions

## Propositions/Hypotheses
- Theoretical propositions made
- Relationships between concepts

## Literature Integration
- How this builds on existing theory
- Key citations and connections

## Implications
- Theoretical implications
- Future research directions

## Tags
- Add relevant tags using #tag format

Text to analyze:
{text}

Generate the structured notes:""",
        
        "output_template": "# {title}\n\n**Source:** {source}\n**Type:** Theory Paper\n**Date Processed:** {date}\n\n{content}"
    },
    
    "method": {
        "prompt_template": """You are analyzing a methodology paper. Create structured notes for Obsidian.

Create notes with these sections:

## Summary
- Brief overview of the methodological contribution

## Method/Approach
- New method or approach presented
- Technical details and procedures

## Validation
- How the method was tested or validated
- Performance metrics or results

## Advantages/Limitations
- Benefits of this approach
- Limitations or constraints

## Applications
- Where this method can be used
- Example applications shown

## Tags
- Add relevant tags using #tag format

Text to analyze:
{text}

Generate the structured notes:""",
        
        "output_template": "# {title}\n\n**Source:** {source}\n**Type:** Methodology Paper\n**Date Processed:** {date}\n\n{content}"
    },
    
    "topic": {
        "prompt_template": """You are analyzing a research paper on a specific topic. Create structured notes for Obsidian.

Create notes with these sections:

## Summary
- Brief overview of the research

## Research Question/Problem
- What problem is being addressed?
- Research questions or objectives

## Key Findings
- Main results or findings
- Important data or statistics

## Methods Used
- Research approach and methods
- Data sources

## Conclusions
- Main conclusions drawn
- Practical implications

## Future Work
- Suggested future research
- Open questions

## Tags
- Add relevant tags using #tag format

Text to analyze:
{text}

Generate the structured notes:""",
        
        "output_template": "# {title}\n\n**Source:** {source}\n**Type:** Research Paper\n**Date Processed:** {date}\n\n{content}"
    }
}

def get_paper_config(paper_type: str) -> dict:
    """Get configuration for a specific paper type"""
    return PAPER_TYPES.get(paper_type, PAPER_TYPES["topic"])  # Default to topic if not found