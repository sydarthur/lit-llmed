# Build Plan: Abstract Enrichment & Hybrid Literature Review Pipeline

**Created:** 2025-12-25
**Status:** Planning
**Priority:** Future Enhancement

---

## Overview

This document outlines a hybrid approach to literature management that combines automated discovery/triage with manual deep review. The goal is to create a practical workflow for OSCM researchers to stay current with journal publications while efficiently identifying papers worthy of deeper analysis.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 1: DISCOVERY & TRIAGE (Automated)              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. FETCH: Pull latest articles from configured journals                │
│     • JBL, POM, JOM via Crossref/OpenAlex APIs                         │
│     • Metadata: title, authors, abstract, DOI, dates                   │
│                                                                         │
│  2. ENRICH: LLM analysis on abstracts                                  │
│     • Categorize by topic/theme                                         │
│     • Identify theory/framework used                                    │
│     • Classify methodology                                              │
│     • Extract key constructs/variables                                  │
│     • Generate relevance tags                                           │
│                                                                         │
│  3. OUTPUT: Structured exports                                          │
│     • Individual markdown notes (Obsidian-ready)                       │
│     • Grouped digests (by topic, method, theory)                       │
│     • Push to Zotero with tags (Auto/JBL, Auto/POM, etc.)             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
                          [Manual Review by User]
                                    ↓
                      ┌──────────────────────────┐
                      │  Interesting paper?      │
                      │  → Download PDF          │
                      │  → Add to "Deep Review"  │
                      │    collection in Zotero  │
                      └──────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 2: DEEP ANALYSIS (On-Demand)                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Watch Zotero "Deep Review" collection for papers with PDFs attached   │
│                                    ↓                                    │
│  Run targeted PDF analysis:                                             │
│     • Extract detailed methodology                                      │
│     • Identify research gaps cited                                      │
│     • Pull specific findings/data                                       │
│     • Map theoretical contributions                                     │
│                                    ↓                                    │
│  Generate rich literature review notes → Obsidian                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Abstract Enrichment (Primary Focus)

### 1.1 LLM Integration for Abstract Analysis

**Objective:** Run LLM on fetched abstracts to extract structured metadata.

**LLM Options:**
| Option | Pros | Cons |
|--------|------|------|
| Local Ollama (llama3, mistral) | Free, private, no API limits | Requires local GPU, variable quality |
| Claude API (Haiku) | High quality, cheap (~$0.001/abstract) | Requires API key, costs scale |
| Claude API (Sonnet) | Best quality | More expensive |

**Recommendation:** Start with Ollama for development, option to use Claude Haiku for production.

**Implementation Location:** `src/features/pdf_enrich/abstract_enricher.py` (new file)

### 1.2 Extraction Schema

**Structured output per article:**

```python
class AbstractEnrichment(BaseModel):
    """Structured extraction from abstract analysis."""

    # Topic Classification
    primary_topic: str  # e.g., "Supply Chain Resilience", "Last-Mile Delivery"
    secondary_topics: List[str]  # Additional themes

    # Theoretical Framework
    theory_used: Optional[str]  # e.g., "Resource-Based View", "Transaction Cost Economics"
    theory_contribution: Optional[str]  # "extends", "tests", "develops", "applies"

    # Methodology
    method_type: str  # "empirical-quantitative", "empirical-qualitative", "analytical", "conceptual", "review"
    method_detail: Optional[str]  # "survey", "experiment", "case study", "simulation", "econometric"
    data_source: Optional[str]  # "primary", "secondary", "archival"

    # Key Elements
    constructs: List[str]  # Key variables/constructs studied
    context: Optional[str]  # Industry/geographic context

    # Auto-generated
    tags: List[str]  # Computed from above fields
    relevance_score: Optional[float]  # 0-1 based on user preferences (future)

    # Summary
    one_line_summary: str  # Single sentence summary
    key_contribution: str  # Main contribution in 1-2 sentences
```

### 1.3 Prompt Engineering

**Base prompt template:**

```
You are an academic research assistant specializing in Operations and Supply Chain Management (OSCM).

Analyze the following journal article abstract and extract structured information.

ARTICLE:
Title: {title}
Authors: {authors}
Journal: {journal}
Year: {year}

ABSTRACT:
{abstract}

INSTRUCTIONS:
1. Identify the primary research topic/theme
2. Note any theoretical framework or theory used
3. Classify the methodology type
4. Extract key constructs or variables studied
5. Identify the industry/geographic context if mentioned
6. Write a one-line summary
7. State the key contribution

Respond in JSON format matching this schema:
{schema}
```

**Topic taxonomy (OSCM-specific):**
- Supply Chain Strategy
- Supply Chain Risk & Resilience
- Sustainable/Green Supply Chain
- Digital Supply Chain / Industry 4.0
- Logistics & Transportation
- Last-Mile Delivery
- Inventory Management
- Procurement & Sourcing
- Manufacturing & Operations
- Service Operations
- Healthcare Operations
- Humanitarian Logistics
- Supplier Relationships
- Quality Management
- Behavioral Operations

**Method taxonomy:**
- Empirical - Survey
- Empirical - Experiment/RCT
- Empirical - Case Study
- Empirical - Econometric/Archival
- Empirical - Mixed Methods
- Analytical - Optimization
- Analytical - Game Theory
- Analytical - Simulation
- Conceptual/Theory Building
- Literature Review/Meta-Analysis

### 1.4 Enhanced Markdown Output

**Individual article note format:**

```markdown
---
title: "{title}"
authors: [{authors}]
journal: "{journal}"
year: {year}
doi: "{doi}"
tags: [{tags}]
topic: "{primary_topic}"
theory: "{theory_used}"
method: "{method_type}"
status: "unread"
---

# {title}

## Quick Summary
{one_line_summary}

## Key Contribution
{key_contribution}

## Classification
- **Topic:** {primary_topic}
- **Theory:** {theory_used} ({theory_contribution})
- **Method:** {method_type} - {method_detail}
- **Context:** {context}
- **Constructs:** {constructs}

## Abstract
{abstract}

## Citation
```bibtex
{bibtex}
```

## Notes
<!-- Your reading notes here -->

```

**Grouped digest format:**

```markdown
# Literature Digest: {date}

## By Topic

### Supply Chain Resilience (5 papers)
1. [Author2024Title](link) - {one_line_summary}
2. ...

### Last-Mile Delivery (3 papers)
...

## By Method

### Empirical - Survey (4 papers)
...

### Analytical - Optimization (2 papers)
...

## By Theory

### Resource-Based View (3 papers)
...
```

### 1.5 Zotero Integration Enhancement

**Current:** Push articles to collections

**Enhanced:**
- Add tags from enrichment (topic, method, theory)
- Add to multiple collections based on classification
- Store enrichment data in Zotero "Extra" field as JSON

**Tag format in Zotero:**
- `topic:supply-chain-resilience`
- `method:survey`
- `theory:rbv`
- `status:unread`

### 1.6 CLI Commands

```bash
# Fetch and enrich (full pipeline)
python -m src.cli fetch-all --email you@example.com --enrich --zotero

# Enrich only (already fetched articles)
python -m src.cli enrich --input output/data/latest.json --output output/enriched/

# Generate grouped digest
python -m src.cli digest --input output/enriched/ --group-by topic method theory

# Filter/query enriched articles
python -m src.cli query --method survey --topic "supply chain resilience" --since 90d
```

---

## Phase 2: Deep PDF Analysis (Future)

### 2.1 Trigger Mechanism

**Option A: Watch Zotero collection**
- Monitor "Deep Review" collection via Zotero API
- When new item with PDF attachment detected → trigger analysis

**Option B: Manual CLI command**
```bash
python -m src.cli analyze-pdf --zotero-key ABC123
python -m src.cli analyze-collection --collection "Deep Review"
```

### 2.2 PDF Processing Strategy

**Approach: Targeted extraction (not full PDF read)**

1. **Text extraction** via pdfplumber/PyMuPDF
2. **Section detection** via heuristics:
   - Find "Introduction", "Literature Review", "Methodology", "Results", "Discussion"
   - Use regex patterns for common headings
3. **Selective LLM analysis:**
   - Send only relevant sections to LLM
   - Skip references, appendices, tables
4. **Cost control:**
   - Estimate token count before sending
   - Use chunking if needed
   - Option to use cheaper models for initial pass

### 2.3 Deep Extraction Schema

```python
class DeepEnrichment(BaseModel):
    """Deep extraction from full PDF analysis."""

    # Research Design
    research_questions: List[str]
    hypotheses: List[str]

    # Methodology Details
    sample_size: Optional[str]
    data_collection_period: Optional[str]
    analysis_technique: Optional[str]
    variables: Dict[str, str]  # variable_name: description

    # Findings
    key_findings: List[str]
    effect_sizes: Optional[Dict[str, str]]

    # Literature Positioning
    research_gaps_cited: List[str]
    key_citations: List[str]  # Most referenced papers

    # Contribution
    theoretical_contribution: Optional[str]
    practical_implications: List[str]
    limitations: List[str]
    future_research: List[str]
```

---

## Implementation Roadmap

### Milestone 1: Basic Abstract Enrichment
- [ ] Create `AbstractEnricher` class with Ollama integration
- [ ] Define Pydantic schema for enrichment output
- [ ] Create prompt templates for OSCM domain
- [ ] Test with sample abstracts

### Milestone 2: Integration with Fetch Pipeline
- [ ] Add `--enrich` flag to `fetch-all` command
- [ ] Modify `FetchJob` to call enricher
- [ ] Store enrichment in Article model
- [ ] Update JSON export to include enrichment

### Milestone 3: Enhanced Output
- [ ] Update `MarkdownDigest` for enriched articles
- [ ] Add grouped digest generation
- [ ] Add tags to Zotero push

### Milestone 4: Query & Filter
- [ ] Add `query` CLI command
- [ ] Implement filtering by enrichment fields
- [ ] Add search across enriched articles

### Milestone 5: Deep Analysis (Future)
- [ ] PDF text extraction pipeline
- [ ] Section detection
- [ ] Deep extraction prompt/schema
- [ ] Integration with Zotero PDF attachments

---

## Configuration

**New config file: `config/enrichment.yaml`**

```yaml
enrichment:
  enabled: true

  # LLM Provider
  provider: "ollama"  # or "anthropic"
  model: "llama3"     # or "claude-3-haiku-20240307"

  # Anthropic settings (if using)
  anthropic:
    api_key: "${ANTHROPIC_API_KEY}"
    max_tokens: 1000

  # Ollama settings
  ollama:
    host: "http://localhost:11434"
    timeout: 60

  # Domain settings
  domain: "oscm"  # Operations & Supply Chain Management

  # Output preferences
  generate_individual_notes: true
  generate_grouped_digest: true
  group_by:
    - topic
    - method
    - theory
```

---

## Dependencies

**New dependencies to add:**

```
# For Ollama
ollama>=0.1.0

# For Claude API (optional)
anthropic>=0.18.0

# For PDF processing (Phase 2)
pdfplumber>=0.10.0  # Already have
pymupdf>=1.23.0     # Better text extraction
```

---

## Cost Estimates

**Phase 1 (Abstract Enrichment):**

| Provider | Cost per Abstract | 100 Articles/Month |
|----------|------------------|-------------------|
| Ollama (local) | $0 | $0 |
| Claude Haiku | ~$0.001 | ~$0.10 |
| Claude Sonnet | ~$0.01 | ~$1.00 |

**Phase 2 (PDF Analysis):**

| Approach | Cost per Paper | Notes |
|----------|---------------|-------|
| Text extraction + Haiku | ~$0.05 | Selective sections |
| Text extraction + Sonnet | ~$0.30 | Better quality |
| Full PDF via Claude | ~$1-5 | Vision processing, expensive |

**Recommendation:** Use Ollama for Phase 1, Claude Haiku for Phase 2 with selective extraction.

---

## Related Files

**Existing (to modify):**
- `src/features/journal_fetch/fetch_job.py` - Add enrichment step
- `src/features/reporting/markdown_reporter.py` - Enhanced templates
- `src/features/obsidian_sync/zotero_client.py` - Add tagging
- `src/core/models.py` - Add enrichment fields
- `src/cli.py` - New commands

**New files:**
- `src/features/pdf_enrich/abstract_enricher.py`
- `src/features/pdf_enrich/prompts/oscm_abstract.py`
- `src/features/pdf_enrich/schemas.py`
- `config/enrichment.yaml`

---

## References

- Twitter thread on Claude Code for literature review (subagent approach)
- Existing `src/features/pdf_enrich/` module (to be revived/refactored)
- Crossref API docs: https://api.crossref.org
- Zotero API docs: https://www.zotero.org/support/dev/web_api/v3/start

---

## Notes

- Start simple: get basic topic/method classification working first
- User can manually curate "Deep Review" collection for expensive PDF analysis
- Consider caching enrichment results to avoid re-processing
- Taxonomy can be customized per user/domain
