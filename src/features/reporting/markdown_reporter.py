"""Generate professional Markdown digests from fetched articles."""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.core.log import get_logger
from src.core.models import Article

LOGGER = get_logger(__name__)


class MarkdownDigest:
    """Generates a Markdown digest of articles."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_digest(
        self,
        articles: List[Article],
        *,
        title: str = "Research Digest",
        filename: Optional[str] = None,
    ) -> Path:
        """Create a Markdown digest for the given articles."""
        if not filename:
            date_str = datetime.now().strftime("%Y-%m-%d")
            filename = f"{date_str}-digest.md"
        
        output_path = self.output_dir / filename
        
        # Group by Journal
        by_journal: Dict[str, List[Article]] = defaultdict(list)
        for article in articles:
            # Fallback to "Unknown Journal" if journal is None/empty
            j_name = article.journal or "Unknown Journal"
            by_journal[j_name].append(article)

        content = [
            f"# {title}",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"**Total Articles:** {len(articles)}",
            "",
            "[TOC]",
            "",
        ]

        sorted_journals = sorted(by_journal.keys())
        for journal_name in sorted_journals:
            journal_articles = by_journal[journal_name]
            content.append(f"## {journal_name}")
            content.append(f"*{len(journal_articles)} new articles*")
            content.append("")
            
            for article in journal_articles:
                content.append(self._format_article(article))
                content.append("---")
                content.append("")

        output_path.write_text("\n".join(content), encoding="utf-8")
        LOGGER.info("markdown_digest_generated", path=output_path, count=len(articles))
        return output_path

    def _format_article(self, article: Article) -> str:
        """Format a single article block."""
        creators = self._format_authors(article)
        year = article.published.year if article.published else "n.d."
        citekey = self._generate_citekey(article, year)
        
        # Title and basic metadata
        lines = [
            f"### {article.title}",
            f"**Authors:** {creators} ({year})",
        ]
        
        if article.doi:
            lines.append(f"**DOI:** [{article.doi}](https://doi.org/{article.doi})")
        
        # Abstract (collapsible)
        if article.abstract:
            # Simple cleanup of abstract text if needed
            abstract_text = article.abstract.strip()
            lines.append("<details>")
            lines.append("<summary><strong>Abstract</strong></summary>")
            lines.append("")
            lines.append(abstract_text)
            lines.append("")
            lines.append("</details>")
            lines.append("")

        # Citation Section
        lines.append("#### Citation")
        lines.append("```bibtex")
        lines.append(f"@article{{{citekey},")
        lines.append(f"  title = {{{article.title}}},")
        lines.append(f"  author = {{{self._bibtex_authors(article)}}},")
        if article.journal:
            lines.append(f"  journal = {{{article.journal}}},")
        lines.append(f"  year = {{{year}}},")
        if article.doi:
            lines.append(f"  doi = {{{article.doi}}}")
        lines.append("}")
        lines.append("```")
        
        return "\n".join(lines)

    def _format_authors(self, article: Article) -> str:
        if not article.authors:
            return "Unknown"
        if len(article.authors) > 3:
            first = article.authors[0]
            return f"{first.given} {first.family} et al."
        return ", ".join(f"{a.given} {a.family}" for a in article.authors)

    def _bibtex_authors(self, article: Article) -> str:
        if not article.authors:
            return "Unknown"
        return " and ".join(f"{a.family}, {a.given}" for a in article.authors)

    def _generate_citekey(self, article: Article, year: str | int) -> str:
        """Generate a unique citation key: AuthorYearTitleWord."""
        if not article.authors:
            author_part = "Anon"
        else:
            author_part = self._clean_str(article.authors[0].family)
        
        title_part = "Title"
        if article.title:
            # First significant word
            words = [w for w in article.title.split() if len(w) > 3]
            if words:
                title_part = self._clean_str(words[0])
            else:
                title_part = self._clean_str(article.title[:10])
                
        return f"{author_part}{year}{title_part}"

    def _clean_str(self, text: str) -> str:
        """Remove non-alphanumeric characters."""
        return re.sub(r'[^a-zA-Z0-9]', '', text)
