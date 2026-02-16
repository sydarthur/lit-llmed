#!/usr/bin/env python3
"""Consolidate digest files into journal-specific files with duplicate tracking."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set


class Article:
    """Simple article representation for tracking."""

    def __init__(self, title: str, doi: str, content: str):
        self.title = title
        self.doi = doi
        self.content = content  # Full markdown content for this article


class DigestProcessor:
    """Process and consolidate digest files."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.digest_dir = project_root / "output" / "enrich" / "digests"
        self.journal_dir = project_root / "output" / "enrich" / "journal_digests"
        self.tracking_file = self.journal_dir / ".processed.json"

        # Create journal digest directory
        self.journal_dir.mkdir(parents=True, exist_ok=True)

        # Load tracking data
        self.tracking = self._load_tracking()

    def _load_tracking(self) -> Dict[str, Dict]:
        """Load tracking JSON with processed DOIs per journal."""
        if not self.tracking_file.exists():
            return {}

        try:
            with open(self.tracking_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load tracking file: {e}")
            return {}

    def _save_tracking(self):
        """Save tracking JSON."""
        try:
            with open(self.tracking_file, 'w', encoding='utf-8') as f:
                json.dump(self.tracking, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error: Could not save tracking file: {e}")

    def _parse_digest_file(self, file_path: Path) -> Dict[str, List[Article]]:
        """Parse a digest file and return articles grouped by journal."""
        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception as e:
            print(f"Error reading {file_path.name}: {e}")
            return {}

        # Extract date from the digest file metadata
        date_match = re.search(r'\*\*Date:\*\* (.+)', content)
        digest_date = date_match.group(1) if date_match else "Unknown Date"

        articles_by_journal: Dict[str, List[Article]] = {}

        # Split by journal sections (## Journal Name)
        journal_sections = re.split(r'^## (.+)$', content, flags=re.MULTILINE)

        # Process each journal section
        for i in range(1, len(journal_sections), 2):
            journal_name = journal_sections[i].strip()
            section_content = journal_sections[i + 1] if i + 1 < len(journal_sections) else ""

            if journal_name == "[TOC]" or not section_content:
                continue

            # Note: Process "Unknown Journal" but warn about it
            if journal_name == "Unknown Journal":
                pass  # Still process, but could log warning if needed

            # Split articles by ### title and ---
            article_blocks = re.split(r'(?=^### )', section_content, flags=re.MULTILINE)

            for block in article_blocks:
                if not block.strip() or not block.startswith('###'):
                    continue

                # Extract DOI
                doi_match = re.search(r'\*\*DOI:\*\* \[([^\]]+)\]', block)
                if not doi_match:
                    continue

                doi = doi_match.group(1)

                # Extract title
                title_match = re.search(r'^### (.+)$', block, re.MULTILINE)
                title = title_match.group(1).strip() if title_match else "Untitled"

                # Try to extract journal from BibTeX citation if available
                actual_journal = journal_name
                if journal_name == "Unknown Journal":
                    # Look for journal field in BibTeX citation
                    bibtex_journal_match = re.search(r'journal\s*=\s*\{([^}]+)\}', block)
                    if bibtex_journal_match:
                        actual_journal = bibtex_journal_match.group(1).strip()
                    else:
                        # Fallback: extract journal code from DOI suffix
                        # DOI format: 10.xxxx/journal-code-rest or 10.xxxx/journal.code.rest
                        doi_parts = doi.split('/', 1)
                        if len(doi_parts) > 1:
                            suffix = doi_parts[1]
                            # Extract journal code (first alphabetic part before - or .)
                            journal_code_match = re.match(r'^([a-zA-Z]+)', suffix)
                            if journal_code_match:
                                journal_code = journal_code_match.group(1).upper()
                                actual_journal = f"{journal_code}"
                            else:
                                actual_journal = f"Unknown ({doi.split('/')[0]})"
                        else:
                            actual_journal = "Unknown Journal"

                # Clean up block - remove trailing "---"
                clean_block = re.sub(r'\n---\s*$', '', block.strip())

                article = Article(title=title, doi=doi, content=clean_block)

                if actual_journal not in articles_by_journal:
                    articles_by_journal[actual_journal] = []

                articles_by_journal[actual_journal].append(article)

        return articles_by_journal

    def _get_processed_dois(self, journal_name: str) -> Set[str]:
        """Get set of already processed DOIs for a journal."""
        if journal_name not in self.tracking:
            return set()
        return set(self.tracking[journal_name].get('dois', []))

    def _add_articles_to_journal_file(self, journal_name: str, articles: List[Article], digest_date: str):
        """Append new articles to a journal-specific digest file."""
        # Sanitize filename
        safe_name = re.sub(r'[^\w\s-]', '', journal_name).strip().replace(' ', '_')
        journal_file = self.journal_dir / f"{safe_name}.md"

        # Filter out already processed articles
        processed_dois = self._get_processed_dois(journal_name)
        new_articles = [a for a in articles if a.doi not in processed_dois]

        if not new_articles:
            return 0

        # Read existing content or create new file
        if journal_file.exists():
            existing_content = journal_file.read_text(encoding='utf-8')
        else:
            existing_content = f"# {journal_name}\n\n"

        # Build new section
        new_section = [
            f"## {digest_date}",
            "",
        ]

        for article in new_articles:
            new_section.append(article.content)
            new_section.append("")

        # Append to file
        updated_content = existing_content + "\n".join(new_section) + "\n"
        journal_file.write_text(updated_content, encoding='utf-8')

        # Update tracking
        if journal_name not in self.tracking:
            self.tracking[journal_name] = {'dois': [], 'last_processed': None}

        for article in new_articles:
            if article.doi not in self.tracking[journal_name]['dois']:
                self.tracking[journal_name]['dois'].append(article.doi)

        self.tracking[journal_name]['last_processed'] = datetime.now().isoformat()

        return len(new_articles)

    def process_all_digests(self):
        """Process all digest files and consolidate into journal-specific files."""
        if not self.digest_dir.exists():
            print(f"Digest directory not found: {self.digest_dir}")
            return

        digest_files = sorted(self.digest_dir.glob("*.md"))

        if not digest_files:
            print(f"No digest files found in {self.digest_dir}")
            return

        print(f"Found {len(digest_files)} digest file(s)")
        print("-" * 60)

        total_added = 0
        journals_updated = set()

        for digest_file in digest_files:
            print(f"\nProcessing: {digest_file.name}")

            articles_by_journal = self._parse_digest_file(digest_file)

            # Extract digest date from filename or content
            # Assuming format: latest_YYYYMMDD_HHMMSS.md or similar
            date_match = re.search(r'(\d{8}_\d{6})', digest_file.name)
            if date_match:
                date_str = date_match.group(1)
                # Parse to readable format
                try:
                    dt = datetime.strptime(date_str, '%Y%m%d_%H%M%S')
                    digest_date = dt.strftime('%Y-%m-%d %H:%M')
                except:
                    digest_date = date_str
            else:
                digest_date = datetime.now().strftime('%Y-%m-%d %H:%M')

            for journal_name, articles in articles_by_journal.items():
                count = self._add_articles_to_journal_file(journal_name, articles, digest_date)
                if count > 0:
                    print(f"  ✓ {journal_name}: {count} new article(s)")
                    total_added += count
                    journals_updated.add(journal_name)
                else:
                    print(f"  - {journal_name}: 0 new (all already processed)")

        # Save tracking data
        self._save_tracking()

        print("-" * 60)
        print(f"\nSummary:")
        print(f"  Total new articles added: {total_added}")
        print(f"  Journals updated: {len(journals_updated)}")
        print(f"\nJournal digests saved to: {self.journal_dir}")
        print(f"Tracking file: {self.tracking_file}")


def main():
    project_root = Path(__file__).parent.parent
    processor = DigestProcessor(project_root)
    processor.process_all_digests()


if __name__ == "__main__":
    main()
