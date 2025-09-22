#!/usr/bin/env python3
"""
Professional RIS file exporter for Zotero integration.
"""

import json
from typing import List, Dict
from pathlib import Path

class RISExporter:
    """Professional RIS file exporter with comprehensive metadata support."""
    
    @staticmethod
    def clean_abstract_text(abstract: str) -> str:
        """Clean JATS markup and other formatting from abstract."""
        if not abstract:
            return ""
        
        # Remove JATS markup
        abstract = abstract.replace('<jats:title>ABSTRACT</jats:title>', '')
        abstract = abstract.replace('<jats:p>', '').replace('</jats:p>', '')
        abstract = abstract.replace('<scp>', '').replace('</scp>', '')
        abstract = abstract.replace('<i>', '').replace('</i>', '')
        abstract = abstract.replace('&amp;', '&')
        abstract = abstract.replace('&lt;', '<').replace('&gt;', '>')
        
        # Remove extra whitespace
        abstract = ' '.join(abstract.split())
        return abstract.strip()
    
    @staticmethod
    def parse_publication_date(date_str: str) -> tuple:
        """Parse publication date and return year, month, day components."""
        if not date_str:
            return None, None, None
        
        parts = date_str.split('-')
        year = parts[0] if len(parts) > 0 else None
        month = parts[1] if len(parts) > 1 else None
        day = parts[2] if len(parts) > 2 else None
        
        return year, month, day
    
    def create_ris_entry(self, article) -> List[str]:
        """Create RIS entry for a single article."""
        # Handle both dict and Article object
        if hasattr(article, 'title'):
            # Article object
            title = article.title
            authors = article.authors or []
            journal_name = article.journal_name
            published_date = article.published_date
            volume = article.volume
            issue = article.issue
            pages = article.pages
            issn = article.issn
            doi = article.doi
            url = article.url
            open_access_url = article.open_access_url
            abstract = article.abstract
            keywords = article.keywords or []
        else:
            # Dictionary
            title = article.get('title')
            authors = article.get('authors', [])
            journal_name = article.get('journal_name')
            published_date = article.get('published_date')
            volume = article.get('volume')
            issue = article.get('issue')
            pages = article.get('pages')
            issn = article.get('issn')
            doi = article.get('doi')
            url = article.get('url')
            open_access_url = article.get('open_access_url')
            abstract = article.get('abstract')
            keywords = article.get('keywords', [])
        
        if not title or title == "Issue Information":
            return []
        
        ris_entry = []
        ris_entry.append("TY  - JOUR")  # Journal article
        
        # Title
        if title:
            ris_entry.append(f"TI  - {title}")
        
        # Authors
        for author in authors:
            ris_entry.append(f"AU  - {author}")
        
        # Journal name
        if journal_name:
            ris_entry.append(f"JO  - {journal_name}")
            ris_entry.append(f"T2  - {journal_name}")  # Secondary title
        
        # Publication date
        year, month, day = self.parse_publication_date(published_date)
        if year:
            ris_entry.append(f"PY  - {year}")
            if month and day:
                ris_entry.append(f"DA  - {year}/{month.zfill(2)}/{day.zfill(2)}")
            elif month:
                ris_entry.append(f"DA  - {year}/{month.zfill(2)}")
        
        # Volume and Issue
        if volume:
            ris_entry.append(f"VL  - {volume}")
        
        if issue:
            ris_entry.append(f"IS  - {issue}")
        
        # Pages
        if pages:
            ris_entry.append(f"SP  - {pages}")
        
        # ISSN
        if issn:
            ris_entry.append(f"SN  - {issn}")
        
        # DOI
        if doi:
            ris_entry.append(f"DO  - {doi}")
        
        # URLs
        if url:
            ris_entry.append(f"UR  - {url}")
        
        # Open access URL as additional link
        if open_access_url and open_access_url != url:
            ris_entry.append(f"L1  - {open_access_url}")
        
        # Abstract
        if abstract:
            clean_abstract = self.clean_abstract_text(abstract)
            if clean_abstract:
                ris_entry.append(f"AB  - {clean_abstract}")
        
        # Keywords
        for keyword in keywords:
            ris_entry.append(f"KW  - {keyword}")
        
        # Additional metadata for Zotero
        ris_entry.append("M1  - Literature Fetcher Import")  # Note field
        
        # End of record
        ris_entry.append("ER  - ")
        
        return ris_entry
    
    def export_to_ris(self, articles: List[Dict], output_file: str) -> str:
        """
        Export articles to RIS format.
        
        Args:
            articles: List of article dictionaries
            output_file: Output file path
            
        Returns:
            Path to created RIS file
        """
        ris_content = []
        article_count = 0
        
        for article in articles:
            entry = self.create_ris_entry(article)
            if entry:
                ris_content.extend(entry)
                ris_content.append("")  # Empty line between entries
                article_count += 1
        
        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(ris_content))
        
        return str(output_path)
    
    def export_from_json(self, json_file: str, output_file: str) -> str:
        """
        Export articles from JSON file to RIS format.
        
        Args:
            json_file: Input JSON file path
            output_file: Output RIS file path
            
        Returns:
            Path to created RIS file
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        
        return self.export_to_ris(articles, output_file)