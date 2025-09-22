#!/usr/bin/env python3
"""
Extract DOIs from literature JSON for easy Zotero import
"""

import json
import sys
from pathlib import Path

def extract_dois_from_json(json_file, output_file=None):
    """Extract DOIs from literature JSON file."""
    with open(json_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    dois = []
    for article in articles:
        if article.get('doi'):
            dois.append(article['doi'])
    
    if output_file:
        with open(output_file, 'w') as f:
            for doi in dois:
                f.write(f"{doi}\n")
        print(f"Saved {len(dois)} DOIs to {output_file}")
    else:
        print("DOIs for Zotero import:")
        print("-" * 40)
        for doi in dois:
            print(doi)
    
    return dois

def clean_abstract_text(abstract):
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

def parse_publication_date(date_str):
    """Parse publication date and return year, month, day components."""
    if not date_str:
        return None, None, None
    
    parts = date_str.split('-')
    year = parts[0] if len(parts) > 0 else None
    month = parts[1] if len(parts) > 1 else None
    day = parts[2] if len(parts) > 2 else None
    
    return year, month, day

def create_ris_format(json_file, output_file):
    """Convert JSON to RIS format for Zotero import with full metadata."""
    with open(json_file, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    ris_content = []
    article_count = 0
    
    for article in articles:
        if not article.get('title') or article.get('title') == "Issue Information":
            continue  # Skip empty entries and issue info
            
        ris_entry = []
        ris_entry.append("TY  - JOUR")  # Journal article
        
        # Title
        if article.get('title'):
            ris_entry.append(f"TI  - {article['title']}")
        
        # Authors
        for author in article.get('authors', []):
            ris_entry.append(f"AU  - {author}")
        
        # Journal name
        if article.get('journal_name'):
            ris_entry.append(f"JO  - {article['journal_name']}")
            ris_entry.append(f"JF  - {article['journal_name']}")  # Full journal name
        
        # Publication date
        year, month, day = parse_publication_date(article.get('published_date'))
        if year:
            ris_entry.append(f"PY  - {year}")
            if month and day:
                ris_entry.append(f"DA  - {year}/{month.zfill(2)}/{day.zfill(2)}")
            elif month:
                ris_entry.append(f"DA  - {year}/{month.zfill(2)}")
        
        # Volume and Issue
        if article.get('volume'):
            ris_entry.append(f"VL  - {article['volume']}")
        
        if article.get('issue'):
            ris_entry.append(f"IS  - {article['issue']}")
        
        # Pages
        if article.get('pages'):
            ris_entry.append(f"SP  - {article['pages']}")
        
        # ISSN
        if article.get('issn'):
            ris_entry.append(f"SN  - {article['issn']}")
        
        # DOI
        if article.get('doi'):
            ris_entry.append(f"DO  - {article['doi']}")
        
        # URLs
        if article.get('url'):
            ris_entry.append(f"UR  - {article['url']}")
        
        # Open access URL as additional link
        if article.get('open_access_url') and article.get('open_access_url') != article.get('url'):
            ris_entry.append(f"L1  - {article['open_access_url']}")
        
        # Abstract
        if article.get('abstract'):
            clean_abstract = clean_abstract_text(article['abstract'])
            if clean_abstract:
                ris_entry.append(f"AB  - {clean_abstract}")
        
        # Keywords
        if article.get('keywords'):
            for keyword in article['keywords']:
                ris_entry.append(f"KW  - {keyword}")
        
        # End of record
        ris_entry.append("ER  - ")
        ris_entry.append("")  # Empty line
        
        ris_content.extend(ris_entry)
        article_count += 1
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(ris_content))
    
    print(f"Created RIS file: {output_file}")
    print(f"Contains {article_count} articles")
    return output_file

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_dois.py <json_file> [--ris]")
        sys.exit(1)
    
    json_file = sys.argv[1]
    
    if "--ris" in sys.argv:
        # Create RIS file
        ris_file = json_file.replace('.json', '.ris')
        create_ris_format(json_file, ris_file)
    else:
        # Extract DOIs
        doi_file = json_file.replace('.json', '_dois.txt')
        extract_dois_from_json(json_file, doi_file)