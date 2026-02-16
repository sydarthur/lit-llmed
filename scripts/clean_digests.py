#!/usr/bin/env python3
"""Clean JATS XML tags from existing markdown digest files."""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path


def clean_jats_tags(text: str) -> str:
    """Remove JATS XML tags and clean up text."""
    # Remove JATS XML tags while preserving content
    cleaned = re.sub(r'</?jats:[^>]+>', '', text)

    # Unescape HTML entities (&amp; -> &, &lt; -> <, etc.)
    cleaned = html.unescape(cleaned)

    # Clean up excessive whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned)

    # Remove leading/trailing whitespace
    cleaned = cleaned.strip()

    return cleaned


def clean_digest_file(file_path: Path) -> bool:
    """Clean a single digest file."""
    try:
        content = file_path.read_text(encoding='utf-8')

        # Check if file has JATS tags
        if '<jats:' not in content:
            print(f"✓ {file_path.name} - already clean")
            return False

        # Pattern to find abstract sections in details blocks
        # Matches content between <summary>Abstract</summary> and </details>
        pattern = r'(<summary><strong>Abstract</strong></summary>\s*\n\s*)(.*?)(\s*</details>)'

        def clean_abstract_match(match):
            prefix = match.group(1)
            abstract_content = match.group(2)
            suffix = match.group(3)

            # Clean the abstract content
            cleaned = clean_jats_tags(abstract_content)

            return f"{prefix}\n{cleaned}\n{suffix}"

        # Replace all abstract sections
        cleaned_content = re.sub(pattern, clean_abstract_match, content, flags=re.DOTALL)

        # Write back
        file_path.write_text(cleaned_content, encoding='utf-8')
        print(f"✓ {file_path.name} - cleaned")
        return True

    except Exception as e:
        print(f"✗ {file_path.name} - error: {e}")
        return False


def main():
    # Find all digest files
    project_root = Path(__file__).parent.parent
    digest_dir = project_root / "output" / "enrich" / "digests"

    if not digest_dir.exists():
        print(f"Digest directory not found: {digest_dir}")
        sys.exit(1)

    # Find all markdown files
    digest_files = list(digest_dir.glob("*.md"))

    if not digest_files:
        print(f"No digest files found in {digest_dir}")
        sys.exit(0)

    print(f"Found {len(digest_files)} digest file(s)")
    print("-" * 60)

    cleaned_count = 0
    for file_path in sorted(digest_files):
        if clean_digest_file(file_path):
            cleaned_count += 1

    print("-" * 60)
    print(f"\nCleaned {cleaned_count} file(s)")


if __name__ == "__main__":
    main()
