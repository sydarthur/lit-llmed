"""Simple one-shot PDF processor that generates Obsidian notes."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pdfplumber

from src.core.log import get_logger
from src.enrich.llm import OllamaLLM
from src.enrich.settings import EnrichmentPaths, resolve_paths

LOGGER = get_logger(__name__)


class PDFProcessor:
    """Extract text from PDFs and push them through the LLM summarizer."""

    def __init__(
        self,
        *,
        paths: Optional[EnrichmentPaths] = None,
        llm: Optional[OllamaLLM] = None,
    ) -> None:
        self.paths = paths or resolve_paths()
        self.llm = llm or OllamaLLM()
        self.paths.ensure()

    def extract_text(self, pdf_path: Path) -> Optional[str]:
        try:
            text = []
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
        except Exception as exc:  # pragma: no cover - pdfplumber I/O errors
            LOGGER.error("pdf_extract_failed", file=str(pdf_path), error=str(exc))
            return None
        content = "\n\n".join(text).strip()
        return content or None

    def process_pdf(self, pdf_path: Path) -> Optional[Dict[str, str]]:
        LOGGER.info("process_pdf_start", path=str(pdf_path))
        text = self.extract_text(pdf_path)
        if not text:
            LOGGER.warning("process_pdf_no_text", path=str(pdf_path))
            return None
        llm_response = self.llm.chat(
            """Create structured notes from the following text suitable for Obsidian. Include:
1. A clear title
2. Key takeaways (3-5 bullet points)
3. Main topics or themes
4. Important quotes or data
5. Relevant tags using #tag format

Text to process:
{body}

Generate Obsidian note:
""".format(body=text)
        )
        if not llm_response:
            LOGGER.error("process_pdf_llm_failed", path=str(pdf_path))
            return None
        note = {
            "title": pdf_path.stem,
            "content": llm_response,
            "source": pdf_path.name,
            "generated_at": datetime.utcnow().isoformat(),
        }
        self._persist_note(note)
        LOGGER.info("process_pdf_complete", path=str(pdf_path))
        return note

    def _persist_note(self, note: Dict[str, str]) -> Path:
        note_path = self.paths.notes_root / f"{note['title']}.md"
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text(
            "# {title}\n\n**Source:** {source}\n**Date Processed:** {timestamp}\n\n{body}".format(
                title=note["title"],
                source=note["source"],
                timestamp=note["generated_at"],
                body=note["content"],
            ),
            encoding="utf-8",
        )
        return note_path

    def process_all(self) -> None:
        pdf_files = list(self.paths.input_root.glob("*.pdf"))
        if not pdf_files:
            LOGGER.info("process_all_no_files", directory=str(self.paths.input_root))
            return
        for pdf_file in pdf_files:
            self.process_pdf(pdf_file)
