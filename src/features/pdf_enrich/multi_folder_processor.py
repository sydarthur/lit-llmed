"""Process PDFs organized by paper type using dedicated prompt templates."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

import pdfplumber

from src.core.log import get_logger
from src.features.pdf_enrich.llm import OllamaLLM
from src.features.pdf_enrich.prompts import get_paper_template
from src.features.pdf_enrich.settings import EnrichmentPaths, resolve_paths

LOGGER = get_logger(__name__)

PAPER_TYPES: Iterable[str] = ("editorial", "theory", "method", "topic")


class MultiFolderProcessor:
    """Apply type-specific prompts to PDFs stored in dedicated folders."""

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
        except Exception as exc:  # pragma: no cover
            LOGGER.error("pdf_extract_failed", file=str(pdf_path), error=str(exc))
            return None
        content = "\n\n".join(text).strip()
        return content or None

    def process_pdf(self, pdf_path: Path, paper_type: str, output_dir: Path) -> Optional[Dict[str, str]]:
        template = get_paper_template(paper_type)
        text = self.extract_text(pdf_path)
        if not text:
            LOGGER.warning("pdf_no_text", file=str(pdf_path))
            return None
        prompt = template["prompt"].format(text=text)
        response = self.llm.chat(prompt)
        if not response:
            LOGGER.error("llm_failed", file=str(pdf_path), paper_type=paper_type)
            return None
        note = template["output"].format(
            title=pdf_path.stem,
            source=pdf_path.name,
            date=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            content=response,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        note_path = output_dir / f"{pdf_path.stem}.md"
        note_path.write_text(note, encoding="utf-8")
        LOGGER.info("note_written", path=str(note_path))
        return {"path": str(note_path), "content": note}

    def process_folder(self, paper_type: str) -> None:
        input_dir = self.paths.input_root / paper_type
        output_dir = self.paths.notes_root / paper_type
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        pdfs = sorted(input_dir.glob("*.pdf"))
        if not pdfs:
            LOGGER.info("no_pdfs_found", folder=str(input_dir))
            return
        LOGGER.info("processing_folder_start", folder=str(input_dir), count=len(pdfs))
        for pdf_path in pdfs:
            self.process_pdf(pdf_path, paper_type, output_dir)
        LOGGER.info("processing_folder_complete", folder=str(input_dir))

    def process_all(self) -> None:
        for paper_type in PAPER_TYPES:
            self.process_folder(paper_type)
