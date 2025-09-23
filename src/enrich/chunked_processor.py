"""Chunked PDF processing for large documents."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pdfplumber

from src.core.log import get_logger
from src.enrich.llm import OllamaLLM
from src.enrich.prompts import get_paper_template
from src.enrich.settings import EnrichmentPaths, resolve_paths

LOGGER = get_logger(__name__)


class ChunkedProcessor:
    """Split PDFs into overlapping chunks and synthesize structured notes."""

    def __init__(
        self,
        *,
        paths: Optional[EnrichmentPaths] = None,
        llm: Optional[OllamaLLM] = None,
        max_chunk_words: int = 800,
        overlap_words: int = 100,
    ) -> None:
        self.paths = paths or resolve_paths()
        self.llm = llm or OllamaLLM()
        self.max_chunk_words = max_chunk_words
        self.overlap_words = overlap_words
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

    def process_pdf(self, pdf_path: Path, paper_type: str) -> Optional[Dict[str, Any]]:
        text = self.extract_text(pdf_path)
        if not text:
            LOGGER.warning("pdf_no_text", file=str(pdf_path))
            return None
        chunks = self._slice_text(text)
        LOGGER.info("chunked_pdf_sliced", file=str(pdf_path), chunks=len(chunks))
        summaries = [summary for summary in (self._summarize_chunk(chunk) for chunk in chunks) if summary]
        if not summaries:
            LOGGER.warning("chunked_no_summaries", file=str(pdf_path))
            return None
        final_note = self._synthesize(pdf_path.stem, paper_type, summaries)
        if not final_note:
            LOGGER.error("chunked_synthesis_failed", file=str(pdf_path))
            return None
        note_path = self._write_note(pdf_path, final_note, paper_type)
        self._write_debug(pdf_path, chunks, summaries, paper_type)
        return {
            "file": str(pdf_path),
            "note_path": str(note_path),
            "chunks": len(chunks),
            "summaries": len(summaries),
        }

    def _slice_text(self, text: str) -> List[Dict[str, Any]]:
        words = text.split()
        chunks = []
        start = 0
        chunk_id = 0
        while start < len(words):
            end = min(start + self.max_chunk_words, len(words))
            chunk_words = words[start:end]
            chunks.append({
                "id": chunk_id,
                "content": " ".join(chunk_words),
            })
            if end == len(words):
                break
            start = max(end - self.overlap_words, start + 1)
            chunk_id += 1
        return chunks

    def _summarize_chunk(self, chunk: Dict[str, Any]) -> Optional[str]:
        prompt = (
            "Summarize this section from an academic paper in 3-5 sentences, highlighting key findings, "
            "concepts, and methods.\n\nSection:\n{content}\n\nSummary:".format(content=chunk["content"])
        )
        return self.llm.chat(prompt)

    def _synthesize(self, title: str, paper_type: str, summaries: List[str]) -> Optional[str]:
        template = get_paper_template(paper_type)
        combined = "\n\n".join(f"Section {i + 1}: {summary}" for i, summary in enumerate(summaries))
        synthesis_prompt = (
            "You have summaries from different sections of an academic paper titled \"{title}\". "
            "Create a comprehensive structured note following this format:\n\n"
            "## Summary\n- Overall summary (2-3 sentences)\n\n"
            "## Key Findings\n- Main findings (bullet list)\n\n"
            "## Main Concepts\n- Important concepts, theories, or methods\n\n"
            "## Methodology\n- Research methods or approaches\n\n"
            "## Implications\n- Practical or theoretical implications\n- Future research directions\n\n"
            "## Tags\n- Relevant #tags\n\nSection summaries:\n{combined}\n\nGenerate the structured note:".format(
                title=title,
                combined=combined,
            )
        )
        response = self.llm.chat(synthesis_prompt)
        if not response:
            return None
        return template["output"].format(
            title=title,
            source=f"{title}.pdf",
            date=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            content=response,
        )

    def _write_note(self, pdf_path: Path, note: str, paper_type: str) -> Path:
        output_dir = self.paths.notes_root / paper_type
        output_dir.mkdir(parents=True, exist_ok=True)
        note_path = output_dir / f"{pdf_path.stem}_chunked.md"
        note_path.write_text(note, encoding="utf-8")
        LOGGER.info("chunked_note_written", path=str(note_path))
        return note_path

    def _write_debug(
        self,
        pdf_path: Path,
        chunks: List[Dict[str, Any]],
        summaries: List[str],
        paper_type: str,
    ) -> None:
        debug_dir = self.paths.debug_root / paper_type
        debug_dir.mkdir(parents=True, exist_ok=True)
        debug_path = debug_dir / f"{pdf_path.stem}_chunks.txt"
        lines = [
            f"Source: {pdf_path.name}",
            f"Total chunks: {len(chunks)}",
            f"Summarized chunks: {len(summaries)}",
            "",
        ]
        for idx, chunk in enumerate(chunks, start=1):
            summary = summaries[idx - 1] if idx - 1 < len(summaries) else "<missing>"
            lines.append(f"CHUNK {idx}")
            lines.append(f"Words: {len(chunk['content'].split())}")
            lines.append(f"Summary: {summary}")
            lines.append("-" * 40)
        debug_path.write_text("\n".join(lines), encoding="utf-8")
        LOGGER.info("chunked_debug_written", path=str(debug_path))

    def process_folder(self, paper_type: str) -> None:
        input_dir = self.paths.input_root / paper_type
        input_dir.mkdir(parents=True, exist_ok=True)
        pdfs = sorted(input_dir.glob("*.pdf"))
        if not pdfs:
            LOGGER.info("chunked_no_pdfs", folder=str(input_dir))
            return
        for pdf_path in pdfs:
            self.process_pdf(pdf_path, paper_type)

    def process_all(self) -> None:
        for paper_type in ("editorial", "theory", "method", "topic"):
            self.process_folder(paper_type)
