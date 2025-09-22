import pdfplumber
import re
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from llm_interface import OllamaLLM
from config import get_paper_config


class ChunkedProcessor:
    def __init__(self, base_input_folder: str = "pdfs_input", base_output_folder: str = "notes_output"):
        self.base_input_folder = Path(base_input_folder)
        self.base_output_folder = Path(base_output_folder)
        self.llm = OllamaLLM()
        
        # Chunking parameters
        self.max_chunk_size = 800  # words per chunk
        self.overlap_size = 100    # words to overlap between chunks
        
        # Create base folders
        self.base_input_folder.mkdir(exist_ok=True)
        self.base_output_folder.mkdir(exist_ok=True)
    
    def extract_text_from_pdf(self, pdf_path: Path) -> Optional[str]:
        """Extract text from PDF"""
        try:
            text = ""
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
            return text.strip() if text.strip() else None
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return None
    
    def split_into_chunks(self, text: str) -> List[Dict[str, any]]:
        """Split text into manageable chunks with basic structure detection"""
        
        # First try to split by potential headers (lines that are short and might be titles)
        lines = text.split('\n')
        sections = []
        current_section = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Heuristic: potential header if line is short, not all caps, and doesn't end with period
            is_potential_header = (
                len(line.split()) <= 8 and 
                len(line) < 100 and
                not line.endswith('.') and
                not line.endswith(',') and
                line != line.upper() and
                any(c.isupper() for c in line)
            )
            
            if is_potential_header and current_section and len(' '.join(current_section).split()) > 50:
                # Save current section and start new one
                sections.append({
                    'header': line,
                    'content': ' '.join(current_section).strip()
                })
                current_section = []
            else:
                current_section.append(line)
        
        # Add final section
        if current_section:
            sections.append({
                'header': 'Final Section',
                'content': ' '.join(current_section).strip()
            })
        
        # If no good sections found, fall back to simple paragraph splitting
        if len(sections) <= 1:
            paragraphs = re.split(r'\n\s*\n', text)
            sections = [{'header': f'Section {i+1}', 'content': p.strip()} 
                       for i, p in enumerate(paragraphs) if p.strip()]
        
        # Now chunk sections that are too long
        final_chunks = []
        for i, section in enumerate(sections):
            content = section['content']
            words = content.split()
            
            if len(words) <= self.max_chunk_size:
                final_chunks.append({
                    'chunk_id': len(final_chunks),
                    'header': section['header'],
                    'content': content,
                    'word_count': len(words)
                })
            else:
                # Split large sections into sub-chunks
                sub_chunks = self._split_long_content(content, section['header'])
                final_chunks.extend(sub_chunks)
        
        return final_chunks
    
    def _split_long_content(self, content: str, base_header: str) -> List[Dict[str, any]]:
        """Split content that's too long into smaller chunks with overlap"""
        words = content.split()
        chunks = []
        start_idx = 0
        chunk_num = 1
        
        while start_idx < len(words):
            end_idx = min(start_idx + self.max_chunk_size, len(words))
            chunk_words = words[start_idx:end_idx]
            
            chunks.append({
                'chunk_id': len(chunks),
                'header': f"{base_header} (Part {chunk_num})",
                'content': ' '.join(chunk_words),
                'word_count': len(chunk_words)
            })
            
            # Move start with overlap
            if end_idx >= len(words):
                break
            start_idx = max(start_idx + self.max_chunk_size - self.overlap_size, start_idx + 1)
            chunk_num += 1
        
        return chunks
    
    def summarize_chunk(self, chunk: Dict[str, any], paper_type: str) -> Optional[str]:
        """Summarize a single chunk"""
        prompt = f"""Summarize this section from an academic paper. Be specific and focus on key information.

Section: {chunk['header']}

Content:
{chunk['content']}

Provide a concise summary (3-5 sentences) focusing on:
- Main points or findings
- Key concepts or methods
- Important data or conclusions

Summary:"""
        
        payload = {
            "model": self.llm.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True
        }
        
        return self.llm._make_request(self.llm.chat_url, payload)
    
    def synthesize_summaries(self, chunk_summaries: List[str], paper_type: str, title: str) -> Optional[str]:
        """Combine chunk summaries into final structured note"""
        config = get_paper_config(paper_type)
        
        combined_summaries = "\n\n".join([f"Section {i+1}: {summary}" 
                                        for i, summary in enumerate(chunk_summaries)])
        
        synthesis_prompt = f"""You have summaries from different sections of an academic paper titled "{title}". 
Create a comprehensive structured note by synthesizing these section summaries.

Section Summaries:
{combined_summaries}

Based on these summaries, create structured notes following this format:

## Summary
- Overall summary of the paper (2-3 sentences)

## Key Findings
- Main findings across all sections (bullet points)

## Main Concepts
- Important concepts, theories, or methods mentioned

## Methodology (if mentioned)
- Research methods or approaches used

## Implications
- Practical or theoretical implications
- Future research directions if mentioned

## Tags
- Relevant tags using #tag format

Generate the structured note:"""
        
        payload = {
            "model": self.llm.model_name,
            "messages": [{"role": "user", "content": synthesis_prompt}],
            "stream": True
        }
        
        return self.llm._make_request(self.llm.chat_url, payload)
    
    def process_pdf_chunked(self, pdf_path: Path, paper_type: str, output_folder: Path) -> Optional[Dict]:
        """Process PDF using chunked approach"""
        print(f"Processing [{paper_type}]: {pdf_path.name}")
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            print(f"Could not extract text from {pdf_path.name}")
            return None
        
        print(f"Extracted {len(text.split())} words of text")
        
        # Split into chunks
        chunks = self.split_into_chunks(text)
        print(f"Split into {len(chunks)} chunks")
        
        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"  Summarizing chunk {i+1}/{len(chunks)}: {chunk['header'][:50]}...")
            summary = self.summarize_chunk(chunk, paper_type)
            if summary:
                chunk_summaries.append(summary)
            else:
                print(f"    Failed to summarize chunk {i+1}")
        
        if not chunk_summaries:
            print("No successful chunk summaries generated")
            return None
        
        print(f"Successfully summarized {len(chunk_summaries)}/{len(chunks)} chunks")
        
        # Synthesize final note
        print("Synthesizing final note...")
        title = pdf_path.stem
        final_content = self.synthesize_summaries(chunk_summaries, paper_type, title)
        
        if final_content:
            # Save output
            config = get_paper_config(paper_type)
            formatted_content = config["output_template"].format(
                title=title,
                source=pdf_path.name,
                date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                content=final_content
            )
            
            output_file = output_folder / f"{title}_chunked.md"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(formatted_content)
            
            print(f"✅ Chunked notes saved to: {output_file}")
            
            # Also save chunk details for debugging
            debug_file = output_folder / f"{title}_chunks_debug.txt"
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(f"Processing Summary for {pdf_path.name}\n")
                f.write(f"Total words: {len(text.split())}\n")
                f.write(f"Number of chunks: {len(chunks)}\n\n")
                
                for i, chunk in enumerate(chunks):
                    f.write(f"CHUNK {i+1}: {chunk['header']}\n")
                    f.write(f"Words: {chunk['word_count']}\n")
                    f.write(f"Summary: {chunk_summaries[i] if i < len(chunk_summaries) else 'FAILED'}\n")
                    f.write("-" * 80 + "\n\n")
            
            return {
                "title": title,
                "chunks_processed": len(chunks),
                "successful_summaries": len(chunk_summaries),
                "final_content": final_content
            }
        
        return None
    
    def process_folder_type(self, paper_type: str):
        """Process all PDFs in a folder using chunked approach"""
        input_folder = self.base_input_folder / paper_type
        output_folder = self.base_output_folder / paper_type
        
        input_folder.mkdir(exist_ok=True)
        output_folder.mkdir(exist_ok=True)
        
        pdf_files = list(input_folder.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {input_folder}")
            return
        
        print(f"Found {len(pdf_files)} PDF files in {paper_type} folder")
        
        for pdf_file in pdf_files:
            try:
                result = self.process_pdf_chunked(pdf_file, paper_type, output_folder)
                if result:
                    print(f"Successfully processed with {result['successful_summaries']}/{result['chunks_processed']} chunks")
                print("=" * 60)
            except Exception as e:
                print(f"Error processing {pdf_file.name}: {e}")
                continue


def main():
    processor = ChunkedProcessor()
    
    if not processor.llm.test_connection():
        print("❌ Cannot connect to Ollama. Please make sure it's running.")
        return
    
    print("🔄 Using CHUNKED processing approach")
    print("=" * 60)
    
    # Process editorial folder as test
    processor.process_folder_type("editorial")


if __name__ == "__main__":
    main()