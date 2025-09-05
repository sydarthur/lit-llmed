import pdfplumber
import os
from pathlib import Path
from typing import Optional, Dict
from llm_interface import OllamaLLM


class PDFProcessor:
    def __init__(self, input_folder: str = "pdfs_input", output_folder: str = "notes_output"):
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.llm = OllamaLLM()
        
        # Create folders if they don't exist
        self.input_folder.mkdir(exist_ok=True)
        self.output_folder.mkdir(exist_ok=True)
    
    def extract_text_from_pdf(self, pdf_path: Path) -> Optional[str]:
        """Extract text from a PDF file"""
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
    
    def process_single_pdf(self, pdf_path: Path) -> Optional[Dict[str, str]]:
        """Process a single PDF file and generate notes"""
        print(f"Processing: {pdf_path.name}")
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            print(f"Could not extract text from {pdf_path.name}")
            return None
        
        print(f"Extracted {len(text)} characters of text")
        
        # Generate notes using LLM
        title = pdf_path.stem  # filename without extension
        notes = self.llm.generate_obsidian_note(text, title)
        
        if notes:
            # Save to output folder
            output_file = self.output_folder / f"{title}.md"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"# {notes['title']}\n\n")
                f.write(f"**Source:** {pdf_path.name}\n")
                f.write(f"**Date Processed:** {str(Path().cwd())}\n\n")
                f.write(notes['content'])
            
            print(f"✅ Notes saved to: {output_file}")
            return notes
        
        return None
    
    def process_all_pdfs(self):
        """Process all PDF files in the input folder"""
        pdf_files = list(self.input_folder.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {self.input_folder}")
            print(f"Please add PDF files to: {self.input_folder.absolute()}")
            return
        
        print(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            try:
                self.process_single_pdf(pdf_file)
                print("-" * 50)
            except Exception as e:
                print(f"Error processing {pdf_file.name}: {e}")
                continue


def main():
    processor = PDFProcessor()
    
    # Check if LLM is connected
    if not processor.llm.test_connection():
        print("❌ Cannot connect to Ollama. Please make sure it's running.")
        return
    
    print(f"📁 Input folder: {processor.input_folder.absolute()}")
    print(f"📁 Output folder: {processor.output_folder.absolute()}")
    print()
    
    processor.process_all_pdfs()


if __name__ == "__main__":
    main()