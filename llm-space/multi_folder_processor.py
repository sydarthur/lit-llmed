import pdfplumber
import os
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime
from llm_interface import OllamaLLM
from config import get_paper_config


class MultiFolderProcessor:
    def __init__(self, base_input_folder: str = "pdfs_input", base_output_folder: str = "notes_output"):
        self.base_input_folder = Path(base_input_folder)
        self.base_output_folder = Path(base_output_folder)
        self.llm = OllamaLLM()
        
        # Create base folders if they don't exist
        self.base_input_folder.mkdir(exist_ok=True)
        self.base_output_folder.mkdir(exist_ok=True)
    
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
    
    def process_pdf_with_config(self, pdf_path: Path, paper_type: str, output_folder: Path) -> Optional[Dict[str, str]]:
        """Process a PDF with specific configuration for the paper type"""
        print(f"Processing [{paper_type}]: {pdf_path.name}")
        
        # Extract text
        text = self.extract_text_from_pdf(pdf_path)
        if not text:
            print(f"Could not extract text from {pdf_path.name}")
            return None
        
        print(f"Extracted {len(text)} characters of text")
        
        # Get configuration for this paper type
        config = get_paper_config(paper_type)
        
        # Create custom prompt
        prompt = config["prompt_template"].format(text=text)
        
        # Generate notes using LLM with custom prompt
        title = pdf_path.stem
        
        # Use the custom prompt directly
        payload = {
            "model": self.llm.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True
        }
        
        response = self.llm._make_request(self.llm.chat_url, payload)
        
        if response:
            # Format output using template
            formatted_content = config["output_template"].format(
                title=title,
                source=pdf_path.name,
                date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                content=response
            )
            
            # Save to appropriate output folder
            output_file = output_folder / f"{title}.md"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(formatted_content)
            
            print(f"✅ Notes saved to: {output_file}")
            return {
                "title": title,
                "content": response,
                "raw_text": text
            }
        
        return None
    
    def process_folder_type(self, paper_type: str):
        """Process all PDFs in a specific paper type folder"""
        input_folder = self.base_input_folder / paper_type
        output_folder = self.base_output_folder / paper_type
        
        # Create folders if they don't exist
        input_folder.mkdir(exist_ok=True)
        output_folder.mkdir(exist_ok=True)
        
        pdf_files = list(input_folder.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {input_folder}")
            return
        
        print(f"Found {len(pdf_files)} PDF files in {paper_type} folder")
        
        for pdf_file in pdf_files:
            try:
                self.process_pdf_with_config(pdf_file, paper_type, output_folder)
                print("-" * 50)
            except Exception as e:
                print(f"Error processing {pdf_file.name}: {e}")
                continue
    
    def process_all_folders(self):
        """Process all paper type folders"""
        paper_types = ["editorial", "theory", "method", "topic"]
        
        for paper_type in paper_types:
            print(f"\n🔍 Processing {paper_type.upper()} papers...")
            self.process_folder_type(paper_type)
    
    def show_folder_structure(self):
        """Display the current folder structure"""
        print("📁 Current folder structure:")
        print(f"Input: {self.base_input_folder.absolute()}")
        for paper_type in ["editorial", "theory", "method", "topic"]:
            input_folder = self.base_input_folder / paper_type
            pdf_count = len(list(input_folder.glob("*.pdf"))) if input_folder.exists() else 0
            print(f"  ├── {paper_type}/ ({pdf_count} PDFs)")
        
        print(f"\nOutput: {self.base_output_folder.absolute()}")
        for paper_type in ["editorial", "theory", "method", "topic"]:
            output_folder = self.base_output_folder / paper_type
            note_count = len(list(output_folder.glob("*.md"))) if output_folder.exists() else 0
            print(f"  ├── {paper_type}/ ({note_count} notes)")


def main():
    processor = MultiFolderProcessor()
    
    # Check if LLM is connected
    if not processor.llm.test_connection():
        print("❌ Cannot connect to Ollama. Please make sure it's running.")
        return
    
    processor.show_folder_structure()
    print()
    
    processor.process_all_folders()


if __name__ == "__main__":
    main()