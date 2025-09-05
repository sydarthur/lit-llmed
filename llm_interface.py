import requests
import json
from typing import Dict, Optional


class OllamaLLM:
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "llama2:latest"):
        self.base_url = base_url
        self.model_name = model_name
        self.chat_url = f"{self.base_url}/api/chat"
        self.generate_url = f"{self.base_url}/api/generate"
    
    def _make_request(self, endpoint: str, payload: Dict) -> Optional[str]:
        """Make request to Ollama API and return response text"""
        try:
            response = requests.post(endpoint, json=payload, stream=True)
            response.raise_for_status()
            
            full_response = ""
            for line in response.iter_lines():
                if line:
                    data = json.loads(line.decode('utf-8'))
                    if 'message' in data and 'content' in data['message']:
                        full_response += data['message']['content']
                    elif 'response' in data:
                        full_response += data['response']
                    
                    if data.get('done', False):
                        break
            
            return full_response.strip()
            
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Ollama: {e}")
            return None
        except json.JSONDecodeError as e:
            print(f"Error parsing response: {e}")
            return None
    
    def summarize_text(self, text: str, max_words: int = 300) -> Optional[str]:
        """Generate a concise summary of the given text"""
        prompt = f"""Please provide a concise summary of the following text in approximately {max_words} words. Focus on the main points, key findings, and important conclusions:

{text}

Summary:"""
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": True
        }
        
        return self._make_request(self.chat_url, payload)
    
    def generate_obsidian_note(self, text: str, title: str = "") -> Optional[Dict[str, str]]:
        """Generate structured note content for Obsidian"""
        prompt = f"""Create structured notes from the following text suitable for Obsidian. Include:

1. A clear title
2. Key takeaways (3-5 bullet points)
3. Main topics/themes
4. Important quotes or data (if any)
5. Relevant tags (use #tag format)

Format the output as structured text with clear sections.

Text to process:
{text}

Generate Obsidian note:"""
        
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": True
        }
        
        response = self._make_request(self.chat_url, payload)
        
        if response:
            return {
                "title": title or "Untitled Note",
                "content": response,
                "raw_text": text
            }
        return None
    
    def test_connection(self) -> bool:
        """Test if Ollama is running and model is available"""
        try:
            test_payload = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True
            }
            
            response = self._make_request(self.chat_url, test_payload)
            return response is not None
            
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False


if __name__ == "__main__":
    # Simple test
    llm = OllamaLLM()
    
    if llm.test_connection():
        print("✅ Connected to Ollama successfully!")
        
        # Test summarization
        sample_text = """
        Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or decisions based on those patterns. Common applications include image recognition, natural language processing, and recommendation systems.
        """
        
        summary = llm.summarize_text(sample_text)
        if summary:
            print(f"\n📝 Summary: {summary}")
        else:
            print("❌ Failed to generate summary")
    else:
        print("❌ Failed to connect to Ollama. Make sure it's running.")