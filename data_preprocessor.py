import os
import re
import glob
from typing import List, Dict, Tuple
import pandas as pd


class DataPreprocessor:
    def __init__(self, transcripts_folder: str = "transcripts/"):
        """
        Initialize the DataPreprocessor with the path to transcripts folder.
        
        Args:
            transcripts_folder (str): Path to the folder containing transcript files
        """
        self.transcripts_folder = transcripts_folder
        self.raw_transcripts = {}
        self.cleaned_transcripts = {}
        self.text_chunks = []
        
    def load_transcripts(self) -> Dict[str, str]:
        """
        Load all .txt files from the transcripts folder.
        
        Returns:
            Dict[str, str]: Dictionary with filename as key and content as value
        """
        if not os.path.exists(self.transcripts_folder):
            os.makedirs(self.transcripts_folder)
            print(f"Created {self.transcripts_folder} folder. Please add your transcript files there.")
            return {}
            
        txt_files = glob.glob(os.path.join(self.transcripts_folder, "*.txt"))
        
        if not txt_files:
            print(f"No .txt files found in {self.transcripts_folder}")
            return {}
            
        for file_path in txt_files:
            filename = os.path.basename(file_path)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    self.raw_transcripts[filename] = content
                    print(f"Loaded: {filename}")
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
                
        return self.raw_transcripts
    
    def clean_text(self, text: str) -> str:
        """
        Clean the transcript text by removing speaker tags, disclaimers, and boilerplate.
        
        Args:
            text (str): Raw transcript text
            
        Returns:
            str: Cleaned text
        """
        # Remove speaker tags (e.g., 'Operator:', 'Analyst:', 'CEO:', etc.)
        text = re.sub(r'\b\w+\s*:\s*', '', text)
        
        # Remove common boilerplate phrases
        boilerplate_patterns = [
            r'safe harbor.*?forward-looking statements.*?(?=\n\n|\Z)',
            r'disclaimer.*?(?=\n\n|\Z)',
            r'this transcript.*?(?=\n\n|\Z)',
            r'please note.*?(?=\n\n|\Z)',
            r'©.*?all rights reserved.*?(?=\n\n|\Z)',
            r'copyright.*?(?=\n\n|\Z)',
        ]
        
        for pattern in boilerplate_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove URLs and email addresses
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'\S+@\S+\.\S+', '', text)
        
        # Remove excessive whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n+', '\n', text)
        
        return text.strip()
    
    def standardize_text(self, text: str) -> str:
        """
        Standardize text format: lowercase, clean punctuation, handle whitespace.
        
        Args:
            text (str): Text to standardize
            
        Returns:
            str: Standardized text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Clean up punctuation - keep sentence structure
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\'\"]', ' ', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s*([\.!\?])\s*', r'\1 ', text)
        text = re.sub(r'\s*([,;:])\s*', r'\1 ', text)
        
        # Handle multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict[str, str]]:
        """
        Segment text into smaller chunks for embedding.
        
        Args:
            text (str): Text to chunk
            chunk_size (int): Maximum number of words per chunk
            overlap (int): Number of words to overlap between chunks
            
        Returns:
            List[Dict[str, str]]: List of text chunks with metadata
        """
        words = text.split()
        chunks = []
        
        if len(words) <= chunk_size:
            return [{"text": text, "word_count": len(words)}]
        
        start = 0
        chunk_id = 0
        
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "word_count": len(chunk_words),
                "start_word": start,
                "end_word": end
            })
            
            chunk_id += 1
            start += chunk_size - overlap
            
        return chunks
    
    def process_all_transcripts(self, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
        """
        Process all loaded transcripts through the complete pipeline.
        
        Args:
            chunk_size (int): Maximum words per chunk
            overlap (int): Overlap between chunks
            
        Returns:
            List[Dict]: List of all processed chunks with metadata
        """
        if not self.raw_transcripts:
            print("No transcripts loaded. Please run load_transcripts() first.")
            return []
        
        all_chunks = []
        
        for filename, raw_text in self.raw_transcripts.items():
            print(f"Processing {filename}...")
            
            # Clean and standardize text
            cleaned_text = self.clean_text(raw_text)
            standardized_text = self.standardize_text(cleaned_text)
            
            self.cleaned_transcripts[filename] = standardized_text
            
            # Create chunks
            chunks = self.chunk_text(standardized_text, chunk_size, overlap)
            
            # Add metadata to each chunk
            for chunk in chunks:
                chunk['source_file'] = filename
                chunk['total_chunks'] = len(chunks)
                all_chunks.append(chunk)
                
            print(f"Created {len(chunks)} chunks from {filename}")
        
        self.text_chunks = all_chunks
        print(f"Total chunks created: {len(all_chunks)}")
        
        return all_chunks
    
    def save_processed_data(self, output_file: str = "processed_data.csv"):
        """
        Save processed chunks to CSV file.
        
        Args:
            output_file (str): Output CSV filename
        """
        if not self.text_chunks:
            print("No processed data to save. Run process_all_transcripts() first.")
            return
        
        df = pd.DataFrame(self.text_chunks)
        df.to_csv(output_file, index=False)
        print(f"Processed data saved to {output_file}")
    
    def load_processed_data(self, input_file: str = "processed_data.csv") -> List[Dict]:
        """
        Load previously processed data from CSV.
        
        Args:
            input_file (str): Input CSV filename
            
        Returns:
            List[Dict]: List of processed chunks
        """
        try:
            df = pd.read_csv(input_file)
            self.text_chunks = df.to_dict('records')
            print(f"Loaded {len(self.text_chunks)} chunks from {input_file}")
            return self.text_chunks
        except Exception as e:
            print(f"Error loading processed data: {str(e)}")
            return []


# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = DataPreprocessor("transcripts/")
    
    # Load transcripts
    transcripts = preprocessor.load_transcripts()
    
    if transcripts:
        # Process all transcripts
        chunks = preprocessor.process_all_transcripts(chunk_size=400, overlap=50)
        
        # Save processed data
        preprocessor.save_processed_data("processed_transcripts.csv")
        
        # Display sample results
        if chunks:
            print("\nSample processed chunk:")
            print(f"Source: {chunks[0]['source_file']}")
            print(f"Chunk ID: {chunks[0]['chunk_id']}")
            print(f"Word Count: {chunks[0]['word_count']}")
            print(f"Text: {chunks[0]['text'][:200]}...")
    else:
        print("Please add .txt files to the transcripts/ folder and run again.")