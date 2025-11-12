import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from tqdm import tqdm
import json


class FinBERTAnalyzer:
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Initialize FinBERT sentiment analyzer.
        
        Args:
            model_name (str): HuggingFace model name for FinBERT
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.classifier = None
        self.sentiment_results = []
        
    def load_model(self):
        """Load the FinBERT model and tokenizer."""
        try:
            print("Loading FinBERT model...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Create pipeline for easier inference
            self.classifier = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            print("FinBERT model loaded successfully!")
            print(f"Using device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
            
        except Exception as e:
            print(f"Error loading FinBERT model: {str(e)}")
            raise
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text chunk.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, float]: Sentiment scores and label
        """
        if not self.classifier:
            raise ValueError("Model not loaded. Please run load_model() first.")
        
        try:
            # Truncate text if too long for the model
            max_length = self.tokenizer.model_max_length - 2  # Account for special tokens
            tokens = self.tokenizer.encode(text, truncation=True, max_length=max_length)
            truncated_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            
            # Get prediction
            result = self.classifier(truncated_text)[0]
            
            # Map labels to standardized format
            label_mapping = {
                'positive': 'positive',
                'negative': 'negative',
                'neutral': 'neutral',
                'POSITIVE': 'positive',
                'NEGATIVE': 'negative',
                'NEUTRAL': 'neutral'
            }
            
            sentiment_label = label_mapping.get(result['label'].lower(), result['label'].lower())
            
            return {
                'label': sentiment_label,
                'score': result['score'],
                'confidence': result['score']
            }
            
        except Exception as e:
            print(f"Error analyzing sentiment for text: {str(e)}")
            return {
                'label': 'neutral',
                'score': 0.0,
                'confidence': 0.0
            }
    
    def analyze_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """
        Analyze sentiment for all text chunks.
        
        Args:
            chunks (List[Dict]): List of text chunks from preprocessor
            
        Returns:
            List[Dict]: Chunks with added sentiment analysis
        """
        if not self.classifier:
            self.load_model()
        
        results = []
        
        print(f"Analyzing sentiment for {len(chunks)} chunks...")
        
        for chunk in tqdm(chunks, desc="Processing chunks"):
            sentiment = self.analyze_sentiment(chunk['text'])
            
            # Add sentiment data to chunk
            enhanced_chunk = chunk.copy()
            enhanced_chunk.update({
                'sentiment_label': sentiment['label'],
                'sentiment_score': sentiment['score'],
                'sentiment_confidence': sentiment['confidence']
            })
            
            results.append(enhanced_chunk)
        
        self.sentiment_results = results
        print("Sentiment analysis completed!")
        
        return results
    
    def calculate_overall_sentiment(self, chunks: List[Dict] = None) -> Dict[str, Dict]:
        """
        Calculate overall sentiment statistics for each transcript and globally.
        
        Args:
            chunks (List[Dict]): Optional chunks list, uses self.sentiment_results if None
            
        Returns:
            Dict[str, Dict]: Overall sentiment statistics
        """
        if chunks is None:
            chunks = self.sentiment_results
            
        if not chunks:
            print("No sentiment results available. Run analyze_chunks() first.")
            return {}
        
        # Group by source file
        file_groups = {}
        for chunk in chunks:
            source_file = chunk.get('source_file', 'unknown')
            if source_file not in file_groups:
                file_groups[source_file] = []
            file_groups[source_file].append(chunk)
        
        results = {}
        
        # Calculate statistics for each file
        for filename, file_chunks in file_groups.items():
            sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
            total_chunks = len(file_chunks)
            weighted_score = 0
            
            for chunk in file_chunks:
                label = chunk.get('sentiment_label', 'neutral')
                score = chunk.get('sentiment_score', 0.0)
                confidence = chunk.get('sentiment_confidence', 0.0)
                
                sentiment_counts[label] += 1
                
                # Calculate weighted score (positive=1, neutral=0, negative=-1)
                if label == 'positive':
                    weighted_score += score * confidence
                elif label == 'negative':
                    weighted_score -= score * confidence
            
            # Calculate percentages
            sentiment_percentages = {
                label: (count / total_chunks) * 100 
                for label, count in sentiment_counts.items()
            }
            
            results[filename] = {
                'total_chunks': total_chunks,
                'sentiment_counts': sentiment_counts,
                'sentiment_percentages': sentiment_percentages,
                'weighted_sentiment_score': weighted_score / total_chunks,
                'dominant_sentiment': max(sentiment_counts, key=sentiment_counts.get)
            }
        
        # Calculate global statistics
        all_labels = [chunk.get('sentiment_label', 'neutral') for chunk in chunks]
        global_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for label in all_labels:
            global_counts[label] += 1
        
        total_global = len(chunks)
        global_percentages = {
            label: (count / total_global) * 100 
            for label, count in global_counts.items()
        }
        
        results['global'] = {
            'total_chunks': total_global,
            'sentiment_counts': global_counts,
            'sentiment_percentages': global_percentages,
            'dominant_sentiment': max(global_counts, key=global_counts.get)
        }
        
        return results
    
    def save_sentiment_results(self, filename: str = "sentiment_results.csv"):
        """
        Save sentiment analysis results to CSV.
        
        Args:
            filename (str): Output filename
        """
        if not self.sentiment_results:
            print("No sentiment results to save.")
            return
        
        df = pd.DataFrame(self.sentiment_results)
        df.to_csv(filename, index=False)
        print(f"Sentiment results saved to {filename}")
    
    def save_sentiment_summary(self, filename: str = "sentiment_summary.json"):
        """
        Save sentiment summary statistics to JSON.
        
        Args:
            filename (str): Output filename
        """
        summary = self.calculate_overall_sentiment()
        
        if summary:
            with open(filename, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"Sentiment summary saved to {filename}")
    
    def load_sentiment_results(self, filename: str = "sentiment_results.csv") -> List[Dict]:
        """
        Load previously calculated sentiment results.
        
        Args:
            filename (str): Input filename
            
        Returns:
            List[Dict]: Loaded sentiment results
        """
        try:
            df = pd.read_csv(filename)
            self.sentiment_results = df.to_dict('records')
            print(f"Loaded {len(self.sentiment_results)} sentiment results from {filename}")
            return self.sentiment_results
        except Exception as e:
            print(f"Error loading sentiment results: {str(e)}")
            return []
    
    def get_sentiment_timeline(self, chunks: List[Dict] = None) -> pd.DataFrame:
        """
        Create a timeline view of sentiment by file and chunk order.
        
        Args:
            chunks (List[Dict]): Optional chunks list
            
        Returns:
            pd.DataFrame: Timeline data for visualization
        """
        if chunks is None:
            chunks = self.sentiment_results
            
        if not chunks:
            return pd.DataFrame()
        
        timeline_data = []
        
        for chunk in chunks:
            timeline_data.append({
                'source_file': chunk.get('source_file', 'unknown'),
                'chunk_id': chunk.get('chunk_id', 0),
                'sentiment_label': chunk.get('sentiment_label', 'neutral'),
                'sentiment_score': chunk.get('sentiment_score', 0.0),
                'sentiment_confidence': chunk.get('sentiment_confidence', 0.0),
                'word_count': chunk.get('word_count', 0)
            })
        
        return pd.DataFrame(timeline_data)


# Example usage
if __name__ == "__main__":
    from data_preprocessor import DataPreprocessor
    
    # Load processed data
    preprocessor = DataPreprocessor()
    chunks = preprocessor.load_processed_data("processed_transcripts.csv")
    
    if chunks:
        # Initialize sentiment analyzer
        analyzer = FinBERTAnalyzer()
        
        # Analyze sentiment
        sentiment_results = analyzer.analyze_chunks(chunks[:10])  # Test with first 10 chunks
        
        # Calculate overall sentiment
        summary = analyzer.calculate_overall_sentiment()
        print("\nSentiment Summary:")
        for filename, stats in summary.items():
            if filename != 'global':
                print(f"\n{filename}:")
                print(f"  Total chunks: {stats['total_chunks']}")
                print(f"  Dominant sentiment: {stats['dominant_sentiment']}")
                print(f"  Sentiment distribution:")
                for label, percentage in stats['sentiment_percentages'].items():
                    print(f"    {label}: {percentage:.1f}%")
        
        # Save results
        analyzer.save_sentiment_results("sentiment_analysis.csv")
        analyzer.save_sentiment_summary("sentiment_summary.json")
    else:
        print("Please run data preprocessing first to generate chunks.")