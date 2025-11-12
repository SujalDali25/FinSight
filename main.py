#!/usr/bin/env python3
"""
Financial Transcript RAG Pipeline - Main Execution Script

This script orchestrates the complete pipeline:
1. Data preprocessing
2. Sentiment analysis with FinBERT
3. RAG pipeline setup with Gemini
4. Optional batch processing

Usage:
    python main.py --api-key YOUR_GEMINI_API_KEY
    python main.py --config config.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

from data_preprocessor import DataPreprocessor
from sentiment_analyzer import FinBERTAnalyzer
from rag_pipeline import GeminiRAGPipeline, setup_rag_pipeline


class FinancialRAGOrchestrator:
    def __init__(self, config: Dict):
        """
        Initialize the orchestrator with configuration.
        
        Args:
            config (Dict): Configuration dictionary
        """
        self.config = config
        self.preprocessor = None
        self.sentiment_analyzer = None
        self.rag_pipeline = None
        
        # Paths
        self.transcripts_folder = config.get('transcripts_folder', 'transcripts/')
        self.output_folder = config.get('output_folder', 'outputs/')
        
        # Ensure output folder exists
        Path(self.output_folder).mkdir(exist_ok=True)
    
    def run_preprocessing(self) -> List[Dict]:
        """
        Run the data preprocessing pipeline.
        
        Returns:
            List[Dict]: Processed text chunks
        """
        print("=" * 60)
        print("PHASE 1: DATA PREPROCESSING")
        print("=" * 60)
        
        self.preprocessor = DataPreprocessor(self.transcripts_folder)
        
        # Load transcripts
        transcripts = self.preprocessor.load_transcripts()
        if not transcripts:
            raise ValueError(f"No transcripts found in {self.transcripts_folder}")
        
        # Process all transcripts
        chunk_size = self.config.get('chunk_size', 400)
        overlap = self.config.get('overlap', 50)
        
        chunks = self.preprocessor.process_all_transcripts(chunk_size, overlap)
        
        # Save processed data
        output_file = os.path.join(self.output_folder, 'processed_transcripts.csv')
        self.preprocessor.save_processed_data(output_file)
        
        print(f"✅ Preprocessing completed: {len(chunks)} chunks created")
        return chunks
    
    def run_sentiment_analysis(self, chunks: List[Dict]) -> List[Dict]:
        """
        Run sentiment analysis on processed chunks.
        
        Args:
            chunks (List[Dict]): Processed text chunks
            
        Returns:
            List[Dict]: Chunks with sentiment analysis
        """
        print("\n" + "=" * 60)
        print("PHASE 3: SENTIMENT ANALYSIS")
        print("=" * 60)
        
        self.sentiment_analyzer = FinBERTAnalyzer()
        
        # Load model
        self.sentiment_analyzer.load_model()
        
        # Analyze sentiment
        max_chunks = self.config.get('max_sentiment_chunks', None)
        if max_chunks:
            chunks_to_analyze = chunks[:max_chunks]
            print(f"Analyzing sentiment for first {max_chunks} chunks (for demo)")
        else:
            chunks_to_analyze = chunks
        
        sentiment_results = self.sentiment_analyzer.analyze_chunks(chunks_to_analyze)
        
        # Calculate overall sentiment
        summary = self.sentiment_analyzer.calculate_overall_sentiment(sentiment_results)
        
        # Save results
        results_file = os.path.join(self.output_folder, 'sentiment_results.csv')
        summary_file = os.path.join(self.output_folder, 'sentiment_summary.json')
        
        self.sentiment_analyzer.save_sentiment_results(results_file)
        self.sentiment_analyzer.save_sentiment_summary(summary_file)
        
        # Display summary
        print("\n📊 SENTIMENT SUMMARY:")
        for filename, stats in summary.items():
            if filename == 'global':
                print(f"\n🌍 Overall Statistics:")
                print(f"  Total chunks analyzed: {stats['total_chunks']}")
                print(f"  Dominant sentiment: {stats['dominant_sentiment']}")
                for label, pct in stats['sentiment_percentages'].items():
                    print(f"  {label.capitalize()}: {pct:.1f}%")
        
        print(f"✅ Sentiment analysis completed: {len(sentiment_results)} chunks analyzed")
        return sentiment_results
    
    def run_rag_pipeline(self, chunks: List[Dict]) -> GeminiRAGPipeline:
        """
        Setup and run the RAG pipeline.
        
        Args:
            chunks (List[Dict]): Processed text chunks
            
        Returns:
            GeminiRAGPipeline: Initialized RAG pipeline
        """
        print("\n" + "=" * 60)
        print("PHASE 4: RAG PIPELINE SETUP")
        print("=" * 60)
        
        api_key = self.config.get('gemini_api_key')
        if not api_key:
            raise ValueError("Gemini API key is required")
        
        # Limit chunks for demo if specified
        max_chunks = self.config.get('max_rag_chunks', None)
        if max_chunks:
            chunks_to_index = chunks[:max_chunks]
            print(f"Building RAG index for first {max_chunks} chunks (for demo)")
        else:
            chunks_to_index = chunks
        
        # Setup RAG pipeline
        self.rag_pipeline = setup_rag_pipeline(
            api_key,
            chunks_to_index,
            force_rebuild=self.config.get('force_rebuild_index', False)
        )
        
        # Display index statistics
        stats = self.rag_pipeline.get_index_stats()
        print("\n📈 RAG INDEX STATISTICS:")
        for key, value in stats.items():
            if key != 'source_distribution':
                print(f"  {key.replace('_', ' ').title()}: {value}")
        
        print("✅ RAG pipeline setup completed")
        return self.rag_pipeline
    
    def run_sample_queries(self):
        """Run sample queries to test the RAG pipeline."""
        if not self.rag_pipeline:
            print("❌ RAG pipeline not initialized")
            return
        
        print("\n" + "=" * 60)
        print("TESTING RAG PIPELINE")
        print("=" * 60)
        
        sample_queries = self.config.get('sample_queries', [
            "What were the key financial highlights mentioned?",
            "How did revenue perform this quarter?",
            "What challenges were discussed?",
            "What guidance was provided?"
        ])
        
        results = []
        for i, query in enumerate(sample_queries, 1):
            print(f"\n🔍 Query {i}: {query}")
            result = self.rag_pipeline.get_gemini_answer(query, k=3)
            
            print(f"📝 Answer: {result['answer'][:300]}...")
            print(f"📊 Used {result.get('num_context_chunks', 0)} context chunks")
            
            if 'error' in result:
                print(f"⚠️ Error: {result['error']}")
            
            results.append({
                'query': query,
                'answer': result['answer'],
                'context_chunks': len(result.get('context_used', [])),
                'timestamp': result.get('timestamp')
            })
        
        # Save sample results
        results_file = os.path.join(self.output_folder, 'sample_queries_results.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Sample queries completed. Results saved to {results_file}")
    
    def run_full_pipeline(self):
        """Run the complete pipeline."""
        print("🚀 Starting Financial Transcript RAG Pipeline")
        print("=" * 80)
        
        try:
            # Phase 1: Preprocessing
            chunks = self.run_preprocessing()
            
            # Phase 3: Sentiment Analysis (if enabled)
            if self.config.get('run_sentiment_analysis', True):
                sentiment_results = self.run_sentiment_analysis(chunks)
            
            # Phase 4: RAG Pipeline
            if self.config.get('run_rag_pipeline', True):
                rag_pipeline = self.run_rag_pipeline(chunks)
                
                # Test with sample queries
                if self.config.get('run_sample_queries', True):
                    self.run_sample_queries()
            
            print("\n" + "=" * 80)
            print("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            
            print(f"\n📁 Output files saved in: {self.output_folder}")
            print("📋 Files created:")
            for file in os.listdir(self.output_folder):
                print(f"  - {file}")
            
            print("\n🖥️  To run the Streamlit dashboard:")
            print("  streamlit run app.py")
            
        except Exception as e:
            print(f"\n❌ Pipeline failed with error: {str(e)}")
            raise


def load_config(config_file: str) -> Dict:
    """Load configuration from JSON file."""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Configuration file {config_file} not found")
        return {}
    except json.JSONDecodeError as e:
        print(f"Error parsing configuration file: {str(e)}")
        return {}


def create_default_config(output_file: str = "config.json"):
    """Create a default configuration file."""
    default_config = {
        "transcripts_folder": "transcripts/",
        "output_folder": "outputs/",
        "chunk_size": 400,
        "overlap": 50,
        "gemini_api_key": "",
        "run_sentiment_analysis": True,
        "run_rag_pipeline": True,
        "run_sample_queries": True,
        "max_sentiment_chunks": 100,
        "max_rag_chunks": 200,
        "force_rebuild_index": False,
        "sample_queries": [
            "What were the key financial highlights mentioned?",
            "How did revenue perform compared to expectations?",
            "What are the main challenges or risks discussed?",
            "What guidance was provided for future quarters?",
            "How did the company perform relative to analyst expectations?",
            "What were the key strategic initiatives mentioned?"
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(default_config, f, indent=2)
    
    print(f"✅ Default configuration created: {output_file}")
    print("📝 Please edit the configuration file and add your Gemini API key")


def setup_directory_structure():
    """Setup the required directory structure."""
    directories = ['transcripts', 'outputs']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"📁 Directory ensured: {directory}/")
    
    # Create a sample transcript if transcripts folder is empty
    transcripts_path = Path('transcripts')
    if not any(transcripts_path.glob('*.txt')):
        sample_content = """
Company Q3 2024 Earnings Call

Thank you for joining our third quarter 2024 earnings call. I'm pleased to report strong financial performance this quarter.

Our revenue increased 15% year-over-year to $2.3 billion, exceeding analyst expectations of $2.1 billion. This growth was driven by strong demand in our core product lines and successful expansion into new markets.

Net income for the quarter was $456 million, or $2.15 per share, compared to $389 million, or $1.83 per share, in the same period last year. This represents a 17% increase in earnings per share.

Looking ahead, we are optimistic about Q4 performance. We expect revenue to be in the range of $2.4 to $2.6 billion, with continued margin expansion due to operational efficiencies.

However, we do face some headwinds including supply chain constraints and increased competition in certain segments. We are actively addressing these challenges through strategic partnerships and investment in technology.

Our balance sheet remains strong with $1.2 billion in cash and cash equivalents. We plan to return value to shareholders through both dividends and share repurchases.

Thank you for your continued confidence in our company. We'll now take questions from analysts.
        """.strip()
        
        with open('transcripts/sample_earnings_call.txt', 'w') as f:
            f.write(sample_content)
        
        print("📄 Sample transcript created: transcripts/sample_earnings_call.txt")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Financial Transcript RAG Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --api-key YOUR_API_KEY
  python main.py --config config.json
  python main.py --create-config
  python main.py --setup-only
        """
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
        help='Gemini API key'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.json',
        help='Configuration file path (default: config.json)'
    )
    
    parser.add_argument(
        '--create-config',
        action='store_true',
        help='Create a default configuration file and exit'
    )
    
    parser.add_argument(
        '--setup-only',
        action='store_true',
        help='Only setup directory structure and exit'
    )
    
    parser.add_argument(
        '--skip-sentiment',
        action='store_true',
        help='Skip sentiment analysis phase'
    )
    
    parser.add_argument(
        '--skip-rag',
        action='store_true',
        help='Skip RAG pipeline phase'
    )
    
    args = parser.parse_args()
    
    # Handle special flags
    if args.create_config:
        create_default_config(args.config)
        return
    
    if args.setup_only:
        setup_directory_structure()
        return
    
    # Setup directory structure
    setup_directory_structure()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.api_key:
        config['gemini_api_key'] = args.api_key
    
    if args.skip_sentiment:
        config['run_sentiment_analysis'] = False
    
    if args.skip_rag:
        config['run_rag_pipeline'] = False
    
    # Validate configuration
    if config.get('run_rag_pipeline', True) and not config.get('gemini_api_key'):
        print("❌ Error: Gemini API key is required for RAG pipeline")
        print("   Use --api-key YOUR_API_KEY or add it to the configuration file")
        sys.exit(1)
    
    # Run the pipeline
    try:
        orchestrator = FinancialRAGOrchestrator(config)
        orchestrator.run_full_pipeline()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()