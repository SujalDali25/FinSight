# Financial Transcript RAG Pipeline

A comprehensive Retrieval-Augmented Generation (RAG) system for financial transcript analysis, featuring sentiment analysis with FinBERT and powered by Google's Gemini API.

## 🌟 Features

- **📄 Automated Text Processing**: Clean and chunk financial transcripts automatically
- **💭 Sentiment Analysis**: Domain-specific sentiment classification using FinBERT
- **🔍 RAG System**: Retrieval-Augmented Generation using Gemini for accurate Q&A
- **📊 Interactive Dashboard**: Streamlit-based dashboard for real-time querying and visualization
- **⚡ Vector Search**: Fast similarity search using FAISS indexing
- **📈 Comprehensive Analytics**: Timeline views and statistical analysis of sentiment trends

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project files
# Navigate to the project directory

# Install dependencies
pip install -r requirements.txt

# Set up your Gemini API key
export GEMINI_API_KEY="your_gemini_api_key_here"
```

### 2. Prepare Your Data

Create a `transcripts/` folder and add your financial transcript files (.txt format):

```bash
mkdir transcripts
# Add your .txt transcript files to this folder
```

### 3. Run the Complete Pipeline

```bash
# Option 1: Quick start with API key
python main.py --api-key YOUR_GEMINI_API_KEY

# Option 2: Create and use configuration file
python main.py --create-config
# Edit config.json with your API key and preferences
python main.py --config config.json

# Option 3: Just setup directory structure
python main.py --setup-only
```

### 4. Launch the Dashboard

```bash
streamlit run app.py
```

## 📁 Project Structure

```
financial-rag-pipeline/
├── data_preprocessor.py      # Phase 1: Text preprocessing and chunking
├── sentiment_analyzer.py     # Phase 3: FinBERT sentiment analysis
├── rag_pipeline.py          # Phase 4: Gemini RAG implementation
├── app.py                   # Phase 5: Streamlit dashboard
├── main.py                  # Main orchestration script
├── requirements.txt         # Python dependencies
├── config.json             # Configuration file (auto-generated)
├── transcripts/            # Input folder for transcript files
├── outputs/                # Output folder for processed data
└── README.md              # This file
```

## 🔧 Configuration

The system uses a `config.json` file for configuration:

```json
{
  "transcripts_folder": "transcripts/",
  "output_folder": "outputs/",
  "chunk_size": 400,
  "overlap": 50,
  "gemini_api_key": "your_api_key_here",
  "run_sentiment_analysis": true,
  "run_rag_pipeline": true,
  "run_sample_queries": true,
  "max_sentiment_chunks": 100,
  "max_rag_chunks": 200,
  "force_rebuild_index": false,
  "sample_queries": [
    "What were the key financial highlights mentioned?",
    "How did revenue perform compared to expectations?"
  ]
}
```

## 📋 Detailed Usage

### Phase 1: Data Preprocessing

```python
from data_preprocessor import DataPreprocessor

preprocessor = DataPreprocessor("transcripts/")
transcripts = preprocessor.load_transcripts()
chunks = preprocessor.process_all_transcripts(chunk_size=400, overlap=50)
preprocessor.save_processed_data("processed_transcripts.csv")
```

### Phase 3: Sentiment Analysis

```python
from sentiment_analyzer import FinBERTAnalyzer

analyzer = FinBERTAnalyzer()
analyzer.load_model()
sentiment_results = analyzer.analyze_chunks(chunks)
summary = analyzer.calculate_overall_sentiment()
```

### Phase 4: RAG Pipeline

```python
from rag_pipeline import GeminiRAGPipeline

pipeline = GeminiRAGPipeline(api_key="your_api_key")
pipeline.build_index(chunks)
result = pipeline.get_gemini_answer("What were the key highlights?")
```

### Phase 5: Dashboard Features

The Streamlit dashboard provides:

- **🤖 Q&A Interface**: Ask questions about your financial transcripts
- **💭 Sentiment Analysis**: Visualize sentiment trends and distributions
- **📊 Statistics**: View system performance and data statistics
- **📖 Documentation**: Complete usage guide

## 🎯 Command Line Options

```bash
# Basic usage
python main.py --api-key YOUR_API_KEY

# Advanced options
python main.py --config config.json --skip-sentiment
python main.py --api-key YOUR_API_KEY --skip-rag
python main.py --create-config
python main.py --setup-only

# Help
python main.py --help
```

## 🔑 API Keys and Setup

### Getting a Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Set it as an environment variable or pass it directly:

```bash
export GEMINI_API_KEY="your_api_key_here"
# OR
python main.py --api-key your_api_key_here
```

## 📊 Sample Outputs

### Sentiment Analysis Results

```
📊 SENTIMENT SUMMARY:

🌍 Overall Statistics:
  Total chunks analyzed: 150
  Dominant sentiment: positive
  Positive: 45.3%
  Neutral: 38.7%
  Negative: 16.0%
```

### RAG Query Example

```
🔍 Query: What were the key financial highlights?

📝 Answer: Based on the earnings call transcript, the key financial 
highlights include: 1) Revenue increased 15% year-over-year to $2.3B, 
exceeding analyst expectations of $2.1B, 2) Net income rose to $456M 
or $2.15 per share, up 17% from the previous year...
```

## 🛠️ Troubleshooting

### Common Issues

1. **"No transcripts found"**
   - Ensure .txt files are in the `transcripts/` folder
   - Check file permissions

2. **"Gemini API key error"**
   - Verify your API key is correct
   - Check your API quota and billing

3. **Memory issues with large datasets**
   - Reduce `max_sentiment_chunks` and `max_rag_chunks` in config
   - Process files in smaller batches

4. **FAISS installation issues**
   - Try `pip install faiss-cpu` for CPU-only version
   - Use `faiss-gpu` if you have CUDA support

### Performance Tips

- Use smaller chunk sizes (200-300 words) for better semantic coherence
- Limit the number of chunks for initial testing
- Consider using GPU acceleration for FinBERT if available

## 📈 Output Files

The system generates several output files:

- `processed_transcripts.csv`: Cleaned and chunked text data
- `sentiment_results.csv`: Detailed sentiment analysis results
- `sentiment_summary.json`: Aggregated sentiment statistics
- `sample_queries_results.json`: Results from test queries
- `gemini_faiss_index.bin`: FAISS search index
- `gemini_embeddings.pkl`: Cached embeddings
- `chunk_metadata.json`: Chunk metadata for retrieval

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve the system.

## 📄 License

This project is provided as-is for educational and research purposes.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the console output for specific error messages
3. Ensure all dependencies are properly installed
4. Verify your API keys and quotas

---

**Happy Analyzing! 📊🚀**
