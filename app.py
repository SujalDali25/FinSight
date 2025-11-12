import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
from datetime import datetime
import numpy as np

# Import our custom modules
from data_preprocessor import DataPreprocessor
from sentiment_analyzer import FinBERTAnalyzer
from rag_pipeline import GeminiRAGPipeline, setup_rag_pipeline

# Page configuration
st.set_page_config(
    page_title="Financial Transcript RAG Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .context-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border-left: 3px solid #28a745;
        margin: 1rem 0;
    }
    .query-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None
    if 'processed_chunks' not in st.session_state:
        st.session_state.processed_chunks = []
    if 'sentiment_results' not in st.session_state:
        st.session_state.sentiment_results = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def load_data():
    """Load and process data"""
    preprocessor = DataPreprocessor()
    
    # Try to load existing processed data
    chunks = preprocessor.load_processed_data("processed_transcripts.csv")
    
    if not chunks:
        st.warning("No processed data found. Please process transcripts first.")
        return [], []
    
    # Try to load sentiment results
    analyzer = FinBERTAnalyzer()
    sentiment_results = analyzer.load_sentiment_results("sentiment_analysis.csv")
    
    return chunks, sentiment_results

def setup_sidebar():
    """Setup sidebar with configuration options"""
    st.sidebar.title("⚙️ Configuration")
    
    # API Key input
    api_key = st.sidebar.text_input(
        "Gemini API Key",
        type="password",
        help="Enter your Google Gemini API key"
    )
    
    if api_key:
        os.environ["GEMINI_API_KEY"] = api_key
    
    # Data processing options
    st.sidebar.header("📁 Data Processing")
    
    if st.sidebar.button("🔄 Process Transcripts"):
        with st.spinner("Processing transcripts..."):
            preprocessor = DataPreprocessor("transcripts/")
            transcripts = preprocessor.load_transcripts()
            
            if transcripts:
                chunks = preprocessor.process_all_transcripts(chunk_size=400, overlap=50)
                preprocessor.save_processed_data("processed_transcripts.csv")
                st.session_state.processed_chunks = chunks
                st.sidebar.success(f"Processed {len(chunks)} chunks!")
            else:
                st.sidebar.error("No transcripts found in 'transcripts/' folder")
    
    if st.sidebar.button("💭 Analyze Sentiment"):
        if st.session_state.processed_chunks:
            with st.spinner("Analyzing sentiment with FinBERT..."):
                analyzer = FinBERTAnalyzer()
                results = analyzer.analyze_chunks(st.session_state.processed_chunks)
                analyzer.save_sentiment_results("sentiment_analysis.csv")
                analyzer.save_sentiment_summary("sentiment_summary.json")
                st.session_state.sentiment_results = results
                st.sidebar.success("Sentiment analysis completed!")
        else:
            st.sidebar.error("Please process transcripts first")
    
    # RAG pipeline setup
    st.sidebar.header("🔍 RAG Pipeline")
    
    if st.sidebar.button("🚀 Build RAG Index"):
        if not api_key:
            st.sidebar.error("Please provide Gemini API key")
        elif not st.session_state.processed_chunks:
            st.sidebar.error("Please process transcripts first")
        else:
            with st.spinner("Building RAG index..."):
                try:
                    st.session_state.rag_pipeline = setup_rag_pipeline(
                        api_key, 
                        st.session_state.processed_chunks,
                        force_rebuild=True
                    )
                    st.sidebar.success("RAG pipeline ready!")
                except Exception as e:
                    st.sidebar.error(f"Error building RAG index: {str(e)}")
    
    # Settings
    st.sidebar.header("⚙️ Settings")
    num_results = st.sidebar.slider("Number of context chunks", 1, 10, 5)
    return num_results

def display_sentiment_analysis():
    """Display sentiment analysis results"""
    st.header("💭 Sentiment Analysis Results")
    
    try:
        # Load sentiment summary
        with open("sentiment_summary.json", "r") as f:
            summary = json.load(f)
        
        # Global sentiment overview
        if 'global' in summary:
            global_stats = summary['global']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Chunks",
                    global_stats['total_chunks']
                )
            
            with col2:
                st.metric(
                    "Dominant Sentiment",
                    global_stats['dominant_sentiment'].title()
                )
            
            with col3:
                positive_pct = global_stats['sentiment_percentages']['positive']
                st.metric(
                    "Positive %",
                    f"{positive_pct:.1f}%"
                )
            
            with col4:
                negative_pct = global_stats['sentiment_percentages']['negative']
                st.metric(
                    "Negative %",
                    f"{negative_pct:.1f}%"
                )
        
        # Sentiment distribution chart
        col1, col2 = st.columns(2)
        
        with col1:
            # Pie chart for global sentiment
            if 'global' in summary:
                labels = ['Positive', 'Neutral', 'Negative']
                values = [
                    summary['global']['sentiment_counts']['positive'],
                    summary['global']['sentiment_counts']['neutral'],
                    summary['global']['sentiment_counts']['negative']
                ]
                
                fig_pie = px.pie(
                    values=values,
                    names=labels,
                    title="Overall Sentiment Distribution",
                    color_discrete_map={
                        'Positive': '#2ecc71',
                        'Neutral': '#95a5a6',
                        'Negative': '#e74c3c'
                    }
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Bar chart by file
            file_data = []
            for filename, stats in summary.items():
                if filename != 'global':
                    file_data.append({
                        'File': filename.replace('.txt', ''),
                        'Positive': stats['sentiment_percentages']['positive'],
                        'Neutral': stats['sentiment_percentages']['neutral'],
                        'Negative': stats['sentiment_percentages']['negative']
                    })
            
            if file_data:
                df_files = pd.DataFrame(file_data)
                fig_bar = px.bar(
                    df_files,
                    x='File',
                    y=['Positive', 'Neutral', 'Negative'],
                    title="Sentiment by File",
                    color_discrete_map={
                        'Positive': '#2ecc71',
                        'Neutral': '#95a5a6',
                        'Negative': '#e74c3c'
                    }
                )
                fig_bar.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_bar, use_container_width=True)
        
        # Timeline view
        if st.session_state.sentiment_results:
            analyzer = FinBERTAnalyzer()
            analyzer.sentiment_results = st.session_state.sentiment_results
            timeline_df = analyzer.get_sentiment_timeline()
            
            if not timeline_df.empty:
                st.subheader("📈 Sentiment Timeline")
                
                # Create timeline chart
                fig_timeline = px.scatter(
                    timeline_df,
                    x='chunk_id',
                    y='source_file',
                    color='sentiment_label',
                    size='sentiment_confidence',
                    title="Sentiment Evolution Across Chunks",
                    color_discrete_map={
                        'positive': '#2ecc71',
                        'neutral': '#95a5a6',
                        'negative': '#e74c3c'
                    }
                )
                fig_timeline.update_layout(height=400)
                st.plotly_chart(fig_timeline, use_container_width=True)
    
    except FileNotFoundError:
        st.info("No sentiment analysis results found. Please run sentiment analysis first.")
    except Exception as e:
        st.error(f"Error displaying sentiment analysis: {str(e)}")

def display_rag_interface(num_results):
    """Display RAG Q&A interface"""
    st.header("🤖 Financial Transcript Q&A")
    
    if not st.session_state.rag_pipeline:
        st.warning("Please build the RAG index first using the sidebar.")
        return
    
    # Query input
    query = st.text_area(
        "Ask a question about the financial transcripts:",
        placeholder="e.g., What were the key financial highlights this quarter?",
        height=100
    )
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        ask_button = st.button("🔍 Ask Question", type="primary")
    with col2:
        clear_history = st.button("🗑️ Clear History")
    
    if clear_history:
        st.session_state.chat_history = []
        st.rerun()
    
    if ask_button and query.strip():
        with st.spinner("Searching transcripts and generating answer..."):
            result = st.session_state.rag_pipeline.get_gemini_answer(query, num_results)
            
            # Add to chat history
            st.session_state.chat_history.append({
                'query': query,
                'result': result,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("💬 Chat History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.expander(f"Q: {chat['query'][:100]}... ({chat['timestamp']})", expanded=(i==0)):
                # Query
                st.markdown(f"""
                <div class="query-box">
                    <strong>🙋 Question:</strong> {chat['query']}
                </div>
                """, unsafe_allow_html=True)
                
                # Answer
                st.markdown(f"""
                <div class="context-box">
                    <strong>🤖 Answer:</strong><br>
                    {chat['result']['answer']}
                </div>
                """, unsafe_allow_html=True)
                
                # Context used
                if chat['result'].get('context_used'):
                    st.markdown("**📄 Sources Used:**")
                    for j, context in enumerate(chat['result']['context_used']):
                        with st.expander(f"Source {j+1}: {context['source_file']} (Similarity: {context['similarity_score']:.3f})"):
                            st.text(context['full_text'])

def display_system_stats():
    """Display system statistics"""
    st.header("📊 System Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Data Statistics")
        if st.session_state.processed_chunks:
            chunk_df = pd.DataFrame(st.session_state.processed_chunks)
            
            # File distribution
            file_counts = chunk_df['source_file'].value_counts()
            fig_files = px.bar(
                x=file_counts.index,
                y=file_counts.values,
                title="Chunks per File",
                labels={'x': 'File', 'y': 'Number of Chunks'}
            )
            fig_files.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_files, use_container_width=True)
            
            # Word count distribution
            fig_words = px.histogram(
                chunk_df,
                x='word_count',
                title="Word Count Distribution",
                nbins=30
            )
            st.plotly_chart(fig_words, use_container_width=True)
    
    with col2:
        st.subheader("🔍 RAG Statistics")
        if st.session_state.rag_pipeline:
            stats = st.session_state.rag_pipeline.get_index_stats()
            
            for key, value in stats.items():
                if key != 'source_distribution':
                    st.metric(key.replace('_', ' ').title(), value)
            
            if 'source_distribution' in stats:
                source_df = pd.DataFrame(
                    list(stats['source_distribution'].items()),
                    columns=['Source', 'Chunks']
                )
                fig_sources = px.pie(
                    source_df,
                    values='Chunks',
                    names='Source',
                    title="Index Source Distribution"
                )
                st.plotly_chart(fig_sources, use_container_width=True)

def main():
    """Main Streamlit application"""
    # Initialize session state
    initialize_session_state()
    
    # Main header
    st.markdown('<h1 class="main-header">📊 Financial Transcript RAG Dashboard</h1>', unsafe_allow_html=True)
    
    # Setup sidebar
    num_results = setup_sidebar()
    
    # Load existing data if available
    if not st.session_state.processed_chunks:
        chunks, sentiment_results = load_data()
        st.session_state.processed_chunks = chunks
        st.session_state.sentiment_results = sentiment_results
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🤖 Q&A Interface", "💭 Sentiment Analysis", "📊 Statistics", "📖 About"])
    
    with tab1:
        display_rag_interface(num_results)
    
    with tab2:
        display_sentiment_analysis()
    
    with tab3:
        display_system_stats()
    
    with tab4:
        st.header("📖 About This Dashboard")
        
        st.markdown("""
        This dashboard provides a comprehensive analysis system for financial transcripts using:
        
        **🔧 Technologies Used:**
        - **Gemini API**: For embeddings and text generation
        - **FinBERT**: For domain-specific sentiment analysis
        - **FAISS**: For efficient similarity search
        - **Streamlit**: For interactive dashboard
        
        **📋 Features:**
        - **Text Processing**: Automatic cleaning and chunking of financial transcripts
        - **Sentiment Analysis**: Domain-specific sentiment classification using FinBERT
        - **RAG System**: Retrieval-Augmented Generation for accurate Q&A
        - **Interactive Dashboard**: Real-time querying and visualization
        
        **🚀 Getting Started:**
        1. Add your transcript files (.txt) to the `transcripts/` folder
        2. Enter your Gemini API key in the sidebar
        3. Click "Process Transcripts" to prepare the data
        4. Click "Analyze Sentiment" to run sentiment analysis
        5. Click "Build RAG Index" to create the search index
        6. Start asking questions in the Q&A tab!
        
        **💡 Tips:**
        - Ask specific questions about financial metrics, guidance, or key highlights
        - The system will only answer based on the provided transcripts
        - Check the sentiment analysis for overall market sentiment trends
        """)

if __name__ == "__main__":
    main()