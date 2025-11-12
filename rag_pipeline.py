import os
import numpy as np
import faiss
import pickle
from typing import List, Dict, Tuple, Optional
import google.generativeai as genai
from tqdm import tqdm
import json
from datetime import datetime


class GeminiRAGPipeline:
    def __init__(self, api_key: str = None):
        """
        Initialize Gemini RAG Pipeline.
        
        Args:
            api_key (str): Google Gemini API key
        """
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Please provide GEMINI_API_KEY as parameter or environment variable")
        
        # Configure Gemini API
        genai.configure(api_key=self.api_key)
        
        # Initialize models
        self.embedding_model = "models/embedding-001"
        self.generation_model = genai.GenerativeModel('gemini-1.5-flash')
        
        # RAG components
        self.text_chunks = []
        self.embeddings = None
        self.faiss_index = None
        self.chunk_metadata = {}
        
        # File paths
        self.index_file = "gemini_faiss_index.bin"
        self.embeddings_file = "gemini_embeddings.pkl"
        self.metadata_file = "chunk_metadata.json"
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 10) -> np.ndarray:
        """
        Generate embeddings for texts using Gemini Embedding API.
        
        Args:
            texts (List[str]): List of texts to embed
            batch_size (int): Number of texts to process at once
            
        Returns:
            np.ndarray: Array of embeddings
        """
        embeddings = []
        
        print(f"Generating embeddings for {len(texts)} texts...")
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch = texts[i:i + batch_size]
            batch_embeddings = []
            
            for text in batch:
                try:
                    # Generate embedding using Gemini
                    result = genai.embed_content(
                        model=self.embedding_model,
                        content=text,
                        task_type="retrieval_document"
                    )
                    batch_embeddings.append(result['embedding'])
                    
                except Exception as e:
                    print(f"Error generating embedding: {str(e)}")
                    # Use zero vector as fallback
                    batch_embeddings.append([0.0] * 768)  # Assuming 768-dim embeddings
            
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings, dtype=np.float32)
    
    def build_index(self, chunks: List[Dict], force_rebuild: bool = False):
        """
        Build FAISS index from text chunks.
        
        Args:
            chunks (List[Dict]): Processed text chunks
            force_rebuild (bool): Whether to rebuild even if index exists
        """
        # Check if index already exists
        if not force_rebuild and self.load_existing_index():
            print("Loaded existing FAISS index.")
            return
        
        print("Building new FAISS index...")
        
        # Extract texts and prepare metadata
        texts = []
        for i, chunk in enumerate(chunks):
            texts.append(chunk['text'])
            self.chunk_metadata[i] = {
                'source_file': chunk.get('source_file', 'unknown'),
                'chunk_id': chunk.get('chunk_id', i),
                'word_count': chunk.get('word_count', 0),
                'text': chunk['text'][:200] + '...' if len(chunk['text']) > 200 else chunk['text'],
                'full_text': chunk['text']
            }
        
        # Generate embeddings
        self.embeddings = self.generate_embeddings(texts)
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(self.embeddings)
        
        # Save index and metadata
        self.save_index()
        
        print(f"FAISS index built with {len(texts)} chunks")
        print(f"Embedding dimension: {dimension}")
    
    def save_index(self):
        """Save FAISS index, embeddings, and metadata to disk."""
        try:
            # Save FAISS index
            faiss.write_index(self.faiss_index, self.index_file)
            
            # Save embeddings
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.embeddings, f)
            
            # Save metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(self.chunk_metadata, f, indent=2)
            
            print("Index, embeddings, and metadata saved successfully.")
            
        except Exception as e:
            print(f"Error saving index: {str(e)}")
    
    def load_existing_index(self) -> bool:
        """
        Load existing FAISS index, embeddings, and metadata.
        
        Returns:
            bool: True if successfully loaded, False otherwise
        """
        try:
            # Check if all files exist
            if not all(os.path.exists(f) for f in [self.index_file, self.embeddings_file, self.metadata_file]):
                return False
            
            # Load FAISS index
            self.faiss_index = faiss.read_index(self.index_file)
            
            # Load embeddings
            with open(self.embeddings_file, 'rb') as f:
                self.embeddings = pickle.load(f)
            
            # Load metadata
            with open(self.metadata_file, 'r') as f:
                self.chunk_metadata = json.load(f)
            
            return True
            
        except Exception as e:
            print(f"Error loading existing index: {str(e)}")
            return False
    
    def search_similar_chunks(self, query: str, k: int = 5) -> List[Dict]:
        """
        Search for similar chunks using FAISS.
        
        Args:
            query (str): Search query
            k (int): Number of top results to return
            
        Returns:
            List[Dict]: Top k similar chunks with metadata
        """
        if self.faiss_index is None:
            raise ValueError("FAISS index not loaded. Please build or load index first.")
        
        # Generate query embedding
        try:
            query_result = genai.embed_content(
                model=self.embedding_model,
                content=query,
                task_type="retrieval_query"
            )
            query_embedding = np.array([query_result['embedding']], dtype=np.float32)
        except Exception as e:
            print(f"Error generating query embedding: {str(e)}")
            return []
        
        # Search FAISS index
        distances, indices = self.faiss_index.search(query_embedding, k)
        
        # Prepare results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if str(idx) in self.chunk_metadata:
                chunk_info = self.chunk_metadata[str(idx)].copy()
                chunk_info['similarity_score'] = float(1 / (1 + distance))  # Convert distance to similarity
                chunk_info['rank'] = i + 1
                results.append(chunk_info)
        
        return results
    
    def construct_prompt(self, query: str, context_chunks: List[Dict]) -> str:
        """
        Construct a detailed prompt for Gemini LLM.
        
        Args:
            query (str): User's query
            context_chunks (List[Dict]): Retrieved context chunks
            
        Returns:
            str: Constructed prompt
        """
        context_text = ""
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk.get('source_file', 'Unknown')
            text = chunk.get('full_text', chunk.get('text', ''))
            context_text += f"\n--- Context {i} (Source: {source}) ---\n{text}\n"
        
        prompt = f"""You are an expert financial analyst with deep knowledge of financial markets, earnings calls, and investor relations. Your task is to provide accurate, insightful answers based solely on the provided financial transcript context.

CONTEXT FROM FINANCIAL TRANSCRIPTS:
{context_text}

USER QUERY: {query}

INSTRUCTIONS:
1. Answer the query based ONLY on the information provided in the context above
2. If the answer cannot be found in the provided context, clearly state: "I cannot find specific information about this in the provided transcripts."
3. When referencing information, mention which source document it comes from
4. Provide specific details, numbers, and quotes when available
5. If there are multiple perspectives or conflicting information in the context, present them clearly
6. Focus on financial insights, trends, and implications when relevant
7. Keep your response concise but comprehensive

RESPONSE:"""
        
        return prompt
    
    def get_gemini_answer(self, query: str, k: int = 5) -> Dict[str, any]:
        """
        Get answer from Gemini using RAG pipeline.
        
        Args:
            query (str): User's query
            k (int): Number of context chunks to retrieve
            
        Returns:
            Dict[str, any]: Response with answer, context, and metadata
        """
        try:
            # Search for relevant chunks
            context_chunks = self.search_similar_chunks(query, k)
            
            if not context_chunks:
                return {
                    'answer': "I couldn't find relevant information in the financial transcripts for your query.",
                    'context_used': [],
                    'query': query,
                    'timestamp': datetime.now().isoformat(),
                    'error': 'No relevant context found'
                }
            
            # Construct prompt
            prompt = self.construct_prompt(query, context_chunks)
            
            # Generate response using Gemini
            response = self.generation_model.generate_content(prompt)
            
            return {
                'answer': response.text,
                'context_used': context_chunks,
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'num_context_chunks': len(context_chunks)
            }
            
        except Exception as e:
            return {
                'answer': f"I encountered an error while processing your query: {str(e)}",
                'context_used': [],
                'query': query,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def batch_query(self, queries: List[str], k: int = 5) -> List[Dict]:
        """
        Process multiple queries in batch.
        
        Args:
            queries (List[str]): List of queries
            k (int): Number of context chunks per query
            
        Returns:
            List[Dict]: List of responses
        """
        results = []
        
        for query in tqdm(queries, desc="Processing queries"):
            result = self.get_gemini_answer(query, k)
            results.append(result)
        
        return results
    
    def get_index_stats(self) -> Dict[str, any]:
        """
        Get statistics about the current index.
        
        Returns:
            Dict[str, any]: Index statistics
        """
        if self.faiss_index is None:
            return {'error': 'No index loaded'}
        
        stats = {
            'total_chunks': self.faiss_index.ntotal,
            'embedding_dimension': self.faiss_index.d,
            'index_type': type(self.faiss_index).__name__
        }
        
        # Source file distribution
        if self.chunk_metadata:
            sources = {}
            for chunk in self.chunk_metadata.values():
                source = chunk.get('source_file', 'unknown')
                sources[source] = sources.get(source, 0) + 1
            stats['source_distribution'] = sources
        
        return stats


# Utility functions for integration
def setup_rag_pipeline(api_key: str, chunks: List[Dict], force_rebuild: bool = False) -> GeminiRAGPipeline:
    """
    Setup and initialize RAG pipeline with data.
    
    Args:
        api_key (str): Gemini API key
        chunks (List[Dict]): Processed text chunks
        force_rebuild (bool): Whether to rebuild index
        
    Returns:
        GeminiRAGPipeline: Initialized pipeline
    """
    pipeline = GeminiRAGPipeline(api_key)
    pipeline.build_index(chunks, force_rebuild)
    return pipeline


# Example usage
if __name__ == "__main__":
    from data_preprocessor import DataPreprocessor
    import os
    
    # Load API key from environment
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Please set GEMINI_API_KEY environment variable")
        exit(1)
    
    # Load processed chunks
    preprocessor = DataPreprocessor()
    chunks = preprocessor.load_processed_data("processed_transcripts.csv")
    
    if not chunks:
        print("No processed chunks found. Please run data preprocessing first.")
        exit(1)
    
    # Initialize RAG pipeline
    print("Initializing RAG pipeline...")
    rag_pipeline = GeminiRAGPipeline(api_key)
    
    # Build index
    rag_pipeline.build_index(chunks[:50])  # Test with first 50 chunks
    
    # Test queries
    test_queries = [
        "What were the key financial highlights mentioned?",
        "How did revenue perform compared to expectations?",
        "What are the main challenges discussed?",
        "What guidance was provided for future quarters?"
    ]
    
    print("\nTesting RAG pipeline with sample queries...")
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = rag_pipeline.get_gemini_answer(query)
        print(f"Answer: {result['answer'][:300]}...")
        print(f"Used {result.get('num_context_chunks', 0)} context chunks")
    
    # Display index stats
    stats = rag_pipeline.get_index_stats()
    print(f"\nIndex Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
                    