from langchain_core.globals import set_verbose, set_debug
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
import logging
import chromadb
import os
from datetime import datetime
from uuid import uuid4

set_debug(True)
set_verbose(True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatPDF:
    """A class for handling PDF ingestion and question answering using RAG."""
    
    class ChatPDFError(Exception):
        """Base exception class for ChatPDF errors"""
        def __str__(self):
            return "ChatPDF Error"
    
    class PDFNotFoundError(ChatPDFError):
        """Raised when the PDF file is not found"""
        def __str__(self):
            return "PDF file not found"
    
    class TextExtractionError(ChatPDFError):
        """Raised when text cannot be extracted from the PDF"""
        def __str__(self):
            return "Failed to extract text from PDF"
    
    class ChunkingError(ChatPDFError):
        """Raised when there are issues with text chunking"""
        def __str__(self):
            return "Error chunking text"
    
    class EmbeddingError(ChatPDFError):
        """Raised when there are issues with embedding generation"""
        def __str__(self):
            return "Error generating embeddings"
    
    class QueryError(ChatPDFError):
        """Raised when there are issues with query processing"""
        def __str__(self):
            return "Error processing query"
    

    def __init__(self, llm_model: str = "hf.co/lmstudio-community/DeepSeek-R1-Distill-Llama-8B-GGUF:Q6_K",
                 embedding_model: str = "hf.co/KimChen/bge-m3-GGUF:Q6_K",
                 chunk_size: int = 1024,
                 chunk_overlap: int = 100):
        """
        Initialize the ChatPDF instance with an LLM and embedding model.
        Args:
            llm_model (str): The name of the LLM model to use
            embedding_model (str): The name of the embedding model to use
            chunk_size (int): The size of text chunks for splitting documents
            chunk_overlap (int): The overlap between consecutive chunks
        """
        self.model = ChatOllama(model=llm_model)
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful assistant answering questions using your knowledge empowered by the uploaded documents you have in your context.
            Context:
            {context}
            
            Question:
            {question}
            
            Answer accurately, comprehensively and relevantly.
            """
        )
        self.client = chromadb.PersistentClient(path="./my_chroma_db")
        self.collection = self.client.get_or_create_collection(
            name="dove",
        )
        
    def __enter__(self):
        """Context manager entry"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with proper cleanup"""
        self.cleanup()
        
    def cleanup(self):
        """Clean up resources properly"""
        if hasattr(self, 'client'):
            try:
                self.client.close()
                logger.info("ChromaDB client connection closed successfully")
            except Exception as e:
                logger.warning(f"Error closing ChromaDB client: {e}")
        
    def clear(self) -> str:
        """
        Reset the vector store and clear all documents.
        Returns:
            str: Status message
        """
        try:
            # Delete the existing collection
            self.client.delete_collection(name="dove")
            # Create a new collection
            self.collection = self.client.create_collection(name="dove")
            logger.info("Vector store cleared successfully")
            return "Vector store cleared successfully"
        except Exception as e:
            error_msg = f"Error clearing vector store: {str(e)}"
            logger.error(error_msg)
            return error_msg
        
    def _get_adaptive_chunk_size(self, total_chars: int) -> tuple[int, int]:
        """
        Calculate adaptive chunk size based on document length.
        Args:
            total_chars (int): Total number of characters in the document
        Returns:
            tuple[int, int]: Tuple of (chunk_size, chunk_overlap)
        """
        if total_chars < 1000:
            return 256, 25  # Small documents
        elif total_chars < 10000:
            return 512, 50  # Medium documents
        else:
            return self.chunk_size, self.chunk_overlap  # Large documents
            
    def ingest(self, pdf_file_path: str) -> str:
        """
        Ingest a PDF file, split its contents, and store the embeddings in the vector store.
        Returns:
            str: Success message if ingestion is successful
        Raises:
            PDFNotFoundError: If the PDF file is not found
            TextExtractionError: If text cannot be extracted from the PDF
            ChunkingError: If there are issues with text chunking
            EmbeddingError: If there are issues with embedding generation
            ChatPDFError: For other unexpected errors
        """
        logger.info(f"Starting ingestion for file: {pdf_file_path}")
        
        if not os.path.exists(pdf_file_path):
            raise self.PDFNotFoundError(f"PDF file not found at path: {pdf_file_path}")
        
        # Try PyPDFLoader first
        try:
            logger.info("Attempting to load PDF with PyPDFLoader")
            docs = PyPDFLoader(file_path=pdf_file_path).load()
            
            # Check if we got any content
            if not docs or all(not doc.page_content.strip() for doc in docs):
                raise self.TextExtractionError("No text content could be extracted from the PDF")
                
        except Exception as e:
            raise self.TextExtractionError(f"Failed to extract text from PDF: {str(e)}")
            
        logger.info(f"Successfully extracted {len(docs)} pages from PDF")
        
        # Get total character count for debugging
        total_chars = sum(len(doc.page_content) for doc in docs)
        logger.info(f"Total characters extracted: {total_chars}")
        
        # Add minimal content if needed
        if total_chars < 10:
            raise self.TextExtractionError("Very little text extracted from the PDF")
            
        # Split documents with adaptive chunk size
        try:
            adaptive_chunk_size, adaptive_overlap = self._get_adaptive_chunk_size(total_chars)
            logger.info(f"Using adaptive chunk size: {adaptive_chunk_size}, overlap: {adaptive_overlap}")
            
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=adaptive_chunk_size,
                chunk_overlap=adaptive_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            chunks = self.text_splitter.split_documents(docs)
            if not chunks:
                # Try with smaller chunk size as fallback
                fallback_chunk_size = max(256, adaptive_chunk_size // 2)
                fallback_overlap = max(25, adaptive_overlap // 2)
                logger.info(f"Retrying with smaller chunk size: {fallback_chunk_size}")
                
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=fallback_chunk_size,
                    chunk_overlap=fallback_overlap,
                    length_function=len,
                    separators=["\n\n", "\n", " ", ""]
                )
                chunks = self.text_splitter.split_documents(docs)
                
                if not chunks:
                    raise self.ChunkingError("Text splitting resulted in empty chunks even with smaller chunk size")
        except self.ChunkingError:
            raise
        except Exception as e:
            raise self.ChunkingError(f"Error during text splitting: {str(e)}")
            
        # Filter metadata and verify chunks have content
        try:
            chunks = filter_complex_metadata(chunks)
            chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
            
            if not chunks:
                raise self.ChunkingError("No valid chunks with content after filtering")
        except self.ChunkingError:
            raise
        except Exception as e:
            raise self.ChunkingError(f"Error during chunk filtering: {str(e)}")
            
        # Log chunk info
        logger.info(f"Created {len(chunks)} chunks from document")
        
        # Test embedding generation with a single chunk
        try:
            test_embedding = self.embeddings.embed_query(chunks[0].page_content)
            if not test_embedding:
                raise self.EmbeddingError("Embedding model returned empty embeddings")
            logger.info(f"Test embedding successful, vector length: {len(test_embedding)}")
        except self.EmbeddingError:
            raise
        except Exception as e:
            raise self.EmbeddingError(f"Error during test embedding: {str(e)}")
            
        # Create vector store
        try:           
            docs = [doc.page_content for doc in chunks]
            self.collection.add(
                documents=docs,
                embeddings = self.embeddings.embed_documents(docs),
                ids=[str(uuid4()) for _ in docs],
                metadatas=[{"timestamp": datetime.now().timestamp()} for _ in docs],
            )
            logger.info("Ingestion completed. Document embeddings stored successfully.")
            return "Document successfully ingested"
        except Exception as e:
            logger.info(str(e))
            raise self.ChatPDFError(f"Error creating vector store: {str(e)}")
        
    def ask(self, query: str, k: int = 5, score_threshold: float = 0.2) -> str:
        """
        Answer a query using the RAG pipeline.
        Args:
            query (str): The question to ask
            k (int): Number of documents to retrieve
            score_threshold (float): Minimum similarity score threshold
        Returns:
            str: The answer from the LLM
        Raises:
            QueryError: If there are issues processing the query
        """
        formatted_input = {
            "context": "There is no context as of now, use your knowledge and mention the fact you have no context and use your knowledge only",
            "question": query,
        }
        logger.info(f"Retrieving context for query: {query}")
        try:
            results = self.collection.query(
                query_embeddings=[self.embeddings.embed_query(query)],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )

            chain = (
                RunnablePassthrough()
                | self.prompt
                | self.model
                | StrOutputParser()
            )

            logger.info("Generating response using the LLM.")

            if not results or not results['documents'][0]:
                return chain.invoke(formatted_input), None
                
            # Format context with document metadata
            context_parts = []
            for idx, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                context_part = f"Document {idx + 1}:\n"
                if 'timestamp' in metadata:
                    timestamp = datetime.fromtimestamp(metadata['timestamp'])
                    context_part += f"[Added on {timestamp.strftime('%Y-%m-%d %H:%M:%S')}]\n"
                context_part += f"{doc.strip()}\n"
                context_parts.append(context_part)
            
            formatted_input = {
                "context": "\n---\n".join(context_parts),
                "question": query,
            }

            logger.info("Generating response using the LLM.")
            return chain.invoke(formatted_input), context_parts
        except Exception as e:
            logger.error(f"Error during query processing: {str(e)}")
            raise self.QueryError(f"Error processing query: {str(e)}")