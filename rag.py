from langchain_core.globals import set_verbose, set_debug
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
import logging
import os

set_debug(True)
set_verbose(True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatPDF:
    """A class for handling PDF ingestion and question answering using RAG."""
    
    def __init__(self, llm_model: str = "hf.co/mradermacher/dictalm2.0-instruct-GGUF:Q6_K",
                 embedding_model: str = "hf.co/KimChen/bge-m3-GGUF:Q6_K"):
        """
        Initialize the ChatPDF instance with an LLM and embedding model.
        """
        self.model = ChatOllama(model=llm_model)
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful assistant answering questions based on the uploaded documents you have in your context.
            Context:
            {context}
            
            Question:
            {question}
            
            Answer accurately, comprehensively and relevantly.
            """
        )
        self.vector_store = None
        self.retriever = None
        
    def ingest(self, pdf_file_path: str):
        """
        Ingest a PDF file, split its contents, and store the embeddings in the vector store.
        """
        logger.info(f"Starting ingestion for file: {pdf_file_path}")
        
        if not os.path.exists(pdf_file_path):
            logger.warning(f"PDF file not found: {pdf_file_path}")
            return "rejected"
        
        # Try PyPDFLoader first
        try:
            logger.info("Attempting to load PDF with PyPDFLoader")
            docs = PyPDFLoader(file_path=pdf_file_path).load()
            
            # Check if we got any content
            if not docs or all(not doc.page_content.strip() for doc in docs):
                logger.warning("PyPDFLoader extracted no text, trying UnstructuredPDFLoader")
                raise ValueError("No text extracted")
                
        except Exception as e:
            logger.warning(f"PyPDFLoader failed: {str(e)}")
            # Fall back to UnstructuredPDFLoader which can handle scanned PDFs better
            try:
                logger.info("Attempting to load PDF with UnstructuredPDFLoader")
                docs = UnstructuredPDFLoader(file_path=pdf_file_path).load()
            except Exception as e2:
                logger.warning(f"UnstructuredPDFLoader also failed: {str(e2)}")
                return "rejected"
        
        # Verify we have content
        if not docs or all(not doc.page_content.strip() for doc in docs):
            logger.warning("No text content found in PDF")
            return "rejected"
            
        logger.info(f"Successfully extracted {len(docs)} pages from PDF")
        
        # Get total character count for debugging
        total_chars = sum(len(doc.page_content) for doc in docs)
        logger.info(f"Total characters extracted: {total_chars}")
        
        # Add minimal content if needed
        if total_chars < 10:
            logger.warning("Very little text extracted")
            return "rejected"
            
        # Split documents
        try:
            chunks = self.text_splitter.split_documents(docs)
            if not chunks:
                logger.warning("Text splitter produced no chunks")
                # Try with a smaller chunk size
                logger.info("Retrying with smaller chunk size")
                backup_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
                chunks = backup_splitter.split_documents(docs)
                if not chunks:
                    logger.warning("Text splitting resulted in empty chunks even with smaller chunk size")
                    return "rejected"
        except Exception as e:
            logger.warning(f"Error during text splitting: {str(e)}")
            return "rejected"
            
        # Filter metadata and verify chunks have content
        try:
            chunks = filter_complex_metadata(chunks)
            chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
            
            if not chunks:
                logger.warning("No valid chunks with content after filtering")
                return "rejected"
        except Exception as e:
            logger.warning(f"Error during chunk filtering: {str(e)}")
            return "rejected"
            
        # Log chunk info
        logger.info(f"Created {len(chunks)} chunks from document")
        
        # Test embedding generation with a single chunk
        try:
            test_embedding = self.embeddings.embed_query(chunks[0].page_content)
            if not test_embedding:
                logger.warning("Embedding model returned empty embeddings")
                return "rejected"
            logger.info(f"Test embedding successful, vector length: {len(test_embedding)}")
        except Exception as e:
            logger.warning(f"Error during test embedding: {str(e)}")
            return "rejected"
            
        # Create vector store
        try:
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory="chroma_db",
            )
            logger.info("Ingestion completed. Document embeddings stored successfully.")
            return "success"
        except Exception as e:
            logger.warning(f"Error creating vector store: {str(e)}")
            return "rejected"
        
    def ask(self, query: str, k: int = 5, score_threshold: float = 0.2):
        """
        Answer a query using the RAG pipeline.
        """
        if not self.vector_store:
            return "No document has been successfully ingested yet. Please ingest a document first."
            
        if not self.retriever:
            try:
                self.retriever = self.vector_store.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={"k": k, "score_threshold": score_threshold},
                )
            except Exception as e:
                logger.warning(f"Error creating retriever: {str(e)}")
                return "Unable to create retriever. Please try again."
            
        logger.info(f"Retrieving context for query: {query}")
        try:
            retrieved_docs = self.retriever.invoke(query)
            
            if not retrieved_docs:
                return "No relevant context found in the document to answer your question."
                
            formatted_input = {
                "context": "\n\n".join(doc.page_content for doc in retrieved_docs),
                "question": query,
            }
            
            # Build the RAG chain
            chain = (
                RunnablePassthrough()   # Passes the input as-is
                | self.prompt           # Formats the input for the LLM
                | self.model            # Queries the LLM
                | StrOutputParser()     # Parses the LLM's output
            )
            
            logger.info("Generating response using the LLM.")
            return chain.invoke(formatted_input)
        except Exception as e:
            logger.warning(f"Error during query processing: {str(e)}")
            return "Error processing query. Please try again."
        
    def clear(self):
        """
        Reset the vector store and retriever.
        """
        logger.info("Clearing vector store and retriever.")
        self.vector_store = None
        self.retriever = None
        return "Memory cleared successfully."
