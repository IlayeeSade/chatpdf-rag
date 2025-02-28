# ChatPDF RAG

A local Retrieval-Augmented Generation (RAG) application for interactive PDF document queries using embeddings and vector storage.

## Quick Start

```bash
# Setup environment
sudo apt update && sudo apt install build-essential gh
gh auth login
gh repo clone IlayeeSade/chatpdf-rag
cd chatpdf-rag

# Install Ollama and required models
curl -fsSL https://ollama.com/install.sh | sh
ollama pull hf.co/mradermacher/dictalm2.0-instruct-GGUF:Q6_K
ollama pull hf.co/KimChen/bge-m3-GGUF:Q6_K

# Setup Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Launch application
streamlit run app.py --server.port 8051 --server.address 0.0.0.0 &
```

## Core Features

- PDF document upload and processing
- Conversational interface for document queries
- Local vector database for efficient retrieval
- Adjustable retrieval parameters for response quality

## Components

```
.
├── app.py              # Streamlit interface
├── rag.py              # Core RAG implementation
├── requirements.txt    # Dependencies
├── chroma_db/          # Vector database (auto-generated)
└── README.md           # Documentation
```

## Configuration Options

### Models (Customizable)
- LLM: `dictalm2.0-instruct-GGUF:Q6_K`
- Embeddings: `bge-m3-GGUF:Q6_K`
- Any Ollama-compatible model can be used via `llm_model` parameter

### Document Processing (Customizable)
- Chunk size: 1024 tokens
- Chunk overlap: 100 tokens

### Retrieval Settings (User-adjustable)
- `k`: Number of retrieved context chunks
- `score_threshold`: Minimum similarity score for relevance

## Dependencies

- Python 3.8+
- Streamlit, LangChain, Ollama
- PyPDF, ChromaDB
- Required packages in `requirements.txt`:

## Usage Guide

1. **Document Upload**
   - Navigate to the upload section
   - Select PDF file(s)

2. **Query Documents**
   - Type questions in the chat input
   - Adjust retrieval parameters for better results

3. **System Reset**
   - Use "Clear Chat" to reset interface and storage

## Troubleshooting

- **Models missing**: Verify Ollama models are pulled correctly
- **Vector store errors**: Delete `chroma_db/` directory and restart
- **Launch issues**: Check dependencies and Python environment

## License

MIT License

## Acknowledgments

- LangChain, Streamlit, Ollama
