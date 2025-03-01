# app.py
import streamlit as st
# Set page config must be the first Streamlit command
st.set_page_config(page_title="IlayeeRAG")

import os
import tempfile
import time
from streamlit_chat import message
from rag import ChatPDF
from reference import create_download_link


def display_messages():
    """Display the chat history."""
    st.subheader("Chat History")
    for msg, is_user in st.session_state["messages"]:
        with st.chat_message("user" if is_user else "assistant"):
            st.markdown(msg, unsafe_allow_html=True)
    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    """Process the user input and generate an assistant response."""
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner("חושב..."):
            try:
                agent_text, context_parts = st.session_state["assistant"].ask(
                    user_text,
                    k=st.session_state["retrieval_k"],
                    score_threshold=st.session_state["retrieval_threshold"],
                )
                
                # If context_parts exists and contains items, create PDF links
                if context_parts and len(context_parts) > 0:
                    # Store the original agent text
                    original_agent_text = agent_text
                    
                    # Add PDF links section
                    agent_text += "\n\n**Source Documents:**\n"
                    
                    # Add links for each context part
                    for i, context in enumerate(context_parts):
                        # Create short preview (first 50 chars)
                        preview = context[:50] + "..." if len(context) > 50 else context
                        
                        # Create PDF link
                        link = create_download_link(context, f"source_document_{i+1}")
                        
                        # Add to agent text
                        agent_text += f"{i+1}. {preview} {link}\n"
                
            except ChatPDF.QueryError as e:
                agent_text = f"Sorry, I encountered an error processing your query: {str(e)}"
            except Exception as e:
                agent_text = f"An unexpected error occurred: {str(e)}"

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))

def handle_file_upload():
    """Handle file upload and ingestion."""
    if "assistant" not in st.session_state:
        st.session_state["assistant"] = ChatPDF()
        st.session_state["messages"] = []
    
    uploaded_files = st.file_uploader(
        "העלה מסמך",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.session_state["assistant"].clear()  # Clear previous documents
        st.session_state["messages"] = []
        st.session_state["user_input"] = ""
        
        for file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False) as tf:
                tf.write(file.getbuffer())
                file_path = tf.name

            with st.spinner(f"מעכל {file.name}..."):
                t0 = time.time()
                try:
                    st.session_state["assistant"].ingest(file_path)
                    t1 = time.time()
                    st.session_state["messages"].append(
                        (f"Successfully ingested {file.name} in {t1 - t0:.2f} seconds", False)
                    )
                except Exception as e:
                    if isinstance(e, ChatPDF.PDFNotFoundError):
                        st.error(f"Could not find the file: {file.name}")
                    elif isinstance(e, ChatPDF.TextExtractionError):
                        st.error(f"Could not extract text from {file.name}: {str(e)}")
                    elif isinstance(e, ChatPDF.ChunkingError):
                        st.error(f"Error processing document chunks in {file.name}: {str(e)}")
                    elif isinstance(e, ChatPDF.EmbeddingError):
                        st.error(f"Error generating embeddings for {file.name}: {str(e)}")
                    elif isinstance(e, ChatPDF.ChatPDFError):
                        st.error(f"Error processing {file.name}: {str(e)}")
                    else:
                        st.error(f"Unexpected error processing {file.name}: {str(e)}")
                finally:
                    os.remove(file_path)

def page():
    """Main app page layout."""
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        st.session_state["assistant"] = ChatPDF()

    st.header("IlayeeRAG")

    st.subheader("העלה מסמך")
    handle_file_upload()

    st.session_state["ingestion_spinner"] = st.empty()

    # Retrieval settings
    st.subheader("Settings")
    st.session_state["retrieval_k"] = st.slider(
        "Number of Retrieved Results (k)", min_value=1, max_value=10, value=5
    )
    st.session_state["retrieval_threshold"] = st.slider(
        "Similarity Score Threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.05
    )

    # Display messages and text input
    display_messages()
    st.text_input("Message", key="user_input", on_change=process_input)

    # Clear chat
    if st.button("Clear Chat"):
        st.session_state["messages"] = []
        if "assistant" in st.session_state:
            st.session_state["assistant"].cleanup()


if __name__ == "__main__":
    page()
