# app.py
import streamlit as st
# Set page config must be the first Streamlit command
st.set_page_config(page_title="IlayeeRAG")

import os
import tempfile
import time
import base64
from streamlit_chat import message
from rag import ChatPDF
from reference import create_download_link
import streamlit.components.v1 as components

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "assistant" not in st.session_state:
    st.session_state["assistant"] = ChatPDF()
if "file_processed" not in st.session_state:
    st.session_state["file_processed"] = False

st.header("IlayeeRAG")

def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        st.session_state["messages"].append((user_text, True))
        st.session_state["user_input"] = ""
        
        with st.spinner("חושב..."):
            try:
                agent_text, context_parts = st.session_state["assistant"].ask(
                    user_text,
                    k=st.session_state["retrieval_k"],
                    score_threshold=st.session_state["retrieval_threshold"],
                )
                
                if context_parts and len(context_parts) > 0:
                    agent_text += "\n\n**Source Documents:**\n"
                    for i, context in enumerate(context_parts):
                        preview = context[:50] + "..." if len(context) > 50 else context
                        link = create_download_link(context, f"source_document_{i+1}")
                        agent_text += f"{i+1}. {preview} {link}\n"
                
            except Exception as e:
                agent_text = f"Error: {str(e)}"

        st.session_state["messages"].append((agent_text, False))

# File upload section
st.subheader("העלה מסמך")

uploaded_files = st.file_uploader(
    "העלה מסמך",
    type=["pdf"],
    accept_multiple_files=True,
    key="file_uploader"
)

music_path = 'bg_music.mp3'
if os.path.exists(music_path):
    st.audio(open(music_path, "rb").read(), format=f"audio/{music_path.split('.')[-1]}")
else:
    st.error(f"Music file not found at: {music_path}")

# Process uploaded files only when files are uploaded
if uploaded_files and not st.session_state["file_processed"]:
    st.session_state["assistant"].clear()  # Clear previous documents
    st.session_state["messages"] = []
    
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
                error_msg = f"Error processing {file.name}: {str(e)}"
                st.error(error_msg)
                st.session_state["messages"].append((error_msg, False))
            finally:
                os.remove(file_path)
    
    st.session_state["file_processed"] = True

# Reset file processed flag when no files are selected
if not uploaded_files:
    st.session_state["file_processed"] = False

# Retrieval settings
st.subheader("Settings")
st.session_state["retrieval_k"] = st.slider(
    "Number of Retrieved Results (k)", min_value=1, max_value=10, value=5
)
st.session_state["retrieval_threshold"] = st.slider(
    "Similarity Score Threshold", min_value=0.0, max_value=1.0, value=0.2, step=0.05
)

# Display messages
st.subheader("Chat History")
for i, (msg, is_user) in enumerate(st.session_state["messages"]):
    st.chat_message("user" if is_user else "assistant").markdown(msg, unsafe_allow_html=True)

st.text_input("Message", key="user_input", on_change=process_input)

# Clear chat
if st.button("Clear Chat"):
    st.session_state["messages"] = []
    if "assistant" in st.session_state:
        st.session_state["assistant"].cleanup()