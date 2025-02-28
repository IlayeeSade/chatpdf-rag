# app.py
import os
import tempfile
import time
import streamlit as st
from streamlit_chat import message
from rag import ChatPDF

st.set_page_config(page_title="IlayeeRAG")

def display_messages():
    """Display the chat history."""
    st.subheader("Chat History")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        avatar = '👩‍🎤' if is_user else '👺'
        message(msg, is_user=is_user, avatar=avatar, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()


def process_input():
    """Process the user input and generate an assistant response."""
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner("חושב..."):
            try:
                agent_text = st.session_state["assistant"].ask(
                    user_text,
                    k=st.session_state["retrieval_k"],
                    score_threshold=st.session_state["retrieval_threshold"],
                )
            except ValueError as e:
                agent_text = str(e)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))

def read_and_save_file():
    """Handle file upload and ingestion."""
    st.session_state["assistant"].clear()
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""
    
    # Create a new list for successfully ingested files
    successful_files = []

    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        with st.session_state["ingestion_spinner"], st.spinner(f"מעכל {file.name}..."):
            t0 = time.time()
            ingest_result = st.session_state["assistant"].ingest(file_path)
            t1 = time.time()

        if ingest_result == "success":
            st.session_state["messages"].append(
                (f"Ingested {file.name} in {t1 - t0:.2f} seconds", False)
            )
            successful_files.append(file)
        else:
            st.error(f"Failed to ingest {file.name}: {ingest_result}")

        os.remove(file_path)

    # Update the file_uploader state to only show successful files
    st.session_state["file_uploader"] = successful_files


def page():
    """Main app page layout."""
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["assistant"] = ChatPDF()

    st.header("IlayeeRAG")

    st.subheader("העלה מסמך")
    st.file_uploader(
        "העלה מסמך",
        type=["pdf"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

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
        st.session_state["assistant"].clear()


if __name__ == "__main__":
    page()
