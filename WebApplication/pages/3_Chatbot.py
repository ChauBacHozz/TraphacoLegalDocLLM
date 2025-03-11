import streamlit as st
import time
from backend.RAGQwenModel import RAGQwen
import os

col1, col2 = st.columns([10, 1])

with col1:
    st.title("Legal chatbot LLM")

with col2:
    if st.button('🤖', use_container_width=True):
        print("Check")

@st.dialog("LLM Setting")
def model_setting(rag_model):

    if st.button("Save"):
        print("SAVED")


def display_tokens(token_stream, container):
    """Display tokens one by one in the Streamlit app."""
    full_text = ""
    for token in token_stream:
        full_text += token
        container.markdown(full_text)  # Update the displayed text
        time.sleep(0.02) 

if "rag_model" in st.session_state:
    rag_model = st.session_state.rag_model

    rag_model.get_model_ready()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])



    # Handle user input
    if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate model response
        with st.spinner("Hãy chờ chút, mô hình đang trả lời!!!!"):
            response = rag_model.rag_answer(prompt)  # Use the loaded model
        st.success("Mô hình đã suy luận xong, thực hiện trả lời!")

        # Typing effect for the assistant's response
        with st.chat_message("assistant"):
            message_container = st.empty()  # Create a container to display the text
            display_tokens(response, message_container)

        st.session_state.messages.append({"role": "assistant", "content": response})

        options = ["Item 1", "Item 2", "Item 3", "Item 4"]

        with st.sidebar.expander("Dropdown View"):
            for item in options:
                st.write(f"- {item}")


else:
    st.write("No data found in session state.")

