import streamlit as st
import time
from backend.RAGQwenModel import RAGQwen
import os

model_exist = False
if "rag_model" in st.session_state:
    rag_model = st.session_state.rag_model

    rag_model.get_model_ready()
    model_exist = True
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
else:
    st.write("No data found in session state.")


@st.dialog("LLM Setting", width="large")
def model_setting(rag_model):
    left_col, right_col = st.columns([3, 8])
    with left_col:
        # Define parameters in a loop to avoid repetition
        params = ["Max New Tokens", "Temperature", "Top P", "Top K"]
        keys = ["max_tokens", "temperature", "top_p", "top_k"]
        
        max_tokens = rag_model.max_new_tokens
        temperature = rag_model.temperature
        top_p = rag_model.top_p
        top_k = rag_model.top_k

        vals = [max_tokens, temperature, top_p, top_k]

        for label, key, val in zip(params, keys, vals):
            label_col, field_col = st.columns([3, 2])  # Adjust for better spacing
            with label_col:
                st.write("####")
                st.markdown(f"**{label}:**")  # Bold text for clarity
            with field_col:
                st.text_input("", key=key, value = val)


    with right_col:
        model_sys_prompt = rag_model.system_prompt
        model_template = rag_model.template
        
        system_prompt = st.text_area(label = "System prompt", height = 80, value = model_sys_prompt)
        template = st.text_area(label = "Template", height = 360, value = model_template)

        if st.button("Save", use_container_width=True):
            print("SAVED")


col1, col2 = st.columns([10, 1])
with col1:
    st.title("Legal chatbot LLM")

with col2:
    if st.button('🤖', use_container_width=True):
        if model_exist:
            model_setting(rag_model)




def display_tokens(token_stream, container):
    """Display tokens one by one in the Streamlit app."""
    full_text = ""
    for token in token_stream:
        full_text += token
        container.markdown(full_text)  # Update the displayed text
        time.sleep(0.02) 

if model_exist:
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




