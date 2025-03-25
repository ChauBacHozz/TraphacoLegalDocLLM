import streamlit as st
import time
import os

model_exist = False
st.title("Legal chatbot LLM")


if "rag_model" in st.session_state:
    # rag_model = st.session_state.rag_model

    st.session_state.rag_model.get_model_ready()
    model_exist = True
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
else:
    st.write("No data found in session state.")
# Initialize session state variable

if "show_dialog" not in st.session_state:
    st.session_state.show_dialog = False

# Function to toggle the dialog visibility
def toggle_dialog():
    st.session_state.show_dialog = not st.session_state.show_dialog

@st.dialog("LLM Setting", width="large")
def model_setting(rag_model):
    left_col, right_col = st.columns([4, 10])
    with left_col:
        # Define parameters in a loop to avoid repetition
        params = ["Max New Tokens", "Temperature", "Top P", "Top K"]
        keys = ["max_tokens", "temperature", "top_p", "top_k"]

        old_max_tokens = rag_model.max_new_tokens
        old_temperature = rag_model.temperature
        old_top_p = rag_model.top_p
        old_top_k = rag_model.top_k

        vals = [old_max_tokens, old_temperature, old_top_p, old_top_k]

        new_vals = {}
        for label, key, val in zip(params, keys, vals):
            label_col, field_col = st.columns([3, 2])  # Adjust for better spacing
            with label_col:
                st.write("####")
                st.markdown(f"**{label}:**")  # Bold text for clarity
            with field_col:
                new_vals[key] = st.text_input("", key=key.upper(), value = val)


    with right_col:
        model_sys_prompt = rag_model.system_prompt
        model_template = rag_model.template
        
        system_prompt = st.text_area(label = "System prompt", height = 90, value = model_sys_prompt)
        template = st.text_area(label = "Template", height = 250, value = model_template)

        if st.button("Save", use_container_width=True):
            saved_success = True
            if int(new_vals["max_tokens"]) and float(new_vals["temperature"]) and float(new_vals["top_p"]) and int(new_vals["top_k"]):
                # Save params to rag model
                rag_model.set_control_params(int(new_vals["max_tokens"]), float(new_vals["temperature"]), float(new_vals["top_p"]), int(new_vals["top_k"]))
                
                rag_model.system_prompt = system_prompt
                if "{context}" in template and "{question}" in template:
                    rag_model.template = template
                else:
                    print("ERRORRRR")
                    saved_success = False
            else:
                saved_success = False

            if saved_success:
                st.toast("✅ saved params to model!")
            else:
                # Print error
                pass







def display_tokens(token_stream, container):
    """Display tokens one by one in the Streamlit app."""
    full_text = ""
    for token in token_stream:
        full_text += token
        container.markdown(full_text)  # Update the displayed text
        time.sleep(0.02) 

if model_exist:
    # Handle user input
    with st.sidebar:
        if st.button('🤖'):
            if model_exist:
                model_setting(st.session_state.rag_model)
    if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Generate model response
        with st.spinner("Hãy chờ chút, mô hình đang trả lời!!!!"):
            response = st.session_state.rag_model.rag_answer(prompt)  # Use the loaded model
        st.success("Mô hình đã suy luận xong, thực hiện trả lời!")

        # Typing effect for the assistant's response
        with st.chat_message("assistant"):
            message_container = st.empty()  # Create a container to display the text
            display_tokens(response, message_container)
        print("Newest temp:", st.session_state.rag_model.temperature)

        st.session_state.messages.append({"role": "assistant", "content": response})

        options = ["Item 1", "Item 2", "Item 3", "Item 4"]

        with st.sidebar.expander("Dropdown View"):
            for item in options:
                st.write(f"- {item}")




