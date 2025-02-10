import streamlit as st
from underthesea import word_tokenize

def count_tokens_underthesea(text):
    tokens = word_tokenize(text, format="text").split()
    return len(tokens)

def on_drection_change():
    st.session_state.direction = st.session_state.direction_pill
    st.session_state.direction_pill = None

if "rag_model" in st.session_state:
    rag_model = st.session_state.rag_model
    rag_model.get_model_ready()
else:
    st.write("Cannot find embedding model")
opts = ["Đề mục", "Nội dung"]
if prompt := st.chat_input("Enter your RAG query..."):
    # with st.chat_message("user"):
        
    #     selection = st.pills

    #     st.markdown(f"Here is retrieval data for query: {prompt}")

    #     retrieval_data = rag_model.get_retrieval_data(prompt)
    #     n_tokens = 0
    #     for context in retrieval_data:
    #         n_tokens += count_tokens_underthesea(context)
    #     print(f"😄 there are {n_tokens} tokens in context")
        
    #     for data in retrieval_data:
    #         st.markdown(data)
    with st.chat_message("assistant"):
        # message_container = st.empty()  # Create a container to display the text
        st.session_state.setdefault("direction", None)
        directions = ["North", "East", "South", "West"]
        st.pills("Direction", options=directions, on_change=on_drection_change, key="direction_pill")
        st.write(f"You selected **{st.session_state.direction}**")
        
# import streamlit as st
# st.set_page_config(layout="wide")
# st.title("Ask Me Anything")

# # Initialize session state for chat messages
# if 'chat_messages' not in st.session_state:
#     st.session_state.chat_messages = []
#     st.session_state.chat_messages = [{"role":"system","content":"you are a helpful assistant"}]

# # Initialize session state for prompt selection
# if 'prompt_selection' not in st.session_state:
#     st.session_state.prompt_selection = None

# if 'direction' not in st.session_state:
#     st.session_state.direction = None

# def on_direction_change():
#     st.session_state.direction = st.session_state.direction_pill
#     st.session_state.direction_pill = None

# if st.sidebar.button(label="clear chat", 
#                        help="clicking this button will clear the whole history",
#                        icon="🗑️"):
#         st.session_state.prompt_selection = None
#         st.session_state["chat_messages"] = [{"role":"system","content":"you are a helpful assistant"}]

# col1, col2 = st.columns([.6,.4])

# with col1:
#      options = ["What is the weather today?", "What is the time now?", 
#                 "How are you doing?", "what is 1+1 equal to ?", "summarize quantum physics in 50 words max"]
     
#      st.session_state.setdefault("direction", None)
#      prompt_selection = st.pills("Prompt suggestions", options, selection_mode="single", 
#                                  label_visibility='hidden',on_change=on_direction_change, key="direction_pill")
#      input_prompt = st.chat_input("Type your message here...")
#      st.session_state.prompt_selection = input_prompt
 
#      if st.session_state.direction is not None:
#           with st.chat_message("user"):
#                     st.write(st.session_state.direction)
#           with st.spinner("searching ..."):  
#                with st.chat_message("assistant"):
#                     r = f"you wrote `{st.session_state.direction}`"
#                     st.write(r)
#           st.session_state.direction = None

#      elif st.session_state.prompt_selection is not None:
#           st.session_state.chat_messages.append({"role": "user", "content": st.session_state.prompt_selection})
#           with st.chat_message("user"):
#                     st.write(str(st.session_state.prompt_selection))
#           with st.spinner("searching ..."):  
#                with st.chat_message("assistant"):
#                     r = f"you wrote `{st.session_state.prompt_selection}`"
#                     st.write(r)
#           st.session_state.prompt_selection = None

# with col2:
#       st.json(st.session_state)