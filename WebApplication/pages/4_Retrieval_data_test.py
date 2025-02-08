import streamlit as st
from underthesea import word_tokenize

def count_tokens_underthesea(text):
    tokens = word_tokenize(text, format="text").split()
    return len(tokens)

if "rag_model" in st.session_state:
    rag_model = st.session_state.rag_model
    rag_model.get_model_ready()
else:
    st.write("Cannot find embedding model")

if prompt := st.chat_input("Enter your RAG query..."):
    with st.chat_message("user"):
        st.markdown(f"Here is retrieval data for query: {prompt}")

        retrieval_data = rag_model.search_query(prompt)
        n_tokens = 0
        for context in retrieval_data:
            n_tokens += count_tokens_underthesea(context)
        print(f"😄 there are {n_tokens} tokens in context")
        
        for data in retrieval_data:
            st.markdown(data)

