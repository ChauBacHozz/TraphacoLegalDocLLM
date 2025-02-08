import streamlit as st


if "rag_model" in st.session_state:
    rag_model = st.session_state.rag_model
    rag_model.get_model_ready()
else:
    st.write("Cannot find embedding model")

if prompt := st.chat_input("Enter your RAG query..."):
    with st.chat_message("user"):
        st.markdown(f"Here is retrieval data for query: {prompt}")

        retrieval_data = rag_model.search_query(prompt)

        for data in retrieval_data:
            st.markdown(data)

