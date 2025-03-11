import streamlit as st
from sentence_transformers import SentenceTransformer
from backend.RAGQwen25 import RAGQwen25
import os

import os
os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"
# PATH = 'D:/VS_Workspace/LLM/.cache'
# os.environ['TRANSFORMERS_CACHE'] = PATH
# os.environ['HF_HOME'] = PATH
# os.environ['HF_DATASETS_CACHE'] = PATH
# os.environ['TORCH_HOME'] = PATH

st.set_page_config(
    page_title = "Home",
    page_icon = "🏠"
)

EMBEDDING_MODEL_NAME = "dangvantuan/vietnamese-document-embedding"

@st.cache_resource
def get_model():
    return RAGQwen25()

        
if "rag_model" not in st.session_state:
    with st.spinner("Loading RAG model"):
        st.session_state.rag_model = get_model()
        
if "embedding_model" not in st.session_state:
    with st.spinner("Loading Embedding model"):
        st.session_state.embedding_model = st.session_state.rag_model.embedding_model

st.write("Welcome to web app")

st.sidebar.success("Select function")

st.markdown(
    """
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.
    **👈 Select a demo from the sidebar** to see some examples
    of what Streamlit can do!
    ### Want to learn more?
    - Check out [streamlit.io](https://streamlit.io)
    - Jump into our [documentation](https://docs.streamlit.io)
    - Ask a question in our [community
        forums](https://discuss.streamlit.io)
    ### See more complex demos
    - Use a neural net to [analyze the Udacity Self-driving Car Image
        Dataset](https://github.com/streamlit/demo-self-driving)
    - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
"""
)