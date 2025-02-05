import streamlit as st
import faiss
import os
import pickle

st.set_page_config(page_title="Upload document", page_icon="📈")

st.markdown("# Database")
st.sidebar.header("Upload your document")
st.write(
    """This function will visualize your document database"""
)

FAISS_INDEX_PATH = "db/faiss_index.bin"
DATA_PATH = "db/data.pkl"
METADATA_PATH = "db/metadata.pkl"

# Load FAISS index, data and metadata if they exist, otherwise initialize
if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH) and os.path.exists(DATA_PATH):
    print("🔄 Loading existing FAISS index, data and metadata...")
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    with open(DATA_PATH, "rb") as f:
        data = pickle.load(f)
else:
    print("Database is not created")


