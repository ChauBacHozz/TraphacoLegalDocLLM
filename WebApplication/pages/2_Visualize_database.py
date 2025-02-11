import streamlit as st
import faiss
import os
import pickle
from icecream import ic
import pandas as pd

st.set_page_config(page_title="Upload document", page_icon="📈", layout="wide")

st.markdown("# Database")
st.sidebar.header("Visualize your document")
st.write(
    """Choose type"""
)

visualize_options = ["By documents", "By chunks"]


selected_option = st.radio("Select an option", visualize_options, key="radio_option") 

# Check if the value has changed and print it
if "previous_option" not in st.session_state:
    st.session_state["previous_option"] = None  # Initialize previous state

if st.session_state["previous_option"] != st.session_state["radio_option"]:
    st.session_state["previous_option"] = st.session_state["radio_option"]
    st.write(f"You selected: {st.session_state['radio_option']}")

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
    ic(metadata)
    df = pd.DataFrame(metadata)
    st.dataframe(df, use_container_width=True)
else:
    print("Database is not created")


