import streamlit as st
import faiss
import os
import pickle
from icecream import ic
import pandas as pd

st.set_page_config(page_title="Upload document", page_icon="📈", layout="wide")

st.markdown("# Database")
st.sidebar.header("Visualize your document")
