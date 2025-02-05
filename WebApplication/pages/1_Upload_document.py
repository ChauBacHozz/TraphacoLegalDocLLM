import streamlit as st
import time
import numpy as np
import docx
from backend.preprocess_docx import (extract_text, normalize_bullets, 
                                     convert_text_list_to_tree, flatten_tree,
                                     preprocess_chunks)

st.set_page_config(page_title="Plotting Demo", page_icon="📈")

st.markdown("# Upload document")
st.sidebar.header("Upload your document")
st.write(
    """This function will upload your document and save to database"""
)

upload_files = st.file_uploader(
    "Choose a doc file", accept_multiple_files = True, type = ['docx']
)

for upload_file in upload_files:
    bytes_data = upload_file.read()
    doc_file = docx.Document(upload_file)
    # Extract text from doc
    extracted_text = extract_text(doc_file)
    print(extracted_text)
    # Merge bullet from extracted text
    full_text = normalize_bullets(extracted_text)
    # Convert text list to tree base to manage content 
    tree = convert_text_list_to_tree(full_text)
    # Flatten tree into list of strings
    flattened_tree = flatten_tree(tree)
    # Split data into chunks
    chunks = [f"{path}: {text}" for path, text in flattened_tree]
    # Preprocess chunks
    preprocessed_chunks = preprocess_chunks(chunks)
    # Extract 'text' atribute from preprocessed_chunks
    texts = [chunk['text'] for chunk in preprocessed_chunks]
