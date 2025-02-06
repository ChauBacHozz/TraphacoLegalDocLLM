import streamlit as st
import time
import numpy as np
import docx
from sentence_transformers import SentenceTransformer
from backend.preprocess_docx import (extract_text, normalize_bullets, 
                                     convert_text_list_to_tree, flatten_tree,
                                     preprocess_chunks)
from backend.save_doc_to_db import save_to_db
from icecream import ic
from collections import OrderedDict


EMBEDDING_MODEL_NAME = "dangvantuan/vietnamese-document-embedding"
st.set_page_config(page_title="Upload document", page_icon="📈")

@st.cache_resource
def get_embedding_model(embedding_model_name):
    return SentenceTransformer(embedding_model_name, trust_remote_code=True)

with st.spinner("Loading Language Embedding model"):
    embedding_model = get_embedding_model(embedding_model_name=EMBEDDING_MODEL_NAME)

st.markdown("# Upload document")
st.sidebar.header("Upload your document")
st.write(
    """This function will upload your document and save to database"""
)

upload_files = st.file_uploader(
    "Choose a doc file", accept_multiple_files = True, type = ['docx']
)


def save_type1_to_db(extracted_text, heading, embedding_model):

    heading_idx = None
    for i, text in enumerate(extracted_text[:100]):
        if "chương" in text.lower():
            heading_idx = i
            break

    if heading_idx:
        extracted_text = extracted_text[heading_idx:]
    else:
        print("ERROR")
        return
    
    full_text = normalize_bullets(extracted_text)
    # Convert text list to tree base to manage content 
    tree = convert_text_list_to_tree(full_text)
    
    # Flatten tree into list of strings
    flattened_tree = flatten_tree(tree)
    # Split data into chunks
    chunks = [text[0] for text in flattened_tree]
    # chunks = [f"{path}: {text}" for path, text in flattened_tree]
    # Preprocess chunks
    preprocessed_chunks = preprocess_chunks(chunks, heading)
    ic(preprocessed_chunks[:3])
    # Extract 'text' atribute from preprocessed_chunks
    texts = [chunk['text'] for chunk in preprocessed_chunks]
    metadata_lst = []
    for chunk in preprocessed_chunks:
        chunk.pop("text")
        metadata_lst.append(chunk)
    save_to_db(texts, metadata_lst, embedding_model)

    
    st.toast(f"Saved {upload_file.name} database✅")

if st.button("Upload to database"):
    for upload_file in upload_files:
        doc_file = docx.Document(upload_file)
        # Extract text from doc
        extracted_text = extract_text(doc_file)
        # Remove special text from extracted_text
        if "\xa0" in extracted_text:
            extracted_text.remove("\xa0")
        # Extract document heading
        doc_number = doc_file.tables[0].rows[1].cells[0].text
        heading = ": ".join(extracted_text[:2])
        heading = heading + " | " + doc_number
        # Type 1
        if "nghị định" in heading.lower() or "thông tư" in heading.lower():
            print("Nghị định hoặc thông tư")
            save_type1_to_db(extracted_text, heading, embedding_model)

        elif "luật" in heading.lower():
            if "sửa đổi" in heading.lower():
                print("Luật sử đổi bổ sung")
            else:
                print("Luật gốc")
        else:
            print("ERROR")
        # Merge bullet from extracted text


