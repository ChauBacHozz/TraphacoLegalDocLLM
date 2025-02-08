import streamlit as st
import time
import numpy as np
import docx
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from sentence_transformers import SentenceTransformer
from backend.preprocess_docx import (extract_text, normalize_bullets, 
                                     convert_text_list_to_tree, flatten_tree,
                                     preprocess_chunks, normalize_appendix_text_bullets)
from backend.save_doc_to_db import save_to_db
from icecream import ic
from collections import OrderedDict
import os
from tqdm import tqdm
from stqdm import stqdm


PATH = 'D:/VS_Workspace/LLM/.cache'
os.environ['TRANSFORMERS_CACHE'] = PATH
os.environ['HF_HOME'] = PATH
os.environ['HF_DATASETS_CACHE'] = PATH
os.environ['TORCH_HOME'] = PATH

EMBEDDING_MODEL_NAME = "dangvantuan/vietnamese-document-embedding"
st.set_page_config(page_title="Upload document", page_icon="📈")

# @st.cache_resource
# def get_embedding_model(embedding_model_name):
#     return SentenceTransformer(embedding_model_name, trust_remote_code=True)

# with st.spinner("Loading Language Embedding model"):
#     embedding_model = get_embedding_model(embedding_model_name=EMBEDDING_MODEL_NAME)
if "embedding_model" in st.session_state:
    embedding_model = st.session_state.embedding_model
else:
    st.write("No data found in session state.")
    
st.markdown("# Upload document")
st.sidebar.header("Upload your document")
st.write(
    """This function will upload your document and save to database"""
)

upload_files = st.file_uploader(
    "Choose a doc file", accept_multiple_files = True, type = ['docx']
)


def save_pre_appendix_text_type1_to_db(extracted_text, heading, embedding_model):
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
    # Extract 'text' atribute from preprocessed_chunks
    texts = [chunk['text'] for chunk in preprocessed_chunks]
    metadata_lst = []
    for chunk in preprocessed_chunks:
        chunk.pop("text")
        metadata_lst.append(chunk)

    batch_size = 3    
    print("☑️ saving pre-appendix data")

    for i in stqdm(range(0, len(metadata_lst), batch_size)):
        save_to_db(texts[i:i+batch_size],metadata_lst[i:i+batch_size], embedding_model)


def save_appendix_text_type1_to_db(document, heading):
    extracted_text = []
    appendix_ids = []
    temp_idx = 0
    remove_idx = None
    for para in document.paragraphs:
        if para.text != "\xa0":
            extracted_text.append(para.text)
            temp_idx += 1
        if para.alignment == WD_PARAGRAPH_ALIGNMENT.CENTER and "phụ lục" in para.text.lower():
            appendix_ids.append(temp_idx)
        if para.alignment == WD_PARAGRAPH_ALIGNMENT.CENTER and "biểu mẫu" in para.text.lower():
            remove_idx = temp_idx - 1
    if remove_idx:
        full_text = extracted_text[appendix_ids[0] - 1:remove_idx]
    else:
        full_text = extracted_text[appendix_ids[0] - 1:]
    normalized_appendix_ids = [ids - appendix_ids[0] for ids in appendix_ids]
    chunks = normalize_appendix_text_bullets(full_text, normalized_appendix_ids)
    # # Convert text list to tree base to manage content 
    # tree = convert_text_list_to_tree(full_text)
    
    # # Flatten tree into list of strings
    # flattened_tree = flatten_tree(tree)
    # # Split data into chunks
    # chunks = [text[0] for text in flattened_tree]
    # # chunks = [f"{path}: {text}" for path, text in flattened_tree]
    # # Preprocess chunks
    preprocessed_chunks = preprocess_chunks(chunks, heading)
    ic(preprocessed_chunks)

    # # Extract 'text' atribute from preprocessed_chunks
    texts = [chunk['text'] for chunk in preprocessed_chunks]
    metadata_lst = []
    for chunk in preprocessed_chunks:
        chunk.pop("text")
        metadata_lst.append(chunk)

    batch_size = 3    
    print("☑️ saving appendix data")
    for i in stqdm(range(0, len(metadata_lst), batch_size)):
        save_to_db(texts[i:i+batch_size],metadata_lst[i:i+batch_size], embedding_model)

    
if st.button("Upload to database"):
    for upload_file in upload_files:
        doc_file = docx.Document(upload_file)
        # Extract text from doc
        extracted_text, appendix_index = extract_text(doc_file)
        # Remove special text from extracted_text
        # if "\xa0" in extracted_text:
        #     extracted_text.remove("\xa0")
        if appendix_index != None:
            pre_appendix_text = extracted_text[:appendix_index - 1]
            appendix_text = extracted_text[appendix_index - 1:]
        else:
            pre_appendix_text = extracted_text
        # Extract document heading
        doc_number = doc_file.tables[0].rows[1].cells[0].text
        heading = ": ".join(extracted_text[:2])
        heading = heading + " | " + doc_number
        # Type 1
        if "nghị định" in heading.lower() or "thông tư" in heading.lower():
            print("Nghị định hoặc thông tư")
            if appendix_index != None:
                print("Có phụ lục")
                save_appendix_text_type1_to_db(doc_file, heading)
            save_pre_appendix_text_type1_to_db(pre_appendix_text, heading, embedding_model)
            st.toast(f"Saved {upload_file.name} database✅")

        elif "luật" in heading.lower():
            if "sửa đổi" in heading.lower():
                print("Luật sử đổi bổ sung")
            else:
                print("Luật gốc")
        else:
            print("ERROR")
        # Merge bullet from extracted text


