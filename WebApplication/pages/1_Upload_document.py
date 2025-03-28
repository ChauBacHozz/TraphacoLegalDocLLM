import streamlit as st
import time
import numpy as np
import docx
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
from sentence_transformers import SentenceTransformer
from WebApplication.backend.preprocessing.preprocess_docx import (extract_text, normalize_bullets, 
                                     convert_text_list_to_tree, flatten_tree,
                                     preprocess_chunks, normalize_appendix_text_bullets,
                                     normalize_modified_text_bullets)
from WebApplication.backend.graph_database.db_uploader.save_doc_to_db import save_origin_doc_to_db, save_modified_doc_to_db
from icecream import ic
from collections import OrderedDict
import os
from tqdm import tqdm
from stqdm import stqdm
import re
from neo4j import GraphDatabase

# WINDOWS_IP = "28.11.5.39"
# URI = "neo4j+s://13d9b8ff.databases.neo4j.io"
# USERNAME = "neo4j"
# PASSWORD = "tDJXOWtq9GSTnXqQyVFmb2xiR3GREbxnU8m9MxxWHwU"

driver = st.session_state.driver
# PATH = 'D:/VS_Workspace/LLM/.cache'
# os.environ['TRANSFORMERS_CACHE'] = PATH
# os.environ['HF_HOME'] = PATH
# os.environ['HF_DATASETS_CACHE'] = PATH
# os.environ['TORCH_HOME'] = PATH

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
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=True)
    
st.markdown("# Upload document")
st.sidebar.header("Upload your document")
st.write(
    """This function will upload your document and save to database"""
)

upload_files = st.file_uploader(
    "Choose a doc file", accept_multiple_files = True, type = ['docx']
)
def save_modified_doc_pre_appendix_type1_to_db(extracted_text, heading, doc_number, driver):
    heading_idx = None
    for i, text in enumerate(extracted_text[:100]):
        if ("Chương" in text or "Mục" in text or "Điều" in text) and not text.isupper():
            heading_idx = i
            break

    if heading_idx:
        extracted_text = extracted_text[heading_idx:]
    else:
        print("ERROR")
        return
    full_text = normalize_modified_text_bullets(extracted_text)

    # # Convert text list to tree base to manage content 
    tree = convert_text_list_to_tree(full_text)
    
    # # Flatten tree into list of strings
    flattened_tree = flatten_tree(tree)
    # # Split data into chunks
    chunks = [text[0] for text in flattened_tree]
    # # chunks = [f"{path}: {text}" for path, text in flattened_tree]
    # # Preprocess chunks
    preprocessed_chunks = preprocess_chunks(chunks, heading, doc_number)
    # Extract 'text' atribute from preprocessed_chunks
    texts = [chunk['content'] for chunk in preprocessed_chunks]
    metadata_lst = []
    for chunk in preprocessed_chunks:
        # chunk.pop("content")
        metadata_lst.append(chunk)

    batch_size = 10    
    print("☑️ saving pre-appendix data")

    for i in stqdm(range(0, len(metadata_lst), batch_size)):
        save_modified_doc_to_db(texts[i:i+batch_size],metadata_lst[i:i+batch_size], driver)

def save_modified_doc_pre_appendix_type2_to_db(extracted_text, heading, doc_number, driver):
    heading_idx = None
    for i, text in enumerate(extracted_text[:100]):
        if ("Chương" in text or "Mục" in text or "Điều" in text) and not text.isupper():
            heading_idx = i
            break

    if heading_idx:
        modified_doc_id = re.search(r"luật dược số (.+)", "".join(extracted_text[:heading_idx]), re.IGNORECASE).group(1).strip().split(" ")[0]

        extracted_text = extracted_text[heading_idx:]
    else:
        print("ERROR")
        return
    full_text = normalize_modified_text_bullets(extracted_text)

    # # Convert text list to tree base to manage content 
    tree = convert_text_list_to_tree(full_text)
    
    # # Flatten tree into list of strings
    flattened_tree = flatten_tree(tree)
    # # Split data into chunks
    chunks = [text[0] for text in flattened_tree]

    # # chunks = [f"{path}: {text}" for path, text in flattened_tree]
    # # Preprocess chunks
    preprocessed_chunks = preprocess_chunks(chunks, heading, doc_number)
    # Extract 'text' atribute from preprocessed_chunks
    texts = [chunk['content'] for chunk in preprocessed_chunks]
    metadata_lst = []
    for chunk in preprocessed_chunks:
        # chunk.pop("content")
        chunk["modified_doc_id"] = modified_doc_id
        metadata_lst.append(chunk)

    batch_size = 10    
    print("☑️ saving pre-appendix data")

    st.write(f"Saving {doc_number} {heading}")
    for i in stqdm(range(0, len(metadata_lst), batch_size)):
        save_modified_doc_to_db(texts[i:i+batch_size],metadata_lst[i:i+batch_size], driver, doc_type=2)

def save_origin_doc_pre_appendix_type1_to_db(extracted_text, heading, doc_number, driver):
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
    
    # save_tree_to_db(tree, driver)
    # Flatten tree into list of strings
    flattened_tree = flatten_tree(tree)
    # ic(flattened_tree)
    # Split data into chunks
    chunks = [text[0] for text in flattened_tree]
    # chunks = [f"{path}: {text}" for path, text in flattened_tree]
    # Preprocess chunks
    preprocessed_chunks = preprocess_chunks(chunks, heading, doc_number)
    # Extract 'text' atribute from preprocessed_chunks
    texts = [chunk['content'] for chunk in preprocessed_chunks]
    metadata_lst = []
    for chunk in preprocessed_chunks:
        # chunk.pop("content")
        metadata_lst.append(chunk)

    batch_size = 10    
    print("☑️ saving pre-appendix data")

    for i in stqdm(range(0, len(metadata_lst), batch_size)):
        save_origin_doc_to_db(texts[i:i+batch_size],metadata_lst[i:i+batch_size], driver)

def save_origin_doc_pre_appendix_type2_to_db(extracted_text, heading, doc_number, driver):
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
    
    # save_tree_to_db(tree, driver)
    # Flatten tree into list of strings
    flattened_tree = flatten_tree(tree)
    # ic(flattened_tree)
    # Split data into chunks
    chunks = [text[0] for text in flattened_tree]
    # chunks = [f"{path}: {text}" for path, text in flattened_tree]
    # Preprocess chunks
    preprocessed_chunks = preprocess_chunks(chunks, heading, doc_number)
    # Extract 'text' atribute from preprocessed_chunks
    texts = [chunk['content'] for chunk in preprocessed_chunks]
    metadata_lst = []
    for chunk in preprocessed_chunks:
        # chunk.pop("content")
        metadata_lst.append(chunk)

    batch_size = 10    
    print("☑️ saving pre-appendix data")

    for i in stqdm(range(0, len(metadata_lst), batch_size)):
        save_origin_doc_to_db(texts[i:i+batch_size],metadata_lst[i:i+batch_size], driver)


def save_origin_doc_appendix_type1_to_db(document, heading, doc_number, driver):
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
    preprocessed_chunks = preprocess_chunks(chunks, heading, doc_number)
    # ic(preprocessed_chunks)

    # # Extract 'text' atribute from preprocessed_chunks
    texts = [chunk['content'] for chunk in preprocessed_chunks]
    metadata_lst = []
    for chunk in preprocessed_chunks:
        # chunk.pop("content")
        metadata_lst.append(chunk)

    batch_size = 10    
    print("☑️ saving appendix data")
    for i in stqdm(range(0, len(metadata_lst), batch_size)):
        save_origin_doc_to_db(texts[i:i+batch_size],metadata_lst[i:i+batch_size], driver)

def save_modified_doc_appendix_type1_to_db(document, heading, doc_number, driver):
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
    ic(full_text)
    normalized_appendix_ids = [ids - appendix_ids[0] for ids in appendix_ids]
    chunks = normalize_appendix_text_bullets(full_text, normalized_appendix_ids)
    ic(chunks)
    # # Convert text list to tree base to manage content 
    # tree = convert_text_list_to_tree(full_text)
    
    # # Flatten tree into list of strings
    # flattened_tree = flatten_tree(tree)
    # # Split data into chunks
    # chunks = [text[0] for text in flattened_tree]
    # # chunks = [f"{path}: {text}" for path, text in flattened_tree]
    # # Preprocess chunks
    preprocessed_chunks = preprocess_chunks(chunks, heading, doc_number)
    # ic(preprocessed_chunks)

    # # Extract 'text' atribute from preprocessed_chunks
    texts = [chunk['content'] for chunk in preprocessed_chunks]
    metadata_lst = []
    for chunk in preprocessed_chunks:
        # chunk.pop("content")
        metadata_lst.append(chunk)

    batch_size = 10    
    print("☑️ saving appendix data")
    for i in stqdm(range(0, len(metadata_lst), batch_size)):
        save_modified_doc_to_db(texts[i:i+batch_size],metadata_lst[i:i+batch_size], driver)

    
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
        # Type 1
        if "nghị định" in heading.lower() or "thông tư" in heading.lower():
            # if "sửa đổi" in heading.lower():
            if "sửa đổi" in heading.lower() or "bổ sung" in heading.lower() or "bãi bỏ" in heading.lower():
                # Văn bản sửa đổi
                if appendix_index != None:
                    print("Có phụ lục")
                    save_modified_doc_appendix_type1_to_db(doc_file, heading, doc_number, driver)
                save_modified_doc_pre_appendix_type1_to_db(pre_appendix_text, heading, doc_number, driver)
            else:
                # Văn bản gốc
                if appendix_index != None:
                    print("Có phụ lục")
                    save_origin_doc_appendix_type1_to_db(doc_file, heading, doc_number, driver)
                save_origin_doc_pre_appendix_type1_to_db(pre_appendix_text, heading, doc_number, driver)
                st.toast(f"Saved {upload_file.name} database✅")

        elif "luật" in heading.lower():
            if "sửa đổi" in heading.lower() or "bổ sung" in heading.lower() or "bãi bỏ" in heading.lower():
                # Văn bản sửa đổi
                save_modified_doc_pre_appendix_type2_to_db(pre_appendix_text, heading, doc_number, driver)
            else:
                # Văn bản gốc
                if appendix_index != None:
                    print("Có phụ lục")
                    save_origin_doc_appendix_type1_to_db(doc_file, heading, doc_number, driver)
                save_origin_doc_pre_appendix_type2_to_db(pre_appendix_text, heading, doc_number, driver)
                st.toast(f"Saved {upload_file.name} database✅")
        else:
            print("ERROR")
        # Merge bullet from extracted text

