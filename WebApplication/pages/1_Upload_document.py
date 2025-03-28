import streamlit as st
import time
import numpy as np
import docx
from sentence_transformers import SentenceTransformer
from collections import OrderedDict
import os
import sys
from backend.graph_database.preprocessing.preprocess_docx import extract_text
from neo4j import GraphDatabase
from backend.document_upload_to_db import (save_modified_doc_appendix_type1_to_db,
                                            save_modified_doc_pre_appendix_type1_to_db,
                                            save_origin_doc_pre_appendix_type1_to_db,
                                            save_origin_doc_appendix_type1_to_db,
                                            save_modified_doc_pre_appendix_type2_to_db,
                                            save_origin_doc_appendix_type2_to_db,
                                            save_origin_doc_pre_appendix_type2_to_db
                                           )

EMBEDDING_MODEL_NAME = "dangvantuan/vietnamese-document-embedding"
driver = st.session_state.driver

st.set_page_config(page_title="Upload document", page_icon="📈")

# Lấy ra embedding_model cho văn bản từ session_state đã load
if "embedding_model" in st.session_state:
    embedding_model = st.session_state.embedding_model
else:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, trust_remote_code=True)

# Xây dựng các thành phần của trang upload document
st.markdown("# Upload document")
st.sidebar.header("Upload your document")
st.write(
    """This function will upload your document and save to database"""
)

upload_files = st.file_uploader(
    "Choose a doc file", accept_multiple_files = True, type = ['docx']
)

"""
    Văn bản pháp luật gồm hai loại:
        - Văn bản thông tư nghị định: Được định nghĩa là loại 1 (type 1)
        - Văn bản luật (e.g Luật Dược): Được định nghĩa là loại 2 (type 2)
    Cấu trúc mỗi văn bản dù là loại 1 hay loại 2 đều gồm cấu trúc hai phần:
        - Nội dung chính - nằm ở phía trước mục lục (main_content)
        - Mục lục (appendix)
    Mỗi văn bản sẽ được chia thành hai loại văn bản:
        - Văn bản gốc (origin_doc)
        - Văn bản bãi bỏ/chỉnh sửa/bổ sung (modified_doc)
    Như vậy có 4 trường hợp xảy ra:
        - 
    Workflow khi upload document như sau:
        - Bóc tách từ file sang text, lấy ra thông tin về phụ lục (là index của dòng 
        bắt đầu phụ lục, nếu không có phụ lục thì biến đó sẽ là None)
        - 
"""
    
if st.button("Upload to database"):
    for upload_file in upload_files:
        doc_file = docx.Document(upload_file)
        # Extract text from doc
        extracted_text, appendix_index = extract_text(doc_file)
        # Remove special text from extracted_text
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
        # Type 2
        elif "luật" in heading.lower():
            if "sửa đổi" in heading.lower() or "bổ sung" in heading.lower() or "bãi bỏ" in heading.lower():
                # Văn bản sửa đổi
                save_modified_doc_pre_appendix_type2_to_db(pre_appendix_text, heading, doc_number, driver)
            else:
                # Văn bản gốc
                if appendix_index != None:
                    print("Có phụ lục")
                    save_origin_doc_appendix_type2_to_db(doc_file, heading, doc_number, driver)
                save_origin_doc_pre_appendix_type2_to_db(pre_appendix_text, heading, doc_number, driver)
                st.toast(f"Saved {upload_file.name} database✅")
        else:
            print("ERROR")
        # Merge bullet from extracted text

