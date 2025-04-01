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
from backend.document_upload_to_db import (
                                            save_type1_origin_appendix_to_db,
                                            save_type1_origin_pre_appendix_to_db,
                                            save_modified_doc_appendix_type1_to_db,
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
        - Phụ lục (appendix)

    Mỗi văn bản sẽ được chia thành hai loại văn bản:
        - Văn bản gốc (origin_doc)
        - Văn bản bãi bỏ/chỉnh sửa/bổ sung (modified_doc)

    Như vậy có 8 trường hợp xảy ra:
        - save_type1_origin_appendix_to_db: Lưu nội dung phụ lục của văn bản thông tư nghị định gốc
        - save_type1_origin_pre_appendix_to_db: Lưu nội dung tiền phụ lục của văn bản thông tư nghị định gốc
        - save_type1_modified_appendix_to_db: Lưu nội dung phụ lục của văn bản thông tư nghị định chỉnh sửa
        - save_type1_modified_pre_appendix_to_db: Lưu nội dung tiền phụ lục của văn bản thông tư nghị định chỉnh sửa

        - save_type2_origin_appendix_to_db: Lưu nội dung phụ lục của văn bản luật gốc
        - save_type2_origin_pre_appendix_to_db: Lưu nội dung tiền phụ lục của văn bản luật gốc
        - save_type2_modified_appendix_to_db: Lưu nội dung phụ lục của văn bản luật chỉnh sửa
        - save_type2_modified_pre_appendix_to_db: Lưu nội dung tiền phụ lục của văn bản luật chỉnh sửa

    Workflow khi upload document như sau:
        - Bóc tách từ file sang text, lấy ra thông tin về phụ lục (là index của dòng 
        bắt đầu phụ lục, nếu không có phụ lục thì biến đó sẽ là None)
        - 
"""
    
if st.button("Upload to database"):
    for upload_file in upload_files:
        doc_file = docx.Document(upload_file)
        # Trích xuất dữ liệu văn bản từ doc_file, khi trích xuất sẽ rà soát xem có phụ lục tồn tại không, nếu có thì chỉ mục (theo dòng)
        # của phục lục sẽ được lưu vào biến appendiz_index, nếu không appendix_index sẽ nhận về None
        extracted_text, appendix_index = extract_text(doc_file)

        if appendix_index != None:
            # Chia văn bản đầu thành văn bản tiền phụ lục và văn bản phụ lục
            pre_appendix_text = extracted_text[:appendix_index - 1]
            appendix_text = extracted_text[appendix_index - 1:]
        else:
            # Trường hợp còn lại toàn bộ văn bản được coi là văn bản tiền phụ lục
            pre_appendix_text = extracted_text
        # Trích xuất thông tin tiêu đề văn bản và mã văn bản
        doc_number = doc_file.tables[0].rows[1].cells[0].text
        heading = ": ".join(extracted_text[:2])

        # Kiểm tra nếu văn bản có là thông tư nghị định hay không (loại 1)
        if "nghị định" in heading.lower() or "thông tư" in heading.lower():
            # Kiểm tra có phải văn bản sửa đổi hây văn bản gốc
            if "sửa đổi" not in heading.lower() and "bổ sung" not in heading.lower() and "bãi bỏ" not in heading.lower():
                # Văn bản gốc
                if appendix_index:
                    save_type1_origin_appendix_to_db(appendix_index, heading, doc_number, driver)
                save_type1_origin_pre_appendix_to_db(pre_appendix_text, heading, doc_number, driver)
            else:
                # Văn bản sửa đổi
                if appendix_index:
                    save_type1_modified_appendix_to_db(appendix_index, heading, doc_number, driver)
                save_type1_modified_pre_appendix_to_db(pre_appendix_text, heading, doc_number, driver)

        # Kiểm tra nếu văn bản có là luật hay không (loại 2)
        elif "luật" in heading.lower():
            if "sửa đổi" not in heading.lower() and "bổ sung" not in heading.lower() and "bãi bỏ" not in heading.lower():
                # Văn bản gốc
                if appendix_index:
                    save_type2_origin_appendix_to_db(appendix_index, heading, doc_number, driver)
                save_type2_origin_pre_appendix_to_db(pre_appendix_text, heading, doc_number, driver)
            else:
                # Văn bản sửa đổi
                if appendix_index:
                    save_type2_modified_appendix_to_db(appendix_index, heading, doc_number, driver)
                save_type2_modified_pre_appendix_to_db(pre_appendix_text, heading, doc_number, driver)
            st.toast(f"Saved {upload_file.name} database✅")
        # Nếu không phải thông tư nghị định và cũng không phải luật, in ra lỗi
        else:
            raise ValueError("Loại văn bản không được hỗ trọ")

