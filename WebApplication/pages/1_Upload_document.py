import streamlit as st
import time
import numpy as np
import docx

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
    text = [] # create a text variable
    for para in doc_file.paragraphs: # loop through the paragraphs of your .docx file
        text.append(para.text)
    st.write(text)