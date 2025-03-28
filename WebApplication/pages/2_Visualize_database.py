import streamlit as st
import faiss
import os
import pickle
from icecream import ic
import pandas as pd
driver = st.session_state.driver

st.set_page_config(page_title="Upload document", page_icon="📈", layout="wide")

st.markdown("# Database")
st.sidebar.header("Visualize your document")

def get_root_nodes(tx):
    query = "MATCH (n:R_Node) RETURN n"
    results = tx.run(query)
    return [res["n"] for res in results]


with driver.session() as session:
    res_lst = []
    r_nodes = session.execute_read(get_root_nodes)
    for r_node in r_nodes:
        if r_node['content']:
            res_lst.append({'Document id':r_node['d_id'], 'Heading':r_node['content']})
st.dataframe(pd.DataFrame(res_lst))
