from langchain.embeddings import GPT4AllEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.vectorstores import FAISS
from transformers import BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.preprocessing import MinMaxScaler
from rank_bm25 import BM25Okapi
import torch
import faiss
import numpy as np
import pickle
from langchain_community.vectorstores import Neo4jVector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores.utils import maximal_marginal_relevance
from neo4j import GraphDatabase
from icecream import ic
from ordered_set import OrderedSet
from collections import OrderedDict
import os
from pyvi import ViTokenizer
from transformers import pipeline, set_seed

os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"

class RetrievalData():
    def __init__(self,embedding_model = "dangvantuan/vietnamese-document-embedding",
                 rerank_model_id = 'itdainb/PhoRanker'
                 ):
        
        # Initialize the embedding model
        # if embedding_model == None:
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model, 
            model_kwargs={"trust_remote_code": True, "device": "cpu"},)
        
        
        self.rerank_model = CrossEncoder(rerank_model_id, max_length=4000, device="cpu")
        self.rerank_model.to("cpu")

        URI = "neo4j+s://13d9b8ff.databases.neo4j.io"
        USERNAME = "neo4j"
        PASSWORD = "tDJXOWtq9GSTnXqQyVFmb2xiR3GREbxnU8m9MxxWHwU"
        self.driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

    def set_control_params(self, max_new_tokens, temperature, top_p, top_k):
        self.max_new_tokens=max_new_tokens
        self.temperature = temperature
        self.top_p=top_p
        self.top_k=top_k
        

    def load_faiss_and_data(self, index_path, path_index_path, data_path, metadata_path):
        index = faiss.read_index(index_path)
        path_index = faiss.read_index(path_index_path)
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        with open(metadata_path, "rb") as f:
            meta_data = pickle.load(f)
        return index, path_index, data, meta_data
    
    def get_model_ready(self):
        self.paper_store = Neo4jVector.from_existing_index(
            embedding=self.embedding_model,
            url="neo4j+s://13d9b8ff.databases.neo4j.io",
            username="neo4j",
            password="tDJXOWtq9GSTnXqQyVFmb2xiR3GREbxnU8m9MxxWHwU",
            index_name="doc_index",
            text_node_property="content"
        )     


    def search_query_from_path(self, query: str, k = 6):
        """
        Perform a similarity search on the vector database.
        
        :param query: The query string.
        :param k: The number of top results to return.
        :return: The top results as a list of strings.
        """
        
        vector_results = self.paper_store.similarity_search_with_score(query, k=k)

        # BM25 Search (Full-Text Index)
        keyword_query = f"""
            CALL db.index.fulltext.queryNodes("full_doc_index", "{query}") 
            YIELD node, score 
            RETURN node.content AS content, node.d_id AS d_id, node.path AS path
            LIMIT {k}
        """
        keyword_results = self.paper_store.query(keyword_query)

        # Convert results to LangChain Document objects
        vector_documents = [doc for doc, _ in vector_results]
        vector_embeddings = np.array([self.embedding_model.embed_query(doc.page_content) for doc in vector_documents])

        keyword_documents = [
            Document(page_content=doc["content"], metadata={"d_id": doc["d_id"], "path": doc["path"]}) 
            for doc in keyword_results
        ]

        # Convert query embedding to NumPy array
        query_embedding = np.array(self.embedding_model.embed_query(query))

        # Merge using MMR
        hybrid_indices = maximal_marginal_relevance(
            query_embedding=query_embedding,
            embedding_list=vector_embeddings,  # Only embeddings, no documents
            lambda_mult=0.9
        )

        # Return re-ranked documents based on MMR indices
        hybrid_results = [vector_documents[i] for i in hybrid_indices]
        final_passages = hybrid_results + keyword_documents

        # Thu gọn các passage bị trùng
        final_dict = {}
        for doc in final_passages:
            key = doc.metadata["d_id"] + " | " + (doc.metadata["path"] or "")
            final_dict[key.strip()] = doc.page_content
        # Sắp xếp theo key thứ tự alphabet
        final_dict = {k: final_dict[k] for k in sorted(final_dict)}
        # ic(final_dict)
        shorten_final_dict = {}
        # Kiểm tra các key trong final dict, nếu có key nào mà key trước đó thuộc key đó thì sẽ lấy key trước đó (cha)
        final_dict_keys_lst = list(final_dict.keys())
        shorten_final_dict[final_dict_keys_lst[0]] = final_dict[final_dict_keys_lst[0]]
        iter = 0
        for i in range(1, len(final_dict_keys_lst)):
            if final_dict_keys_lst[iter].strip() in final_dict_keys_lst[i]:
                continue
            shorten_final_dict[final_dict_keys_lst[i]] = final_dict[final_dict_keys_lst[i]]
            iter = i

        # Thực hiện rerank
        tokenized_query = ViTokenizer.tokenize(query)
        tokenized_sentences = [ViTokenizer.tokenize(sent) for sent in shorten_final_dict.values()]

        tokenized_pairs = [[tokenized_query, sent] for sent in tokenized_sentences]
        scores = self.rerank_model.predict(tokenized_pairs)
        scores_dict = {}

        for i, key in enumerate(shorten_final_dict.keys()):
            scores_dict[key] = scores[i]

        shorten_final_dict = OrderedDict(shorten_final_dict)
        shorten_final_dict = OrderedDict(sorted(shorten_final_dict.items(), reverse=True, key=lambda x: scores_dict[x[0]]))
        # Làm giàu thông tin retrieval data
        def get_sub_nodes(tx, doc_id, path):
            query_sub_info = """ MATCH (n:Doc_Node {d_id: $d_id})
                                WHERE n.path STARTS WITH $path 
                                RETURN n ORDER BY n.path ASC
                             """
            result = tx.run(query_sub_info, d_id = doc_id, path = path)
            result = list(result)
            return [Document(page_content=doc["n"]["content"], metadata={"d_id": doc["n"]["d_id"], "path": doc["n"]["path"], "bullet_type": doc["n"]["bullet_type"]}) for doc in result if doc["n"]["path"] != path]
        
        def get_modified_nodes(tx, doc_id, content):
            query = """ 
            MATCH (modifier:Modified_Node)-[:MODIFIED]->(x:Origin_Node {d_id: $d_id, content: $content})
            RETURN modifier
            """
            result = tx.run(query, d_id = doc_id, content = content)
            return [record["modifier"] for record in result] or []  # Ensure it returns an empty list

        def get_modified_path(tx, doc_id, id):
            query = """
            MATCH path = (root:R_Node:Modified_Node {d_id: $d_id})-[:CONTAIN*]->(t:Modified_Node {d_id: $d_id, id: $id})
            WHERE NOT (root)<-[]-()
            UNWIND nodes(path) AS node
            WITH node, head(nodes(path)) AS root_node 
            WHERE node <> root_node  
            RETURN DISTINCT node
            """
            result = tx.run(query, d_id = doc_id, id = id)
            path = [record["node"] for record in result]
            return path
        # Hồ sơ đề nghị điều chỉnh nội dung Chứng chỉ hành nghề dược gồm những gì?
        def get_modified_sub_nodes(tx, doc_id, content, bullet_type, bullet):
            query = """
            MATCH (b:Doc_Node:Modified_Node {d_id: $d_id, content: $content, bullet_type: $bullet_type, bullet: $bullet})-[:CONTAIN*1..]->(subnodes)
            RETURN subnodes ORDER BY elementId(subnodes)
            """
            result = tx.run(query, d_id = doc_id, content = content, bullet_type = bullet_type, bullet = bullet)
            subnodes = [record["subnodes"] for record in result]
            return subnodes
        
        origin_results = []
        origin_results.append("Nội dung gốc:")
        modified_results = OrderedSet()
        modified_results.add("Nội dung sửa đổi, bãi bỏ, bổ sung:")
        # ic(shorten_final_dict)
        for key, val in shorten_final_dict.items():
            doc_id = key.split(" | ")[0]
            path = key.split(" | ")[1]
            origin_results.append(str(doc_id + " " + path + " | " + val))
            with self.driver.session() as session:
                modified_nodes = session.read_transaction(get_modified_nodes, doc_id, val)
                for modified_node in modified_nodes:
                    modified_results.add(modified_node["d_id"] + " " + modified_node["bullet_type"] + " " + modified_node["bullet"] + " | " + modified_node["modified_purpose"] + " nội dung thuộc văn bản " + doc_id + " như sau " + modified_node["content"])
                    modified_sub_nodes = session.read_transaction(get_modified_sub_nodes, modified_node["d_id"], modified_node["content"], modified_node["bullet_type"], modified_node["bullet"])
                    for modified_sub_node in modified_sub_nodes:
                        modified_results.add(modified_sub_node["content"])
                    m_paths = session.read_transaction(get_modified_path, modified_node["d_id"], modified_node["id"])
                    m_path = OrderedSet()
                    for p in m_paths:
                        m_path.add(p["bullet_type"] + " " + p["bullet"])
                    m_path = " ".join(list(m_path))
                    origin_results[-1] = origin_results[-1].rstrip(".;")
                    if origin_results[-1] != ":":
                        origin_results[-1] += ";"
                    origin_results[-1] = origin_results[-1] + " (Được " + modified_node["modified_purpose"] + " ở " + m_path + " thuộc văn bản " + modified_node["d_id"] + ");"
            #     final_results.append(modified_nodes)
            if len(path) > 0:
                # Get sub nodes
                with self.driver.session() as session:
                    nodes_list = session.read_transaction(get_sub_nodes, doc_id, path)
                    for node in nodes_list:
                        if node.metadata["bullet_type"] in node.metadata["path"].split(" > ")[-1].split(" ")[0]:
                            origin_results[-1] = origin_results[-1] + "\n" + node.metadata["path"].split(" > ")[-1].split(" ")[0] + " " + node.page_content.strip()
                        else:
                            origin_results[-1] = origin_results[-1] + "\n" + node.metadata["bullet_type"] + " " + node.page_content.strip()

                        origin_results[-1] = origin_results[-1].rstrip(".;")
                        if origin_results[-1] != ":":
                            origin_results[-1] += ";"
                        modified_nodes = session.read_transaction(get_modified_nodes, node.metadata["d_id"], node.page_content)
                        for modified_node in modified_nodes:
                            modified_results.add(modified_node["d_id"] + " " + modified_node["bullet_type"] + " " + modified_node["bullet"] + " | " + modified_node["modified_purpose"] + " nội dung thuộc văn bản " + doc_id + " như sau " + modified_node["content"])
                            modified_sub_nodes = session.read_transaction(get_modified_sub_nodes, modified_node["d_id"], modified_node["content"], modified_node["bullet_type"], modified_node["bullet"])
                            for modified_sub_node in modified_sub_nodes:
                                modified_results.add(modified_sub_node["content"])
                            m_paths = session.read_transaction(get_modified_path, modified_node["d_id"], modified_node["id"])
                            m_path = OrderedSet()
                            for p in m_paths:
                                m_path.add(p["bullet_type"] + " " + p["bullet"])
                            m_path = " ".join(list(m_path))
                            origin_results[-1] = origin_results[-1] + " (Được " + modified_node["modified_purpose"] + " ở " + m_path + " thuộc văn bản " + modified_node["d_id"] + ");"
                        # final_results.append(modified_nodes)
                    # for node in nodes_list:
                    #     final_results.append(node.metadata["d_id"] + " " + node.metadata["path"] + " | " + node.page_content.strip())
        modified_results = list(modified_results)
        return origin_results, modified_results
                        
    
    def get_retrieval_data(self, query: str):
        # CHECK IF QUERY IS A HEADER OR NOT
        origin_context, modified_context = self.search_query_from_path(query)
        return origin_context, modified_context
    


    
    def rag_answer(self, prompt):
        origin_context, modified_context = self.get_retrieval_data(prompt)
        # origin_context.insert(0, "Nội dung gốc")
        # modified_context.insert(0, "Nội dung sửa đổi, bãi bỏ, bổ sung")
        context_list = origin_context + modified_context
        # n_tokens = 0
        # for context in context_list:
        #     n_tokens += self.count_tokens_underthesea(context)

        context = "\n".join(context_list)
        # print(f"😄 there are {n_tokens} tokens in context")
        return context

