from langchain.vectorstores import FAISS
from langchain.embeddings import GPT4AllEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoProcessor, Gemma3ForConditionalGeneration
from transformers import AutoConfig
from transformers import pipeline
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
from rank_bm25 import BM25Okapi
import torch
import faiss
import numpy as np
import pickle
import os
# from underthesea import word_tokenize
from langchain_community.vectorstores import Neo4jVector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores.utils import maximal_marginal_relevance
from neo4j import GraphDatabase
from icecream import ic
from ordered_set import OrderedSet
import os
os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"
# PATH = 'D:/VS_Workspace/LLM/.cache'
# os.environ['TRANSFORMERS_CACHE'] = PATH
# os.environ['HF_HOME'] = PATH
# os.environ['HF_DATASETS_CACHE'] = PATH
# os.environ['TORCH_HOME'] = PATH

class RAGGemma():
    def __init__(self, vector_db_path = "vectorstores/db_faiss", 
                 embedding_model = None,
                 model_file = "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
                 ):
        
        self.vector_db_path = vector_db_path
        self.model_file = model_file
        # Initialize the embedding model
        # if embedding_model == None:
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="dangvantuan/vietnamese-document-embedding", 
            model_kwargs={"trust_remote_code": True},)
        # else:
        #     print("Founded existing embedding model")
        #     self.embedding_model = embedding_model


        # Load the FAISS vector database with the embedding model
        # self.db = FAISS.load_local(folder_path=vector_db_path, embeddings=self.embedding_model, allow_dangerous_deserialization = True)

        self.template = '''Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể. Hãy tuân thủ nghiêm ngặt các nguyên tắc sau:

        - Chỉ trả lời dựa trên thông tin có trong ngữ cảnh được cung cấp ({context}), không sử dụng bất kỳ thông tin nào ngoài ngữ cảnh.
        - Phải nêu rõ câu trả lời được lấy từ nội dung của văn bản nào, đề mục như thế nào.
        - Nếu ngữ cảnh chứa câu trả lời, hãy cung cấp câu trả lời chính xác, đầy đủ, bao gồm toàn bộ nội dung liên quan từ ngữ cảnh (văn bản, đề mục, và các chi tiết cụ thể), không bỏ sót thông tin quan trọng.
        - Trích dẫn đầy đủ và chính xác các văn bản, điều khoản, khoản, hoặc đề mục được nêu trong ngữ cảnh để tránh thiếu sót.
        - Nếu ngữ cảnh không chứa câu trả lời, chỉ từ chối trả lời bằng cách nêu rõ không có thông tin, không suy luận hay bổ sung thêm.

        Hãy trả lời câu hỏi sau dựa trên ngữ cảnh:

        ### Ngữ cảnh :
        {context}

        ### Câu hỏi :
        {question}? Các nội dung này có bị sửa đổi, bãi bỏ, thêm không? Nếu có thì chỉ rõ văn bản nào, đề mục cụ thể?

        ### Trả lời :
        - Nếu có thông tin: Dựa trên ngữ cảnh được cung cấp, {question} [đã bị sửa đổi/bãi bỏ/được thêm] bởi [văn bản cụ thể], tại [đề mục cụ thể], với nội dung chi tiết như sau: [trích dẫn đầy đủ nội dung liên quan từ ngữ cảnh].
        - Nếu không có thông tin: Dựa trên ngữ cảnh được cung cấp, không có thông tin về việc {question} bị sửa đổi, bãi bỏ hay được thêm.'''

        # Khởi tạo mô hình LLM và tokenizer
        self.model, self.tokenizer = self.load_huggingface_model(self.model_file)
        WINDOWS_IP = "28.11.5.39"
        URI = "neo4j+s://13d9b8ff.databases.neo4j.io"
        USERNAME = "neo4j"
        PASSWORD = "tDJXOWtq9GSTnXqQyVFmb2xiR3GREbxnU8m9MxxWHwU"

        self.driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))
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

        
    # def count_tokens_underthesea(self, text):
    #     tokens = word_tokenize(text, format="text").split()
    #     return len(tokens)

    def search_query_from_path(self, query: str, k = 3):
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
            final_dict[key] = doc.page_content
        # Sắp xếp theo key thứ tự alphabet
        final_dict = {k: final_dict[k] for k in sorted(final_dict)}
        # ic(final_dict)
        shorten_final_dict = {}
        # Kiểm tra các key trong final dict, nếu có key nào mà key trước đó thuộc key đó thì sẽ lấy key trước đó (cha)
        final_dict_keys_lst = list(final_dict.keys())
        shorten_final_dict[final_dict_keys_lst[0]] = final_dict[final_dict_keys_lst[0]]
        for i in range(1, len(final_dict_keys_lst)):
            if final_dict_keys_lst[i-1] in final_dict_keys_lst[i]:
                continue
            shorten_final_dict[final_dict_keys_lst[i]] = final_dict[final_dict_keys_lst[i]]
        # Làm giàu thông tin retrieval data
        def get_sub_nodes(tx, doc_id, path):
            query_sub_info = """ MATCH (n:Doc_Node {d_id: $d_id})
                                WHERE n.path STARTS WITH $path 
                                RETURN n ORDER BY elementId(n)
                             """
            result = tx.run(query_sub_info, d_id = doc_id, path = path)
            result = list(result)
            return [Document(page_content=doc["n"]["content"], metadata={"d_id": doc["n"]["d_id"], "path": doc["n"]["path"]}) for doc in result if doc["n"]["path"] != path]
        
        def get_sub_nodes_lv1(tx, doc_id, path):
            query_sub_info = """ MATCH (n:Doc_Node:Origin_Node {d_id: $d_id, path: $path})-[:CONTAIN]->(m:Doc_Node:Origin_Node {d_id: $d_id})
                                RETURN m ORDER BY elementId(m)
                             """
            result = tx.run(query_sub_info, d_id = doc_id, path = path)
            result = list(result)
            return [Document(page_content=doc["m"]["content"], metadata={"d_id": doc["m"]["d_id"], "path": doc["m"]["path"]}) for doc in result if doc["m"]["path"] != path]
        
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
                    m_paths = session.read_transaction(get_modified_path, modified_node["d_id"], modified_node["id"])
                    m_path = OrderedSet()
                    for p in m_paths:
                        m_path.add(p["bullet_type"] + " " + p["bullet"])
                    m_path = " ".join(list(m_path))
                    origin_results[-1] = origin_results[-1] + " (Được " + modified_node["modified_purpose"] + " ở " + m_path + " thuộc văn bản " + modified_node["d_id"] + ")."
            #     final_results.append(modified_nodes)
            if len(path) > 0:
                # Get sub nodes
                with self.driver.session() as session:
                    nodes_list = session.read_transaction(get_sub_nodes, doc_id, path)
                    for node in nodes_list:
                        origin_results[-1] = origin_results[-1] + "\n" + node.metadata["path"].split(" > ")[-1].split(" ")[0] + " " + node.page_content.strip()
                        modified_nodes = session.read_transaction(get_modified_nodes, node.metadata["d_id"], node.page_content)
                        for modified_node in modified_nodes:
                            modified_results.add(modified_node["d_id"] + " " + modified_node["bullet_type"] + " " + modified_node["bullet"] + " | " + modified_node["modified_purpose"]  + " nội dung thuộc văn bản " + doc_id  +  " như sau " + modified_node["content"])
                            m_paths = session.read_transaction(get_modified_path, modified_node["d_id"], modified_node["id"])
                            m_path = OrderedSet()
                            for p in m_paths:
                                m_path.add(p["bullet_type"] + " " + p["bullet"])
                            m_path = " ".join(list(m_path))
                            origin_results[-1] = origin_results[-1] + " (Được " + modified_node["modified_purpose"] + " ở " + m_path + " thuộc văn bản " + modified_node["d_id"] + ")."
                        # final_results.append(modified_nodes)
                    # for node in nodes_list:
                    #     final_results.append(node.metadata["d_id"] + " " + node.metadata["path"] + " | " + node.page_content.strip())


        # ic(final_results)
        modified_results = list(modified_results)
        return origin_results, modified_results
                        



        # final_passages_full = []
        # final_passages_path = []
        # for passage in final_results:
        #     if "path" in passage.metadata.keys():
        #         if passage.metadata["path"]:
        #             path = passage.metadata["path"]
        #         else:
        #             path = ""
        #         final_passages_full.append(str(passage.metadata["d_id"]) + path + passage.page_content.strip())
        #         final_passages_path.append(str(passage.metadata["d_id"]) + path)
        #     else:
        #         final_passages_full.append(str(passage.metadata["d_id"]) + passage.page_content.strip())
        #         final_passages_path.append(str(passage.metadata["d_id"]))
        # return final_passages_full, final_passages_path # Combine with keyword-based retrieval



        final_passages = [ doc for score, doc in hybrid_results]
        return final_passages
    
    def get_retrieval_data(self, query: str):
        # CHECK IF QUERY IS A HEADER OR NOT
        res = self.search_query_from_path(query)
        return res
    
    def load_huggingface_model(self,model_file):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Tải trọng số được lượng hóa trước theo định dạng 4 bit
            bnb_4bit_quant_type="nf4",  # Sử dụng loại lượng hóa "nf4" cho trọng số 4 bit
            bnb_4bit_compute_dtype=torch.bfloat16,  # Sử dụng torch.bfloat16 cho các phép tính trung gian
            bnb_4bit_use_double_quant=True,  # Sử dụng độ chính xác kép để lượng hóa kích hoạt
        )
        # quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold = 6.0)
        # model = Gemma3ForConditionalGeneration.from_pretrained(model_file, device_map="auto")
        # config = AutoConfig.from_pretrained(model_file)
        # for key, value in vars(config.text_config).items():
        #     setattr(config, key, value)
        
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_file, device_map="auto"
        ).eval()
        tokenizer = AutoProcessor.from_pretrained(model_file)
        return model, tokenizer

    
    def rag_answer(self, prompt):
        origin_context, modified_context = self.get_retrieval_data(prompt)
        # origin_context.insert(0, "Nội dung gốc")
        # modified_context.insert(0, "Nội dung sửa đổi, bãi bỏ, bổ sung")
        context_list = origin_context + modified_context
        # n_tokens = 0
        # for context in context_list:
        #     n_tokens += self.count_tokens_underthesea(context)
        # print(f"😄 there are {n_tokens} tokens in context")
        context = "\n".join(context_list)
        # print("\n\n\nCONTEXT:", context)
        # print("\n\n"), 
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": '''Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể. Hãy tuân thủ nghiêm ngặt các nguyên tắc sau:

        - Chỉ trả lời dựa trên thông tin có trong ngữ cảnh được cung cấp ({context}), không sử dụng bất kỳ thông tin nào ngoài ngữ cảnh.
        - Phải nêu rõ câu trả lời được lấy từ nội dung của văn bản nào, đề mục như thế nào.
        - Nếu ngữ cảnh chứa câu trả lời, hãy cung cấp câu trả lời chính xác, đầy đủ, bao gồm toàn bộ nội dung liên quan từ ngữ cảnh (văn bản, đề mục, và các chi tiết cụ thể), không bỏ sót thông tin quan trọng.
        - Trích dẫn đầy đủ và chính xác các văn bản, điều khoản, khoản, hoặc đề mục được nêu trong ngữ cảnh để tránh thiếu sót.
        - Nếu ngữ cảnh không chứa câu trả lời, chỉ từ chối trả lời bằng cách nêu rõ không có thông tin, không suy luận hay bổ sung thêm.'''}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Ngữ cảnh:" + context},
                    {"type": "text", "text": "Dựa trên ngữ cảnh đã cho, hãy trả lời câu hỏi sau: "+ prompt}
                ]
            }
        ]
        
        # Get tokenized inputs and move to device directly
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(self.model.device, dtype=torch.bfloat16)
        
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=2040, do_sample=False)
            generation = generation[0][input_len:]
        
        decoded = self.tokenizer.decode(generation, skip_special_tokens=True)
        print("Decoded:", decoded)
        return decoded