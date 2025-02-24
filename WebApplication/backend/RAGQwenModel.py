from langchain.vectorstores import FAISS
from langchain.embeddings import GPT4AllEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
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
from underthesea import word_tokenize
from langchain_community.vectorstores import Neo4jVector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores.utils import maximal_marginal_relevance
from neo4j import GraphDatabase
from icecream import ic
# PATH = 'D:/VS_Workspace/LLM/.cache'
# os.environ['TRANSFORMERS_CACHE'] = PATH
# os.environ['HF_HOME'] = PATH
# os.environ['HF_DATASETS_CACHE'] = PATH
# os.environ['TORCH_HOME'] = PATH

class RAGQwen():
    def __init__(self, vector_db_path = "vectorstores/db_faiss", 
                 embedding_model = None,
                 model_file = "AITeamVN/Vi-Qwen2-3B-RAG",
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


        self.system_prompt = "Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể."
        self.template = '''Chú ý các yêu cầu sau:
        - Câu trả lời phải chính xác và đầy đủ nếu ngữ cảnh có câu trả lời. 
        - Chỉ sử dụng các thông tin có trong ngữ cảnh được cung cấp.
        - Chỉ cần từ chối trả lời và không suy luận gì thêm nếu ngữ cảnh không có câu trả lời.
        - Nếu nhiều nội dung được lấy từ cùng 1 khoản trong tài liệu đã cho, trả về toàn bộ nội dung trong khoản đó một cách chính xác nhất, không thực hiện tóm tắt lại.
        Hãy trả lời câu hỏi dựa trên ngữ cảnh:
        ### Ngữ cảnh :
        {context}

        ### Câu hỏi :
        {question}

        ### Trả lời :'''

        # Khởi tạo mô hình LLM và tokenizer
        self.model, self.tokenizer = self.load_huggingface_model(self.model_file)
        WINDOWS_IP = "28.11.5.39"
        URI = f"bolt://{WINDOWS_IP}:7687"
        USERNAME = "neo4j"
        PASSWORD = "phongthang2012"

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
            url="bolt://28.11.5.39:7687",
            username="neo4j",
            password="phongthang2012",
            index_name="doc_index",
            text_node_property="content"
        )     

        
    def count_tokens_underthesea(self, text):
        tokens = word_tokenize(text, format="text").split()
        return len(tokens)

    def search_query_from_path(self, query: str, k =10):
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


        # Làm giàu thông tin retrieval data
        def get_sub_info(tx, doc_id, path):
            query_sub_info = """ MATCH (n:Doc_Node {d_id: $d_id})
                                WHERE n.path STARTS WITH $path 
                                RETURN n ORDER BY elementId(n)
                             """
            result = tx.run(query_sub_info, d_id = doc_id, path = path)
            return [record["n"] for record in result]
        
        final_results = []
        for passage in final_passages:
            doc_id = passage.metadata["d_id"]
            path = passage.metadata["path"]
            if path:
                with self.driver.session() as session:
                    nodes_list = session.read_transaction(get_sub_info, doc_id, path)
                    ic(nodes_list)



        final_passages_full = []
        final_passages_path = []
        for passage in final_passages:
            if "path" in passage.metadata.keys():
                if passage.metadata["path"]:
                    path = passage.metadata["path"]
                else:
                    path = ""
                final_passages_full.append(str(passage.metadata["d_id"]) + path + passage.page_content.strip())
                final_passages_path.append(str(passage.metadata["d_id"]) + path)
            else:
                final_passages_full.append(str(passage.metadata["d_id"]) + passage.page_content.strip())
                final_passages_path.append(str(passage.metadata["d_id"]))
        return final_passages_full, final_passages_path # Combine with keyword-based retrieval



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
        model = AutoModelForCausalLM.from_pretrained(model_file, device_map="auto", quantization_config=quantization_config)
        tokenizer = AutoTokenizer.from_pretrained(model_file)
        return model, tokenizer

    # # Read the vector database (FAISS)
    # def read_vectors_db(self):
    #     embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
    #     db = FAISS.load_local(self.vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    #     return db

    # # Perform similarity search on the vector database
    # def search_vector_db(self,query, k=2):
    #     db = self.read_vectors_db()
    #     results = db.similarity_search(query, k=k)
    #     return [result.page_content for result in results]
    
    def rag_answer(self, prompt):
        context_list, _ = self.get_retrieval_data(prompt)
        n_tokens = 0
        for context in context_list:
            n_tokens += self.count_tokens_underthesea(context)
        print(f"😄 there are {n_tokens} tokens in context")
        context = "\n".join(context_list)
        # print("\n\n\nCONTEXT:", context)
        # print("\n\n")
        conversation = [{"role": "system", "content": self.system_prompt }]
        conversation.append({"role": "user", "content": self.template.format(context = context, question = prompt)})
        with torch.inference_mode():
            text = self.tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True)
            model_inputs = self.tokenizer(text,return_tensors="pt").to(self.model.device)

            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=2048,
                temperature = 0.1,
                top_p=0.95,
                top_k=40,
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response