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

PATH = 'D:/VS_Workspace/LLM/.cache'
os.environ['TRANSFORMERS_CACHE'] = PATH
os.environ['HF_HOME'] = PATH
os.environ['HF_DATASETS_CACHE'] = PATH
os.environ['TORCH_HOME'] = PATH

class RAGQwen():
    def __init__(self, vector_db_path = "vectorstores/db_faiss", 
                 embedding_model = None,
                 model_file = "AITeamVN/Vi-Qwen2-3B-RAG",
                 ):
        
        self.vector_db_path = vector_db_path
        self.model_file = model_file
        # Initialize the embedding model
        # if embedding_model == None:
        self.embedding_model = SentenceTransformer('dangvantuan/vietnamese-document-embedding', trust_remote_code=True)
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
        # self.model, self.tokenizer = self.load_huggingface_model(self.model_file)
    def load_faiss_and_data(self, index_path, data_path, metadata_path):
        index = faiss.read_index(index_path)
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        with open(metadata_path, "rb") as f:
            meta_data = pickle.load(f)
        return index, data, meta_data
    def get_model_ready(self):
        self.index, self.loaded_data, self.loaded_metadata = self.load_faiss_and_data("db/faiss_index.bin", "db/data.pkl", "db/metadata.pkl")
        self.texts = [data for data in self.loaded_data]
        self.tokenized_docs = [doc.split() for doc in self.texts]
    def count_tokens_underthesea(self, text):
        tokens = word_tokenize(text, format="text").split()
        return len(tokens)
    def search_query(self, query: str):
        """
        Perform a similarity search on the vector database.
        
        :param query: The query string.
        :param k: The number of top results to return.
        :return: The top results as a list of strings.
        """
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        bm25 = BM25Okapi(self.tokenized_docs)
        
        n_contexts = 6
        # Dense Retrieval (FAISS)
        D, I = self.index.search(query_embedding, k=n_contexts)  # Retrieve top-3 similar docs
        dense_results = [self.texts[i] for i in I[0]]
        dense_scores = D[0]

        # Sparse Retrieval (BM25)
        query_tokens = query.split()
        bm25_scores = bm25.get_scores(query_tokens)
        top_k_bm25 = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:n_contexts]
        sparse_results = [self.texts[i] for i in top_k_bm25]
        sparse_scores = [bm25_scores[i] for i in top_k_bm25]


        scaler = MinMaxScaler()

        # Normalize FAISS scores
        dense_scores = np.array(dense_scores).reshape(-1, 1)
        sparse_scores = np.array(sparse_scores).reshape(-1, 1)

        dense_scores = scaler.fit_transform(dense_scores).flatten()
        sparse_scores = scaler.fit_transform(sparse_scores).flatten()

        # Weighted Hybrid Scoring (Adjust Weights as Needed)
        hybrid_results = []
        for i, doc in enumerate(dense_results):
            hybrid_results.append((0.6 * dense_scores[i], doc))  # 60% weight to Dense
            
        for i, doc in enumerate(sparse_results):
            hybrid_results.append((0.4 * sparse_scores[i], doc))  # 40% weight to Sparse

        # Sort by Final Score
        hybrid_results.sort(reverse=True, key=lambda x: x[0])
        final_passages = [doc for _, doc in hybrid_results]
        return final_passages
    
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

    # Read the vector database (FAISS)
    def read_vectors_db(self):
        embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
        db = FAISS.load_local(self.vector_db_path, embedding_model, allow_dangerous_deserialization=True)
        return db

    # Perform similarity search on the vector database
    def search_vector_db(self,query, k=2):
        db = self.read_vectors_db()
        results = db.similarity_search(query, k=k)
        return [result.page_content for result in results]
    
    def rag_answer(self, prompt):
        context_list = self.search_query(prompt)
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