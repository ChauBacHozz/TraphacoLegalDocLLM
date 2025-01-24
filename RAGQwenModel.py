from langchain.vectorstores import FAISS
from langchain.embeddings import GPT4AllEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import BitsAndBytesConfig

import torch


class RAGQwen():
    def __init__(self, vector_db_path = "vectorstores/db_faiss", 
                 embedding_model_file = "models/all-MiniLM-L6-v2-f16.gguf",
                 model_file = "AITeamVN/Vi-Qwen2-7B-RAG",
                 ):
        self.vector_db_path = vector_db_path

        # Initialize the embedding model
        self.embedding_model = GPT4AllEmbeddings(model_file=embedding_model_file)

        # Load the FAISS vector database with the embedding model
        self.db = FAISS.load_local(folder_path=vector_db_path, embeddings=self.embedding_model, allow_dangerous_deserialization = True)


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
    def search_query(self, query: str, k: int = 30):
        """
        Perform a similarity search on the vector database.
        
        :param query: The query string.
        :param k: The number of top results to return.
        :return: The top results as a list of strings.
        """
        results = self.db.similarity_search(query, k=k)
        return [result.page_content for result in results]
    
    def load_huggingface_model(self,model_file):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Tải trọng số được lượng hóa trước theo định dạng 4 bit
            bnb_4bit_quant_type="nf4",  # Sử dụng loại lượng hóa "nf4" cho trọng số 4 bit
            bnb_4bit_compute_dtype=torch.bfloat16,  # Sử dụng torch.bfloat16 cho các phép tính trung gian
            bnb_4bit_use_double_quant=True,  # Sử dụng độ chính xác kép để lượng hóa kích hoạt
        )
        # quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold = 6.0)
        model = AutoModelForCausalLM.from_pretrained(model_file, device_map="auto", quantization_config=quantization_config,)
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