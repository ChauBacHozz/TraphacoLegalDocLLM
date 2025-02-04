import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from langchain.vectorstores import FAISS
from langchain.embeddings import GPT4AllEmbeddings
from typing import List, Dict, Tuple

vector_db_path = "vectorstores/db_faiss"

# Initialize the embedding model
# embedding_model = GPT4AllEmbeddings("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
from langchain.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs = {"device": "cpu"},
    encode_kwargs = {"normalize_embeddings": True}
)
# Load the FAISS vector database with the embedding model
# db = FAISS.load_local(folder_path=vector_db_path, embeddings=embedding_model, allow_dangerous_deserialization = True)

# # Perform a similarity search
# def search_query(query: str, k: int = 30):
#     """
#     Perform a similarity search on the vector database.
    
#     :param query: The query string.
#     :param k: The number of top results to return.
#     :return: The top results as a list of strings.
#     """
#     results = db.similarity_search(query, k=k)
#     return [result.page_content for result in results]


from transformers import AutoTokenizer, pipeline, GenerationConfig
from ipex_llm.transformers import AutoModelForCausalLM
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import BitsAndBytesConfig
import numpy as np
from icecream import ic
# from sentence_transformers import SentenceTransformer

import faiss
import pickle
import torch
generation_config = GenerationConfig(use_cache=True)

# Define the path for the model
model_file = "Qwen/Qwen2-1.5B-Instruct"
# model_file = "vilm/vinallama-7b-chat"

# Define the FAISS vector store path
vector_db_path = "vectorstores/db_faiss"

# Load the Hugging Face model and tokenizer
def load_huggingface_model(model_file):
    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,  # Tải trọng số được lượng hóa trước theo định dạng 4 bit
    #     bnb_4bit_quant_type="nf4",  # Sử dụng loại lượng hóa "nf4" cho trọng số 4 bit
    #     bnb_4bit_compute_dtype=torch.bfloat16,  # Sử dụng torch.bfloat16 cho các phép tính trung gian
    #     bnb_4bit_use_double_quant=True,  # Sử dụng độ chính xác kép để lượng hóa kích hoạt
    # )
    # quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold = 6.0)
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-1.5B-Instruct",
                                             load_in_4bit=True,
                                             cpu_embedding=True,
                                             trust_remote_code=True)
    model = model.to('xpu')
    tokenizer = AutoTokenizer.from_pretrained(model_file, trust_remote_code=True)
    print('Successfully loaded Tokenizer and optimized Model!')
    return model, tokenizer

# Read the vector database (FAISS)
# def read_vectors_db():
#     embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
#     db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
#     return db

def load_faiss_and_metadata(index_path, metadata_path):
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

# Load index and metadata
index, loaded_metadata = load_faiss_and_metadata("faiss_index.bin", "metadata.pkl")
# sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("FAISS index and metadata loaded successfully!")
# Perform similarity search on the vector database
def query_faiss(query, top_k=5):
    query_embedding = embedding_model.embed_query(query)  # Fix: Use embed_query instead of encode
    query_embedding = np.array(query_embedding).reshape(1, -1)  
    ic(query_embedding)
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx != -1:
            metadata = loaded_metadata[idx]
            metadata['score'] = dist
            results.append(metadata)
    return results
# Load the model and tokenizer
model, tokenizer = load_huggingface_model(model_file)


system_prompt = "Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể."
template = '''Chú ý các yêu cầu sau:
- Câu trả lời phải chính xác và đầy đủ nếu ngữ cảnh có câu trả lời. 
- Chỉ sử dụng các thông tin có trong ngữ cảnh được cung cấp.
- Chỉ cần từ chối trả lời và không suy luận gì thêm nếu ngữ cảnh không có câu trả lời.
Hãy trả lời câu hỏi dựa trên ngữ cảnh:
### Ngữ cảnh :
{context}

### Câu hỏi :
{question}

### Trả lời :'''





question = '''Quy định chi tiết về hồ sơ đề nghị cấp Chứng chỉ hành nghề dược?'''
context_list = query_faiss(question, top_k = 3)
print("\n\n\nContext:")
print("\n\n")
ic(context_list)
context_list = [context["text"] for context in context_list]
context = "\n".join(context_list)
conversation = [{"role": "system", "content": system_prompt }]
conversation.append({"role": "user", "content": template.format(context = context, question = question)})

with torch.inference_mode():
    text = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True)
    model_inputs = tokenizer(text,return_tensors="pt").to('xpu')
    attention_mask = model_inputs["attention_mask"]
    _ = model.generate(model_inputs.input_ids,
                        do_sample=False,
                        max_new_tokens=32,
                        generation_config=generation_config) # warm-up
    generated_ids = model.generate(
        model_inputs.input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1000,
        generation_config=generation_config,
        temperature = 0.2,
        #top_p=0.95,
        #top_k=40,
    ).cpu()
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)
# import textwrap

# # Example text (you can replace this with your query result)
# text = response

# # Function to wrap text
# def wrap_text(text, width=80):
#     wrapper = textwrap.TextWrapper(width=width, replace_whitespace=False)
#     wrapped_text = "\n".join([wrapper.fill(line) for line in text.splitlines()])
#     return wrapped_text                                                                                                                                                                                            

# # Wrap and print the text
# wrapped_text = wrap_text(text, width=140)
# print(wrapped_text)



