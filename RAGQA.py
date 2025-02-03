import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from langchain.vectorstores import FAISS
from langchain.embeddings import GPT4AllEmbeddings
from typing import List, Dict, Tuple

vector_db_path = "vectorstores/db_faiss"

# Initialize the embedding model
embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")

# Load the FAISS vector database with the embedding model
db = FAISS.load_local(folder_path=vector_db_path, embeddings=embedding_model, allow_dangerous_deserialization = True)

# Perform a similarity search
def search_query(query: str, k: int = 30):
    """
    Perform a similarity search on the vector database.
    
    :param query: The query string.
    :param k: The number of top results to return.
    :return: The top results as a list of strings.
    """
    results = db.similarity_search(query, k=k)
    return [result.page_content for result in results]


from transformers import AutoTokenizer, pipeline, GenerationConfig
from ipex_llm.transformers import AutoModelForCausalLM
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import BitsAndBytesConfig

import torch
generation_config = GenerationConfig(use_cache=True)

# Define the path for the model
model_file = "thangvip/vwen2.5-1.5b-evol"
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
    model = AutoModelForCausalLM.from_pretrained(model_file, cpu_embedding=True, trust_remote_code=True, load_in_4bit = True)
    model = model.to('xpu')
    tokenizer = AutoTokenizer.from_pretrained(model_file, trust_remote_code=True)
    print('Successfully loaded Tokenizer and optimized Model!')
    return model, tokenizer

# Read the vector database (FAISS)
def read_vectors_db():
    embedding_model = GPT4AllEmbeddings(model_file="models/all-MiniLM-L6-v2-f16.gguf")
    db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    return db

# Perform similarity search on the vector database
def search_vector_db(query, k=2):
    db = read_vectors_db()
    results = db.similarity_search(query, k=k)
    return [result.page_content for result in results]
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





question = '''Hồ sơ đề nghị cấp chứng chỉ hành nghề dược gồm những nội dung nào, trả lời một cách chi tiết và đầy đủ nhất'''
context_list = search_vector_db(question, k = 2)
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
        temperature = 0.5,
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



