
import faiss
import pickle
import os
from icecream import ic
import base64
import hashlib


def save_to_db(new_texts, new_metadata, model):
    # Paths for FAISS index and metadata
    FAISS_INDEX_PATH = "db/faiss_index.bin"
    FAISSPATH_INDEX_PATH = "db/faiss_path_index.bin"
    DATA_PATH = "db/data.pkl"
    METADATA_PATH = "db/metadata.pkl"

    # Load FAISS index and metadata if they exist, otherwise initialize
    def load_or_initialize_faiss():
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH) and os.path.exists(DATA_PATH):
            print("🔄 Loading existing FAISS index, data and metadata...")
            index = faiss.read_index(FAISS_INDEX_PATH)
            path_index = faiss.read_index(FAISSPATH_INDEX_PATH)

            with open(METADATA_PATH, "rb") as f:
                metadata = pickle.load(f)
            with open(DATA_PATH, "rb") as f:
                data = pickle.load(f)
        else:
            print("🆕 Creating a new FAISS index...")
            index = None  # Initialize index as None (to be created later)
            path_index = None
            metadata = []  # Empty metadata list
            data = []  # Empty metadata list
        return index, path_index, data, metadata

    # Save the FAISS index and metadata
    def save_faiss_and_metadata(index, path_index, data, metadata):
        faiss.write_index(index, FAISS_INDEX_PATH)
        faiss.write_index(path_index, FAISSPATH_INDEX_PATH)

        with open(METADATA_PATH, "wb") as f:
            pickle.dump(metadata, f)
        with open(DATA_PATH, "wb") as f:
            pickle.dump(data, f)
        print("✅ FAISS index, data and metadata saved successfully!")

    # Load or initialize FAISS
    index, path_index, data, metadata = load_or_initialize_faiss()

    # Convert new documents to embeddings
    new_embeddings = model.encode(new_texts, convert_to_numpy=True)
    new_path_embeddings = model.encode([mtdata["path"] for mtdata in new_metadata], convert_to_numpy=True)


    # If FAISS index does not exist, create it
    if index is None:
        embedding_size = new_embeddings.shape[1]  # Get the embedding dimension
        index = faiss.IndexFlatL2(embedding_size)  # Create FAISS index
        path_embedding_size = new_path_embeddings.shape[1]  # Get the embedding dimension
        path_index = faiss.IndexFlatL2(path_embedding_size)  # Create FAISS index
        print(f"🛠️ Created FAISS index with dimension {embedding_size}")

    # Append new embeddings to FAISS index
    index.add(new_embeddings)

    path_index.add(new_path_embeddings)

    # Append new data
    data.extend(new_texts)


    # Add id to current new_metadata
    for i, mtdata in enumerate(new_metadata):
        # bytes_representation = mtdata["path"].encode(encoding="utf-8") 
        bytes_representation = str(mtdata["doc_id"] + mtdata["path"] + str(len(metadata) + i + 1)).encode("utf-8")
        hash_object = hashlib.sha256(bytes_representation)  # Use SHA-256 (or hashlib.md5 for a smaller hash)
        hash_int = int(hash_object.hexdigest(), 16) % (10**10)
        mtdata["id"] = hash_int
    # Append new data
    metadata.extend(new_metadata)

    # Save the updated FAISS index and metadata
    save_faiss_and_metadata(index, path_index, data, metadata)

    # Verify update
    index, path_index, data, metadata = load_or_initialize_faiss()
    print(f"📌 Updated FAISS Index: {index.ntotal} entries")
    print(f"📌 Updated Data: {len(data)} entries")
    print(f"📌 Updated Metadata: {len(metadata)} entries")

