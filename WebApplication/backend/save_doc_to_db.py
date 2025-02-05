
import faiss
import pickle
import os
from icecream import ic

def save_to_db(new_texts, new_metadata, model):
    # Paths for FAISS index and metadata
    FAISS_INDEX_PATH = "db/faiss_index.bin"
    DATA_PATH = "db/data.pkl"
    METADATA_PATH = "db/metadata.pkl"

    # Load FAISS index and metadata if they exist, otherwise initialize
    def load_or_initialize_faiss():
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH) and os.path.exists(DATA_PATH):
            print("🔄 Loading existing FAISS index, data and metadata...")
            index = faiss.read_index(FAISS_INDEX_PATH)
            with open(METADATA_PATH, "rb") as f:
                metadata = pickle.load(f)
            with open(DATA_PATH, "rb") as f:
                data = pickle.load(f)
        else:
            print("🆕 Creating a new FAISS index...")
            index = None  # Initialize index as None (to be created later)
            metadata = []  # Empty metadata list
            data = []  # Empty metadata list
        return index, data, metadata

    # Save the FAISS index and metadata
    def save_faiss_and_metadata(index, data, metadata):
        faiss.write_index(index, FAISS_INDEX_PATH)
        with open(METADATA_PATH, "wb") as f:
            pickle.dump(metadata, f)
        with open(DATA_PATH, "wb") as f:
            pickle.dump(data, f)
        print("✅ FAISS index, data and metadata saved successfully!")

    # Load or initialize FAISS
    index, data, metadata = load_or_initialize_faiss()

    # Convert new documents to embeddings
    new_embeddings = model.encode(new_texts, convert_to_numpy=True)

    # If FAISS index does not exist, create it
    if index is None:
        embedding_size = new_embeddings.shape[1]  # Get the embedding dimension
        index = faiss.IndexFlatL2(embedding_size)  # Create FAISS index
        print(f"🛠️ Created FAISS index with dimension {embedding_size}")

    # Append new embeddings to FAISS index
    index.add(new_embeddings)

    # Append new data
    data.extend(new_texts)

    # Append new data
    metadata.extend(new_metadata)

    # Save the updated FAISS index and metadata
    save_faiss_and_metadata(index, data, metadata)

    # Verify update
    index, data, metadata = load_or_initialize_faiss()
    print(f"📌 Updated FAISS Index: {index.ntotal} entries")
    print(f"📌 Updated Data: {len(data)} entries")
    print(f"📌 Updated Metadata: {len(metadata)} entries")

