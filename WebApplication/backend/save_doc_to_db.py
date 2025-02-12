
import faiss
import pickle
import os
from icecream import ic
import base64
import hashlib


def save_to_db(new_texts, new_metadata, driver):
    # Paths for FAISS index and metadata
    # FAISS_INDEX_PATH = "db/faiss_index.bin"
    # FAISSPATH_INDEX_PATH = "db/faiss_path_index.bin"
    # DATA_PATH = "db/data.pkl"
    # METADATA_PATH = "db/metadata.pkl"

    # # Load FAISS index and metadata if they exist, otherwise initialize
    # def load_or_initialize_faiss():
    #     if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(METADATA_PATH) and os.path.exists(DATA_PATH):
    #         print("🔄 Loading existing FAISS index, data and metadata...")
    #         index = faiss.read_index(FAISS_INDEX_PATH)
    #         path_index = faiss.read_index(FAISSPATH_INDEX_PATH)

    #         with open(METADATA_PATH, "rb") as f:
    #             metadata = pickle.load(f)
    #         with open(DATA_PATH, "rb") as f:
    #             data = pickle.load(f)
    #     else:
    #         print("🆕 Creating a new FAISS index...")
    #         index = None  # Initialize index as None (to be created later)
    #         path_index = None
    #         metadata = []  # Empty metadata list
    #         data = []  # Empty metadata list
    #     return index, path_index, data, metadata

    # # Save the FAISS index and metadata
    # def save_faiss_and_metadata(index, path_index, data, metadata):
    #     faiss.write_index(index, FAISS_INDEX_PATH)
    #     faiss.write_index(path_index, FAISSPATH_INDEX_PATH)

    #     with open(METADATA_PATH, "wb") as f:
    #         pickle.dump(metadata, f)
    #     with open(DATA_PATH, "wb") as f:
    #         pickle.dump(data, f)
    #     print("✅ FAISS index, data and metadata saved successfully!")

    # # Load or initialize FAISS
    # index, path_index, data, metadata = load_or_initialize_faiss()

    # # Convert new documents to embeddings
    # new_embeddings = model.encode(new_texts, convert_to_numpy=True)
    # new_path_embeddings = model.encode([mtdata["path"] for mtdata in new_metadata], convert_to_numpy=True)


    # # If FAISS index does not exist, create it
    # if index is None:
    #     embedding_size = new_embeddings.shape[1]  # Get the embedding dimension
    #     index = faiss.IndexFlatL2(embedding_size)  # Create FAISS index
    #     path_embedding_size = new_path_embeddings.shape[1]  # Get the embedding dimension
    #     path_index = faiss.IndexFlatL2(path_embedding_size)  # Create FAISS index
    #     print(f"🛠️ Created FAISS index with dimension {embedding_size}")

    # # Append new embeddings to FAISS index
    # index.add(new_embeddings)

    # path_index.add(new_path_embeddings)

    # # Append new data
    # data.extend(new_texts)


    # # Add id to current new_metadata
    # for i, mtdata in enumerate(new_metadata):
    #     # bytes_representation = mtdata["path"].encode(encoding="utf-8") 
    #     bytes_representation = str(mtdata["doc_id"] + mtdata["path"] + str(len(metadata) + i + 1)).encode("utf-8")
    #     hash_object = hashlib.sha256(bytes_representation)  # Use SHA-256 (or hashlib.md5 for a smaller hash)
    #     hash_int = int(hash_object.hexdigest(), 16) % (10**10)
    #     mtdata["id"] = hash_int
    # # Append new data
    # metadata.extend(new_metadata)

    # # Save the updated FAISS index and metadata
    # save_faiss_and_metadata(index, path_index, data, metadata)

    # # Verify update
    # index, path_index, data, metadata = load_or_initialize_faiss()
    # print(f"📌 Updated FAISS Index: {index.ntotal} entries")
    # print(f"📌 Updated Data: {len(data)} entries")
    # print(f"📌 Updated Metadata: {len(metadata)} entries")
    def create_graph(tx, metadata):
        root_node_content = metadata["heading"]
        root_id = metadata["doc_id"]
        content = metadata["content"]

        # Create root node
        tx.run("MERGE (p: P_Node {content: $content, root_id: $id})", content = root_node_content, id = root_id)

        # Create content node
        tx.run("MERGE (p: C_Node {content: $content, root_id: $id})", content = content, id = root_id)

        # Create middle nodes
        middle_node_names = metadata["middle_path"].split(" > ")
        for middle_node in middle_node_names:
            tx.run("MERGE (p: M_Node {content: $content, root_id: $id})", content = middle_node, id = root_id)
        # Connect root node to first middle node
        tx.run("""
            MATCH (a:P_Node {content: $p_content, root_id: $id}), (b:M_Node {content: $m_content, root_id: $id})
            MERGE (a)-[:PARRENT]->(b)
        """, p_content=root_node_content, m_content=middle_node_names[0], id = root_id)
        # Connect last middle node to content node
        tx.run("""
            MATCH (a:M_Node {content: $m_content, root_id: $id}), (b:C_Node {content: $c_content, root_id: $id})
            MERGE (a)-[:CONTAIN]->(b)
        """, m_content=middle_node_names[-1], c_content=content, id = root_id)
        # Connect middle nodes
        for i in range(len(middle_node_names) - 1):
            tx.run("""
                MATCH (a:M_Node {content: $node1, root_id: $id}), (b:M_Node {content: $node2, root_id: $id})
                MERGE (a)-[:NEXT]->(b)
            """, node1=middle_node_names[i], node2=middle_node_names[i + 1], id = root_id)

        # Create nodes
        # for node in nodes:
        #     tx.run("MERGE (n:Node {name: $name})", name=node)

        # # Create relationships
        # for i in range(len(nodes) - 1):
        #     tx.run("""
        #         MATCH (a:Node {name: $node1}), (b:Node {name: $node2})
        #         MERGE (a)-[:NEXT]->(b)
        #     """, node1=nodes[i], node2=nodes[i + 1])
    for mtdata in new_metadata:
        # paths = mtdata["path"].split(">")
        with driver.session() as session:
            session.execute_write(create_graph, mtdata)

