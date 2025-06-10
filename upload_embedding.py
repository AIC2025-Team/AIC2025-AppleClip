from src.core.milvus_setup import MilvusSetup
import os
import numpy as np

def upload_embedding(collection_name: str, data: list):
    res = MilvusSetup.insert_data(collection_name, data)
    return res

def create_index(collection_name: str, field_name: str, index_params: dict):
    collection = MilvusSetup.get_collection(collection_name)
    if collection.has_index():  # Không truyền field_name vào đây
        print(f"Index already exists in '{collection_name}' collection.")
        return
    print(f"Creating index for field '{field_name}' in '{collection_name}' collection with params: {index_params}")
    MilvusSetup.create_index(collection_name, field_name, index_params)
    print(f"Index created for '{field_name}' in '{collection_name}' collection.")

def load_embeddings(embeddings_files_folder: str):
    embeddings = []
    for filename in os.listdir(embeddings_files_folder):
        if filename.endswith(".npy"):
            file_path = embeddings_files_folder + "/" + filename
            embedding = np.load(file_path).tolist()
            embeddings.append({"frame_path": file_path, "embedding": embedding})
    if not embeddings:
        print("No embeddings found in the specified folder.")
        return []
    print(f"Loaded {len(embeddings)} embeddings from '{embeddings_files_folder}'.")
    return embeddings

if __name__ == "__main__":
    MilvusSetup.connect_milvus()
    # Example usage
    clip_embeddings_folder = "embeddings/clip_embeddings"
    beit_embeddings_folder = "embeddings/beit_embeddings"
    # Load embeddings
    clip_embeddings = load_embeddings(clip_embeddings_folder)
    beit_embeddings = load_embeddings(beit_embeddings_folder)
    # Upload embeddings
    upload_embedding("clip_model_collection", clip_embeddings)
    upload_embedding("beit_model_collection", beit_embeddings)

    # Create index for embeddings
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 128}
    }

    create_index("clip_model_collection", "embedding", index_params)
    create_index("beit_model_collection", "embedding", index_params)
    print("Embedding upload and indexing completed.")