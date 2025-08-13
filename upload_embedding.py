from src.core.milvus_setup import MilvusSetup
import os
import numpy as np

def upload_embedding(collection_name: str, data: list):
    if not data:
        print("No data to upload.")
        return
    res = MilvusSetup.insert_data(collection_name, data)
    print(f"Uploaded {len(data)} embeddings to '{collection_name}' collection.")
    return res

def create_index(collection_name: str, field_name: str, index_params: dict):
    collection = MilvusSetup.get_collection(collection_name)
    if collection.has_index():
        print(f"Index already exists in '{collection_name}' collection.")
        return
    print(f"Creating index for field '{field_name}' in '{collection_name}' with params: {index_params}")
    MilvusSetup.create_index(collection_name, field_name, index_params)
    print(f"Index created for '{field_name}' in '{collection_name}' collection.")

def load_embeddings(embeddings_files_folder: str):
    embeddings = []
    
    for root, _, files in os.walk(embeddings_files_folder):
        folder_name = os.path.basename(root)  
        
        for filename in files:
            if filename.endswith(".npy"):
                file_path = os.path.join(root, filename)
                embedding = np.load(file_path).astype(np.float32).tolist()
                
                relative_path = f"{folder_name}/{filename}"
                
                embeddings.append({
                    "frame_path": relative_path,
                    "embedding": embedding
                })
    
    if not embeddings:
        print("No embeddings found in the specified folder.")
        return []
    
    print(f"Loaded {len(embeddings)} embeddings from '{embeddings_files_folder}'.")
    return embeddings

if __name__ == "__main__":
    MilvusSetup.connect_milvus()
    
    clip_embeddings_folder = "embeddings/clip_embeddings"
    collection_name = "clip_model_collection"
    
    # 1. Load
    clip_embeddings = load_embeddings(clip_embeddings_folder)

    # 2. Upload
    upload_embedding(collection_name, clip_embeddings)

    # 3. Create index
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 128}
    }
    create_index(collection_name, "embedding", index_params)
    
    print("Embedding upload and indexing completed.")
