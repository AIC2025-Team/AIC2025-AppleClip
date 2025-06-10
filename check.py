from pymilvus import Collection, utility
from src.core.milvus_setup import MilvusSetup

def check_collection_exists(collection_name: str):
    collection = Collection(collection_name)
    print(collection.num_entities)

if __name__ == "__main__":
    MilvusSetup.connect_milvus()

    clip_model_collection = "clip_model_collection"
    beit_model_collection = "beit_model_collection"

    print(f"Checking collection '{clip_model_collection}'...")
    check_collection_exists(clip_model_collection)

    print(f"Checking collection '{beit_model_collection}'...")
    check_collection_exists(beit_model_collection)
    
    print("Collection checks completed.")