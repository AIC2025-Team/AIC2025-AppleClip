from pymilvus import utility
from src.core.milvus_setup import MilvusSetup



if __name__ == "__main__":
    MilvusSetup.connect_milvus()
    
    clip_model_collection = "clip_model_collection"

    MilvusSetup.drop_collection(clip_model_collection)
    
    print("All specified collections have been deleted.")