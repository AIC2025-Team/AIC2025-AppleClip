from pymilvus import connections, Collection, utility
import os

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")


def connect_milvus():
    connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    print(f"Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")

def create_collection(collection_name: str, dim: int = 128):
    if utility.has_collection(collection_name):
        print(f"Collection '{collection_name}' already exists")
        return Collection(collection_name)

    from pymilvus import FieldSchema, CollectionSchema, DataType

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields, description="Vector collection")
    collection = Collection(name=collection_name, schema=schema)
    print(f"Created collection '{collection_name}'")
    return collection
