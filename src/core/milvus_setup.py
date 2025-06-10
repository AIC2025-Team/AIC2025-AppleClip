from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
import os

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")


class MilvusSetup:
    @staticmethod
    def connect_milvus():
        connections.connect(
            alias="default",
            host=MILVUS_HOST,
            port=MILVUS_PORT,
        )
        print(f"Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")

    @staticmethod
    @staticmethod
    def create_collection(collection_name: str, dim: int):
        if not utility.has_collection(collection_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="frame_path", dtype=DataType.VARCHAR, max_length=512, is_primary=False),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
            ]
            schema = CollectionSchema(fields=fields, description=f"Collection for {collection_name} embeddings")
            collection = Collection(name=collection_name, schema=schema)
            
            print(f"Collection '{collection_name}' created successfully.")
        else:
            print(f"Collection '{collection_name}' already exists.")

    @staticmethod
    def drop_collection(collection_name: str):
        if utility.has_collection(collection_name):
            utility.drop_collection(collection_name)
            print(f"Collection '{collection_name}' deleted successfully.")
        else:
            print(f"Collection '{collection_name}' does not exist.")

    @staticmethod
    def check_collection_exists(collection_name: str):
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            print(f"Collection '{collection_name}' exists with {collection.num_entities} entities.")
        else:
            print(f"Collection '{collection_name}' does not exist.")

    @staticmethod
    def list_collections():
        collections = utility.list_collections()
        print("Collections in Milvus:")
        for collection in collections:
            print(f"- {collection}")
        return collections
    
    @staticmethod
    def get_collection(collection_name: str) -> Collection:
        if utility.has_collection(collection_name):
            return Collection(collection_name)
        else:
            raise ValueError(f"Collection '{collection_name}' does not exist.")
        
    @staticmethod
    def get_collection_info(collection_name: str):
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            return {
                "name": collection.name,
                "num_entities": collection.num_entities,
                "schema": collection.schema.to_dict(),
            }
        else:
            raise ValueError(f"Collection '{collection_name}' does not exist.")
        
    @staticmethod
    def get_query_segment_info(collection_name: str):
        if utility.has_collection(collection_name):
            segments_info = utility.get_query_segment_info(collection_name)
            print(f"Segments info for '{collection_name}': {segments_info}")
            return segments_info
        else:
            raise ValueError(f"Collection '{collection_name}' does not exist.")
        
    @staticmethod
    def flush_collection(collection_name: str):
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            collection.flush()
            print(f"Flushed collection '{collection_name}' successfully.")
        else:
            raise ValueError(f"Collection '{collection_name}' does not exist.")
    
    @staticmethod
    def insert_data(collection_name: str, data: list):
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            res = collection.insert(data)
            print(f"Inserted data into '{collection_name}': {res}")
            collection.flush()
            return res
        else:
            raise ValueError(f"Collection '{collection_name}' does not exist.")
        
    @staticmethod
    def create_index(collection_name: str, field_name: str, index_params: dict):
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            collection.create_index(field_name, index_params)
            print(f"Index created for '{field_name}' in collection '{collection_name}'.")
        else:
            raise ValueError(f"Collection '{collection_name}' does not exist.")
        
    @staticmethod
    def search(collection_name: str, data: list, anns_field: str, limit: int, params: dict, output_fields: list):
        if utility.has_collection(collection_name):
            collection = Collection(collection_name)
            collection.load()
            results = collection.search(
                data=data,
                anns_field=anns_field,
                param=params,
                limit=limit,
                output_fields=output_fields
            )
            print(f"Search completed in collection '{collection_name}'.")
            return results
        else:
            raise ValueError(f"Collection '{collection_name}' does not exist.")