from src.utils.embedding_models import get_clip_embedding
from src.core.milvus_setup import MilvusSetup
from src.search.dtos.search_result_dto import SearchResult

class SearchService:
    def __init__(self):
        pass

    def search(self, query: str, top_k: int = 10):
        if not query:
            raise ValueError("Query cannot be empty")
        query_embedding = get_clip_embedding(query)

        collection = MilvusSetup.get_collection("clip_model_collection")
        if not collection.has_index():
            raise RuntimeError("Collection does not have an index. Please create an index before searching.")

        results = MilvusSetup.search(
            collection_name="clip_model_collection",
            data=query_embedding,
            anns_field="embedding",
            limit=top_k,
            params={"metric_type": "COSINE", "params": {"nprobe": 10}},
            output_fields=["frame_path"]
        )
        hits = results[0]
        return [SearchResult(frame_path=hit.entity.get("frame_path"), distance=hit.distance) for hit in hits]

    def get_suggestions(self, query: str):
        return self.search_engine.get_suggestions(query)

    def get_search_history(self, user_id: str):
        return self.search_engine.get_search_history(user_id)