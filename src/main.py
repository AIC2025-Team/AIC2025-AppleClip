from fastapi import FastAPI
from src.api.routes import include_all_routers
from src.core.milvus_setup import MilvusSetup
from src.core.elastic_search import get_elasticsearch_client


def create_app() -> FastAPI:
    app = FastAPI(title="AIC2025", version="0.1.0")
    MilvusSetup.connect_milvus()
    MilvusSetup.create_collection("clip_model_collection", dim=1024)
    MilvusSetup.create_collection("beit_model_collection", dim=768)
    get_elasticsearch_client()
    include_all_routers(app)
    return app
