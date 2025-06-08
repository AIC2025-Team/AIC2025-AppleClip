from fastapi import FastAPI
from src.api.routes import include_all_routers
from src.core.milvus_setup import connect_milvus, create_collection
from src.core.elastic_search import get_elasticsearch_client


def create_app() -> FastAPI:
    app = FastAPI(title="AIC2025", version="0.1.0")
    connect_milvus()
    get_elasticsearch_client()
    include_all_routers(app)
    return app
