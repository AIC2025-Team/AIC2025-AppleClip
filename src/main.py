from fastapi import FastAPI
from src.api.routes import include_all_routers
from src.core.milvus_setup import MilvusSetup


def create_app() -> FastAPI:
    app = FastAPI(title="AIC2025", version="0.1.0")
    MilvusSetup.connect_milvus()
    MilvusSetup.create_collection("clip_model_collection", dim=1024)
    include_all_routers(app)
    return app
