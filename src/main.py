from fastapi import FastAPI
from src.api.routes import include_all_routers
from src.core.milvus_setup import connect_milvus, create_collection

def create_app() -> FastAPI:
    app = FastAPI(title="AIC2025", version="0.1.0")
    connect_milvus()
    include_all_routers(app)
    return app
