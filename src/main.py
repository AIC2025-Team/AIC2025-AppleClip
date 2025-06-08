from fastapi import FastAPI
from src.api.routes import include_all_routers

def create_app() -> FastAPI:
    app = FastAPI(title="AIC2025", version="0.1.0")
    include_all_routers(app)
    return app
