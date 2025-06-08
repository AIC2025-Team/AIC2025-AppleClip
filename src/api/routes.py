from fastapi import FastAPI
from src.main_controller import router as MainController

def include_all_routers(app: FastAPI):
    app.include_router(MainController)