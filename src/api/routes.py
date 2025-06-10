from fastapi import FastAPI
from src.main_controller import router as MainController
from src.search.serach_controller import router as SearchController

def include_all_routers(app: FastAPI):
    app.include_router(MainController)
    app.include_router(SearchController)