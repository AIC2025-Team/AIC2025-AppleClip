from src.main_service import MainService
from fastapi import APIRouter

class MainController:
    def __init__(self):
        self.main_service = MainService()
        self.router = APIRouter(prefix="", tags=["main"])
        self.router.add_api_route("/", self.hello_world, methods=["GET"])

    def hello_world(self):
        return self.main_service.hello_world()
    
# Router export
main_controller = MainController()
router = main_controller.router
