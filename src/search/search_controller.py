from src.search.search_service import SearchService
from fastapi import APIRouter, Query


class SearchController:
    def __init__(self, search_service):
        self.router = APIRouter(prefix="/search", tags=["search"])
        self.router.add_api_route("/", self.search, methods=["GET"])
        self.search_service = search_service

    def search(
        self,
        query: str = Query(..., description="The search query string"),
        top_k: int = Query(10, description="Number of top results to return"),
    ):
        # translated_text = self.search_service.translate(query)
        print(query)
        return self.search_service.search(query, top_k)


search_controller = SearchController(SearchService())
router = search_controller.router
