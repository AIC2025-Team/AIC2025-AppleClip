from pydantic import BaseModel

class SearchResult(BaseModel):
    frame_path: str
    distance: float