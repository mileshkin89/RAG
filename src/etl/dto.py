from pydantic import BaseModel, Field
from typing import Optional


class RecipesDTO(BaseModel):
    recipe_title: str
    url: str
    record_health: Optional[str] = ""
    vote_count: int = Field(default=0, ge=0)
    rating: float = Field(default=0.0, ge=0.0, le=5.0)
    description: Optional[str] = ""
    cuisine: Optional[str] = ""
    course: Optional[str] = ""
    diet: Optional[str] = ""
    prep_time: Optional[str] = ""
    cook_time: Optional[str] = ""
    ingredients: Optional[str] = ""
    instructions: Optional[str] = ""
    author: Optional[str] = ""
    tags: Optional[str] = ""
    category: Optional[str] = ""
