from dataclasses import dataclass
from datetime import timedelta


@dataclass
class RecipesDTO:
    recipe_title: str
    url: str
    record_health: str
    vote_count: int
    rating: float
    description: str
    cuisine: str
    course: str
    diet: str
    prep_time: str
    cook_time: str
    ingredients: str
    instructions: str
    author: str
    tags: str
    category: str
