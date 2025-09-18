# etl/load_recipes_to_chroma.py

import csv
import hashlib
import time
from chromadb.utils import embedding_functions

from db.client import chromadb_client
from dto import RecipesDTO
from config import config
from logger import get_logger

logger = get_logger(__name__)

CHUNK_SIZE = 1000
csv_file = config.DATASET_DIR / "food_recipes.csv"

# Store only unique "recipe_title" hashes
seen_titles = set()


embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=config.EMBEDDER_NAME
)
collection_name = config.COLLECTION_NAME

# Try to get existing collection, or create new one with embedding_function
try:
    collection = chromadb_client.get_collection(name=collection_name)
    logger.info(f"Using existing collection '{collection_name}'")
except ValueError:
    # Collection does not exist, create new one with embedding_function
    collection = chromadb_client.create_collection(
        name=collection_name,
        embedding_function=embedding_func
    )
    logger.info(f"Created new collection '{collection_name}'")



def _hasher(title: str) -> str:
    """
    Return SHA256 hash of recipe_title.

    Args:
        title (str): recipe_title string.

    Returns:
        str: hash in hex format.
    """
    return hashlib.sha256(title.strip().lower().encode("utf-8")).hexdigest()


def _row_to_dto(row: dict) -> RecipesDTO:
    """Converts a csv string to a DTO with safe data retrieval"""
    return RecipesDTO(
        recipe_title=row.get("recipe_title", "").strip(),
        url=row.get("url", "").strip(),
        record_health=row.get("record_health", "").strip(),
        vote_count=int(row.get("vote_count") or 0),
        rating=float(row.get("rating") or 0.0),
        description=row.get("description", "").strip(),
        cuisine=row.get("cuisine", "").strip(),
        course=row.get("course", "").strip(),
        diet=row.get("diet", "").strip(),
        prep_time=row.get("prep_time", "").strip(),
        cook_time=row.get("cook_time", "").strip(),
        ingredients=row.get("ingredients", "").strip(),
        instructions=row.get("instructions", "").strip(),
        author=row.get("author", "").strip(),
        tags=row.get("tags", "").strip(),
        category=row.get("category", "").strip(),
    )


def _validate_row(row: dict) -> bool:
    """
    Validate a row before inserting into Chroma.

    Checks:
        1. rows items is not empty or NaN/null/none
        2. title is not a duplicate (based on hash)

    Args:
        row (dict): A row from CSV file.

    Returns:
        bool: True if valid, False if invalid.
    """
    for item in row.values():
        if not item or str(item).strip().lower() in {"nan", "null", "none"}:
            logger.warning(f"Skipped row with empty or invalid item: {row}")
            return False

    title = row.get("recipe_title", "").strip()

    title_hash = _hasher(title)
    if title_hash in seen_titles:
        logger.warning(f"Skipped duplicate title: {title}")
        return False

    seen_titles.add(title_hash)
    return True


def load_in_db(documents, metadatas, ids):
    """
    Add documents into Chroma.

    Args:
        documents (list[str]): List of texts.
        metadatas (list[dict]): List of metadata dicts.
        ids (list[str]): List of document IDs.
    """

    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    logger.info(f"Inserted {len(documents)} rows into collection '{collection_name}'")


def main():
    documents, metadatas, ids = [], [], []
    total_inserted = 0

    start = time.perf_counter()

    # Extract
    with open(csv_file, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for idx, row in enumerate(reader, start=1):
            if not _validate_row(row):
                continue

            # Transform
            try:
                dto = _row_to_dto(row)
            except Exception as e:
                logger.warning(f"Skipping row {idx}: {e}")
                continue

            text = f"{dto.recipe_title}. {dto.description}. {dto.instructions}."
            documents.append(text)

            metadata = {
                "title": dto.recipe_title,
                "url": dto.url,
                "cuisine": dto.cuisine,
                "course": dto.course,
                "diet": dto.diet,
                "author": dto.author,
                "prep_time": dto.prep_time,
                "cook_time": dto.cook_time,
            }
            metadatas.append(metadata)

            ids.append(f"recipe_{idx:05d}")

            if len(documents) >= CHUNK_SIZE:
                load_in_db(documents, metadatas, ids)
                total_inserted += len(documents)
                documents, metadatas, ids = [], [], []

        # Load
        if documents:
            load_in_db(documents, metadatas, ids)
            total_inserted += len(documents)

    total = time.perf_counter() - start

    logger.info(f"ETL finished in {total:.3f}s")
    logger.info(f"ETL finished. Total rows: {idx}. Total inserted: {total_inserted} rows into '{collection_name}'")


if __name__ == "__main__":
    main()


