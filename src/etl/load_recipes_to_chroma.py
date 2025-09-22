"""
ETL pipeline for loading recipe data from a CSV file into ChromaDB.

This module provides Extract, Transform, and Load (ETL) functionality:
- Extract: read recipe rows from a CSV file.
- Transform: validate and convert rows into DTOs, documents, metadata, and unique IDs.
- Load: batch upsert documents into a ChromaDB collection.

The pipeline ensures row validation, duplicate handling via deterministic IDs,
and progress tracking with tqdm.
"""

# etl.load_recipes_to_chroma.py

import csv
import hashlib
import time
from pathlib import Path
from typing import Iterator, Tuple, List, Optional

from pydantic import ValidationError
from tqdm import tqdm

from db.collection import get_or_create_collection
from dto import RecipesDTO
from config import config
from logger import get_logger


logger = get_logger(__name__)

BATCH_SIZE = config.BATCH_SIZE
csv_file = config.DATASET_PATH


def _hasher(text: str) -> str:
    """
    Generate a SHA256 hash for the given text.

    Args:
        text (str): Input text.

    Returns:
        str: Hexadecimal SHA256 hash of the normalized text.
    """
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()


def _row_to_dto(row: dict) -> Optional[RecipesDTO]:
    """
    Convert a CSV row dictionary into a RecipesDTO object with Pydantic validation.

    Args:
        row (dict): Dictionary representing a single CSV row.

    Returns:
        RecipesDTO | None: Data transfer object otherwise None.
    """
    try:
        dto = RecipesDTO(**row)
        return dto
    except ValidationError as e:
        logger.warning(f"String validation error: {e}")
        return None


def _validate_row(row: dict) -> bool:
    """
    Validate a CSV row by checking for empty or invalid values.

    Args:
        row (dict): Dictionary representing a single CSV row.
        retries (int, optional): Number of retries for validation. Defaults to 3.

    Returns:
        bool: True if the row is valid, False otherwise.
    """
    for key, value in row.items():
        if not value or str(value).strip().lower() in {"nan", "null", "none"}:
            logger.warning(f"Skipped row with empty or invalid item. item[key]: `{key}`, item[value]: `{value}`, row: {row}")
            return False
    return True


def _load_in_db(
    documents: List[str],
    metadatas: List[dict],
    ids: List[str],
    batch_size: int = int(BATCH_SIZE / 4),
):
    """
    Insert or update documents in ChromaDB in batches.

    Args:
        documents (List[str]): List of document strings.
        metadatas (List[dict]): List of metadata dictionaries.
        ids (List[str]): List of unique IDs.
        batch_size (int, optional): Number of rows per batch. Defaults to BATCH_SIZE/4.
    """
    collection = get_or_create_collection(config.COLLECTION_NAME)

    with tqdm(total=len(ids), desc="Inserting into DB") as pbar:
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i+batch_size]
            batch_metas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]

            collection.upsert(
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids,
            )

            # update progress by batch size
            pbar.update(len(batch_ids))

            logger.info(f"Upserted {len(batch_docs)} rows into collection '{config.COLLECTION_NAME}'")


def extract(file_path: Path) -> Iterator[dict]:
    """
    Extract rows from a CSV file.

    Args:
        file_path (Path): Path to the CSV file.

    Yields:
        tuple[int, dict]: Row index (1-based) and row dictionary.
    """
    with open(file_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader, start=1):
            yield idx, row


def transform(idx: int, row: dict) -> Tuple[str, dict, str] | None:
    """
    Transform a raw CSV row into (document, metadata, id).

    Args:
        idx (int): Row index.
        row (dict): Row dictionary.

    Returns:
        tuple[str, dict, str] | None: Document string, metadata dict, and unique ID.
            Returns None if row is invalid.
    """
    if not _validate_row(row):
        return None

    try:
        dto = _row_to_dto(row)
    except Exception as e:
        logger.warning(f"Skipping row {idx}: {e}")
        return None

    if dto is None:
        return None

    documents = f"{dto.recipe_title}. {dto.description}. {dto.instructions}."
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

    # unique ID based on title + URL
    text = dto.recipe_title + dto.url
    doc_id = f"{_hasher(text)}"

    return documents, metadata, doc_id


def load(batch: List[Tuple[str, dict, str]]) -> int:
    """
    Load a batch of documents into ChromaDB.

    Args:
        batch (List[Tuple[str, dict, str]]): List of (document, metadata, id) tuples.

    Returns:
        int: Number of inserted rows.
    """
    if not batch:
        return 0

    documents, metadatas, ids = zip(*batch)
    _load_in_db(list(documents), list(metadatas), list(ids))
    return len(batch)



def main():
    """
    Run the ETL pipeline:
    - Extract rows from the CSV file.
    - Transform each row into a document with metadata and unique ID.
    - Load batches into ChromaDB.
    """
    total_inserted = 0
    total_rows = 0
    batch = []

    start = time.perf_counter()

    for idx, row in extract(csv_file):
        total_rows = idx

        result = transform(idx, row)
        if result:
            batch.append(result)

        if len(batch) % BATCH_SIZE == 0:
            total_inserted += load(batch)
            batch = []

    total_inserted += load(batch)

    total = time.perf_counter() - start

    logger.info(f"ETL finished in {total:.3f}s")
    logger.info(f"Total rows: {total_rows}. Total upserted: {total_inserted} rows into '{config.COLLECTION_NAME}'")



if __name__ == "__main__":
    main()
