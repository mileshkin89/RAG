# db.collection.py

from chromadb.errors import NotFoundError

from db.client import chromadb_client
from config import config
from db.embedding_func import get_embedding_func
from logger import get_logger

logger = get_logger(__name__)


def get_or_create_collection(collection_name: str = config.COLLECTION_NAME):
    """
    Get an existing ChromaDB collection or create a new one if it does not exist.

    This function tries to retrieve a collection by name. If the collection
    is not found, it creates a new collection with the specified embedding
    function.

    Args:
        collection_name (str, optional): Name of the collection.
            Defaults to `config.COLLECTION_NAME`.

    Returns:
        chromadb.api.types.Collection: A ChromaDB collection object, either
        existing or newly created.
    """
    try:
        collection = chromadb_client.get_collection(name=collection_name)
        logger.info(f"Using existing collection '{collection_name}'")
        return collection
    except (ValueError, NotFoundError):
        embedding_func = get_embedding_func(config.EMBEDDER_NAME)
        collection = chromadb_client.create_collection(
            name=collection_name,
            embedding_function=embedding_func
        )
        logger.info(f"Created new collection '{collection_name}'")
        return collection