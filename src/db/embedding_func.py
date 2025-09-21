# db.embedding_func.py

from chromadb.utils import embedding_functions

from config import config


def get_embedding_func(model_name: str =config.EMBEDDER_NAME):
    """
    Create and return a SentenceTransformer embedding function for ChromaDB.

    Args:
        model_name (str, optional): The name of the embedding model to use.
            Defaults to `config.EMBEDDER_NAME`.

    Returns:
        chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction:
        Embedding function instance that generates vector embeddings from text.
    """
    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )
    return embedding_func