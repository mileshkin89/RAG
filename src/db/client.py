# db/client.py

import chromadb

from config import config


chromadb_client = chromadb.PersistentClient(path=f"{config.DB_DIR}/recipes")



