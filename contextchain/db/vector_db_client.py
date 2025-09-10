from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Dict, Optional
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Singleton client instance
_db_client = None

class ChromaClient:
    """A client for managing ChromaDB vector database operations."""
    
    def __init__(self, db_path: str = "./contextchain_chromadb"):
        """
        Initialize the ChromaDB client with a persistent storage path.

        Args:
            db_path (str): The file system path to store the vector database.
        """
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Default embedding model
        logger.info(f"✓ ChromaDB client initialized at: {self.db_path.resolve()}")

    def heartbeat(self) -> bool:
        """Check if the ChromaDB client is operational."""
        try:
            self.client.heartbeat()
            return True
        except Exception as e:
            logger.error(f"✗ Heartbeat failed: {e}")
            return False

    def create_collection(self, collection_name: str) -> bool:
        """
        Create a new collection in the vector database.

        Args:
            collection_name (str): Name of the collection to create.

        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            self.client.create_collection(name=collection_name, embedding_function=None)  # Use custom embedding
            logger.info(f"✓ Created collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"✗ Failed to create collection {collection_name}: {e}")
            return False

    def add_documents(self, collection_name: str, documents: List[str], metadata: Optional[List[Dict]] = None, batch_size: int = 1000) -> Dict:
        """
        Add documents to a collection with embeddings.

        Args:
            collection_name (str): Target collection name.
            documents (List[str]): List of text documents to add.
            metadata (Optional[List[Dict]]): Optional metadata for each document.
            batch_size (int): Number of documents to process per batch.

        Returns:
            Dict: Metrics (e.g., {'added': count, 'time_taken': seconds}).
        """
        try:
            collection = self.client.get_collection(name=collection_name)
            start_time = time.time()
            ids = [f"doc_{i}" for i in range(len(documents))]
            embeddings = self.embedding_model.encode(documents, batch_size=batch_size).tolist()
            
            # Process in batches to handle large data
            for i in range(0, len(documents), batch_size):
                batch_ids = ids[i:i + batch_size]
                batch_docs = documents[i:i + batch_size]
                batch_embeds = embeddings[i:i + batch_size]
                batch_meta = metadata[i:i + batch_size] if metadata else None
                collection.add(
                    ids=batch_ids,
                    documents=batch_docs,
                    embeddings=batch_embeds,
                    metadatas=batch_meta
                )
            
            metrics = {
                'added': len(documents),
                'time_taken': time.time() - start_time
            }
            logger.info(f"✓ Added {len(documents)} documents to {collection_name}")
            return metrics
        except Exception as e:
            logger.error(f"✗ Failed to add documents to {collection_name}: {e}")
            raise

    def search(self, collection_name: str, query: str, k: int = 5) -> Dict:
        """
        Search the collection for documents similar to the query.

        Args:
            collection_name (str): Target collection name.
            query (str): Query text to search.
            k (int): Number of results to return.

        Returns:
            Dict: Search results and metrics (e.g., {'results': [...], 'time_taken': seconds}).
        """
        try:
            collection = self.client.get_collection(name=collection_name)
            start_time = time.time()
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=k
            )
            metrics = {
                'results': results,
                'time_taken': time.time() - start_time
            }
            logger.info(f"✓ Search completed in {collection_name} for query: {query}")
            return metrics
        except Exception as e:
            logger.error(f"✗ Failed to search in {collection_name}: {e}")
            raise

def get_vector_db_client(db_path: str = "./contextchain_chromadb") -> ChromaClient:
    """
    Get or initialize a singleton ChromaDB client.

    Args:
        db_path (str): The file system path to store the vector database.

    Returns:
        ChromaClient: An initialized and connected ChromaDB client instance.
    """
    global _db_client
    if _db_client:
        return _db_client

    try:
        _db_client = ChromaClient(db_path)
        if _db_client.heartbeat():
            return _db_client
        else:
            raise Exception("ChromaDB client heartbeat failed.")
    except Exception as e:
        logger.error(f"✗ Failed to initialize ChromaDB client at {db_path}: {e}")
        raise Exception(f"Fatal: Could not set up the vector database.")