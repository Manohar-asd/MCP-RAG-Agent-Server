import chromadb
from typing import List, Dict, Optional

# Initialize Chroma client (persistent storage in ./chroma_data)
client = chromadb.PersistentClient(path="./chroma_data")

def init_collection(name: str = "rag_documents"):
    """
    Initialize or get a collection in Chroma
    """
    try:
        collection = client.get_collection(name=name)
    except Exception:
        collection = client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )
    return collection

def upsert(
    collection_name: str,
    ids: List[str],
    embeddings: List[List[float]],
    texts: List[str],
    metadatas: Optional[List[Dict]] = None
):
    """
    Upsert documents with embeddings to Chroma
    
    Args:
        collection_name: Name of collection
        ids: List of document IDs
        embeddings: List of embedding vectors
        texts: List of document texts
        metadatas: Optional list of metadata dicts
    """
    collection = init_collection(collection_name)
    
    if metadatas is None:
        metadatas = [{} for _ in ids]
    
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )
    
    return {"status": "upserted", "count": len(ids)}

def query(
    collection_name: str,
    query_embeddings: List[List[float]],
    top_k: int = 5
):
    """
    Query documents by embedding similarity
    
    Args:
        collection_name: Name of collection
        query_embeddings: List of query embedding vectors
        top_k: Number of results to return
    
    Returns:
        Query results with IDs, distances, documents, and metadata
    """
    collection = init_collection(collection_name)
    
    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=top_k
    )
    
    return results
