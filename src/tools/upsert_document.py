import os
from src.tools.embed_text import handler as embed_handler
from src.vector_store.chroma_store import upsert

COLLECTION_NAME = "rag_documents"

def tool_definition():
    return {
        "name": "upsert_document",
        "description": "Upsert a document to vector store with embedding",
        "input_schema": {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Unique document ID"},
                "text": {"type": "string", "description": "Document text to embed and store"},
                "metadata": {
                    "type": "object",
                    "description": "Optional metadata (source, author, etc)",
                    "additionalProperties": True
                }
            },
            "required": ["id", "text"]
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "id": {"type": "string"},
                "embedding_dim": {"type": "integer"}
            }
        }
    }

def handler(payload: dict):
    """
    Upsert document: embed text and store to Chroma
    """
    doc_id = payload.get("id", "").strip()
    text = payload.get("text", "").strip()
    metadata = payload.get("metadata", {})
    
    if not doc_id:
        return {"error": "id cannot be empty"}
    if not text:
        return {"error": "text cannot be empty"}
    
    # Generate embedding using embed_text tool
    embed_result = embed_handler({"text": text})
    
    if "error" in embed_result:
        return embed_result
    
    embedding = embed_result.get("embedding")
    embedding_dim = embed_result.get("dim")
    
    # Upsert to Chroma
    try:
        upsert(
            collection_name=COLLECTION_NAME,
            ids=[doc_id],
            embeddings=[embedding],
            texts=[text],
            metadatas=[metadata if metadata else {}]
        )
        
        return {
            "status": "success",
            "id": doc_id,
            "embedding_dim": embedding_dim
        }
    except Exception as e:
        return {"error": str(e)}
