import os
from src.tools.embed_text import handler as embed_handler
from src.vector_store.chroma_store import query

COLLECTION_NAME = "rag_documents"

def tool_definition():
    return {
        "name": "vector_search",
        "description": "Search vector store with text query",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query text"},
                "top_k": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 5
                }
            },
            "required": ["query"]
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "text": {"type": "string"},
                            "score": {"type": "number"},
                            "metadata": {"type": "object"}
                        }
                    }
                },
                "count": {"type": "integer"}
            }
        }
    }

def handler(payload: dict):
    """
    Vector search: embed query and find similar documents
    """
    query_text = payload.get("query", "").strip()
    top_k = payload.get("top_k", 5)
    
    if not query_text:
        return {"error": "query cannot be empty"}
    
    if not isinstance(top_k, int) or top_k < 1:
        top_k = 5
    
    # Generate embedding for query
    embed_result = embed_handler({"text": query_text})
    
    if "error" in embed_result:
        return embed_result
    
    query_embedding = embed_result.get("embedding")
    
    # Query Chroma
    try:
        results = query(
            collection_name=COLLECTION_NAME,
            query_embeddings=[query_embedding],
            top_k=top_k
        )
        
        # Format results
        formatted_results = []
        if results and results.get("ids") and len(results["ids"]) > 0:
            for i, doc_id in enumerate(results["ids"][0]):
                # Chroma returns distances (smaller = more similar)
                # Convert to similarity score (1 - distance) for cosine
                distance = results["distances"][0][i]
                score = 1 - distance
                
                formatted_results.append({
                    "id": doc_id,
                    "text": results["documents"][0][i],
                    "score": round(score, 4),
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {}
                })
        
        return {
            "results": formatted_results,
            "count": len(formatted_results)
        }
    except Exception as e:
        return {"error": str(e)}
