import os
from src.clients.azure_openai import get_azure_client
from src.tools.vector_search import handler as vector_search_handler

def tool_definition():
    return {
        "name": "rag_answer",
        "description": "Retrieve documents from vector store and generate answer using Azure OpenAI chat",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Question or search query"},
                "top_k": {
                    "type": "integer",
                    "description": "Number of document chunks to retrieve",
                    "default": 5
                }
            },
            "required": ["query"]
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "sources": {
                    "type": "array",
                    "items": {"type": "object"}
                },
                "chunks_used": {
                    "type": "array",
                    "items": {"type": "object"}
                }
            }
        }
    }

def handler(payload: dict):
    """
    RAG Answer: Search vector store, build context, and generate answer with Azure OpenAI
    """
    query = payload.get("query", "").strip()
    top_k = payload.get("top_k", 5)
    
    if not query:
        return {"error": "query cannot be empty"}
    
    if not isinstance(top_k, int) or top_k < 1:
        top_k = 5
    
    # Step 1: Retrieve documents from vector store
    try:
        search_result = vector_search_handler({
            "query": query,
            "top_k": top_k
        })
        
        if "error" in search_result:
            return search_result
        
        results = search_result.get("results", [])
        
        # Step 2: Check if we have results with sufficient score
        if not results:
            return {
                "answer": "I don't have enough context to answer this question.",
                "sources": [],
                "chunks_used": []
            }
        
        # Filter results with minimum score threshold (0.5)
        min_score_threshold = 0.5
        high_score_results = [r for r in results if r.get("score", 0) >= min_score_threshold]
        
        if not high_score_results:
            return {
                "answer": "I don't have enough context to answer this question.",
                "sources": [],
                "chunks_used": results  # Still return what was found for debugging
            }
        
        # Step 3: Build context string from retrieved chunks
        context_chunks = []
        sources = []
        
        for idx, chunk in enumerate(high_score_results, 1):
            context_chunks.append(f"[Chunk {idx}] {chunk.get('text', '')}")
            
            # Collect unique sources from metadata
            metadata = chunk.get("metadata", {})
            if metadata and metadata not in sources:
                sources.append(metadata)
        
        context_string = "\n\n".join(context_chunks)
        
        # Step 4: Build prompt for Azure OpenAI
        system_prompt = """You are a helpful assistant answering questions based on provided context. 
Answer the question using only the information in the context. 
If the context doesn't contain enough information to answer, say so explicitly.
Keep your answer clear, concise, and well-organized."""
        
        user_prompt = f"""Context:
{context_string}

Question: {query}

Please provide a clear and concise answer based on the context above."""
        
        # Step 5: Call Azure OpenAI chat
        try:
            chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
            if not chat_deployment:
                return {"error": "AZURE_OPENAI_CHAT_DEPLOYMENT missing in .env"}
            
            client = get_azure_client()
            
            response = client.chat.completions.create(
                model=chat_deployment,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            
            # Step 6: Return structured response
            return {
                "answer": answer,
                "sources": sources,
                "chunks_used": high_score_results
            }
            
        except Exception as e:
            return {"error": f"Azure OpenAI chat failed: {str(e)}"}
    
    except Exception as e:
        return {"error": f"RAG answer generation failed: {str(e)}"}
