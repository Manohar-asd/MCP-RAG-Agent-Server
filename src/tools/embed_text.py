import os
from src.clients.azure_openai import get_azure_client

def tool_definition():
    return {
        "name": "embed_text",
        "description": "Generate embedding vector for input text using Azure OpenAI",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"}
            },
            "required": ["text"]
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "embedding": {"type": "array", "items": {"type": "number"}},
                "dim": {"type": "integer"}
            }
        }
    }

def handler(payload: dict):
    text = payload.get("text", "").strip()
    if not text:
        return {"error": "text cannot be empty"}

    deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    if not deployment:
        return {"error": "AZURE_OPENAI_EMBEDDING_DEPLOYMENT missing in .env"}

    client = get_azure_client()

    resp = client.embeddings.create(
        model=deployment,
        input=text
    )

    embedding = resp.data[0].embedding
    return {"embedding": embedding, "dim": len(embedding)}
