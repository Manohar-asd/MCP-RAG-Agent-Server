from src.tools import health, embed_text, upsert_document, vector_search, rag_answer

TOOLS = {
    "health_check": health,
    "embed_text": embed_text,
    "upsert_document": upsert_document,
    "vector_search": vector_search,
    "rag_answer": rag_answer
}

def list_tools():
    return [TOOLS[name].tool_definition() for name in TOOLS]

def call_tool(tool_name: str, payload: dict):
    if tool_name not in TOOLS:
        raise ValueError(f"Tool not found: {tool_name}")
    return TOOLS[tool_name].handler(payload)
