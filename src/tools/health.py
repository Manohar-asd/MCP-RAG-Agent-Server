def tool_definition():
    return {
        "name": "health_check",
        "description": "Check if MCP server is running",
        "input_schema": {"type": "object", "properties": {}},
        "output_schema": {"type": "object", "properties": {"status": {"type": "string"}}},
    }

def handler(payload: dict):
    return {"status": "ok"}
