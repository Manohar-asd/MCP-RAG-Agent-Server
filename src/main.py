import logging
import os
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.clients.azure_openai import get_azure_client
from src.registry import list_tools, call_tool
from src.tools.rag_answer import handler as rag_answer_handler
from src.tools.upsert_document import handler as upsert_document_handler

app = FastAPI(title="MCP Server")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agent")

class ToolCallRequest(BaseModel):
    tool_name: str
    payload: dict = {}


class AgentRequest(BaseModel):
    message: str


def classify_intent(message: str) -> str:
    """Classify intent using Azure OpenAI chat or simple prefix rules."""
    lowered = message.lower().strip()

    # Shortcut: explicit store/save prefix
    if lowered.startswith("store:") or lowered.startswith("save:"):
        return "store"

    # Try LLM classification; fall back to QA on error/missing config
    chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
    if not chat_deployment:
        logger.warning("AZURE_OPENAI_CHAT_DEPLOYMENT missing; defaulting to QA intent")
        return "qa"

    try:
        client = get_azure_client()
        response = client.chat.completions.create(
            model=chat_deployment,
            messages=[
                {
                    "role": "system",
                    "content": "Classify the user's intent. Respond with 'store' if they want to save knowledge, otherwise respond with 'qa'. Reply with a single word: store or qa.",
                },
                {"role": "user", "content": message},
            ],
            temperature=0,
            max_tokens=5,
        )
        intent = response.choices[0].message.content.strip().lower()
        return "store" if "store" in intent else "qa"
    except Exception as exc:
        logger.warning("Intent classification failed; defaulting to QA intent: %s", exc)
        return "qa"

@app.get("/tools")
def tools():
    return {"tools": list_tools()}

@app.post("/tool-call")
def tool_call(req: ToolCallRequest):
    try:
        result = call_tool(req.tool_name, req.payload)
        return {"tool_name": req.tool_name, "result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/agent")
def agent(req: AgentRequest):
    """
    Simple agent loop that routes between QA and storage workflows.
    - Classify intent (QA vs store) using Azure OpenAI chat and prefix hints
    - Calls rag_answer for Q&A
    - Calls upsert_document when prefixed with store:/save:
    """
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message cannot be empty")

    actions_taken = []

    try:
        intent = classify_intent(message)
        logger.info("Agent intent resolved: %s", intent)

        if intent == "store":
            # strip leading directive
            content = message
            if ":" in message:
                content = message.split(":", 1)[1].strip() or message

            doc_id = f"agent-{uuid4().hex[:8]}"
            payload = {
                "id": doc_id,
                "text": content,
                "metadata": {"source": "agent", "origin": "agent_workflow"},
            }

            result = upsert_document_handler(payload)
            actions_taken.append({"tool": "upsert_document", "payload": payload, "result": result})

            if isinstance(result, dict) and result.get("error"):
                reply = f"Failed to store: {result['error']}"
            else:
                reply = "Stored in vector database."
        else:
            qa_payload = {"query": message, "top_k": 5}
            qa_result = rag_answer_handler(qa_payload)
            actions_taken.append({"tool": "rag_answer", "payload": qa_payload, "result": qa_result})

            if isinstance(qa_result, dict) and qa_result.get("error"):
                reply = f"Unable to answer: {qa_result['error']}"
            else:
                reply = qa_result.get("answer") if isinstance(qa_result, dict) else None
                if not reply:
                    reply = "I don't have enough context to answer."
    except Exception as exc:
        logger.exception("Agent workflow failed: %s", exc)
        reply = "I don't have enough context to answer."
        actions_taken.append({"error": str(exc)})

    return {"reply": reply, "actions_taken": actions_taken}
