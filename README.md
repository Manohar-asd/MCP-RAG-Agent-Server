# MCP-RAG-Agent-Server

MCP (Model Context Protocol) RAG (Retrieval-Augmented Generation) Agent Server with Azure OpenAI and ChromaDB vector store integration.

## Features

- **Azure OpenAI Integration**: Embed text and chat with Azure OpenAI models
- **Vector Store**: ChromaDB-based persistent vector storage with semantic search
- **RAG Tools**: Upsert documents with embeddings and perform vector similarity search
- **MCP Compatible**: FastAPI-based server for tool registration and execution

## Architecture

```
src/
├── clients/
│   ├── azure_openai.py      # Azure OpenAI client initialization
│   └── __init__.py
├── tools/
│   ├── health.py            # Health check tool
│   ├── embed_text.py        # Text embedding tool
│   ├── upsert_document.py   # Document upsert with embedding
│   ├── vector_search.py     # Vector similarity search
│   └── __pycache__/
├── vector_store/
│   ├── chroma_store.py      # ChromaDB wrapper functions
│   └── __init__.py
├── config.py
├── main.py                  # FastAPI server
└── registry.py              # Tool registry and dispatcher
```

## Setup

### Prerequisites

- Python 3.8+
- Azure OpenAI API credentials

### Installation

1. Clone repository and install dependencies:
```bash
pip install -r requirements.txt
```

2. Create `.env` file with Azure OpenAI credentials:
```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_KEY=your-api-key
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-3-small
AZURE_OPENAI_CHAT_DEPLOYMENT=gpt-4
```

### Running the Server

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

Server starts at: `http://localhost:8000`

## Available Tools

### 1. health_check
Check server status.

**Input**: `{}`

**Output**:
```json
{"status": "ok"}
```

### 2. embed_text
Generate embedding vector for text using Azure OpenAI.

**Input**:
```json
{
  "text": "hello world"
}
```

**Output**:
```json
{
  "embedding": [0.1234, -0.5678, ...],
  "dim": 1536
}
```

### 3. upsert_document
Embed text and store document in vector store.

**Input**:
```json
{
  "id": "doc-001",
  "text": "The quick brown fox",
  "metadata": {
    "source": "book",
    "author": "anonymous",
    "page": 42
  }
}
```

**Output**:
```json
{
  "status": "success",
  "id": "doc-001",
  "embedding_dim": 1536
}
```

### 4. vector_search
Search stored documents by semantic similarity.

**Input**:
```json
{
  "query": "quick animals",
  "top_k": 5
}
```

**Output**:
```json
{
  "results": [
    {
      "id": "doc-001",
      "text": "The quick brown fox",
      "score": 0.8765,
      "metadata": {
        "source": "book",
        "author": "anonymous"
      }
    }
  ],
  "count": 1
}
```

### 5. rag_answer
Retrieve relevant documents and generate answers using Azure OpenAI chat. Combines vector search with LLM reasoning.

**Input**:
```json
{
  "query": "What is machine learning?",
  "top_k": 5
}
```

**Output**:
```json
{
  "answer": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It uses algorithms to analyze data, identify patterns, and make predictions.",
  "sources": [
    {
      "source": "documentation",
      "topic": "AI/ML"
    }
  ],
  "chunks_used": [
    {
      "id": "doc-001",
      "text": "Machine learning enables systems to learn from data patterns...",
      "score": 0.92,
      "metadata": {
        "source": "documentation",
        "topic": "AI/ML"
      }
    }
  ]
}
```

## API Endpoints

### GET /tools
List all registered MCP tools with their schemas.

**Response**:
```json
{
  "tools": [
    {
      "name": "health_check",
      "description": "Check if MCP server is running",
      "input_schema": {...},
      "output_schema": {...}
    },
    ...
  ]
}
```

### POST /tool-call
Execute a registered tool.

**Request**:
```json
{
  "tool_name": "embed_text",
  "payload": {
    "text": "hello world"
  }
}
```

**Response**:
```json
{
  "tool_name": "embed_text",
  "result": {
    "embedding": [...],
    "dim": 1536
  }
}
```

### POST /agent
Agentic workflow that classifies intent and routes to MCP tools internally.

- If the message starts with `store:` or `save:`, it stores the text via `upsert_document`.
- Otherwise, it runs `rag_answer` to answer using the vector store.

**Request (Q&A)**:
```json
{
  "message": "Explain what vector databases do"
}
```

**Request (Store)**:
```json
{
  "message": "store: Transformers are attention-based models used in LLMs."
}
```

**Response Shape**:
```json
{
  "reply": "string",
  "actions_taken": [
    {
      "tool": "rag_answer",
      "payload": {"query": "...", "top_k": 5},
      "result": {...}
    }
  ]
}
```

Safe fallback: If intent classification or tool execution fails, the agent replies with a friendly message and includes any errors in `actions_taken`.

## Swagger UI

Interactive API documentation available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Vector Store

ChromaDB persists data to `./chroma_data` directory. The vector store:

- Uses cosine distance metric
- Automatically initializes `rag_documents` collection on first use
- Stores document embeddings alongside text and metadata
- Returns similarity scores (0-1) where 1 = exact match

## Usage Example

```bash
# 1. Embed and store documents
curl -X POST http://localhost:8000/tool-call \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "upsert_document",
    "payload": {
      "id": "doc-1",
      "text": "Python is a great programming language",
      "metadata": {"type": "article"}
    }
  }'

## Usage Example

```bash
# 1. Embed and store documents
curl -X POST http://localhost:8000/tool-call \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "upsert_document",
    "payload": {
      "id": "doc-1",
      "text": "Python is a great programming language used for data science",
      "metadata": {"type": "article", "topic": "programming"}
    }
  }'

curl -X POST http://localhost:8000/tool-call \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "upsert_document",
    "payload": {
      "id": "doc-2",
      "text": "Machine learning algorithms power modern AI applications",
      "metadata": {"type": "article", "topic": "AI"}
    }
  }'

# 2. Search similar documents
curl -X POST http://localhost:8000/tool-call \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "vector_search",
    "payload": {
      "query": "programming languages",
      "top_k": 5
    }
  }'

# 3. Generate answer with RAG (Retrieval-Augmented Generation)
curl -X POST http://localhost:8000/tool-call \
  -H "Content-Type: application/json" \
  -d '{
    "tool_name": "rag_answer",
    "payload": {
      "query": "What programming language is used for data science?",
      "top_k": 3
    }
  }'
```

**RAG Answer Response Example**:
```json
{
  "tool_name": "rag_answer",
  "result": {
    "answer": "Python is a great programming language used for data science. It provides powerful libraries and frameworks that make data analysis and modeling efficient.",
    "sources": [
      {"type": "article", "topic": "programming"}
    ],
    "chunks_used": [
      {
        "id": "doc-1",
        "text": "Python is a great programming language used for data science",
        "score": 0.95,
        "metadata": {"type": "article", "topic": "programming"}
      }
    ]
  }
}
```

## RAG Workflow

The `rag_answer` tool implements a complete RAG pipeline:

1. **Query Understanding**: User submits a question
2. **Vector Search**: Retrieves top-k semantically similar documents from ChromaDB
3. **Context Building**: Constructs a prompt with numbered chunks and metadata
4. **LLM Generation**: Sends context + query to Azure OpenAI chat model
5. **Answer Assembly**: Returns answer with source attribution and chunk details

**Score Threshold**: Results with similarity score < 0.5 are filtered out to ensure quality context.

**Fallback Response**: If no high-confidence matches, returns "I don't have enough context to answer this question."

## Dependencies

- **fastapi**: Web framework
- **uvicorn**: ASGI server
- **openai**: Azure OpenAI SDK
- **chromadb**: Vector database
- **python-dotenv**: Environment configuration
- **pydantic**: Data validation

## License

MIT