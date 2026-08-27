"""
FastAPI entrypoint.

Run with:  uvicorn app.main:app --reload
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from app.mcp.tool_client import load_mcp_tools
from app.agents.graph import build_graph
from app.memory.long_term import get_long_term_memory

from fastapi import UploadFile, File
from app.storage.s3_client import upload_document, list_documents
from app.rag.ingest import ingest_text
from fastapi.staticfiles import StaticFiles
_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once when the server starts: connect to the MCP server,
    # discover tools, and build the LangGraph agent — all reused
    # across every incoming request.
    registry, tools = await load_mcp_tools()
    _state["mcp_registry"] = registry
    _state["graph"] = build_graph(tools)

    yield  # server runs and handles requests here

    # Runs once when the server shuts down: clean up the MCP subprocess.
    await registry.close()


app = FastAPI(title="Enterprise AI Workflow Automation Platform", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


class WorkflowRequest(BaseModel):
    message: str
    user_id: str = "demo-user"
    thread_id: str = "default-thread"


class WorkflowResponse(BaseModel):
    response: str
    route: str

from fastapi.responses import FileResponse

@app.get("/")
async def serve_frontend():
    return FileResponse("static/index.html")
@app.post("/workflow/run", response_model=WorkflowResponse)
async def run_workflow(req: WorkflowRequest):
    graph = _state["graph"]
    config = {"configurable": {"thread_id": req.thread_id}}

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=req.message)], "user_id": req.user_id, "route": ""},
        config=config,
    )
    final_message = result["messages"][-1]
    return WorkflowResponse(response=final_message.content, route=result.get("route", "general"))


@app.get("/memory/{user_id}")
async def get_memory(user_id: str):
    memory = get_long_term_memory()
    return memory.recall_all(user_id)


@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/documents/upload")
async def upload_document_endpoint(file: UploadFile = File(...)):
    contents = await file.read()

    # 1. Store the raw file in cloud storage (Supabase, S3-compatible)
    storage_key = upload_document(contents, file.filename)

    # 2. Decode and ingest into the RAG pipeline (chunk + embed + Qdrant)
    text = contents.decode("utf-8", errors="ignore")
    chunk_count = ingest_text(text, source_name=file.filename)

    return {
        "filename": file.filename,
        "storage_key": storage_key,
        "chunks_ingested": chunk_count,
    }


@app.get("/documents")
async def list_documents_endpoint():
    return {"documents": list_documents()}