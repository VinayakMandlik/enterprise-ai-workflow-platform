"""
FastAPI entrypoint.

Run with:  uvicorn app.main:app --reload
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from app.mcp.tool_client import load_mcp_tools
from app.agents.graph import build_graph, run_intake_agent, run_scheduled_digest_agent
from app.memory.long_term import get_long_term_memory
from app.storage.s3_client import upload_document, list_documents, delete_document
from app.rag.ingest import ingest_text, extract_pdf_text
from app.rag.vector_store import delete_by_source
from app.scheduler import start_scheduler, stop_scheduler

_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once when the server starts: connect to the MCP server,
    # discover tools, and build the LangGraph agent — all reused
    # across every incoming request.
    registry, tools = await load_mcp_tools()
    _state["mcp_registry"] = registry
    _state["graph"] = build_graph(tools)

    # Start the Scheduled Agent's background timer.
    start_scheduler()

    yield  # server runs and handles requests here

    # Runs once when the server shuts down: clean up.
    stop_scheduler()
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
async def upload_document_endpoint(file: UploadFile = File(...), user_id: str = "demo-user"):
    contents = await file.read()

    # 1. Store the raw file in cloud storage (Supabase, S3-compatible)
    storage_key = upload_document(contents, file.filename)

    # 2. Extract text depending on file type, then ingest into RAG pipeline
    if file.filename.lower().endswith(".pdf"):
        text = extract_pdf_text(contents)
    else:
        text = contents.decode("utf-8", errors="ignore")

    chunk_count = ingest_text(text, source_name=file.filename)

    # 3. Document Intake Agent: automatically analyze the document the
    # moment it's uploaded, without waiting for a user question.
    intake_result = run_intake_agent(text, file.filename, user_id=user_id)

    return {
        "filename": file.filename,
        "storage_key": storage_key,
        "chunks_ingested": chunk_count,
        "intake_analysis": intake_result["analysis"],
    }


@app.get("/documents")
async def list_documents_endpoint():
    return {"documents": list_documents()}


@app.delete("/documents/{filename}")
async def delete_document_endpoint(filename: str, user_id: str = "demo-user"):
    storage_key = f"documents/{filename}"
    delete_document(storage_key)
    delete_by_source(filename)

    # Also remove the Document Intake Agent's saved analysis for this
    # file — otherwise stale summaries linger in the digest forever.
    memory = get_long_term_memory()
    memory.forget(user_id=user_id, key=f"document_intake:{filename}")

    return {"deleted": filename}


@app.post("/agents/digest")
async def trigger_digest_endpoint(user_id: str = "demo-user"):
    """Manually trigger the Scheduled Agent's digest on demand (in
    addition to it running automatically on a timer)."""
    return run_scheduled_digest_agent(user_id)