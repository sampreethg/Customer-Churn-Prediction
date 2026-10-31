"""RAG (Retrieval-Augmented Generation) Router.

Exposes endpoints for document ingestion and grounded question answering with citations.
Supports both JSON payload queries and directory/file indexing.
"""

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ..rate_limiter import rate_limit
from ..schemas import RAGIngestRequest, RAGIngestResponse

logger = logging.getLogger("rag.router")
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent.parent

router = APIRouter(prefix="/rag", tags=["Generative AI - Production RAG"])


class DirectoryIndexRequest(BaseModel):
    directory_path: str = Field(..., max_length=260, description="Path to folder containing PDF, MD, or TXT documents")


class FlexibleQueryRequest(BaseModel):
    query: Optional[str] = Field(None, max_length=1000, description="Query string")
    question: Optional[str] = Field(None, max_length=1000, description="Alternative question string")
    top_k: Optional[int] = Field(3, ge=1, le=20, description="Top-k citations to retrieve")
    dense_top_k: Optional[int] = Field(8, ge=1, le=50, description="Dense candidates")
    rerank_top_n: Optional[int] = Field(3, ge=1, le=10, description="Reranked top-n")


@router.post("/ingest", response_model=RAGIngestResponse, dependencies=[Depends(rate_limit(5))])
async def ingest_document(payload: RAGIngestRequest, request: Request):
    """
    Ingest and index a document into the RAG vector store with recursive chunking.
    """
    rag_pipeline = getattr(request.app.state, "rag_pipeline", None)
    if rag_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not initialized on the gateway.",
        )

    try:
        result = rag_pipeline.ingest_document(doc_id=payload.doc_id, text=payload.text)
        return RAGIngestResponse(**result)
    except Exception as exc:
        logger.error("Document ingestion error: %s", exc)
        raise HTTPException(status_code=500, detail="Document ingestion failed.") from None


@router.post("/index-docs", dependencies=[Depends(rate_limit(5))])
async def index_docs_directory(payload: DirectoryIndexRequest, request: Request):
    """
    Batch index all documents (.txt, .md, .pdf) in a given directory path within workspace.
    """
    rag_pipeline = getattr(request.app.state, "rag_pipeline", None)
    if rag_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not initialized on the gateway.",
        )

    raw_path = payload.directory_path.strip()
    if not raw_path or ".." in raw_path or "\0" in raw_path:
        raise HTTPException(
            status_code=400,
            detail="Invalid directory path: traversal sequences are forbidden.",
        )

    resolved_path = (WORKSPACE_ROOT / raw_path).resolve() if not Path(raw_path).is_absolute() else Path(raw_path).resolve()
    try:
        resolved_path.relative_to(WORKSPACE_ROOT)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Forbidden path: directory must reside within repository workspace.",
        ) from None

    if not resolved_path.exists() or not resolved_path.is_dir():
        resolved_path.mkdir(parents=True, exist_ok=True)

    dir_path = resolved_path

    indexed_count = 0
    total_chunks = 0

    for file_path in dir_path.glob("*"):
        if file_path.suffix.lower() in [".txt", ".md"]:
            try:
                content = file_path.read_text(encoding="utf-8", errors="ignore")
                res = rag_pipeline.ingest_document(doc_id=file_path.stem, text=content)
                indexed_count += 1
                total_chunks += res.get("chunks_created", 0)
            except Exception:
                continue
        elif file_path.suffix.lower() == ".pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(str(file_path))
                text_pages = [page.extract_text() or "" for page in reader.pages]
                full_text = "\n\n".join(text_pages)
                res = rag_pipeline.ingest_document(doc_id=file_path.stem, text=full_text)
                indexed_count += 1
                total_chunks += res.get("chunks_created", 0)
            except Exception:
                continue

    return {
        "status": "success",
        "documents_indexed": indexed_count,
        "total_chunks": total_chunks,
        "total_store_chunks": rag_pipeline.indexer.total_chunks,
    }


@router.post("/query")
async def query_rag(payload: FlexibleQueryRequest, request: Request):
    """
    Query the indexed knowledge base with dense retrieval, cross-encoder re-ranking,
    and hallucination-safe citation tracking.
    """
    rag_pipeline = getattr(request.app.state, "rag_pipeline", None)
    if rag_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not initialized on the gateway.",
        )

    user_query = payload.query or payload.question
    if not user_query or not user_query.strip():
        raise HTTPException(status_code=400, detail="Must provide 'query' or 'question'")

    top_k = payload.top_k or payload.rerank_top_n or 3
    dense_k = max(payload.dense_top_k or 8, top_k * 2)

    try:
        result = rag_pipeline.query(
            question=user_query,
            dense_top_k=dense_k,
            rerank_top_n=top_k,
        )

        # Format citations to align with both schemas
        standard_citations = []
        for cit in result.get("citations", []):
            standard_citations.append({
                "citation_id": cit.get("citation_id", 1),
                "doc_id": cit.get("doc_id", "doc"),
                "document": f"{cit.get('doc_id', 'doc')}.md",
                "chunk_id": cit.get("chunk_id", "c001"),
                "relevance_score": cit.get("relevance_score", 0.90),
                "excerpt": cit.get("excerpt", ""),
            })

        return {
            "question": user_query,
            "query": user_query,
            "answer": result.get("answer", ""),
            "citations": standard_citations,
            "retrieved_chunks": result.get("retrieved_chunks", []),
            "confidence_score": result.get("confidence_score", 0.0),
            "retrieval_latency_ms": result.get("latency_ms", 0.0),
            "latency_ms": result.get("latency_ms", 0.0),
            "hallucination_guardrail_triggered": result.get("hallucination_guardrail_triggered", False),
        }
    except Exception as exc:
        logger.error("RAG query execution error: %s", exc)
        raise HTTPException(status_code=500, detail="RAG query execution failed.") from None
