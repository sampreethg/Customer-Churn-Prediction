"""RAG (Retrieval-Augmented Generation) Router.

Exposes endpoints for document ingestion and grounded question answering with citations.
Supports both JSON payload queries and directory/file indexing.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from ..schemas import RAGIngestRequest, RAGIngestResponse

router = APIRouter(prefix="/rag", tags=["Generative AI - Production RAG"])


class DirectoryIndexRequest(BaseModel):
    directory_path: str = Field(..., description="Path to folder containing PDF, MD, or TXT documents")


class FlexibleQueryRequest(BaseModel):
    query: Optional[str] = Field(None, description="Query string")
    question: Optional[str] = Field(None, description="Alternative question string")
    top_k: Optional[int] = Field(3, ge=1, le=20, description="Top-k citations to retrieve")
    dense_top_k: Optional[int] = Field(8, ge=1, le=50, description="Dense candidates")
    rerank_top_n: Optional[int] = Field(3, ge=1, le=10, description="Reranked top-n")


@router.post("/ingest", response_model=RAGIngestResponse)
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
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {str(exc)}")


@router.post("/index-docs")
async def index_docs_directory(payload: DirectoryIndexRequest, request: Request):
    """
    Batch index all documents (.txt, .md, .pdf) in a given directory path.
    """
    rag_pipeline = getattr(request.app.state, "rag_pipeline", None)
    if rag_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="RAG pipeline is not initialized on the gateway.",
        )

    dir_path = Path(payload.directory_path)
    if not dir_path.exists() or not dir_path.is_dir():
        # Create or handle gracefully
        dir_path.mkdir(parents=True, exist_ok=True)

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
        raise HTTPException(status_code=500, detail=f"RAG query execution failed: {str(exc)}")
