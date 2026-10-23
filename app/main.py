"""HTTP layer. Routes validate input, call a service and return a schema. No business logic lives here."""
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from app.config import Settings, get_settings
from app.container import build_services
from app.parsing import UnsupportedFileType
from app.schemas import DocumentInfo, DocumentRecord, QueryRequest, QueryResponse
from app.services.ingest import EmptyDocument, info_of


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    services = build_services(settings)
    app = FastAPI(title="EMS RAG proof of concept", version="0.1.0")

    def _record_or_404(doc_id: str) -> DocumentRecord:
        record = services.index.documents.get(doc_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Document not found")
        return record

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "llm_enabled": services.query.answerer.enabled,
            "embedder": settings.embedder,
            "reranker": settings.reranker,
        }

    @app.post("/documents", response_model=DocumentInfo, status_code=201)
    async def upload(
        file: UploadFile = File(...),
        access_level: str = Query("public", min_length=1, max_length=32),
    ) -> DocumentInfo:
        content = await file.read()
        if len(content) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File is too large")
        try:
            return services.ingest.ingest(file.filename or "upload", content, access_level)
        except UnsupportedFileType as exc:
            raise HTTPException(status_code=415, detail=str(exc)) from exc
        except EmptyDocument as exc:
            raise HTTPException(status_code=422, detail="No readable text was found in this file") from exc

    @app.get("/documents/{doc_id}", response_model=DocumentInfo)
    def get_document(doc_id: str) -> DocumentInfo:
        return info_of(_record_or_404(doc_id))

    @app.get("/documents/{doc_id}/download")
    def download(doc_id: str) -> Response:
        record = _record_or_404(doc_id)
        safe_name = record.name.replace('"', "")
        return Response(
            content=record.content,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{safe_name}"'},
        )

    @app.post("/query", response_model=QueryResponse)
    def query(request: QueryRequest) -> QueryResponse:
        if not request.question.strip():
            raise HTTPException(status_code=422, detail="Question is empty")
        return services.query.answer(request.question, tuple(request.access_levels))

    return app


app = create_app()
