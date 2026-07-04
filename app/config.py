"""
Global configuration for the hackathon project.

The code is compatible with Pydantic v1/v2. If `pydantic-settings`
is installed, settings are validated through BaseSettings. Otherwise a
small stdlib fallback reads the same environment variables. This keeps
local hackathon runs from breaking on fresh Python environments.
"""

from __future__ import annotations

import os
from pathlib import Path

from .runtime.profiles import (
    apply_runtime_profile_policy,
    bool_from_env_or_profile,
    profile_default,
    runtime_profile_from_environment,
    str_from_env_or_profile,
)


_RUNTIME_PROFILE = runtime_profile_from_environment()

try:  # Pydantic v2 path
    from pydantic_settings import BaseSettings, SettingsConfigDict
    from pydantic import Field

    class Settings(BaseSettings):
        qdrant_host: str = Field("localhost", validation_alias="QDRANT_HOST")
        qdrant_port: int = Field(6333, validation_alias="QDRANT_PORT")
        qdrant_collection: str = Field("documents", validation_alias="QDRANT_COLLECTION")

        neo4j_uri: str = Field("bolt://localhost:7687", validation_alias="NEO4J_URI")
        neo4j_user: str = Field("neo4j", validation_alias="NEO4J_USER")
        neo4j_password: str = Field("password", validation_alias="NEO4J_PASSWORD")
        neo4j_database: str = Field("neo4j", validation_alias="NEO4J_DATABASE")
        kg_backend: str = Field("auto", validation_alias="KG_BACKEND")

        runtime_profile: str = Field(_RUNTIME_PROFILE, validation_alias="RUNTIME_PROFILE")
        retrieval_mode: str = Field(profile_default("RETRIEVAL_MODE", "bm25", _RUNTIME_PROFILE), validation_alias="RETRIEVAL_MODE")
        embedding_model: str = Field(profile_default("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", _RUNTIME_PROFILE), validation_alias="EMBEDDING_MODEL")
        embedding_model_path: str = Field("", validation_alias="EMBEDDING_MODEL_PATH")
        reranker_model: str = Field("BAAI/bge-reranker-v2-m3", validation_alias="RERANKER_MODEL")

        chunk_size: int = Field(400, validation_alias="CHUNK_SIZE")
        chunk_overlap: int = Field(50, validation_alias="CHUNK_OVERLAP")
        data_dir: Path = Field(Path("data"), validation_alias="DATA_DIR")
        metadata_db_path: Path = Field(Path("data/outbox.sqlite3"), validation_alias="METADATA_DB_PATH")
        catalog_db_path: Path = Field(Path("data/catalog.sqlite3"), validation_alias="CATALOG_DB_PATH")
        direct_qdrant_projection: bool = Field(profile_default("DIRECT_QDRANT_PROJECTION", False, _RUNTIME_PROFILE), validation_alias="DIRECT_QDRANT_PROJECTION")
        enable_local_embeddings: bool = Field(profile_default("ENABLE_LOCAL_EMBEDDINGS", False, _RUNTIME_PROFILE), validation_alias="ENABLE_LOCAL_EMBEDDINGS")
        eager_local_embeddings: bool = Field(profile_default("EAGER_LOCAL_EMBEDDINGS", False, _RUNTIME_PROFILE), validation_alias="EAGER_LOCAL_EMBEDDINGS")
        retrieval_query_expansion: bool = Field(profile_default("RETRIEVAL_QUERY_EXPANSION", True, _RUNTIME_PROFILE), validation_alias="RETRIEVAL_QUERY_EXPANSION")

        extraction_mode: str = Field(profile_default("EXTRACTION_MODE", "deterministic", _RUNTIME_PROFILE), validation_alias="EXTRACTION_MODE")
        extraction_min_confidence: float = Field(0.55, validation_alias="EXTRACTION_MIN_CONFIDENCE")
        extraction_enable_llm: bool = Field(profile_default("EXTRACTION_ENABLE_LLM", False, _RUNTIME_PROFILE), validation_alias="EXTRACTION_ENABLE_LLM")
        extraction_on_ingest: bool = Field(False, validation_alias="EXTRACTION_ON_INGEST")
        extraction_audit_dir: Path = Field(Path("data/extraction_audit"), validation_alias="EXTRACTION_AUDIT_DIR")

        parser_backend: str = Field("auto", validation_alias="PARSER_BACKEND")
        enable_ocr: bool = Field(False, validation_alias="ENABLE_OCR")
        ocr_backend: str = Field("none", validation_alias="OCR_BACKEND")
        scanned_pdf_min_text_chars: int = Field(50, validation_alias="SCANNED_PDF_MIN_TEXT_CHARS")
        parser_audit_dir: Path = Field(Path("data/parser_audit"), validation_alias="PARSER_AUDIT_DIR")

        analytics_max_facts: int = Field(30, validation_alias="ANALYTICS_MAX_FACTS")
        analytics_max_sources: int = Field(12, validation_alias="ANALYTICS_MAX_SOURCES")
        analytics_max_graph_nodes: int = Field(50, validation_alias="ANALYTICS_MAX_GRAPH_NODES")
        analytics_max_graph_edges: int = Field(80, validation_alias="ANALYTICS_MAX_GRAPH_EDGES")
        answer_synthesis_mode: str = Field(profile_default("ANSWER_SYNTHESIS_MODE", "template", _RUNTIME_PROFILE), validation_alias="ANSWER_SYNTHESIS_MODE")

        enable_llm: bool = Field(profile_default("ENABLE_LLM", False, _RUNTIME_PROFILE), validation_alias="ENABLE_LLM")
        llm_provider: str = Field(profile_default("LLM_PROVIDER", "offline", _RUNTIME_PROFILE), validation_alias="LLM_PROVIDER")
        llm_base_url: str = Field("http://localhost:11434", validation_alias="LLM_BASE_URL")
        llm_model: str = Field("qwen2.5:7b-instruct", validation_alias="LLM_MODEL")
        llm_api_key: str = Field("", validation_alias="LLM_API_KEY")
        llm_referer: str = Field("http://localhost:8501", validation_alias="LLM_REFERER")
        llm_app_title: str = Field("Scientific Knowledge Graph Demo", validation_alias="LLM_APP_TITLE")
        llm_timeout_seconds: int = Field(20, validation_alias="LLM_TIMEOUT_SECONDS")
        mistral_api_key: str = Field("", validation_alias="MISTRAL_API_KEY")
        mistral_base_url: str = Field("https://api.mistral.ai/v1", validation_alias="MISTRAL_BASE_URL")
        mistral_model: str = Field("mistral-small-latest", validation_alias="MISTRAL_MODEL")
        mistral_timeout_seconds: int = Field(60, validation_alias="MISTRAL_TIMEOUT_SECONDS")
        mistral_max_tokens: int = Field(1200, validation_alias="MISTRAL_MAX_TOKENS")
        mistral_temperature: float = Field(0.2, validation_alias="MISTRAL_TEMPERATURE")
        openrouter_api_key: str = Field("", validation_alias="OPENROUTER_API_KEY")
        openrouter_base_url: str = Field("https://openrouter.ai/api/v1", validation_alias="OPENROUTER_BASE_URL")
        openrouter_model: str = Field("", validation_alias="OPENROUTER_MODEL")
        yandex_api_key: str = Field("", validation_alias="YANDEX_API_KEY")
        yandex_folder_id: str = Field("", validation_alias="YANDEX_FOLDER_ID")
        yandex_model_uri: str = Field("", validation_alias="YANDEX_MODEL_URI")
        yandex_base_url: str = Field("https://ai.api.cloud.yandex.net/v1", validation_alias="YANDEX_BASE_URL")

        ingest_url_allow_private: bool = Field(False, validation_alias="INGEST_URL_ALLOW_PRIVATE")
        ingest_url_max_bytes: int = Field(10_485_760, validation_alias="INGEST_URL_MAX_BYTES")
        ingest_url_timeout_seconds: int = Field(10, validation_alias="INGEST_URL_TIMEOUT_SECONDS")
        max_upload_mb: int = Field(25, validation_alias="MAX_UPLOAD_MB")
        max_upload_files: int = Field(20, validation_alias="MAX_UPLOAD_FILES")
        allowed_upload_extensions: str = Field(
            ".pdf,.docx,.pptx,.xlsx,.csv,.html,.htm,.txt,.md",
            validation_alias="ALLOWED_UPLOAD_EXTENSIONS",
        )

        model_config = SettingsConfigDict(env_file=".env", extra="ignore")

except Exception:  # pragma: no cover - fallback for very small environments
    class Settings:
        qdrant_host: str = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port: int = int(os.getenv("QDRANT_PORT", "6333"))
        qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "documents")

        neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password: str = os.getenv("NEO4J_PASSWORD", "password")
        neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")
        kg_backend: str = os.getenv("KG_BACKEND", "auto")

        runtime_profile: str = _RUNTIME_PROFILE
        retrieval_mode: str = str_from_env_or_profile("RETRIEVAL_MODE", "bm25", _RUNTIME_PROFILE)
        embedding_model: str = str_from_env_or_profile("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", _RUNTIME_PROFILE)
        embedding_model_path: str = os.getenv("EMBEDDING_MODEL_PATH", "")
        reranker_model: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

        chunk_size: int = int(os.getenv("CHUNK_SIZE", "400"))
        chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))
        data_dir: Path = Path(os.getenv("DATA_DIR", "data"))
        metadata_db_path: Path = Path(os.getenv("METADATA_DB_PATH", "data/outbox.sqlite3"))
        catalog_db_path: Path = Path(os.getenv("CATALOG_DB_PATH", "data/catalog.sqlite3"))
        direct_qdrant_projection: bool = bool_from_env_or_profile("DIRECT_QDRANT_PROJECTION", False, _RUNTIME_PROFILE)
        enable_local_embeddings: bool = bool_from_env_or_profile("ENABLE_LOCAL_EMBEDDINGS", False, _RUNTIME_PROFILE)
        eager_local_embeddings: bool = bool_from_env_or_profile("EAGER_LOCAL_EMBEDDINGS", False, _RUNTIME_PROFILE)
        retrieval_query_expansion: bool = bool_from_env_or_profile("RETRIEVAL_QUERY_EXPANSION", True, _RUNTIME_PROFILE)

        extraction_mode: str = str_from_env_or_profile("EXTRACTION_MODE", "deterministic", _RUNTIME_PROFILE)
        extraction_min_confidence: float = float(os.getenv("EXTRACTION_MIN_CONFIDENCE", "0.55"))
        extraction_enable_llm: bool = bool_from_env_or_profile("EXTRACTION_ENABLE_LLM", False, _RUNTIME_PROFILE)
        extraction_on_ingest: bool = os.getenv("EXTRACTION_ON_INGEST", "false").lower() in {"1", "true", "yes"}
        extraction_audit_dir: Path = Path(os.getenv("EXTRACTION_AUDIT_DIR", "data/extraction_audit"))

        parser_backend: str = os.getenv("PARSER_BACKEND", "auto")
        enable_ocr: bool = os.getenv("ENABLE_OCR", "false").lower() in {"1", "true", "yes"}
        ocr_backend: str = os.getenv("OCR_BACKEND", "none")
        scanned_pdf_min_text_chars: int = int(os.getenv("SCANNED_PDF_MIN_TEXT_CHARS", "50"))
        parser_audit_dir: Path = Path(os.getenv("PARSER_AUDIT_DIR", "data/parser_audit"))

        analytics_max_facts: int = int(os.getenv("ANALYTICS_MAX_FACTS", "30"))
        analytics_max_sources: int = int(os.getenv("ANALYTICS_MAX_SOURCES", "12"))
        analytics_max_graph_nodes: int = int(os.getenv("ANALYTICS_MAX_GRAPH_NODES", "50"))
        analytics_max_graph_edges: int = int(os.getenv("ANALYTICS_MAX_GRAPH_EDGES", "80"))
        answer_synthesis_mode: str = str_from_env_or_profile("ANSWER_SYNTHESIS_MODE", "template", _RUNTIME_PROFILE)

        enable_llm: bool = bool_from_env_or_profile("ENABLE_LLM", False, _RUNTIME_PROFILE)
        llm_provider: str = str_from_env_or_profile("LLM_PROVIDER", "offline", _RUNTIME_PROFILE)
        llm_base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:11434")
        llm_model: str = os.getenv("LLM_MODEL", "qwen2.5:7b-instruct")
        llm_api_key: str = os.getenv("LLM_API_KEY", "")
        llm_referer: str = os.getenv("LLM_REFERER", "http://localhost:8501")
        llm_app_title: str = os.getenv("LLM_APP_TITLE", "Scientific Knowledge Graph Demo")
        llm_timeout_seconds: int = int(os.getenv("LLM_TIMEOUT_SECONDS", "20"))
        mistral_api_key: str = os.getenv("MISTRAL_API_KEY", "")
        mistral_base_url: str = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1")
        mistral_model: str = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
        mistral_timeout_seconds: int = int(os.getenv("MISTRAL_TIMEOUT_SECONDS", "60"))
        mistral_max_tokens: int = int(os.getenv("MISTRAL_MAX_TOKENS", "1200"))
        mistral_temperature: float = float(os.getenv("MISTRAL_TEMPERATURE", "0.2"))
        openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")
        openrouter_base_url: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        openrouter_model: str = os.getenv("OPENROUTER_MODEL", "")
        yandex_api_key: str = os.getenv("YANDEX_API_KEY", "")
        yandex_folder_id: str = os.getenv("YANDEX_FOLDER_ID", "")
        yandex_model_uri: str = os.getenv("YANDEX_MODEL_URI", "")
        yandex_base_url: str = os.getenv("YANDEX_BASE_URL", "https://ai.api.cloud.yandex.net/v1")

        ingest_url_allow_private: bool = os.getenv("INGEST_URL_ALLOW_PRIVATE", "false").lower() in {"1", "true", "yes"}
        ingest_url_max_bytes: int = int(os.getenv("INGEST_URL_MAX_BYTES", "10485760"))
        ingest_url_timeout_seconds: int = int(os.getenv("INGEST_URL_TIMEOUT_SECONDS", "10"))
        max_upload_mb: int = int(os.getenv("MAX_UPLOAD_MB", "25"))
        max_upload_files: int = int(os.getenv("MAX_UPLOAD_FILES", "20"))
        allowed_upload_extensions: str = os.getenv("ALLOWED_UPLOAD_EXTENSIONS", ".pdf,.docx,.pptx,.xlsx,.csv,.html,.htm,.txt,.md")


settings = Settings()
apply_runtime_profile_policy(settings)
