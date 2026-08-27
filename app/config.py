import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ---- LLM provider switch ----
    # "groq"   -> free, fast cloud inference (default)
    # "openai" -> uses OPENAI_API_KEY
    # "azure"  -> uses AZURE_OPENAI_* vars (real enterprise setup)
    llm_provider: str = os.getenv("LLM_PROVIDER", "groq")

    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    azure_openai_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_openai_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_openai_deployment: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
    azure_openai_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")

    # ---- Embeddings ----
    # Groq doesn't serve embedding models, so we use Hugging Face's
    # free inference API just for embeddings.
    huggingface_api_key: str = os.getenv("HUGGINGFACE_API_KEY", "")
    huggingface_embedding_model: str = os.getenv(
        "HUGGINGFACE_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # ---- Vector store switch ----
    vector_store: str = os.getenv("VECTOR_STORE", "qdrant")
    qdrant_url: str = os.getenv("QDRANT_URL", "")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "enterprise_docs")

    # ---- Tracing ----
    langsmith_enabled: bool = os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    langsmith_api_key: str = os.getenv("LANGCHAIN_API_KEY", "")
    langsmith_project: str = os.getenv("LANGCHAIN_PROJECT", "enterprise-ai-workflow")

    # ---- Memory persistence ----
    memory_backend: str = os.getenv("MEMORY_BACKEND", "sqlite")
    memory_db_path: str = os.getenv("MEMORY_DB_PATH", "./data/memory.db")


    # ---- Cloud document storage (Supabase Storage, S3-compatible API) ----
    r2_endpoint_url: str = os.getenv("R2_ENDPOINT_URL", "")
    r2_access_key_id: str = os.getenv("R2_ACCESS_KEY_ID", "")
    r2_secret_access_key: str = os.getenv("R2_SECRET_ACCESS_KEY", "")
    r2_region: str = os.getenv("R2_REGION", "us-east-1")
    r2_bucket_name: str = os.getenv("R2_BUCKET_NAME", "enterprise-documents")

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()