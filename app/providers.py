"""
Provider factories.

Every place in the codebase that needs an LLM or an embedding model calls
get_llm() / get_embeddings() from here — never a provider SDK directly.
This is the seam that makes the platform provider-agnostic: swapping
Groq for Azure OpenAI later means changing one field in .env, not
touching any agent or RAG code.
"""
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings

from app.config import get_settings


def get_llm(temperature: float = 0.0) -> BaseChatModel:
    settings = get_settings()

    if settings.llm_provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=temperature,
        )

    if settings.llm_provider == "azure":
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_api_key,
            azure_deployment=settings.azure_openai_deployment,
            api_version=settings.azure_openai_api_version,
            temperature=temperature,
        )

    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            temperature=temperature,
        )

    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider}")


def get_embeddings() -> Embeddings:
    """
    Runs a local embedding model (downloads once, then fully offline).
    This avoids Hugging Face's Inference API permission restrictions
    entirely, and has zero rate limits since it runs on your own CPU.
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    settings = get_settings()
    return HuggingFaceEmbeddings(model_name=settings.huggingface_embedding_model)