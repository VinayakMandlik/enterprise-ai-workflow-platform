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

    if settings.llm_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=temperature,
        )

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
    Uses fastembed (ONNX Runtime based) instead of sentence-transformers
    (PyTorch based) — dramatically lighter memory footprint, critical
    for running on memory-constrained free-tier hosting like Render.
    """
    from langchain_community.embeddings import FastEmbedEmbeddings

    settings = get_settings()
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
def extract_text(message) -> str:
    """
    Normalizes an LLM response's .content into a plain string,
    regardless of provider. Some providers (Gemini) return a list of
    structured content blocks instead of a plain string — this
    function handles both shapes consistently everywhere in the app.
    """
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)