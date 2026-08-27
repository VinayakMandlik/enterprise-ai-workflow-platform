"""
Cloud document storage client — S3-compatible API (works with Supabase
Storage, Cloudflare R2, or real AWS S3 with zero code changes, only
different .env values).
"""
import boto3
from app.config import get_settings

_client = None


def get_s3_client():
    global _client
    if _client is not None:
        return _client

    settings = get_settings()
    _client = boto3.client(
        "s3",
        endpoint_url=settings.r2_endpoint_url,
        aws_access_key_id=settings.r2_access_key_id,
        aws_secret_access_key=settings.r2_secret_access_key,
        region_name=settings.r2_region,
    )
    return _client


def upload_document(file_bytes: bytes, filename: str) -> str:
    """
    Uploads a document to cloud storage. Returns the storage key
    (path) it was saved under.
    """
    settings = get_settings()
    client = get_s3_client()

    key = f"documents/{filename}"
    client.put_object(
        Bucket=settings.r2_bucket_name,
        Key=key,
        Body=file_bytes,
    )
    return key


def download_document(key: str) -> bytes:
    """Downloads a document's raw bytes from cloud storage."""
    settings = get_settings()
    client = get_s3_client()

    response = client.get_object(Bucket=settings.r2_bucket_name, Key=key)
    return response["Body"].read()


def list_documents() -> list[str]:
    """Lists all document keys currently in the bucket."""
    settings = get_settings()
    client = get_s3_client()

    response = client.list_objects_v2(
        Bucket=settings.r2_bucket_name, Prefix="documents/"
    )
    return [obj["Key"] for obj in response.get("Contents", [])]