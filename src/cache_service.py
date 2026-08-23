"""Safe local JSON/NumPy caches keyed by PDF content hash."""

from hashlib import sha256
import json
import logging
import os
from pathlib import Path
import shutil
import tempfile
import uuid

import numpy as np


CACHE_VERSION = 1
CACHE_ROOT = Path(__file__).resolve().parents[1] / ".cache" / "defence_policy_rag"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
logger = logging.getLogger(__name__)


def _document_directory(document_hash: str) -> Path:
    """Resolve a hash to a contained cache directory."""
    if len(document_hash) != 64 or any(char not in "0123456789abcdef" for char in document_hash):
        raise ValueError("Invalid document hash.")
    return CACHE_ROOT / document_hash


def _prepare_directory(document_hash: str) -> Path:
    """Create a private local cache directory where the platform permits it."""
    directory = _document_directory(document_hash)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        directory.chmod(0o700)
    except OSError:
        pass
    return directory


def chunk_fingerprint(chunks: list[dict]) -> str:
    """Identify ordered chunk text used to produce an embedding matrix."""
    digest = sha256()
    for chunk in chunks:
        digest.update(chunk["text"].encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _json_bytes(payload: dict) -> bytes:
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _atomic_write(destination: Path, data: bytes) -> None:
    """Write and fsync a unique same-directory temporary file before replacement."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_save_numpy(destination: Path, matrix: np.ndarray) -> str:
    """Stream a NumPy matrix to a unique temporary file and replace atomically."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp.npy", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as handle:
            np.save(handle, matrix, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        digest = _file_digest(temporary)
        temporary.replace(destination)
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def _valid_document_record(record: dict, document_hash: str) -> bool:
    """Reject malformed cache structures before they reach the Streamlit session."""
    if not isinstance(record, dict):
        return False
    if record.get("content_hash", document_hash) != document_hash:
        return False
    pages = record.get("pages")
    chunks = record.get("chunks")
    if not isinstance(pages, list) or not isinstance(chunks, list):
        return False
    for page in pages:
        if (
            not isinstance(page, dict)
            or not isinstance(page.get("page_number"), int)
            or not isinstance(page.get("text"), str)
            or not isinstance(page.get("extraction_method"), str)
        ):
            return False
    for chunk in chunks:
        if (
            not isinstance(chunk, dict)
            or not isinstance(chunk.get("text"), str)
            or not isinstance(chunk.get("page_number"), int)
            or not isinstance(chunk.get("chunk_id"), str)
            or not isinstance(chunk.get("extraction_method"), str)
        ):
            return False
    return True


def _cacheable_record(record: dict) -> dict:
    """Whitelist processing fields so credentials or unrelated state cannot persist."""
    page_fields = (
        "source_filename", "document_id", "page_number", "text",
        "extraction_method", "ocr_error",
    )
    chunk_fields = (
        "text", "source_filename", "document_id", "page_number", "chunk_id",
        "extraction_method",
    )
    safe_record = {
        key: record.get(key)
        for key in (
            "filename", "document_id", "content_hash", "file_size",
            "safeguard_warning", "status", "error", "failure_stage",
        )
        if key in record
    }
    safe_record["pages"] = [
        {key: page.get(key) for key in page_fields if key in page}
        for page in record.get("pages", [])
    ]
    safe_record["chunks"] = [
        {key: chunk.get(key) for key in chunk_fields if key in chunk}
        for chunk in record.get("chunks", [])
    ]
    return safe_record


def relabel_document(record: dict, filename: str, document_id: str) -> dict:
    """Copy cached content while applying current upload metadata."""
    pages = [
        {**page, "source_filename": filename, "document_id": document_id}
        for page in record["pages"]
    ]
    chunks = [
        {**chunk, "source_filename": filename, "document_id": document_id}
        for chunk in record["chunks"]
    ]
    return {
        **record,
        "filename": filename,
        "document_id": document_id,
        "content_hash": document_id,
        "pages": pages,
        "chunks": chunks,
    }


def load_document(document_hash: str, filename: str) -> dict | None:
    """Load cached page/chunk data, returning None for missing or corrupt entries."""
    cache_file = _document_directory(document_hash) / "document.json"
    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
        if payload.get("cache_version") != CACHE_VERSION:
            return None
        record = payload["record"]
        if payload.get("document_hash", document_hash) != document_hash:
            return None
        expected_digest = payload.get("record_sha256")
        if expected_digest and expected_digest != sha256(_json_bytes(record)).hexdigest():
            logger.warning("Ignoring document cache with failed integrity check: %s", document_hash)
            return None
        if not _valid_document_record(record, document_hash):
            return None
        return relabel_document(record, filename, document_hash)
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        logger.warning("Ignoring unreadable document cache entry: %s", document_hash)
        return None


def save_document(document_hash: str, record: dict) -> None:
    """Persist extracted pages and chunks atomically as JSON."""
    directory = _prepare_directory(document_hash)
    record = _cacheable_record(record)
    if not _valid_document_record(record, document_hash):
        raise ValueError("Document record is not safe to cache.")
    record_digest = sha256(_json_bytes(record)).hexdigest()
    payload = {
        "cache_version": CACHE_VERSION,
        "document_hash": document_hash,
        "record_sha256": record_digest,
        "record": record,
    }
    _atomic_write(directory / "document.json", _json_bytes(payload))


def load_embeddings(
    document_hash: str,
    chunks: list[dict],
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> np.ndarray | None:
    """Load an embedding matrix only when its chunk fingerprint still matches."""
    directory = _document_directory(document_hash)
    metadata_file = directory / "embeddings.json"
    matrix_file = directory / "embeddings.npy"
    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        if (
            metadata.get("cache_version") != CACHE_VERSION
            or metadata.get("chunk_fingerprint") != chunk_fingerprint(chunks)
            or metadata.get("chunk_count") != len(chunks)
            or metadata.get("embedding_model", embedding_model) != embedding_model
        ):
            return None
        expected_matrix_digest = metadata.get("matrix_sha256")
        if expected_matrix_digest and expected_matrix_digest != _file_digest(matrix_file):
            logger.warning("Ignoring embedding cache with failed integrity check: %s", document_hash)
            return None
        matrix = np.load(matrix_file, allow_pickle=False)
        if (
            matrix.ndim != 2
            or len(matrix) != len(chunks)
            or metadata.get("vector_dimension", matrix.shape[1]) != matrix.shape[1]
        ):
            return None
        return np.asarray(matrix, dtype="float32")
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        logger.warning("Ignoring unreadable embedding cache entry: %s", document_hash)
        return None


def save_embeddings(
    document_hash: str,
    chunks: list[dict],
    embeddings: np.ndarray,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> None:
    """Persist a validated per-document embedding matrix and metadata."""
    directory = _prepare_directory(document_hash)
    matrix = np.asarray(embeddings, dtype="float32")
    if matrix.ndim != 2 or len(matrix) != len(chunks):
        raise ValueError("Embedding matrix does not match cached chunks.")
    matrix_file = directory / "embeddings.npy"
    matrix_digest = _atomic_save_numpy(matrix_file, matrix)

    metadata = {
        "cache_version": CACHE_VERSION,
        "chunk_count": len(chunks),
        "chunk_fingerprint": chunk_fingerprint(chunks),
        "embedding_model": embedding_model,
        "vector_dimension": matrix.shape[1],
        "dtype": "float32",
        "matrix_sha256": matrix_digest,
        "vector_index_type": "FAISS IndexFlatIP",
        "similarity_metric": "cosine via normalized inner product",
    }
    _atomic_write(directory / "embeddings.json", _json_bytes(metadata))


def cache_statistics() -> dict:
    """Return non-sensitive local cache counts and total disk usage."""
    try:
        files = [path for path in CACHE_ROOT.rglob("*") if path.is_file()]
        document_count = sum(
            1
            for directory in CACHE_ROOT.iterdir()
            if directory.is_dir() and len(directory.name) == 64
        ) if CACHE_ROOT.exists() else 0
        return {
            "documents": document_count,
            "files": len(files),
            "bytes": sum(path.stat().st_size for path in files),
        }
    except OSError:
        return {"documents": 0, "files": 0, "bytes": 0}


def clear_local_cache() -> int:
    """Atomically detach and remove only the application-owned cache directory."""
    root = CACHE_ROOT.absolute()
    if not root.exists():
        return 0
    if root.is_symlink():
        raise ValueError("Refusing to clear a symbolic-link cache path.")
    if root == Path(root.anchor) or len(root.parts) < 3:
        raise ValueError("Refusing to clear an unsafe cache path.")
    statistics = cache_statistics()
    detached = root.with_name(f".{root.name}.clearing-{uuid.uuid4().hex}")
    root.replace(detached)
    shutil.rmtree(detached)
    return statistics["documents"]
