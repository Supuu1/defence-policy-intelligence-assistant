"""Regression checks for safe restart persistence and cache recovery."""

from hashlib import sha256
from pathlib import Path
import json
import sys
import tempfile

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import src.cache_service as cache  # noqa: E402


def record_for(document_hash):
    return {
        "filename": "policy.pdf",
        "document_id": document_hash,
        "content_hash": document_hash,
        "file_size": 1234,
        "status": "Processed",
        "error": "",
        "GEMINI_API_KEY": "must-never-be-persisted",
        "pages": [
            {
                "source_filename": "policy.pdf",
                "document_id": document_hash,
                "page_number": 1,
                "text": "The policy becomes effective in March 2026.",
                "extraction_method": "Native text",
                "ocr_error": "",
            }
        ],
        "chunks": [
            {
                "source_filename": "policy.pdf",
                "document_id": document_hash,
                "page_number": 1,
                "chunk_id": "page-1-chunk-1",
                "text": "The policy becomes effective in March 2026.",
                "extraction_method": "Native text",
            }
        ],
    }


def main():
    original_root = cache.CACHE_ROOT
    try:
        with tempfile.TemporaryDirectory(prefix="defence-cache-test-") as directory:
            cache.CACHE_ROOT = Path(directory) / "document-cache"
            document_hash = sha256(b"unchanged PDF bytes").hexdigest()
            record = record_for(document_hash)
            cache.save_document(document_hash, record)

            document_file = cache.CACHE_ROOT / document_hash / "document.json"
            raw_document = document_file.read_text(encoding="utf-8")
            assert "must-never-be-persisted" not in raw_document
            loaded = cache.load_document(document_hash, "renamed-policy.pdf")
            assert loaded is not None and loaded["filename"] == "renamed-policy.pdf"
            assert loaded["pages"][0]["extraction_method"] == "Native text"

            matrix = np.asarray([[0.1, 0.2, 0.3]], dtype="float32")
            cache.save_embeddings(document_hash, record["chunks"], matrix)
            loaded_matrix = cache.load_embeddings(document_hash, record["chunks"])
            assert loaded_matrix is not None and np.allclose(matrix, loaded_matrix)
            assert cache.load_embeddings(
                document_hash, record["chunks"], embedding_model="different-model"
            ) is None

            metadata_file = cache.CACHE_ROOT / document_hash / "embeddings.json"
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            assert metadata["vector_index_type"] == "FAISS IndexFlatIP"
            assert metadata["matrix_sha256"]

            # Corruption is isolated and treated as a cache miss, never a crash.
            document_file.write_text("{broken json", encoding="utf-8")
            assert cache.load_document(document_hash, "policy.pdf") is None
            assert cache.load_embeddings(document_hash, record["chunks"]) is not None
            matrix_file = cache.CACHE_ROOT / document_hash / "embeddings.npy"
            matrix_file.write_bytes(b"corrupt matrix")
            assert cache.load_embeddings(document_hash, record["chunks"]) is None

            stats = cache.cache_statistics()
            assert stats["documents"] == 1 and stats["bytes"] > 0
            assert cache.clear_local_cache() == 1
            assert not cache.CACHE_ROOT.exists()
            assert cache.clear_local_cache() == 0

            temporary_files = list(Path(directory).rglob("*.tmp"))
            assert temporary_files == []
    finally:
        cache.CACHE_ROOT = original_root

    print("Persistence, integrity, model metadata, secret exclusion, corruption, and clearing checks passed.")


if __name__ == "__main__":
    main()
