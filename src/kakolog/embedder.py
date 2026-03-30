import functools

from sentence_transformers import SentenceTransformer

MODEL_NAME = "cl-nagoya/ruri-v3-30m"


@functools.lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME, device="cpu")


def embed_query(text: str) -> list[float]:
    model = get_model()
    vec = model.encode(f"検索クエリ: {text}", normalize_embeddings=True)
    return vec.tolist()


def embed_document(text: str) -> list[float]:
    model = get_model()
    vec = model.encode(f"検索文書: {text}", normalize_embeddings=True)
    return vec.tolist()


def embed_documents(texts: list[str]) -> list[list[float]]:
    model = get_model()
    vecs = model.encode([f"検索文書: {t}" for t in texts], normalize_embeddings=True, batch_size=8)
    return vecs.tolist()
