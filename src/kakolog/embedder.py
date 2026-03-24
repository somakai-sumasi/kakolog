from sentence_transformers import SentenceTransformer

MODEL_NAME = "cl-nagoya/ruri-v3-30m"
_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME, device="cpu")
    return _model


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
    prefixed = [f"検索文書: {t}" for t in texts]
    vecs = model.encode(prefixed, normalize_embeddings=True, batch_size=8)
    return vecs.tolist()
