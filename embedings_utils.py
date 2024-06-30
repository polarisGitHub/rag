from config import select_embedding_model
from domain.information import EmbeddingModelInfo
from sentence_transformers import SentenceTransformer

__embedding_model_repo = {
    "m3e_large": EmbeddingModelInfo("m3e_large", "/home/polaris_he/cached_model/m3e-large"),
    "zpoint_large_embedding_zh": EmbeddingModelInfo("zpoint_large_embedding_zh", "/home/polaris_he/cached_model/zpoint_large_embedding_zh"),
}
select_model = __embedding_model_repo[select_embedding_model]
embedding_model = SentenceTransformer(select_model.model_path)


def encode(contents: list[str]) -> list[list[float]]:
    return embedding_model.encode(contents)
