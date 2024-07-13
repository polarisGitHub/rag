from domain.information import EmbeddingModelInfo
from sentence_transformers import SentenceTransformer


class Embeddings(object):

    def __init__(self, model_info: EmbeddingModelInfo) -> None:
        self.__model_info = model_info
        self.embedding_model = SentenceTransformer(model_info.model_path)

    def encode(self, content: str) -> list[float]:
        return self.embedding_model.encode(content, normalize_embeddings=True)

    def batch_encode(self, contents: list[str]) -> list[list[float]]:
        return self.embedding_model.encode(contents, normalize_embeddings=True)
