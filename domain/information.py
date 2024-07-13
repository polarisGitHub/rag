class EmbeddingModelInfo(object):

    def __init__(self, model_name: str, model_path: str) -> None:
        self.model_name: str = model_name
        self.model_path: str = model_path

    def __str__(self) -> str:
        return f"EmbeddingModel: model_name={self.model_name}, model_path={self.model_path}"


class RerankerModelInfo(object):
    def __init__(self, model_name: str, model_path: str) -> None:
        self.model_name: str = model_name
        self.model_path: str = model_path

    def __str__(self) -> str:
        return f"RerankerModel: model_name={self.model_name}, model_path={self.model_path}"


class OllamaModelInfo(object):
    def __init__(self, model_name: str) -> None:
        self.model_name: str = model_name

    def __str__(self) -> str:
        return f"OllamaModel: model_name={self.model_name}"


class MilvusInfo(object):
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port

    def __str__(self) -> str:
        return f"Milvus: host={self.host}, port={self.port}"


class ElasticSearchInfo(object):
    def __init__(self, uri) -> None:
        self.uri = uri

    def __str__(self) -> str:
        return f"ElasticSearch: uri={self.uri}"
