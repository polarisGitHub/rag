class EmbeddingModelInfo(object):

    def __init__(self, model_name: str, model_path: str) -> None:
        self.model_name: str = model_name
        self.model_path: str = model_path


class OllamaModelInfo(object):
    def __init__(self, model_name: str) -> None:
        self.model_name: str = model_name


class MilvuslInfo(object):
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
