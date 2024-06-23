class TextExtractInfo(object):
    def __init__(
        self,
        name: str,
        prefix: str,
        file_path: str,
        extract_file_path: str,
        split_file_path: str,
        start_page: int = 0,
        end_page: int = -1,
    ) -> None:
        self.name: str = name
        self.prefix: str = prefix
        self.file_path: str = prefix + file_path
        self.extract_file_path: str = prefix + extract_file_path
        self.split_file_path: str = prefix + split_file_path
        self.start_page = start_page
        self.end_page = end_page

    def get_source(self):
        return "book"


class EmbeddingModelInfo(object):

    def __init__(self, model_name: str, model_path: str) -> None:
        self.model_name: str = model_name
        self.model_path: str = model_path


class OllamaModelInfo(object):
    def __init__(self, model_name: str) -> None:
        self.model_name: str = model_name
