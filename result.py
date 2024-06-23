from enum import Enum


class ResultIndexType(Enum):
    BOOK = 1
    WEB_SITE = 2


class Result(object):

    def __init__(self) -> None:
        self.pk: str = ""
        self.previous_pk: str = ""
        self.index: ResultIndex = ResultIndex()
        self.content: str = ""
        self.above: list[Result] = []
        self.below: list[Result] = []


class ResultIndex(object):

    def __init__(self, type: ResultIndexType) -> None:
        # book,websit
        self.type: ResultIndexType = type

        # book
        self.book: str = ""
        self.catalog: list[str] = ""

        # website
        self.url: str = ""

    def is_book(self) -> bool:
        return self.type == ResultIndexType.BOOK
    
    def is_website(self) -> bool:
        return self.type == ResultIndexType.WEB_SITE
