import config
from elasticsearch import Elasticsearch, helpers
from domain.information import ElasticSearchInfo


class EsSearch(object):

    def __init__(self, elasticSearchInfo: ElasticSearchInfo) -> None:
        self._es_info = elasticSearchInfo
        self.elasticsearch = Elasticsearch(hosts=[elasticSearchInfo.uri])

    def create_index(self, index: str = None, body: dict = None) -> Elasticsearch:
        if index and body and not self.elasticsearch.indices.exists(index=index):
            self.elasticsearch.indices.create(index=index, body=body)

    def bulk_insert(self, index: str, data: list[dict]) -> None:
        actions = []
        for item in data:
            actions.append(
                {
                    "_index": index,
                    "_id": item["id"],
                    "_source": {
                        "content_type": item["content_type"],
                        "previous_id": item["previous_id"],
                        "next_id": item["next_id"],
                        "parent_id": item["parent_id"],
                        "text": item["text"],
                        "paragraph": item["paragraph"],
                        "meta": item["meta"],
                    },
                }
            )
        helpers.bulk(self.elasticsearch, actions=actions)

    def search(self, index: str, question: str, content_type: str, limit: int, analyzer: str = "hanlp_standard"):
        body = {
            "_source": config.elasticsearch["query_fields"],
            "query": {
                "bool": {
                    "should": [
                        {
                            "match": {
                                "text": {
                                    "query": question,
                                    "analyzer": analyzer,
                                },
                            }
                        },
                        {
                            "match_phrase": {
                                "text": {
                                    "query": question,
                                    "analyzer": analyzer,
                                    "slop": 5,
                                }
                            }
                        },
                    ],
                    "filter": [{"term": {"content_type": content_type}}],
                }
            },
            "size": limit,
        }
        return self.__parse_search_results(self.elasticsearch.search(index=index, body=body))

    def query_by_parent_id(self, index: str, parent_ids: list[str]):
        body = {
            "_source": config.elasticsearch["query_fields"],
            "query": {
                "bool": {
                    "should": [
                        {
                            "terms": {"_id": parent_ids},
                        },
                    ],
                    "filter": [{"term": {"content_type": "fragment"}}],
                }
            },
        }
        return self.__parse_search_results(self.elasticsearch.search(index=index, body=body))

    def __parse_search_results(self, resp):
        results = []
        if resp.body:
            for item in resp.body["hits"]["hits"]:
                data = item["_source"]
                data["id"] = item["_id"]
                results.append(data)
        return results
