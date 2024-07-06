from elasticsearch import Elasticsearch, helpers


def init_elasticsearch(uri: str, index: str = None, body: dict = None) -> Elasticsearch:
    es = Elasticsearch(hosts=[uri])
    if index and body and not es.indices.exists(index=index):
        es.indices.create(index=index, body=body)
    return es


def bulk_insert(elasticsearch: Elasticsearch, index: str, data: list[dict]) -> None:
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
    helpers.bulk(elasticsearch, actions=actions)


def search(elasticsearch: Elasticsearch, index: str, question: str, content_type: str, limit: int, analyzer: str = "hanlp_standard"):
    body = {
        "_source": ["text", "content_type"],
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
    resp = elasticsearch.search(index=index, body=body)
    if resp.body:
        return resp.body["hits"]["hits"]
    return []


if __name__ == "__main__":
    import config

    es = init_elasticsearch(config.elasticsearch["uri"], config.elasticsearch["index"])
    resp = search(es, config.elasticsearch["index"], "小孩在家很活泼，在外面却很内向", "fragment", 5)
    print(resp)
