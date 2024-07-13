import config
from domain.information import MilvusInfo
from pymilvus import connections, db, CollectionSchema, Collection


def init_milvus(embeding_model_name, milvus_info: MilvusInfo) -> None:
    create_connection(milvus_info)

    milvus_model_conf = config.milvus[embeding_model_name]
    database_name = milvus_model_conf["database_name"]

    if database_name not in db.list_database():
        db.create_database(database_name)
    db.using_database(database_name)

    collection = Collection(name=milvus_model_conf["collection_name"], schema=CollectionSchema(fields=milvus_model_conf["fields"]))

    for index in config.milvus["vectors_indexes"]:
        collection.create_index(
            field_name=index["field_name"],
            name=index["name"],
            index_params=index["index_params"],
        )

    for index in config.milvus["indexes"]:
        collection.create_index(
            field_name=index["field_name"],
            name=index["name"],
        )


def create_connection(milvus_info: MilvusInfo, database: str = None) -> None:
    connections.connect(
        host=milvus_info.host,
        port=milvus_info.port,
    )
    if database:
        db.using_database(database)


def upsert(collection_name: str, data: list[str]) -> None:
    collection = Collection(collection_name)
    collection.upsert(data)


def search(vector, limit: int, collection: Collection, expr: str = None):
    return drill_search_result(
        collection.search(
            expr=expr,
            data=[vector],
            anns_field="embedding",
            param=config.milvus["search_params"],
            limit=limit,
            output_fields=config.milvus["output_fields"],
        )
    )


def drill_search_result(results):
    result_list = []
    for result in results:
        for r in result:
            fields = r.entity.fields
            result_list.append(
                {
                    "id": fields["id"],
                    "content_type": fields["content_type"],
                    "previous_id": fields["previous_id"],
                    "next_id": fields["next_id"],
                    "parent_id": fields["parent_id"],
                    "paragraph": fields["paragraph"],
                    "text": fields["text"],
                    "meta": fields["meta"],
                    "distance": r.distance,
                }
            )
    return result_list
