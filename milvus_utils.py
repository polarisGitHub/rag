import config
from domain.information import MilvuslInfo
from pymilvus import connections, db, CollectionSchema, Collection


def init_milvus(embeding_model_name, milvus_info: MilvuslInfo) -> None:
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


def create_connection(milvus_info: MilvuslInfo) -> None:
    connections.connect(
        host=milvus_info.host,
        port=milvus_info.port,
    )


def upsert(collection_name: str, data: list[str]) -> None:
    collection = Collection(collection_name)
    collection.upsert(data)
