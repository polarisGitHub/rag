import config
from pymilvus import connections, db, CollectionSchema, Collection


milvus_model_conf = config.milvus[config.select_embedding_model_info.model_name]

database_name = milvus_model_conf["database_name"]
conn = connections.connect(
    host=config.milvus["host"],
    port=config.milvus["port"],
)

if database_name not in db.list_database():
    database = db.create_database(database_name)
db.using_database(database_name)

collection = Collection(name=milvus_model_conf["collection_name"], schema=CollectionSchema(fields=milvus_model_conf["fields"]))

for index in config.milvus["vectors_indexes"]:
    collection.create_index(field_name=index['field_name'],index_params=index['index_params'])
    
for index in config.milvus["indexes"]:
    collection.create_index(field_name=index['field_name'],name=index['name'])