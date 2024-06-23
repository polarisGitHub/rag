import json
import codecs
import config
import hashlib
from tqdm import tqdm
from pymilvus import connections, Collection


extract_file_info = config.select_extract_file_info
read_content_path = extract_file_info.split_file_path

batch_size, contents, batch_json = 8, [], []
with codecs.open(read_content_path, mode="r", encoding="utf-8") as f:
    contents = f.read()
json_array = json.loads(contents)

for i in range(0, len(json_array), batch_size):
    batch_json.append(json_array[i : i + batch_size])

milvus = config.milvus[config.select_embedding_model_info.model_name]
conn = connections.connect(
    host=config.milvus["host"],
    port=config.milvus["port"],
    db_name=milvus["database_name"],
)

previous_pk = ""
collection = Collection(milvus["collection_name"])
for i in tqdm(range(0, len(batch_json))):
    batch_item, data, batch_contents = batch_json[i], [], []
    for item in batch_item:
        batch_contents.append(item["content"])
    results = config.select_embedding_model.encode(batch_contents)
    for j in range(0, len(batch_item)):
        content = batch_contents[j]
        pk = hashlib.sha256(content.encode("utf-8")).hexdigest()
        data.append(
            {
                "pk": pk,
                "previous_pk": previous_pk,
                "source": extract_file_info.get_source(),
                "content": content,
                "embedding": results[j],
                "meta": {}, # todo
            }
        )
        previous_pk = pk
    collection.upsert(data)
