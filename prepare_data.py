import os
import json
import codecs
import config
from tqdm import tqdm
from embedings_utils import encode
from milvus_utils import init_milvus, upsert
from domain.information import MilvuslInfo


class DataConstructor(object):

    def __init__(self, prefix: str, milvus_info: MilvuslInfo) -> None:
        self.prefix = prefix
        self.milvus_info = milvus_info
        self.json_array = []
        self.embeding_data = []

        init_milvus(embeding_model_name=config.select_embedding_model, milvus_info=milvus_info)
        for file in os.listdir(prefix):
            if file.endswith(".json"):
                with codecs.open(prefix + file, mode="r", encoding="utf-8") as f:
                    self.json_array += json.load(f)

    def embeding(self, batch_size: int) -> None:
        batch_json = []
        for i in range(0, len(self.json_array), batch_size):
            batch_json.append(self.json_array[i : i + batch_size])
        # embedings
        for i in tqdm(range(0, len(batch_json))):
            batch_item = batch_json[i]
            batch_contents = [item["text"] for item in batch_item]
            results = encode(batch_contents)
            for j in range(len(batch_item)):
                item = batch_item[j]
                self.embeding_data.append(
                    {
                        "id": item["id"],
                        "content_type": item["content_type"],
                        "previous_id": item["previous_id"],
                        "next_id": item["next_id"],
                        "parent_id": item["parent_id"],
                        "text": item["text"],
                        "embedding": results[j],
                        "paragraph": "->".join(item["paragraph"]),
                        "meta": {
                            "isbn": item["isbn"],
                            "name": item["name"],
                            "category": item["category"],
                        },
                    }
                )

    def similarity(self):
        pass

    def milvus_upsert(self, collection_name, batch_size: int) -> None:
        for i in tqdm(range(0, len(self.embeding_data), batch_size)):
            upsert(collection_name=collection_name, data=self.embeding_data[i : i + batch_size])


if __name__ == "__main__":
    data_construct = DataConstructor(
        prefix="data/processed/",
        milvus_info=MilvuslInfo(
            host=config.milvus["host"],
            port=config.milvus["port"],
        ),
    )
    print("embeding")
    data_construct.embeding(batch_size=8)
    print("insert milvus")
    collection_name = config.milvus[config.select_embedding_model]["collection_name"]
    data_construct.milvus_upsert(collection_name=collection_name, batch_size=200)
