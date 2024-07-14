import os
import json
import codecs
import config
import milvus_utils
from tqdm import tqdm
from embedings import Embeddings
from es_search import EsSearch
from domain.information import MilvusInfo, EmbeddingModelInfo, ElasticSearchInfo


class DataConstructor(object):

    def __init__(self, embedings_data: list = []) -> None:
        self.json_array = []
        self.embedings_data = embedings_data

    def load_json(self, prefix: str) -> None:
        temp_data_list = []
        for file in os.listdir(prefix):
            if file.endswith(".json"):
                with codecs.open(prefix + file, mode="r", encoding="utf-8") as f:
                    temp_data_list += json.load(f)

        for item in temp_data_list:
            self.json_array.append(
                {
                    "id": item["id"],
                    "content_type": item["content_type"],
                    "previous_id": item["previous_id"],
                    "next_id": item["next_id"],
                    "parent_id": item["parent_id"],
                    "text": item["text"],
                    "paragraph": "->".join(item["paragraph"]),
                    "meta": {
                        "isbn": item["isbn"],
                        "name": item["name"],
                        "category": item["category"],
                    },
                }
            )

    def embeddings_init(self, model: EmbeddingModelInfo):
        self.embeddings_model = Embeddings(model)

    def embeddings(self, batch_size: int) -> None:
        batch_json = []
        for i in range(0, len(self.json_array), batch_size):
            batch_json.append(self.json_array[i : i + batch_size])
        # embedings
        for i in tqdm(range(0, len(batch_json))):
            batch_item = batch_json[i]
            batch_contents = [item["text"] for item in batch_item]
            results = self.embeddings_model.encode(batch_contents)
            for j in range(len(batch_item)):
                item = batch_item[j]
                self.embedings_data.append(
                    {
                        "id": item["id"],
                        "content_type": item["content_type"],
                        "previous_id": item["previous_id"],
                        "next_id": item["next_id"],
                        "parent_id": item["parent_id"],
                        "text": item["text"],
                        "embeddings": results[j],
                        "paragraph": item["paragraph"],
                        "meta": item["meta"],
                    }
                )

    def compute_distance_nearby(self):
        length = len(self.embedings_data)
        self.embedings_data[0]["previous_distance"], self.embedings_data[length - 1]["previous_distance"] = -1, -1
        for i in tqdm(range(length - 1)):
            data = self.embedings_data[i]
            next_data = self.embedings_data[i + 1]
            distance = data["embeddings"] @ next_data["embeddings"].T
            data["next_distance"], next_data["previous_distance"] = distance, distance

    def write_to_json(self, json_file: str) -> None:
        json_list = []
        for data in self.embedings_data:
            json_list.append(
                {
                    "id": data["id"],
                    "content_type": data["content_type"],
                    "previous_id": data["previous_id"],
                    "next_id": data["next_id"],
                    "parent_id": data["parent_id"],
                    "text": data["text"],
                    "embeddings": data["embeddings"].tolist(),
                    "paragraph": data["paragraph"],
                    "meta": data["meta"],
                }
            )
        with codecs.open(json_file, mode="w", encoding="utf-8") as f:
            json.dump(json_list, f, ensure_ascii=False)
        json_list.clear()

    def milvus_init(self, milvus_info: MilvusInfo) -> None:
        self.milvus_info = milvus_info
        milvus_utils.init_milvus(
            embeding_model_name=config.select_embedding_model,
            milvus_info=self.milvus_info,
        )

    def milvus_upsert(self, collection_name, batch_size: int) -> None:
        for i in tqdm(range(0, len(self.embeding_data), batch_size)):
            milvus_utils.upsert(collection_name=collection_name, data=self.embeding_data[i : i + batch_size])

    def elasticsearch_init(self, esInfo: ElasticSearchInfo, index: str = None, body: dict = None):
        self.elasticsearch_index = index
        self.elasticsearch = EsSearch(esInfo)

    def elasticsearch_insert(self, batch_size: int) -> None:
        for i in tqdm(range(0, len(self.json_array), batch_size)):
            self.elasticsearch.bulk_insert(
                elasticsearch=self.elasticsearch,
                index=self.elasticsearch_index,
                data=self.json_array[i : i + batch_size],
            )


if __name__ == "__main__":
    data_constructor = DataConstructor()

    print("load")
    data_constructor.load_json(prefix="data/processed/")

    # print("insert elasticsearch")
    # data_constructor.elasticsearch_init(
    #     config.elasticsearch["uri"],
    #     index=config.elasticsearch["index"],
    #     body=config.elasticsearch["body"],
    # )
    # data_constructor.elasticsearch_insert(batch_size=500)

    print("embeding")
    data_constructor.embeddings_init(config.select_embeddings_model)
    data_constructor.embeddings(batch_size=8)
    data_constructor.compute_distance_nearby()
    print("write json")
    data_constructor.write_to_json("data/embeddings.json")

    # print("insert milvus")
    # data_constructor.milvus_init(
    #     milvus_info=MilvuslInfo(
    #         host=config.milvus["host"],
    #         port=config.milvus["port"],
    #     )
    # )
    # collection_name = config.milvus[config.select_embedding_model]["collection_name"]
    # data_constructor.milvus_upsert(collection_name=collection_name, batch_size=200)
