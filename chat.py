import config
import ollama
from pymilvus import Collection
from milvus_utils import create_connection
from domain.information import MilvuslInfo
from embedings_utils import encode


def recall(question: str, limit: int, collection: Collection, expr: str = None):
    embeding_result = encode(question)

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
                    }
                )
        return result_list

    return drill_search_result(
        collection.search(
            expr=expr,
            data=[embeding_result],
            anns_field="embedding",
            param=config.milvus["search_params"],
            limit=limit,
            output_fields=config.milvus["output_fields"],
        )
    )


def chat(query_results: list, question: str, context: list = None):
    ans_list = []
    for result in query_results:
        ans_list.append(
            {
                "content": result['text'],
            }
        )

    data = f"\n\n".join([f"[{i}] {ans['content']}" for i, ans in enumerate(ans_list)])
    prompt = f"""
阅读以下知识点，每个知识点以[x]开头，其中x是数字

{data}

使用上述知识点，并用中文回答这个问题：{question}
        """

    print(prompt)
    output = ollama.generate(
        model=config.select_ollama_model_info.model_name,
        prompt=prompt,
        context=context,
    )

    return output["response"], output["context"]


if __name__ == "__main__":
    info = config.milvus[config.select_embedding_model]
    create_connection(
        milvus_info=MilvuslInfo(
            host=config.milvus["host"],
            port=config.milvus["port"],
        ),
        database=info["database_name"],
    )
    collection = Collection(info["collection_name"])

    questions, context = [
        "小孩比较固执，不顺着他的意思就会一直哭闹，说什么都不听",
        "你和他沟通，他经常就闷着不说话，怎么说都不动",
    ], None
    for q in questions:
        recall_result = recall(question=q, limit=10, collection=collection)
        ans, context = chat(query_results=recall_result, question=q, context=context)
        print(
            f"\n============================================================\n问题：{q}\n回答：{ans}\n============================================================"
        )
