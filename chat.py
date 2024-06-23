import config
import ollama
from result import Result, ResultIndex, ResultIndexType
from pymilvus import connections, Collection


def init_milvus():
    milvus = config.milvus[config.select_embedding_model_info.model_name]
    connections.connect(
        host=config.milvus["host"],
        port=config.milvus["port"],
        db_name=milvus["database_name"],
    )
    collection = Collection(milvus["collection_name"])
    collection.load()
    return collection


def recall(question: str, limit: int, collection: Collection, expr: str = None):
    embeding_result = config.select_embedding_model.encode(question)

    def drill_search_result(results):
        result_list = []
        for result in results:
            for r in result:
                fields = r.entity.fields
                result_list.append(
                    {
                        "pk": fields["pk"],
                        "previous_pk": fields["previous_pk"],
                        "source": fields["source"],
                        "meta": fields["meta"],
                        "content": fields["content"],
                        "above": [],
                        "below": [],
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


# 扩充上下文
def context_augmentation(query_results: list, expansion: int, collection: Collection):
    def fetch_context_pk(results: list):
        pk_list, previous_pk_list = [], []
        for r in results:
            if len(r["above"]) == 0:
                pk_list.append(r["pk"])
            else:
                previous_pk_list.append(r["above"][0]["previous_pk"])
            if len(r["below"]) == 0:
                previous_pk_list.append(r["previous_pk"])
            else:
                pk_list.append(r["below"][-1]["pk"])
        return pk_list, previous_pk_list

    def drill_query_result(results):
        result_list = []
        for result in results:
            result_list.append(
                {
                    "pk": result["pk"],
                    "previous_pk": result["previous_pk"],
                    "source": result["source"],
                    "meta": result["meta"],
                    "content": result["content"],
                }
            )
        return result_list

    context_result = []
    pk_list, previous_pk_list = fetch_context_pk(query_results)

    # 上文
    context_result += drill_query_result(
        collection.query(
            expr="pk in {}".format(previous_pk_list),
            output_fields=config.milvus["output_fields"],
        )
    )

    # 下文
    context_result += drill_query_result(
        collection.query(
            expr="previous_pk in {}".format(pk_list),
            output_fields=config.milvus["output_fields"],
        )
    )

    pk_map = {}
    for item in context_result:
        pk_map[item["pk"]] = item
        pk_map[item["previous_pk"]] = item

    for item in query_results:
        # 处理上文
        above = {}
        if len(item["above"]) == 0 and item["previous_pk"] in pk_map:
            above = pk_map[item["previous_pk"]]
        elif len(item["above"]) > 0 and item["above"][0]["previous_pk"] in pk_map:
            above = pk_map[item["above"][0]["previous_pk"]]
        if len(above) > 0:
            item["above"].insert(0, above)

        # 处理下文
        below = {}
        if (len(item["below"])) == 0 and item["pk"] in pk_map:
            below = pk_map[item["pk"]]
        elif len(item["below"]) > 0 and item["below"][-1]["pk"] in pk_map:
            below = pk_map[item["below"][-1]["pk"]]
        if len(below) > 0:
            item["below"].append(below)

    for i in range(expansion):
        context_augmentation(query_results, 0, collection)

    # 需要定义下策略
    # def build_title(title: list, special_title: str) -> str:
    #     title_str = "->".join(title)
    #     if len(special_title) != 0:
    #         title_str += "({})".format(special_title)
    #     return title_str

    # # 删除不在同一个章节的
    # for result in query_results:
    #     title = build_title(result["title"], result["special_title"])
    #     result["above"] = list(filter(lambda x: title == build_title(x["title"], x["special_title"]), result["above"]))
    #     result["below"] = list(filter(lambda x: title == build_title(x["title"], x["special_title"]), result["below"]))
    return query_results


def chat(query_results: list, question: str, context: list = None):
    ans_list = []
    for result in query_results:
        content = ""
        for r in result["above"] + [result] + result["below"]:
            content += r["content"] + "。"
        ans_list.append(
            {
                "content": content,
            }
        )

    data = f"\n\n".join([f"[{i}] {ans['content']}" for i, ans in enumerate(ans_list)])
    prompt = f"""
阅读以下知识点，每个知识点以[x]开头，其中x是数字

{data}

使用上述知识点，并用中文回答这个问题：{question}
最后在总结问题回答的基础上，给出解决方案
        """

    output = ollama.generate(
        model=config.select_ollama_model_info.model_name,
        prompt=prompt,
        context=context,
    )

    return output["response"], output["context"]


if __name__ == "__main__":
    collection = init_milvus()

    questions, context = [
        "小孩比较固执，不顺着他的意思就会一直哭闹，说什么都不听",
        "你和他沟通，他经常就闷着不说话，怎么说都不动",
    ], None
    for q in questions:
        recall_result = recall(question=q, limit=5, collection=collection)
        recall_result = context_augmentation(query_results=recall_result, expansion=1, collection=collection)
        ans, context = chat(query_results=recall_result, question=q, context=context)
        print(f"\n============================================================\n问题：{q}\n回答：{ans}\n============================================================")
