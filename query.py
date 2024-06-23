import config
import ollama
from pymilvus import connections, Collection


def init_milvus():
    milvus = config.milvus[config.select_embedding_model_info["name"]]
    connections.connect(
        host=config.milvus["host"],
        port=config.milvus["port"],
        db_name=milvus["database_name"],
    )
    collection = Collection(milvus["collection_name"])
    collection.load()
    return collection


def recall(question: str, limit: int, collection: Collection):
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
                        "content": fields["content"],
                        "title": fields["title"],
                        "special_title": fields["special_title"],
                        "above": [],
                        "below": [],
                    }
                )
        return result_list

    return drill_search_result(
        collection.search(
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
                    "content": result["content"],
                    "title": result["title"],
                    "special_title": result["special_title"],
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

    def build_title(title: list, special_title: str) -> str:
        title_str = "->".join(title)
        if len(special_title) != 0:
            title_str += "({})".format(special_title)
        return title_str

    # 删除不在同一个章节的
    for result in query_results:
        title = build_title(result["title"], result["special_title"])
        result["above"] = list(filter(lambda x: title == build_title(x["title"], x["special_title"]), result["above"]))
        result["below"] = list(filter(lambda x: title == build_title(x["title"], x["special_title"]), result["below"]))
    return query_results


def chat(query_results: list, question: str):
    ans_list = []
    for result in query_results:
        content = ""
        for r in result["above"] + [result] + result["below"]:
            content += r["content"] + "。"
        ans_list.append(
            {
                "book": "育儿百科",
                "title": "->".join(result["title"]),
                "content": content,
            }
        )

    context = "\n\n".join([f"[citation:{i}] {ans['content']} <{ans['book']}:{ans['title']}>" for i, ans in enumerate(ans_list)])

    user_prompt = f"用户的问题是：{question}"
    system_prompt = f"""
        当你收到用户的问题时，请编写清晰、简洁、准确的回答。
        你会收到一组与问题相关的上下文，每个上下文都以参考编号开始，如[citation:x]，其中x是一个数字；每个上下文都以参考文献结束，如<book:directory>，其中book是书名，directory目录。
        请使用这些上下文，并在适当的情况下在每个句子的末尾引用上下文。

        你的答案必须是正确的，并且使用公正和专业的语气写作。请限制在8192个tokens之内。
        不要提供与问题无关的信息，也不要重复。
        不允许在答案中添加编造成分，如果给定的上下文没有提供足够的信息，就说“缺乏关于xx的信息”。

        请用参考编号和参考文献引用上下文，参考编号格式为[citation:x]，参考文献格式为<book:directory>。
        如果一个句子来自多个上下文，请列出所有适用的引用，如[citation:3][citation:5]。
        除了代码和特定的名字和引用，你的答案必须用与问题相同的语言编写，如果问题是中文，则回答也是中文。

        这是一组上下文：

        {context}
        """
    response = ollama.chat(
        model=config.select_ollama_model_info["model_name"],
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response["message"]["content"]


if __name__ == "__main__":
    collection = init_milvus()

    questions = [
        "小孩比较固执，不顺着他的意思就会一直哭闹，说什么都不听",
        "你和他沟通，他经常就闷着不说话，怎么说都不动",
        "小孩说话慢，现在两岁半了说话还是不太清楚",
    ]
    for q in questions:
        recall_result = recall(question=q, limit=15, collection=collection)
        recall_result = context_augmentation(query_results=recall_result, expansion=4, collection=collection)
        ans = chat(query_results=recall_result, question=q)
        print("问题：{}\n回答：{}\n============================================================".format(q, ans))
