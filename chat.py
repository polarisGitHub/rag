import config
import ollama
from pymilvus import Collection
import milvus_utils

from rerank import Reranker
from embedings import Embeddings
from es_search import EsSearch
from domain.information import MilvusInfo


def chat(query_results: list, question: str, context: list = None):
    ans_list = []
    for result in query_results:
        ans_list.append(
            {
                "content": result["text"],
                "score": result["score"],
            }
        )

    data = f"\n".join([f"[{i}][{ans['score']:.3f}] {ans['content']}" for i, ans in enumerate(ans_list)])
    prompt = f"""
请阅读以下知识点，每个知识点以[x][y]开头，其中x和y都是数字。x是序号，y是知识点的评分

------------------------------------
{data}
------------------------------------

使用上述知识点，而不是已有知识回答问题。需要给出两点以上建议并总结

作为一个智能助手，你的回答要尽可能严谨。回答可以忽略评分较低的知识点，如果信息不足，可以询问更多信息。



请回答问题：{question}
    """

    print(prompt)
    output = ollama.generate(
        model=config.select_ollama_model_info.model_name,
        prompt=prompt,
        system="你是一个人工智能助手",
        context=context,
    )

    return output["response"], output["context"]


def vector_recall(question: str, model: Embeddings):
    vector = model.encode(question)
    result = milvus_utils.search(
        vector=vector,
        expr="content_type=='fragment'",
        limit=8,
        collection=collection,
    )

    result += milvus_utils.search(
        vector=vector,
        expr="content_type=='sentence'",
        limit=8,
        collection=collection,
    )
    return result


def elasticsearch_recall(question: str, es_search: EsSearch):
    result = es_search.search(
        config.elasticsearch["index"],
        question=question,
        content_type="fragment",
        limit=5,
    )
    result += es_search.search(
        config.elasticsearch["index"],
        question=question,
        content_type="sentence",
        limit=5,
    )
    return result


def merge_and_expand(vector_result, elasticsearch_result):
    return vector_result + elasticsearch_result


if __name__ == "__main__":
    # 多路召回，重排序，llm
    question = "小孩在家里很活泼，到了外面就变得很内向"

    # init es
    elastic_search = EsSearch(config.elasticsearch["info"])
    elasticsearch_result = elasticsearch_recall(question=question, es_search=elastic_search)

    # init milvus
    info = config.milvus[config.select_embeddings_model.model_name]
    milvus_utils.create_connection(
        milvus_info=MilvusInfo(
            host=config.milvus["host"],
            port=config.milvus["port"],
        ),
        database=info["database_name"],
    )
    collection = Collection(info["collection_name"])
    collection.load()
    # init embeddings
    embeddings = Embeddings(config.select_embeddings_model)
    vector_result = vector_recall(question=question, model=embeddings)

    # 文本合并和扩充
    result = merge_and_expand(vector_result, elasticsearch_result)

    # 重排序
    # init reranker
    reranker = Reranker(config.select_reranker_model)
    rerank_text = reranker.rerank(text=question, data=result, text_mapper=lambda x: x["text"])
    
    # 提取重排序的作为llm输入
    high_score_rerank_text = list(filter(lambda x: x["score"] > 0, rerank_text))
    if len(high_score_rerank_text) == 0 or len(high_score_rerank_text) < 5:
        search_text = rerank_text[0:5]
    else:
        search_text = high_score_rerank_text[0:10]

    # chat
    ans, context = chat(
        query_results=[
            {
                "text": item["data"]["text"],
                "score": item["score"],
            }
            for item in search_text
        ],
        question=question,
    )
    print(ans)
