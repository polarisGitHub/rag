import config
import ollama
from pymilvus import Collection
from milvus_utils import create_connection
from domain.information import MilvuslInfo
from embedings_utils import encode





def chat(query_results: list, question: str, context: list = None):
    ans_list = []
    for result in query_results:
        ans_list.append(
            {
                "content": result["text"],
            }
        )

    data = f"\n".join([f"[{i}] {ans['content']}" for i, ans in enumerate(ans_list)])
    prompt = f"""
请阅读以下知识点，每个知识点以[x]开头，其中x是数字

------------------------------------
{data}
------------------------------------

使用上述知识点，而不是已有知识回答问题。

作为一个智能助手，你的回答要尽可能严谨。如果不能回答，请回答不知道，不要编造数据。

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
    collection.load()

    question = "小孩在家里很活泼，到了外面就变得很内向"
    recall_result = recall(
        question=question,
        expr="content_type=='fragment'",
        limit=5,
        collection=collection,
    )
    
    recall_result += recall(
        question=question,
        expr="content_type=='sentence'",
        limit=5,
        collection=collection,
    )
    ans, context = chat(query_results=recall_result, question=question)
    print(ans)
