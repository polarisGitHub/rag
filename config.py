from entity.model_info import EmbeddingModelInfo, OllamaModelInfo, TextExtractInfo
from pymilvus import FieldSchema, DataType
from sentence_transformers import SentenceTransformer

# 信息提取
__extract_info = {
    "cybyc": TextExtractInfo(
        name="育儿百科",
        prefix="data/cybyc/",
        file_path="美国儿科学会育儿百科(第7版).pdf",
        extract_file_path="extract.json",
        split_file_path="split.json",
        start_page=42,
        end_page=1235,
    ),
}
select_extract_file_name = "cybyc"
select_extract_file_info = __extract_info[select_extract_file_name]

# embedding模型
__embedding_model_repo = {
    "m3e_large": EmbeddingModelInfo("m3e_large", "/home/polaris_he/cached_model/m3e-large"),
    "zpoint_large_embedding_zh": EmbeddingModelInfo("zpoint_large_embedding_zh", "/home/polaris_he/cached_model/zpoint_large_embedding_zh"),
}
__select_embedding_model_name = "zpoint_large_embedding_zh"
select_embedding_model_info = __embedding_model_repo[__select_embedding_model_name]
select_embedding_model = SentenceTransformer(select_embedding_model_info.model_path)

# llm模型
__ollama_model_repo = {
    "llama3": OllamaModelInfo("llama3"),
    "qwen2:7b": OllamaModelInfo("qwen2:7b"),
    "wangshenzhi/llama3-8b-chinese-chat-ollama-q8": OllamaModelInfo("wangshenzhi/llama3-8b-chinese-chat-ollama-q8"),
}
__select_ollama_model_name = "wangshenzhi/llama3-8b-chinese-chat-ollama-q8"
select_ollama_model_info = __ollama_model_repo[__select_ollama_model_name]

# 向量数据库
milvus_common_fields = [
    FieldSchema(name="pk", dtype=DataType.VARCHAR, max_length=64, is_primary=True, description="primary id use sha256"),
    FieldSchema(name="previous_pk", dtype=DataType.VARCHAR, max_length=64, description="previous primary id"),
    FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=16, description="source:book,website"),
    FieldSchema(name="meta", dtype=DataType.JSON),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=1024),
]
milvus = {
    "host": "127.0.0.1",
    "port": 19530,
    "search_params": {
        "metric_type": "COSINE",
        "params": {"nlist": 128},
    },
    "output_fields": [
        "pk",
        "previous_pk",
        "source",
        "meta",
        "content",
    ],
    "vectors_indexes": [
        {
            "field_name": "embedding",
            "index_params": {"metric_type": "COSINE", "index_type": "IVF_FLAT", "params": {"nlist": 4096}},
        }
    ],
    "indexes": [
        {
            "field_name": "previous_pk",
            "name": "idx_previous_pk",
        }
    ],
    "m3e_large": {
        "database_name": "rag",
        "collection_name": "m3e_large",
        "fields": milvus_common_fields
        + [
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
        ],
    },
    "zpoint_large_embedding_zh": {
        "database_name": "rag",
        "collection_name": "zpoint_large_embedding_zh",
        "fields": milvus_common_fields
        + [
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1792),
        ],
    },
}
