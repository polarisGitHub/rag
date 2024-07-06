from pymilvus import FieldSchema, DataType
from domain.information import EmbeddingModelInfo, OllamaModelInfo

# embedding模型
select_embedding_model = "zpoint_large_embedding_zh"

# llm模型
__select_ollama_model_name = "qwen2:7b"
__ollama_model_repo = {
    "llama3": OllamaModelInfo("llama3"),
    "qwen2:7b": OllamaModelInfo("qwen2:7b"),
    "wangshenzhi/llama3-8b-chinese-chat-ollama-q8": OllamaModelInfo("wangshenzhi/llama3-8b-chinese-chat-ollama-q8"),
    "gemma2": OllamaModelInfo("gemma2"),
}
select_ollama_model_info = __ollama_model_repo[__select_ollama_model_name]

# 向量数据库
milvus_common_fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=64, is_primary=True, description="primary id"),
    ##
    FieldSchema(name="content_type", dtype=DataType.VARCHAR, max_length=16, description="fragment,sentence"),
    FieldSchema(name="previous_id", dtype=DataType.VARCHAR, max_length=64, description="previous text"),
    FieldSchema(name="next_id", dtype=DataType.VARCHAR, max_length=64, description="next text"),
    FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=64, description="parent text"),
    ##
    FieldSchema(name="paragraph", dtype=DataType.VARCHAR, max_length=512, description="paragraph"),
    FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048, description="text"),
    FieldSchema(name="meta", dtype=DataType.JSON, description="meta"),
]
milvus = {
    "host": "127.0.0.1",
    "port": 19530,
    "search_params": {
        "metric_type": "L2",
        "params": {"nprobe": 64},
    },
    "output_fields": [
        "id",
        "content_type",
        "previous_id",
        "next_id",
        "parent_id",
        "paragraph",
        "text",
        "meta",
    ],
    "vectors_indexes": [
        {
            "field_name": "embeddings",
            "index_params": {"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}},
            "name": "idx_embeddings",
        }
    ],
    "indexes": [
        {
            "field_name": "previous_id",
            "name": "idx_previous_id",
        },
        {
            "field_name": "next_id",
            "name": "idx_next_id",
        },
        {
            "field_name": "parent_id",
            "name": "idx_parent_id",
        },
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
            FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=1792),
        ],
    },
}

# 全文检索

elasticsearch = {
    "uri":"http://localhost:9200",
    "index": "rag",
    "body": {
        "settings": {
            "number_of_replicas": 0,
        },
        "mappings": {
            "properties": {
                "content_type": {"type": "keyword"},
                "previous_id": {"type": "keyword"},
                "next_id": {"type": "keyword"},
                "parent_id": {"type": "keyword"},
                "text": {
                    "type": "text",
                    "analyzer": "hanlp_index",
                    "search_analyzer": "hanlp_standard",
                },
                "paragraph": {"type": "keyword"},
                "meta": {"type": "object"},
            }
        },
    },
}
