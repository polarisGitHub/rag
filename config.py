from pymilvus import FieldSchema, DataType
from sentence_transformers import SentenceTransformer

__select_embedding_model_name = "zpoint_large_embedding_zh"
__select_ollama_model_name = 'qwen2:7b'

__embedding_model_repo = {
    "m3e_large": {
        "name": "m3e_large",
        "path": "/home/polaris_he/cached_model/m3e-large",
    },
    "zpoint_large_embedding_zh": {
        "name": "zpoint_large_embedding_zh",
        "path": "/home/polaris_he/cached_model/zpoint_large_embedding_zh",
    },
}

__ollama_model_repo = {
    "llama3":{
        "model_name":'llama3',
    },
    "qwen2:7b":{
        "model_name":'qwen2:7b',
    }
}


select_embedding_model_info = __embedding_model_repo[__select_embedding_model_name]
select_ollama_model_info = __ollama_model_repo[__select_ollama_model_name]

select_embedding_model = SentenceTransformer(select_embedding_model_info["path"])

milvus_common_fields = [
    FieldSchema(name="pk", dtype=DataType.VARCHAR, max_length=64, is_primary=True, description="primary id use sha256"),
    FieldSchema(name="previous_pk", dtype=DataType.VARCHAR, max_length=64, description="previous primary id"),
    FieldSchema(name="content", dtype=DataType.VARCHAR, description="content", max_length=1024),
    FieldSchema(name="title", dtype=DataType.JSON, description="title"),
    FieldSchema(name="special_title", dtype=DataType.VARCHAR, description="special_title", max_length=100),
]
milvus = {
    "host": "127.0.0.1",
    "port": 19530,
    "search_params": {
        "metric_type": "COSINE",
        "params": {"nlist": 64},
    },
    "output_fields": [
        "pk",
        "previous_pk",
        "content",
        "title",
        "special_title",
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
            "name": "idx_pprevious_pk",
        }
    ],
    "m3e_large": {
        "database_name": "test_rag",
        "collection_name": "m3e_large",
        "fields": milvus_common_fields
        + [
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
        ],
    },
    "zpoint_large_embedding_zh": {
        "database_name": "test_rag",
        "collection_name": "zpoint_large_embedding_zh",
        "fields": milvus_common_fields
        + [
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1792),
        ],
    },
}