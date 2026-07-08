from dotenv import load_dotenv
load_dotenv()

import yaml, os, hashlib, traceback
from typing import Union, List, Iterable, Tuple, Optional, Set , Dict
from datetime import datetime

from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from langchain.storage import LocalFileStore

from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan, bulk

from .s3_io import load_jsonl_from_s3

with open('config/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# .env 파일에서 엔드포인트와 API 키를 가져옵니다.
ELASTIC_ENDPOINT = os.environ.get("ES_ENDPOINT")
ELASTIC_API_KEY = os.environ.get("ELASTIC_API_KEY")

ES_INDEXS_K1 = config['elasticsearch']['kgs_index_k1']
ES_INDEX_B = config['elasticsearch']['kgs_index_b']
ES_INDEX_M = config['elasticsearch']['kgs_index_m']
ES_INDEX_EFC = config['elasticsearch']['kgs_ef_construction']

EMBEDDING_MODEL = config['openai']['embeddings']['models'][1]
EMBEDDING_DIM = config['openai']['embeddings']['dim'][1]
EMBEDDING_BATCH_SIZE = 100

INDEX_NAME = None
INPUT_FILE_PATH = None


def get_es_client(timeout=60) -> Elasticsearch:
    """ Elasticsearch 클라이언트 생성기 """

    return Elasticsearch(
        ELASTIC_ENDPOINT,
        api_key=ELASTIC_API_KEY,
        verify_certs=True,
        ssl_assert_hostname=False, # 필요에 따라 조정
        ssl_show_warn=False,
        request_timeout=timeout,
        max_retries=3,
        retry_on_timeout=True
    )


def get_es_vectorstore(index_name: str) -> ElasticsearchStore:
    es_client = get_es_client()
    
    return ElasticsearchStore(
        index_name=index_name,
        embedding=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        es_connection=es_client,
        vector_query_field="vector_field", 
        query_field="text"
    )

_CACHED_EMBEDDER = None

def get_cached_embedder():
    """ 한 번 임베딩 된 텍스트를 캐시에 저장하고 반환하는 함수 """
    global _CACHED_EMBEDDER
    if _CACHED_EMBEDDER is None:
        store = LocalFileStore("/app/cache/embeddings/") 
        underlying_embedder = OpenAIEmbeddings(
            model=config["openai"]["embeddings"]["models"][1]
        )
        _CACHED_EMBEDDER = CacheBackedEmbeddings.from_bytes_store(
            underlying_embedder, 
            store, 
            namespace=underlying_embedder.model
        )
    return _CACHED_EMBEDDER


def create_es_schema(
    embedding_dim: int = EMBEDDING_DIM,
    k1:float = ES_INDEXS_K1,
    b:float = ES_INDEX_B,
    M: int = ES_INDEX_M,
    EFC: int = ES_INDEX_EFC,
    metadata: Union[dict, None] = None)-> dict :

    if metadata is None:
        metadata = {}

    return {
        "settings": {
            "index": {
                "similarity": {"default": {"type": "BM25", "k1": k1, "b": b}},
            },
            "analysis": {
                "analyzer": {
                    "kiwi_ws": {
                        "type": "custom",
                        "tokenizer": "whitespace",
                        "filter": ["lowercase"]
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "text": {"type": "text", "analyzer": "standard"},
                "metadata": {
                    "type": "object",
                    "properties": metadata
                },
                "vector_field": {
                    "type": "dense_vector",
                    "dims": embedding_dim,
                    "index": True,
                    "similarity": "dot_product",
                    "index_options": {"type": "hnsw", "m": M, "ef_construction": EFC}
                }
            }
        }
    }

def build_elasticsearch_index(client: Elasticsearch, index_name: str, body: dict):
    """ Elasticsearch Index 생성기 """
    if client.indices.exists(index=index_name):
        print(f"✅ 인덱스 '{index_name}'에 연결 성공")
        return
    
    # ES는 create 시 settings와 mappings를 함께 전달
    client.indices.create(index=index_name, 
                            settings=body["settings"], 
                            mappings=body["mappings"])
    print(f"✅ 인덱스 '{index_name}'를 Kiwi/Whitespace 분석기반으로 생성했습니다.")


def generate_bulk_actions(docs, ids, vectors, index_name: str):
    """ Elasticsearch Index 형태 생성기 """

    for i, doc in enumerate(docs):
        yield {
            "_index": index_name,
            "_id": ids[i],
            "_source": {
                "text": doc.page_content,
                "metadata": doc.metadata,
                "vector_field": vectors[i]
            }
        }

def delete_elastic_documents(
        client: Elasticsearch, 
        index_name: str, 
        ids_to_delete: Iterable[str],
        id_field_path: str , 
        batch_size: int = 200, 
        assume_keyword: bool = True
) -> int:
    """ Elasticsearch 동기화(중복 또는 업데이트내역 삭제) """

    if not ids_to_delete:
        print("--- 삭제할 문서가 없습니다. ---")
        return 0

    ids = list(set(ids_to_delete))
    
    print(f"\n--- '{index_name}'에서 '{id_field_path}' 기준으로 {len(ids)}건 삭제 시작 ---")
    deleted_total = 0

    for i in range(0, len(ids), batch_size):
        batch = ids[i : i + batch_size]

        if id_field_path.lower() in ["id", "_id"]:
            body = {"query": {"ids": {"values": batch}}}
        else:
            if assume_keyword:
                body = {"query": {"terms": {id_field_path: batch}}}
            else:
                body = {
                    "query": {
                        "bool": {
                            "should": [{"match_phrase": {id_field_path: v}} for v in batch],
                            "minimum_should_match": 1,
                        }
                    }
                }

        try:
            resp = client.delete_by_query(
                index=index_name,
                body=body,
                conflicts="proceed",
                refresh=True,
                wait_for_completion=True,
                slices="auto",
                request_timeout=120,
            )
            deleted = int(resp.get("deleted", 0))
            deleted_total += deleted
            print(f"  · 배치 {i//batch_size + 1}: 문서 {deleted}건 삭제")
        except Exception as e:
            print(f"  ! 배치 {i//batch_size + 1} 오류: {e}")

    print(f"✅ 삭제 완료: 총 문서 {deleted_total}건")
    return deleted_total

def elasticsearch_uploader(client : Elasticsearch,
                        index_name: str, 
                        input_file_path: str):
    """
    [S3 jsonl -> Elasticsearch Index] (OpenSearch와 거의 동일, bulk 헬퍼 사용)
    """
    docs_to_index, doc_ids = load_jsonl_from_s3(input_file_path)

    if not docs_to_index:
        print("❌ 처리할 문서가 없습니다.")
        return
    
    print(f"✅ '{input_file_path}' 임베딩 시작")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    all_vectors = []
    
    for i in range(0, len(docs_to_index), EMBEDDING_BATCH_SIZE):
        batch_docs = docs_to_index[i : i + EMBEDDING_BATCH_SIZE]
        batch_texts = [doc.page_content for doc in batch_docs]
        print(f"  ✅ 문서 {i+1}번부터 {i+len(batch_texts)}번까지 임베딩 중...")
        batch_vectors = embeddings.embed_documents(batch_texts)
        all_vectors.extend(batch_vectors)

    print(f"✅ 총 {len(all_vectors)}개 문서의 임베딩 생성 완료.")

    print("Elasticsearch에 데이터 적재 시작...")
    action_generator = generate_bulk_actions(docs_to_index, doc_ids, all_vectors, index_name)

    success, errors = bulk(
        client.options(request_timeout=180),
        actions=action_generator,      
        chunk_size=400,
        max_chunk_bytes=9 * 1024 * 1024,
        raise_on_error=False,
    )
    
    print("\n" + "="*50)
    print("🎉 데이터 적재 완료!")
    print(f"  - 성공: {success} 건")
    print(f"  - 실패: {len(errors)} 건")
    if errors:
        print("실패 내역 (최대 5개):", errors[:5])
    print("="*50)

# ===================================================
# 🏛️ Elasticsearch PDF
# ===================================================
def get_indexed_ids_and_mtime_es(client: Elasticsearch, index_name: str) -> Tuple[Set[str], Dict[str, Optional[str]]]:
    """
    Elasticsearch: 인덱스 전체를 스캔하여 'metadata.drive_file_id'와 'metadata.modified_time'을 반환합니다.
    (OpenSearch와 동일, elasticsearch.helpers.scan 사용)
    """
    if not client.indices.exists(index=index_name):
        return set(), {}

    existing_ids: Set[str] = set()
    id_to_mtime: Dict[str, Optional[str]] = {} 
    q = {"query": {"exists": {"field": "metadata.drive_file_id"}}, "_source": ["metadata.drive_file_id", "metadata.modified_time"]}

    try:
        for hit in scan(client, index=index_name, query=q, size=1000, scroll="5m"):
            meta = hit.get("_source", {}).get("metadata", {})
            fid, mtime = meta.get("drive_file_id"), meta.get("modified_time")
            if not fid: continue
            
            existing_ids.add(fid)
            cur = id_to_mtime.get(fid)
            if mtime and (cur is None or mtime > cur): # 문자열로 비교
                 id_to_mtime[fid] = mtime
        
        print(f"✅ 기존 문서 {len(existing_ids)}개의 ID 스캔 완료")
        return existing_ids, id_to_mtime
        
    except Exception as e:
        print(f"🚨 get_indexed_ids_and_mtime (Elasticsearch) 함수 실행 중 오류: {e}")
        traceback.print_exc()
        return set(), {}


def need_update_es(drive_mtime: Optional[str], indexed_mtime: Optional[str]) -> bool:
    """
    Elasticsearch: Drive(str)와 ES(str)의 수정 시간을 비교합니다. (OpenSearch와 100% 동일)
    """
    if not drive_mtime or not indexed_mtime:
        return True 
    try:
        to_dt = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))
        return to_dt(drive_mtime) > to_dt(indexed_mtime)
    except Exception:
        return drive_mtime != indexed_mtime

def generate_chunk_id_es(file_id: str, chunk_index: int) -> str:
    """Elasticsearch: SHA256 해시 기반의 청크 ID를 생성합니다. (OpenSearch와 100% 동일)"""
    unique_string = f"{file_id}-{chunk_index}"
    return hashlib.sha256(unique_string.encode('utf-8')).hexdigest()

# ===================================================
# 🏛️ Elasticsearch Sheet
# ===================================================
def get_existing_doc_ids_sheet_es(client: Elasticsearch, index_name: str) -> set:
    """Elasticsearch: KGS 인덱스에 존재하는 모든 문서의 ID(_id)를 집합(set)으로 반환합니다. (OpenSearch와 동일)"""
    
    if not client.indices.exists(index=index_name):
        return set()
    
    try:
        results = scan(client, index=index_name, query={"query": {"match_all": {}}}, _source=False)
        id_set = {hit["_id"] for hit in results}
        print(f"✅ 기존 문서 {len(id_set)}개의 ID를 스캔 완료")
        return id_set

    except Exception as e:
        print(f"🚨 get_existing_doc_ids (Sheet/Elasticsearch) 함수 실행 중 오류: {e}")
        traceback.print_exc()
        return set()

def generate_sheet_id_es(id_string: str) -> str:
    """Elasticsearch: SHA256 해시 기반의 시트 ID를 생성합니다. (OpenSearch와 100% 동일)"""
    return hashlib.sha256(id_string.encode("utf-8")).hexdigest()