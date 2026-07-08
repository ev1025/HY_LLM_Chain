from dotenv import load_dotenv
load_dotenv()

import yaml, os, hashlib, traceback, boto3
from typing import Union, List, Iterable, Tuple, Optional, Set , Dict
from datetime import datetime

from langchain_openai import OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain.storage import LocalFileStore

from opensearchpy import OpenSearch, RequestsHttpConnection
from opensearchpy.helpers import scan, bulk
from opensearchpy.helpers.signer import AWSV4SignerAuth

from .s3_io import load_jsonl_from_s3

with open('config/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

KGS_INDEX = config['aws']['opensearch']['kgs_index']

OPENSEARCH_ENDPOINT = os.environ.get("OS_ENDPOINT")
AWS_REGION = config['aws']['region']
SERVICE = config['aws']['service']

OPEN_INDEX_K1 = config['aws']['opensearch']['kgs_index_k1']
OPEN_INDEX_B = config['aws']['opensearch']['kgs_index_b']
OPEN_INDEX_M = config['aws']['opensearch']['kgs_index_m']
OPEN_INDEX_EFC = config['aws']['opensearch']['kgs_ef_construction']
OPEN_INDEX_EFS = config['aws']['opensearch']['kgs_ef_search']


EMBEDDING_MODEL = config['openai']['embeddings']['models'][1]
EMBEDDING_DIM = config['openai']['embeddings']['dim'][1]
EMBEDDING_BATCH_SIZE = 100

session = boto3.Session()
credentials = session.get_credentials()
awsauth = AWSV4SignerAuth(credentials, AWS_REGION, SERVICE)

INDEX_NAME = None
INPUT_FILE_PATH = None

def get_opensearch_client(timeout = 60) -> OpenSearch:
    """ Opensearch 클라이언트 생성기 """

    return OpenSearch(
        hosts=[{"host": OPENSEARCH_ENDPOINT, "port": 443, "scheme": "https"}],
        http_auth=awsauth,
        http_compress=True,
        connection_class=RequestsHttpConnection,
        use_ssl=True, verify_certs=True,
        timeout=timeout, max_retries=3, retry_on_timeout=True
    )


def get_opensearch_vectorstore(index_name, 
                                ef_search=100, 
                                m=16, 
                                ef_construction=128):
    return OpenSearchVectorSearch(
        index_name=index_name,
        embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        opensearch_url=f"https://{OPENSEARCH_ENDPOINT}",
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        engine=config['retriever']['vector_store']['engine'],
        space_type=config['retriever']['vector_store']['space_type'],                            
        timeout=60
    )

_CACHED_EMBEDDER = None

def get_cached_embedder():
    """ 한 번 임베딩 된 텍스트를 캐시에 저장하고 반환하는 함수 """
    global _CACHED_EMBEDDER
    if _CACHED_EMBEDDER is None:
        # (1) 캐시를 저장할 로컬 파일 저장소
        store = LocalFileStore("/app/cache/embeddings/") # 경로는 서버 환경에 맞게 수정
        
        # (2) 원본 임베딩 모델
        underlying_embedder = OpenAIEmbeddings(
            model=config["openai"]["embeddings"]["models"][1]
        )
        
        # (3) 캐시 래퍼 적용
        _CACHED_EMBEDDER = CacheBackedEmbeddings.from_bytes_store(
            underlying_embedder, 
            store, 
            namespace=underlying_embedder.model # 모델별로 캐시 네임스페이스 분리
        )
    return _CACHED_EMBEDDER


def create_open_schema(
    embedding_dim: int = EMBEDDING_DIM,
    k1: float = OPEN_INDEX_K1,
    b: float = OPEN_INDEX_B,
    M: int = OPEN_INDEX_M,
    EFC: int = OPEN_INDEX_EFC,
    EFS: int = OPEN_INDEX_EFS,
    metadata: Union[dict, None] = None) -> dict:

    if metadata is None:
        metadata = {}

    return {
        "settings": {
            "index": {
                "knn": True,
                "similarity": {"default": {"type": "BM25", "k1": k1, "b": b}},
                "analysis": {
                    "tokenizer": {"nori_tokenizer": {"type": "nori_tokenizer"}},
                    "analyzer": {"korean_nori": { # 커스텀 토크나이저
                        "type": "custom", "tokenizer": "nori_tokenizer",
                        "filter": ["lowercase", "nori_readingform"] # [소문자, 한자를 한글로]
                    }},
                },
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
                    "type": "knn_vector",
                    "dimension": embedding_dim,
                    "method": {
                        "name": "hnsw", "space_type": "cosinesimil", "engine": "faiss",
                        "parameters": {"ef_search": EFS, "m": M, "ef_construction": EFC}
                    }
                }
            }
        }
    }

def build_opensearch_index(client: OpenSearch, index_name: str, body: dict):
    """ Opensearch Index생성기 """
    if client.indices.exists(index=index_name):
        print(f"✅ 인덱스 '{index_name}'에 연결 성공")
        return
    
    client.indices.create(index=index_name, body=body)
    print(f"✅ 인덱스 '{index_name}'를 Nori 분석기로 생성했습니다.")


def generate_bulk_actions(docs: List, ids: List, vectors: List, index_name: str):
    """ Opensearch Index 형태 생성기 """

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

def delete_opensearch_documents(
        client: OpenSearch, 
        index_name: str, 
        ids_to_delete: Iterable[str],
        id_field_path: str , 
        batch_size: int = 200, 
        assume_keyword: bool = True
) -> int:
    """ Opensearch 동기화(중복 또는 업데이트내역 삭제) """
 
    if not ids_to_delete:
        print("--- 삭제할 문서가 없습니다. ---")
        return 0

    ids = list(set(ids_to_delete))  # 중복 제거

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
                # 텍스트 필드만 존재할 때 임시 대안
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
                params={
                    "conflicts": "proceed",
                    "refresh": "true",
                    "wait_for_completion": "true",
                    "slices": "auto",
                },
                request_timeout=120,
            )
            deleted = int(resp.get("deleted", 0))
            deleted_total += deleted
            print(f"  · 배치 {i//batch_size + 1}: 문서 {deleted}건 삭제")
        except Exception as e:
            print(f"  ! 배치 {i//batch_size + 1} 오류: {e}")

    print(f"✅ 삭제 완료: 총 문서 {deleted_total}건")
    return deleted_total

def opensearch_uploader(client : OpenSearch,
                        index_name: str, 
                        input_file_path: str):
    """
    [S3 jsonl -> Opensearch Index]

    JSONL에서 문서를 로드해 임베딩을 배치 계산하고, Bulk API로 OpenSearch에 적재합니다.

    임베딩은 EMBEDDING_BATCH_SIZE 단위로 나눠 계산하며,
    Bulk 전송은 chunk_size(문서 수)와 max_chunk_bytes(바이트 상한)를 함께 제한해
    HTTP 413(Request Entity Too Large)을 방지합니다.

    Args:
        CLIENT (Opensearch) : Opensearch 클라이언트 객체
        INDEX_NAME (str): 적재 대상 인덱스명.
        INPUT_FILE_PATH (str): JSONL 파일 경로(id, page_content, metadata 포함).

    Workflow:
        1) get_opensearch_client()로 연결 생성
        2) ensure_index()로 스키마 보장
        3) JSONL 파싱 → Document/ID 배열 구성
        4) OpenAI 임베딩 배치 계산
        5) generate_bulk_actions()로 액션 생성 후 bulk 적재

    Notes:
        - raise_on_error=False로 실패 항목을 수집해 로깅합니다.
        - request_timeout을 충분히 크게 두고, max_chunk_bytes를 10MB 미만(예: 9MB)으로 설정하세요.
        - 인덱스의 매핑에서 metadata.* 필드는 검색/필터링 용도로 keyword/integer로 정의되어야 합니다.
    """

    # --- 반환받은 S3_URI에서 jsonl 데이터 불러오기 ---
    docs_to_index, doc_ids = load_jsonl_from_s3(input_file_path)

    if not docs_to_index:
        print("❌ 처리할 문서가 없습니다.")
        return
    
    # --- 데이터 임베딩 ---
    print(f"✅ '{input_file_path}' 임베딩 시작")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    all_vectors = []
    
    for i in range(0, len(docs_to_index), EMBEDDING_BATCH_SIZE):
        batch_docs = docs_to_index[i : i + EMBEDDING_BATCH_SIZE]
        batch_texts = [doc.page_content for doc in batch_docs]
        print(f"  ✅ 문서 {i+1}번부터 {i+len(batch_texts)}번까지 임베딩 중...")
        # 동기 메서드 embed_documents 사용
        batch_vectors = embeddings.embed_documents(batch_texts)
        all_vectors.extend(batch_vectors)

    print(f"✅ 총 {len(all_vectors)}개 문서의 임베딩 생성 완료.")

    # --- 데이터 Opensearch DB에 적재 (opensearch-py의 동기 bulk 헬퍼 사용) ---
    print("OpenSearch에 데이터 적재 시작...")
    
    action_generator = generate_bulk_actions(docs_to_index, doc_ids, all_vectors, index_name)

    # chunk_size, max_chunk_bytes 둘 다 넣으면 둘 중 작은 조건을 우선
    success, errors = bulk(
    client,
    actions=action_generator,      
    chunk_size=400,                   # 문서 개수 기준. 
    max_chunk_bytes=9 * 1024 * 1024,  # 바이트 기준 (최대 10MB라서 9MB로 제한)
    request_timeout=180,
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
# 🏛️ OpenSearch PDF
# ===================================================
def get_indexed_ids_and_mtime_opensearch(client: OpenSearch, index_name: str) -> Tuple[Set[str], Dict[str, Optional[str]]]:
    """
    OpenSearch: 인덱스 전체를 스캔하여 'metadata.drive_file_id'와 'metadata.modified_time'을 반환합니다.
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
        print(f"🚨 get_indexed_ids_and_mtime (OpenSearch) 함수 실행 중 오류: {e}")
        traceback.print_exc()
        return set(), {}


def need_update_opensearch(drive_mtime: Optional[str], indexed_mtime: Optional[str]) -> bool:
    """
    OpenSearch: Drive(str)와 OpenSearch(str)의 수정 시간을 비교합니다.
    """
    if not drive_mtime or not indexed_mtime:
        return True # 둘 중 하나라도 값이 없으면 업데이트 대상
    try:
        # 둘 다 문자열(str)이므로 datetime으로 변환하여 비교
        to_dt = lambda s: datetime.fromisoformat(s.replace("Z", "+00:00"))
        return to_dt(drive_mtime) > to_dt(indexed_mtime)
    except Exception:
        # 파싱 실패 시 단순 문자열 비교
        return drive_mtime != indexed_mtime

def generate_chunk_id_opensearch(file_id: str, chunk_index: int) -> str:
    """OpenSearch: SHA256 해시 기반의 청크 ID를 생성합니다."""
    unique_string = f"{file_id}-{chunk_index}"
    return hashlib.sha256(unique_string.encode('utf-8')).hexdigest()

# ===================================================
# 🏛️ OpenSearch Sheet
# ===================================================
def get_existing_doc_ids_sheet_opensearch(client: OpenSearch, index_name: str) -> set:
    """OpenSearch: KGS 인덱스에 존재하는 모든 문서의 ID(_id)를 집합(set)으로 반환합니다."""
    
    if not client.indices.exists(index=index_name):
        return set()
    
    try:
        # _source=False로 설정하여 ID만 가져옵니다.
        results = scan(client, index=index_name, query={"query": {"match_all": {}}}, _source=False)
        id_set = {hit["_id"] for hit in results}
        print(f"✅ 기존 문서 {len(id_set)}개의 ID를 스캔 완료")
        return id_set

    except Exception as e:
        print(f"🚨 get_existing_doc_ids (Sheet/OpenSearch) 함수 실행 중 오류: {e}")
        traceback.print_exc()
        return set()

def generate_sheet_id_opensearch(id_string: str) -> str:
    """OpenSearch: SHA256 해시 기반의 시트 ID를 생성합니다."""
    return hashlib.sha256(id_string.encode("utf-8")).hexdigest()

if __name__ == "__main__":
    opensearch_uploader(INDEX_NAME, INPUT_FILE_PATH)