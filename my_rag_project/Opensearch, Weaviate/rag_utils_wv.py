from dotenv import load_dotenv
load_dotenv()

import os, yaml, traceback, uuid
from typing import List, Tuple, Optional, Set , Dict, Iterable
from datetime import datetime, timezone

import weaviate
from weaviate.classes.data import DataObject
from weaviate.classes.query import Filter
from weaviate.auth import AuthApiKey

from langchain_core.documents import Document

from .s3_io import load_jsonl_from_s3

with open('config/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

WV_INDEX_SUSI = config['weaviate']['susi_index']
WV_INDEX_KGS = config['weaviate']['kgs_index']
WV_INDEX_K1 = config['weaviate']['kgs_index_k1']
WV_INDEX_B = config['weaviate']['kgs_index_b']
WV_INDEX_EFC = config['weaviate']['kgs_ef_construction']
WV_INDEX_M = config['weaviate']['kgs_index_m']
WV_INDEX_EFS = config['weaviate']['kgs_ef_search']


def get_weaviate_client() -> weaviate.WeaviateClient:
    """ Weaviate v4 클라이언트를 생성합니다."""
    try:
        url = os.environ["WEAVIATE_URL"]
        api_key = os.environ["WEAVIATE_API_KEY"]
        openai_key = os.environ.get("WEAVIATE_OPENAI_KEY")
    except KeyError as e:
        print(f"🚨 .env 파일에 {e} 환경 변수가 설정되지 않았습니다.")
        raise e

    headers = {}
    if openai_key:
        headers["X-OpenAI-Api-Key"] = openai_key
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=url,
        auth_credentials=AuthApiKey(api_key), 
        headers=headers
    )
  
    return client

def create_weaviate_schema(client: weaviate.WeaviateClient, 
                            index_name: str, 
                            properties: List[dict],
                            efc: int = WV_INDEX_EFC,
                            m: int = WV_INDEX_M,
                            efs: int = WV_INDEX_EFS,
                            k1: float = WV_INDEX_K1,
                            b: float =  WV_INDEX_B):
    """ Weaviate v4: 클래스(인덱스) 스키마를 생성합니다 """
    
    if client.collections.exists(index_name):
        print(f"✅ 스키마(컬렉션) '{index_name}'에 연결 성공")
        return

    class_obj = {
        "class": index_name,  
        "description": f"{index_name} data",
        "vectorizer": "text2vec-openai",
        "vectorIndexType": "hnsw",            
        "vectorIndexConfig": {                
            "efConstruction": efc,
            "ef": efs,
            "maxConnections": m,
        },
        "moduleConfig": {
            "text2vec-openai": {
                "vectorizeClassName": True
            }            
        },    
        "invertedIndexConfig": {
            "bm25": {
                "k1": k1,
                "b": b
            }
        },
        "properties": properties
    }
    
    client.collections.create_from_dict(class_obj)
    print(f"✅ 스키마(컬렉션) '{index_name}'를 생성했습니다.")

def generate_weaviate_objects(docs: List[Document], ids: List[str], index_name: str) -> Iterable[DataObject]:
    """
    Susi/Kgs 인덱스에 따라 LangChain Document를 Weaviate DataObject로 변환하는 생성기(generator)입니다.
    """
    
    # Susi 인덱스(PDF)용 데이터 매핑
    if index_name.startswith(WV_INDEX_SUSI):
        for i, doc in enumerate(docs):
            meta = doc.metadata
            properties = {
                "text": doc.page_content,
                "text_morph": meta.get("text_morph"), 
                "drive_file_id": meta.get("drive_file_id"),
                "university": meta.get("university"),
                "year": meta.get("year"),
                "admission_type": meta.get("admission_type"),
                "document_type": meta.get("document_type"),
                "modified_time": meta.get("modified_time"),
                "metadata": {
                    "source": meta.get("source"),
                    "chunk_number": meta.get("chunk_number"),
                    "start_index": meta.get("start_index"),
                }
            }
            final_properties = {k: v for k, v in properties.items() if v is not None}
            yield DataObject(properties=final_properties, uuid=ids[i])
    
    # Kgs 인덱스(Sheet)용 데이터 매핑
    elif index_name.startswith(WV_INDEX_KGS):
        for i, doc in enumerate(docs):
            meta = doc.metadata
            properties = {
                "text": doc.page_content, 
                "question": meta.get("question", ""), 
                "text_morph": meta.get("text_morph"), 
                "question_morph": meta.get("question_morph"), 
                "university": meta.get("university"),
                "year": meta.get("year"),
                "category1": meta.get("category1"),
                "category2": meta.get("category2"),
                "category3": meta.get("category3"),
            }
            final_properties = {k: v for k, v in properties.items() if v is not None}
            yield DataObject(properties=final_properties, uuid=ids[i])

def weaviate_uploader(client: weaviate.WeaviateClient, index_name: str, input_file_path: str):
    """ S3 JSONL → Weaviate 적재 """
    docs_to_index, doc_ids = load_jsonl_from_s3(input_file_path)

    if not docs_to_index:
        print("처리할 문서가 없습니다.")
        return

    try:
        collection = client.collections.get(index_name)
        object_generator = generate_weaviate_objects(docs_to_index, doc_ids, index_name)

        # 200개씩 수동 배치 업로드
        total_success = 0
        BATCH_SIZE = 200 
        batch: List[DataObject] = [] # 배치 리스트 초기화

        for obj in object_generator:
            batch.append(obj)
            
            # 배치가 200개에 도달하면 업로드 실행
            if len(batch) >= BATCH_SIZE:
                print(f"  ✅ 문서 {total_success + 1}번부터 {total_success + len(batch)}번까지 업로드 중...")
                result = collection.data.insert_many(batch)
                
                if result.has_errors:
                    print(f"  🚨 배치 업로드 중 오류 발생: {result.errors}")
                
                total_success += len(result.uuids)
                batch = [] # 배치 리스트 초기화
        
        # 루프가 끝난 후, 200개가 안 되어 남아있던 나머지 배치를 업로드
        if batch:
            print(f"  ✅ 문서 {total_success + 1}번부터 {total_success + len(batch)}번까지 업로드 중...")
            result = collection.data.insert_many(batch)
            
            if result.has_errors:
                print(f"  🚨 배치 업로드 중 오류 발생: {result.errors}")
            
            total_success += len(result.uuids)
        
        print(f"🎉 Weaviate 데이터 적재 완료! (총 {total_success} 건)")
        
    except Exception as e:
        print(f"🚨 Weaviate Uploader 오류: {e}")
        raise e


def delete_weaviate_documents(client: weaviate.WeaviateClient, index_name: str, ids_to_delete: List[str], id_field_path: str):
    """
    id_field_path가 'id' 또는 '_id'인 경우: Native UUID로 간주하여 delete_by_id()를 호출합니다.
    - 그 외의 경우: Property로 간주하여 delete_many(where=...)를 호출합니다.
    """
    if not ids_to_delete:
        print("✅ 삭제할 문서가 없습니다. (0건)")
        return

    print(f"--- '{index_name}'에서 '{id_field_path}' 기준으로 {len(ids_to_delete)}건 삭제 시작 ---")
    
    try:
        # 1. 삭제를 수행할 컬렉션 객체를 가져옵니다.
        collection = client.collections.get(index_name)
        
        # 2. KGS 파이프라인의 경우 (id_field_path == "id")
        #    Native UUID 기준으로 개별 삭제를 수행합니다.
        if id_field_path.lower() in ["id", "_id"]:
            successful_deletes = 0
            failed_deletes = 0
            
            for uid in ids_to_delete:
                try:
                    collection.data.delete_by_id(uid)
                    successful_deletes += 1
                except Exception as e:
                    print(f"  ⚠️  ID {uid} 삭제 실패: {e}")
                    failed_deletes += 1
            print(f"✅ 삭제 완료: 총 문서 {successful_deletes}건 (실패: {failed_deletes})")

        # 3. Susi 파이프라인의 경우 (id_field_path == "drive_file_id")
        else:
            print(f"  > '{id_field_path}' Property 기준으로 'delete_many' 실행...")
            
            # 필터 생성
            where_filter = Filter.by_property(id_field_path).contains_any(list(map(str, ids_to_delete)))
            
            # 배치 삭제 실행
            result = collection.data.delete_many(
                where=where_filter
            )
            print(f"✅ Weaviate 배치 삭제 완료. 결과: {result}")
            print(f"✅ 삭제 완료: 총 문서 {result.successful}건 (실패: {result.failed})")

    except Exception as e:
        print(f"🚨 Weaviate 배치 삭제 중 심각한 오류 발생: {e}")

# ===================================================
# 🏛️ Weaviate PDF
# ===================================================
def _to_datetime_utc(x) -> Optional[datetime]:
    """ 입력값(str, datetime, None)을 UTC 타임존 정보가 포함된 datetime 객체로 변환합니다. """
    if not x:
        return None
    
    if isinstance(x, datetime):
        return x if x.tzinfo else x.replace(tzinfo=timezone.utc)
    
    if isinstance(x, str):
        try:
            s = x.replace("Z", "+00:00")
            return datetime.fromisoformat(s)
        except ValueError:
            return None 
    
    return None

def get_indexed_ids_and_mtime_weaviate(client: weaviate.WeaviateClient, index_name: str) -> Tuple[Set[str], Dict[str, Optional[datetime]]]:
    """ 실제 스키마에 맞춰 drive_file_id와 modified_time을 스캔합니다. """
    
    if not client.collections.exists(index_name):
        return set(), {}

    existing_ids: Set[str] = set()
    id_to_mtime: Dict[str, Optional[datetime]] = {} 
    
    try:
        collection = client.collections.get(index_name)
        for item in collection.iterator(
            include_vector=False,
            return_properties=["drive_file_id", "modified_time"]
        ):
            props = item.properties or {}
            fid = props.get("drive_file_id")
            mtime_raw = props.get("modified_time") # String 타입으로 반환됨
            
            if not fid: 
                continue
            
            existing_ids.add(fid)
            mt = _to_datetime_utc(mtime_raw) # String -> Datetime 변환
            
            cur = id_to_mtime.get(fid)
            if mt and (cur is None or mt > cur):
                 id_to_mtime[fid] = mt # datetime 객체로 저장
                 
        print(f"✅ 기존 문서 {len(existing_ids)}개의 ID 스캔 완료")
        return existing_ids, id_to_mtime

    except Exception as e:
        print(f"🚨 get_indexed_ids_and_mtime (Weaviate) 함수 실행 중 오류: {e}")
        traceback.print_exc()
        return set(), {}

def need_update_weaviate(drive_mtime: Optional[str], indexed_mtime: Optional[datetime]) -> bool:
    """ Drive(str)와 Weaviate(datetime)의 수정 시간을 비교합니다. """
    d1 = _to_datetime_utc(drive_mtime) # Drive(str → datetime)
    d2 = indexed_mtime                 # Weaviate(datetime)

    if d2 is None: return True
    if d1 is None: return False
    return d1 > d2

def generate_chunk_id_weaviate(file_id: str, chunk_index: int) -> str:
    """ PDF용 UUIDv5 ID를 생성합니다."""
    unique_string = f"{file_id}-{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_string))

# ===================================================
# 🏛️ Weaviate Sheet
# ===================================================
def get_existing_doc_ids_sheet_weaviate(client: weaviate.WeaviateClient, index_name: str) -> set:
    """ KGS 인덱스에 존재하는 모든 문서의 ID(uuid)를 집합(set)으로 반환합니다. """
    
    if not client.collections.exists(index_name):
        return set()
    
    try:
        collection = client.collections.get(index_name)
        id_set = set()
        
        # return_properties=[]로 설정하여 UUID만 가져옵니다.
        for item in collection.iterator(
            include_vector=False,
            return_properties=[] 
        ):
            id_set.add(str(item.uuid))
            
        print(f"✅ 기존 문서 {len(id_set)}개의 ID를 스캔 완료")
        return id_set
    
    except Exception as e:
        print(f"🚨 get_existing_doc_ids (Sheet/Weaviate) 함수 실행 중 오류: {e}")
        traceback.print_exc()
        return set()

def generate_sheet_id_weaviate(id_string: str) -> str:
    """ Sheet용 UUIDv5 ID를 생성합니다. """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, id_string))
