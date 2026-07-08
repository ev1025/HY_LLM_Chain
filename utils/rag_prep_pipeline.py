import asyncio
from typing import Callable, Dict, Any

from .s3_io import save_jsonl_to_s3, chunk_to_jsonl

def run_rag_pipeline(
    client: Any,                    # Weaviate 또는 OpenSearch 클라이언트
    index_name: str,
    schema_definition: Any,         # Weaviate의 'properties' 또는 OpenSearch의 'metadata'
    s3_bucket: str,
    s3_file_path: str,
    prepare_function: Callable,     # 예: rag_prep_pdf.prepare_pdf_data
    source_name: str,
    delete_id_field: str,           # 삭제 기준 필드 (예: "drive_file_id" 또는 "id")
    is_async: bool,
    db_strategy: Dict[str, Callable] # DB별 구현 함수(전략) 묶음
):
    """
    RAG 데이터 동기화 파이프라인 범용 함수.
    (스키마 생성 -> 데이터 준비 -> S3/DB 업로드 -> 이전 데이터 삭제)
    
    db_strategy 딕셔너리는 다음 키를 포함해야 합니다:
    - "create_schema": (client, index_name, schema_definition) -> None
    - "uploader": (client, index_name, s3_uri) -> None
    - "delete_docs": (client, index_name, delete_ids, id_field_path) -> int
    """
    
    # 1단계: DB별 스키마 생성
    try:
        # db_strategy에서 "create_schema" 함수를 가져와 실행
        db_strategy["create_schema"](client, index_name, schema_definition)
    except Exception as e:
        print(f"🚨 스키마 생성 중 오류 발생: {e}")
        return # 스키마 생성 실패 시 중단

    # 2단계: 데이터 준비 (is_async 플래그에 따라 다르게 실행)
    if is_async:
        docs_to_upsert, delete_ids = asyncio.run(prepare_function(client, index_name))
    else:
        docs_to_upsert, delete_ids = prepare_function(client, index_name)

    # 3단계: S3 및 DB 적재
    if docs_to_upsert:
        serializable_docs = chunk_to_jsonl(docs_to_upsert)
        # S3 저장 시 DB="RAG"로 공통화 (필요시 'db' 인자 수정)
        s3_uri = save_jsonl_to_s3(db="RAG", rows=serializable_docs, bucket=s3_bucket, s3_file_path=s3_file_path)
        print(f"✅ S3 업로드 완료: {s3_uri}")
        
        try:
            # db_strategy에서 "uploader" 함수를 가져와 실행
            db_strategy["uploader"](client, index_name, s3_uri)
            print(f"✅ Target DB 업로드 완료: {index_name}")
        except Exception as e:
            print(f"🚨 DB 업로더 실행 중 오류 발생: {e}")
            
            
    # 4단계: DB에서 이전 문서 제거
    if delete_ids:
        try:
            # db_strategy에서 "delete_docs" 함수를 가져와 실행
            db_strategy["delete_docs"](
                client=client,
                index_name=index_name,
                ids_to_delete=delete_ids,
                id_field_path=delete_id_field  # 인자 이름 통일
            )
        except Exception as e:
            print(f"🚨 DB 삭제 실행 중 오류 발생: {e}")

    print(f"\n🎉 {source_name}의 동기화 작업이 완료되었습니다.")



