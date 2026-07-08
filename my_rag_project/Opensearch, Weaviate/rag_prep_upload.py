import yaml, asyncio, os
from kiwipiepy import Kiwi

from .rag_prep_pipeline import run_rag_pipeline
from .rag_prep_pdf import prepare_pdf_data
from .rag_prep_sheet import prepare_sheet_data
from . import rag_utils_wv, rag_utils_open, rag_utils_es

with open('config/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

S3_BUCKET_NAME = config['aws']['s3']['bucket']
S3_SUSI_FILE_PATH = config['aws']['s3']['susi_file_path']     
S3_KGS_FILE_PATH = config['aws']['s3']['kgs_file_path'] 

kiwi_analyzer = Kiwi() 

def get_morph(chunk: str) -> str:
    """Kiwi 분석기로 형태소 문자열을 생성합니다."""
    tokens = kiwi_analyzer.tokenize(chunk)
    chunk_tokens_list = [token.form for token in tokens]
    
    return " ".join(chunk_tokens_list)

if __name__ == "__main__":
    if os.name == 'nt': # Windows 비동기 정책
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    for DB_TARGET in ['OPENSEARCH','WEAVIATE','ELASTICSEARCH', ]: # 업로드할 VectorDB 목록 
        db_strategy = {}

        # 업데이트 확인용 필드
        KGS_DELETE_FIELD = "id"
        SUSI_DELETE_FIELD = "drive_file_id" if DB_TARGET== "WEAVIATE" else "metadata.drive_file_id"

        try:
            # ===================== 
            # WEAVIATE 요소 정의
            # ===================== 
            if DB_TARGET == "WEAVIATE":
                client = rag_utils_wv.get_weaviate_client()
                
                # 1. DB 전략 매핑
                db_strategy = {
                    "create_schema": rag_utils_wv.create_weaviate_schema,
                    "uploader": rag_utils_wv.weaviate_uploader,
                    "delete_docs": rag_utils_wv.delete_weaviate_documents
                }
                
                # 2. 인덱스/스키마
                SUSI_INDEX = config['weaviate']['susi_index']
                KGS_INDEX = config['weaviate']['kgs_index']

                susi_schema  = [
                    {"name": "text",         "dataType": ["text"], "tokenization": "word", "indexSearchable": True,},
                    {"name": "text_morph",   "dataType": ["text"], "tokenization": "whitespace", "indexSearchable": True,}, 
                    {"name": "drive_file_id","dataType": ["text"], "tokenization": "field", "indexSearchable": False, "indexFilterable": True},
                    {"name": "university",   "dataType": ["text"], "tokenization": "field", "indexSearchable": False, "indexFilterable": True},
                    {"name": "year",         "dataType": ["int"]},
                    {"name": "admission_type","dataType":["text"], "tokenization": "field", "indexSearchable": False, "indexFilterable": True},
                    {"name": "document_type","dataType":["text"], "tokenization": "field", "indexSearchable": False, "indexFilterable": True},
                    {"name": "modified_time","dataType":["date"]},  
                    {
                    "name": "metadata",
                    "dataType": ["object"],
                    "nestedProperties": [
                        {"name": "source",       "dataType": ["text"], "tokenization": "field"},
                        {"name": "chunk_number", "dataType": ["int"]},
                        {"name": "start_index",  "dataType": ["int"]},
                        ]
                        },
                    ]   

                kgs_schema = [
                    # --- 원본 필드 ---
                    {"name": "question", "dataType": ["text"], "tokenization": "word", "indexSearchable": True,},
                    {"name": "text",   "dataType": ["text"], "tokenization": "word", "indexSearchable": True,},
                    
                    # --- 형태소 분석 필드 (Kiwi 분석 완료) ---
                    {"name": "question_morph", "dataType": ["text"], "tokenization": "whitespace", "indexSearchable": True,},
                    {"name": "text_morph",   "dataType": ["text"], "tokenization": "whitespace", "indexSearchable": True,}, 
                    
                    # --- 필터링용 키워드 필드 ---
                    {"name": "university", "dataType": ["text"], "tokenization": "field", "indexSearchable": False, "indexFilterable": True},
                    {"name": "year",       "dataType": ["int"]},
                    {"name": "category1",  "dataType": ["text"], "tokenization": "field", "indexSearchable": False, "indexFilterable": True},
                    {"name": "category2",  "dataType": ["text"], "tokenization": "field", "indexSearchable": False, "indexFilterable": True},
                    {"name": "category3",  "dataType": ["text"], "tokenization": "field", "indexSearchable": False, "indexFilterable": True},
                ]

                # 3. PDF(Susi) 데이터 준비 함수 래핑
                prepare_susi = lambda c, i: prepare_pdf_data(
                    get_ids_func=lambda: rag_utils_wv.get_indexed_ids_and_mtime_weaviate(c, i),
                    need_update_func=rag_utils_wv.need_update_weaviate,
                    generate_chunk_id_func=rag_utils_wv.generate_chunk_id_weaviate,
                    get_morph_func=get_morph
                )
                
                # 4. Sheet(KGS) 데이터 준비 함수 래핑
                prepare_kgs = lambda c, i: prepare_sheet_data(
                    get_existing_doc_ids_func=lambda: rag_utils_wv.get_existing_doc_ids_sheet_weaviate(c, i),
                    generate_sheet_id_func=rag_utils_wv.generate_sheet_id_weaviate,
                    get_morph_func=get_morph
                )
                
                
            # ===================== 
            # OPENSEARCH 요소 정의
            # ===================== 
            elif DB_TARGET == "OPENSEARCH":
                client = rag_utils_open.get_opensearch_client()

                SUSI_INDEX = config['aws']['opensearch']['susi_index']
                KGS_INDEX = config['aws']['opensearch']['kgs_index']
                k1 = config['aws']['opensearch']['kgs_index_k1']
                b = config['aws']['opensearch']['kgs_index_b']

                susi_schema = {
                    "source": {"type": "keyword"},
                    "university": {"type": "keyword"},
                    "year": {"type": "integer"},
                    "admission_type": {"type": "keyword"},
                    "document_type": {"type": "keyword"},
                    "chunk_number": {"type": "integer"},
                    "drive_file_id": {"type": "keyword"},
                    "start_index": {"type": "integer"},
                    "modified_time": {"type": "keyword"}
                }
                
                kgs_schema = {
                    "question": {"type": "text", "analyzer": "standard"}, 
                    "question_morph": {"type": "text", "analyzer": "korean_nori"}, 
                    "text_morph": {"type": "text", "analyzer": "korean_nori"},
                    "university": {"type": "keyword"},
                    "year": {"type": "integer"},
                    "category1": {"type": "keyword"},
                    "category2": {"type": "keyword"},
                    "category3": {"type": "keyword"},
                }
                # DB 전략 매핑
                db_strategy = {
                    "create_schema": rag_utils_open.build_opensearch_index,
                    "uploader": rag_utils_open.opensearch_uploader,
                    "delete_docs": rag_utils_open.delete_opensearch_documents
                }
        


                # PDF(Susi) 데이터 준비 함수 래핑
                prepare_susi = lambda c, i: prepare_pdf_data(
                    get_ids_func=lambda: rag_utils_open.get_indexed_ids_and_mtime_opensearch(c, i),
                    need_update_func=rag_utils_open.need_update_opensearch,
                    generate_chunk_id_func=rag_utils_open.generate_chunk_id_opensearch,
                    get_morph_func=None # OpenSearch는 Nori 분석기 사용
                )
                
                # Sheet(KGS) 데이터 준비 함수 래핑
                prepare_kgs = lambda c, i: prepare_sheet_data(
                    get_existing_doc_ids_func=lambda: rag_utils_open.get_existing_doc_ids_sheet_opensearch(c, i),
                    generate_sheet_id_func=rag_utils_open.generate_sheet_id_opensearch,
                    get_morph_func=None # OpenSearch는 Nori 분석기 사용
                )
                
            
            #  ===================== 
            # ELASTICSEARCH 요소 정의
            # ===================== 
            elif DB_TARGET == "ELASTICSEARCH":
                client = rag_utils_es.get_es_client()

                SUSI_INDEX = config['elasticsearch']['susi_index']
                KGS_INDEX = config['elasticsearch']['kgs_index']
                k1 = config['elasticsearch']['kgs_index_k1']
                b = config['elasticsearch']['kgs_index_b']
                
                susi_schema = {
                    "source": {"type": "keyword"},
                    "university": {"type": "keyword"},
                    "year": {"type": "integer"},
                    "admission_type": {"type": "keyword"},
                    "document_type": {"type": "keyword"},
                    "chunk_number": {"type": "integer"},
                    "drive_file_id": {"type": "keyword"},
                    "start_index": {"type": "integer"},
                    "modified_time": {"type": "keyword"}
                }

                kgs_schema = {
                    "question": {"type": "text", "analyzer": "standard"}, 
                    "question_morph": {"type": "text", "analyzer": "kiwi_ws"}, 
                    "text_morph": {"type": "text", "analyzer": "kiwi_ws"},
                    "university": {"type": "keyword"},
                    "year": {"type": "integer"},
                    "category1": {"type": "keyword"},
                    "category2": {"type": "keyword"},
                    "category3": {"type": "keyword"},
                }    

                # DB 전략 매핑
                db_strategy = {
                    "create_schema": rag_utils_es.build_elasticsearch_index,
                    "uploader": rag_utils_es.elasticsearch_uploader,
                    "delete_docs": rag_utils_es.delete_elastic_documents
                }   

                # PDF(Susi) 데이터 준비 함수 래핑
                prepare_susi = lambda c, i: prepare_pdf_data(
                    get_ids_func=lambda: rag_utils_es.get_indexed_ids_and_mtime_es(c, i),
                    need_update_func=rag_utils_es.need_update_es,
                    generate_chunk_id_func=rag_utils_es.generate_chunk_id_es,
                    get_morph_func=get_morph # <- Kiwi 함수 전달
                )
                
                # Sheet(KGS) 데이터 준비 함수 래핑
                prepare_kgs = lambda c, i: prepare_sheet_data(
                    get_existing_doc_ids_func=lambda: rag_utils_es.get_existing_doc_ids_sheet_es(c, i),
                    generate_sheet_id_func=rag_utils_es.generate_sheet_id_es,
                    get_morph_func=get_morph # <- Kiwi 함수 전달
                )

            else:
                print(f"🚨 에러: DB_TARGET('{DB_TARGET}')을(를) 인식할 수 없습니다.")
            

            # =======================
            # VectorDB 파이프라인 실행
            # =======================
            if client:
                try:
                    if DB_TARGET == "WEAVIATE":
                        susi_schema_body = susi_schema
                        kgs_schema_body = kgs_schema
                    elif DB_TARGET == "OPENSEARCH":
                        susi_schema_body = rag_utils_open.create_open_schema(metadata=susi_schema,
                                                                                k1=k1,
                                                                                b=b)
                        kgs_schema_body = rag_utils_open.create_open_schema(metadata=kgs_schema,
                                                                                k1=k1,
                                                                                b=b)
                    else: # ElasticSearch
                        susi_schema_body = rag_utils_es.create_es_schema(metadata=susi_schema,
                                                                                k1=k1,
                                                                                b=b)
                        kgs_schema_body = rag_utils_es.create_es_schema(metadata=kgs_schema,
                                                                                k1=k1,
                                                                                b=b)
                        
                    print("="*8, f"[{DB_TARGET}] Susi (PDF) 파이프라인 실행 시작", "="*8)

                    run_rag_pipeline(
                        client=client,
                        index_name=SUSI_INDEX,
                        schema_definition=susi_schema_body, # 생성된 본문 전달
                        s3_bucket=S3_BUCKET_NAME,
                        s3_file_path=S3_SUSI_FILE_PATH,
                        prepare_function=prepare_susi, # 래핑된 함수
                        source_name=f"{DB_TARGET} Susi (PDF)",
                        delete_id_field=SUSI_DELETE_FIELD,
                        is_async=True,
                        db_strategy=db_strategy
                    )

                    print("\n", "="*8, f"[{DB_TARGET}] KGS Google Sheet 파이프라인 실행 시작", "="*8)
                        
                    run_rag_pipeline(
                        client=client,
                        index_name=KGS_INDEX,
                        schema_definition=kgs_schema_body, # 생성된 본문 전달
                        s3_bucket=S3_BUCKET_NAME,
                        s3_file_path=S3_KGS_FILE_PATH,
                        prepare_function=prepare_kgs, # 래핑된 함수
                        source_name=f"{DB_TARGET} KGS (Sheet)",
                        delete_id_field=KGS_DELETE_FIELD,
                        is_async=False,
                        db_strategy=db_strategy
                    )

                except Exception as e:
                    print(f"🚨 {DB_TARGET} 파이프라인 생성 오류: {e}")
                finally:
                    if hasattr(client, 'close'):
                        client.close()
                        print(f"🚨 {DB_TARGET} 클라이언트 연결을 종료했습니다.\n\n")
            else:
                print(f"🚨 {DB_TARGET}는 존재하지 않는 DB입니다.")

        except Exception as outer_e:
            print(f'🚨 {DB_TARGET} 업로드 중 치명적 오류 발생: {outer_e}')
            continue