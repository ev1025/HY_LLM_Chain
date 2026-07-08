import yaml, asyncio, os
from kiwipiepy import Kiwi

from .rag_prep_pipeline import run_rag_pipeline
from .rag_prep_pdf import prepare_pdf_data
from .rag_prep_sheet import prepare_sheet_data
from . import rag_utils_es

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
    
        db_strategy = {}

        # 업데이트 확인용 필드
        KGS_DELETE_FIELD = "id"
        SUSI_DELETE_FIELD = "metadata.drive_file_id"

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

            # =======================
            # VectorDB 파이프라인 실행
            # =======================
        if client:
            susi_schema_body = rag_utils_es.create_es_schema(metadata=susi_schema, k1=k1, b=b)
            kgs_schema_body = rag_utils_es.create_es_schema(metadata=kgs_schema, k1=k1, b=b)
                    
            print("="*8, f"[Susi (PDF)] 파이프라인 업로드 시작", "="*8)

            run_rag_pipeline(
                client=client,
                index_name=SUSI_INDEX,
                schema_definition=susi_schema_body, # 생성된 본문 전달
                s3_bucket=S3_BUCKET_NAME,
                s3_file_path=S3_SUSI_FILE_PATH,
                prepare_function=prepare_susi, # 래핑된 함수
                source_name=f"Susi (PDF)",
                delete_id_field=SUSI_DELETE_FIELD,
                is_async=True,
                db_strategy=db_strategy
            )

            print("\n", "="*8, f"[KGS (Google Sheet)] 파이프라인 실행 시작", "="*8)
                
            run_rag_pipeline(
                client=client,
                index_name=KGS_INDEX,
                schema_definition=kgs_schema_body, # 생성된 본문 전달
                s3_bucket=S3_BUCKET_NAME,
                s3_file_path=S3_KGS_FILE_PATH,
                prepare_function=prepare_kgs, # 래핑된 함수
                source_name=f"KGS (Google Sheet)",
                delete_id_field=KGS_DELETE_FIELD,
                is_async=False,
                db_strategy=db_strategy
                )


        if hasattr(client, 'close'):
            client.close()
            print(f"🚨 클라이언트 연결을 종료했습니다.\n\n")