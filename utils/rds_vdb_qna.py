import asyncio
import os
import pandas as pd
from pathlib import Path
from typing import List
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector

from .rds_db_setup import get_or_create_async_db_engine

def prep_csv(csv_path: str) -> List[Document]:
    """'
    질문정리.csv' 파일을 읽어 LangChain Document 리스트로 변환합니다.
    """
    try:
        df = pd.read_csv(csv_path)
        df.dropna(subset=["HY AI 답변"], inplace=True)

        langchain_docs = []
        for _, row in df.iterrows():
            content = f"{row['HY AI 답변']}"
            metadata = {
                "category1": row.get('분류1', ''), 
                "category2": row.get('분류2', ''),
                "category3": row.get('분류3', ''), 
            }
            langchain_docs.append(Document(page_content=content, metadata=metadata))

        print(f"전처리 완료: 총 {len(langchain_docs)}개의 문서를 준비했습니다.")
        return langchain_docs
    except Exception as e:
        print(f"❌ CSV 처리 중 오류 발생: {e}")
        return []


async def unique_filter(db_engine: AsyncEngine, collection_name: str, all_docs: List[Document]) -> List[Document]:
    """
    DB와 비교하여 새로운 문서만 필터링합니다.
    """
    try:
        async with db_engine.connect() as conn:
            # 컬렉션 아이디 추출
            result = await conn.execute(
                text("SELECT uuid FROM langchain_pg_collection WHERE name = :name"),
                {"name": collection_name}
            )
            collection_id = result.scalar_one_or_none()

            # 기존문서의 앞 100글자와 비교해서 중복확인
            existing_snippets = set()
            if collection_id:
                rows_result = await conn.execute(
                    text("SELECT LEFT(document, 100) FROM langchain_pg_embedding WHERE collection_id = :cid"),
                    {"cid": collection_id}
                )
                existing_snippets = {r[0] for r in rows_result.fetchall()}
            
            unique_docs = [doc for doc in all_docs if doc.page_content[:100] not in existing_snippets]
            print(f"중복 검사 완료: {len(unique_docs)}개의 새로운 문서를 찾았습니다.")
            return unique_docs
        
    except Exception as e:
        print(f"❌ 중복 검사 중 오류 발생: {e}")
        return []

async def upload_docs_to_vectorstore(
    db_engine: AsyncEngine, collection_name: str, docs_to_add: List[Document], embedding_model: OpenAIEmbeddings
):
    """
    필터링된 문서를 Vector Store에 업로드합니다.
    """
    if not docs_to_add:
        print("추가할 새로운 문서가 없습니다.")
        return

    print(f"Vector Store에 {len(docs_to_add)}개 문서 업로드")
    try:
        await PGVector.afrom_documents(
            documents=docs_to_add,
            embedding=embedding_model,
            collection_name=collection_name,
            connection=db_engine,
            pre_delete_collection=False
        )
        print(f"{len(docs_to_add)}개의 문서를 추가했습니다.")
    except Exception as e:
        print(f"❌ 업로드 중 오류 발생: {e}")


async def all_pipeline():
    """
    Q&A 데이터 처리 및 저장을 위한 전체 파이프라인을 실행합니다.
    """
    # 현재 스크립트 파일(prep_car.py)의 절대 경로를 가져옵니다.
    current_file_path = Path(__file__).resolve()
    
    # 루트 디렉토리 경로를 설정합니다. (현재 파일의 부모 폴더('utils')의 부모 폴더)
    ROOT_DIR = current_file_path.parent.parent
    base_dir = ROOT_DIR / 'data' / 'qna'      # 폴더 지정
    STATS_CSV_PATH = base_dir / "질문정리.csv" # 전처리 완료 통합 데이터
    DB_NAME = "hy_rag_db"

    QNA_COLLECTION_NAME = "qna_documents"

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    
    db_engine = await get_or_create_async_db_engine(DB_NAME)

    try:
        all_qna_docs = prep_csv(STATS_CSV_PATH)
        
        if all_qna_docs:
            unique_qna_docs = await unique_filter(db_engine, QNA_COLLECTION_NAME, all_qna_docs)
            await upload_docs_to_vectorstore(db_engine, QNA_COLLECTION_NAME, unique_qna_docs, embedding_model)
    finally:
        await db_engine.dispose()
        print("\n--- 모든 작업이 완료되었습니다. ---")


if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(all_pipeline())