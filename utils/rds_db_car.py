import asyncio
import os
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path

from sqlalchemy import text, MetaData, Table, Column, Integer, String, Float
from sqlalchemy.ext.asyncio import AsyncEngine

from utils.rds_db_setup import get_or_create_async_db_engine

"""
전처리된 입결 데이터를 RDS에 적재된 데이터와 비교하여
중복이 아닌 데이터만 RDS에 적재합니다.
"""


def get_car_documents_table(table_name = "car_documents")-> Table:
    """
    스키마를 지정하는 함수입니다.
    """
    metadata_obj = MetaData()
    admission_stats_table = Table(
        table_name,
        metadata_obj,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("세부", String, nullable=True), # 경우에 따라 없을 수 있으므로 nullable=True 유지
        Column("대학", String, nullable=False), 
        Column("전형", String, nullable=False), 
        Column("학과", String, nullable=False),
        Column("년도", Integer, nullable=False), 
        Column("모집인원", Integer),  
        Column("경쟁률", Float),
        Column("충원율", Float),
        Column("최저충족비율", Float),
        Column("충원+최저충족", Float),
        Column("50%-70% 컷차이", Float),
        Column("입결0.5", Float),
        Column("입결0.7", Float),
        Column("1-1차", String),
        Column("1-1비중", Float),
        Column("1-2차", String),
        Column("1-2비중", Float),
        Column("2-1차", String),
        Column("2-1비중", Float),
        Column("2-2차", String),
        Column("2-2비중", Float),
    )
    return metadata_obj, admission_stats_table

def prep_csv(csv_path: str) -> List[Dict]:
    """
    CSV 파일을 읽고 결측값을 숫자는 0, 글자는 None으로 변환
    """
    try:
        df = pd.read_csv(csv_path)

        # 문자열과 숫자 컬럼 목록을 명확하게 구분합니다.
        string_cols = ["세부", "대학", "전형", "학과", "1-1차", "1-2차", "2-1차", "2-2차"]
        numeric_cols = [
            '50%-70% 컷차이', '경쟁률', '모집인원', '입결0.5', '입결0.7', '최저충족비율',
            '충원+최저충족', '충원율', '1-1비중', '1-2비중', '2-1비중', '2-2비중', '년도'
        ]
        integer_cols = ['모집인원', '년도']

        # 숫자 컬럼들을 숫자로 변환 (변환 불가 시 NaN 처리)
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0)


        for col in string_cols:
            # 문자 컬럼의 빈 값(NaN 등)은 None으로 바꿔 DB에 NULL로 저장되게 합니다.
            # .astype(object)를 통해 다양한 타입의 빈 값을 None으로 일괄 처리합니다.
            df[col] = df[col].where(pd.notna(df[col]), None)

        # 정수여야 하는 컬럼의 타입을 int로 강제 변환합니다.
        for col in integer_cols:
            df[col] = df[col].fillna(0).astype(int)

        records = df.to_dict('records')
    
        print(f"전처리 완료: 총 {len(records)}개의 데이터를 준비했습니다.")
        return records

    except Exception as e:
        print(f"❌ CSV 처리 중 오류 발생: {e}")
        return []


async def unique_filter_load(db_engine: AsyncEngine, new_records: List[Dict]) -> List[Dict]:
    """
    DB의 기존 데이터를 제외하고 새로 추가할 데이터만 필터링합니다.
    """
    try:
        metadata_obj, admission_stats_table = get_car_documents_table()

        async with db_engine.begin() as conn:
            await conn.run_sync(metadata_obj.create_all)

        # 중복 데이터 확인을 위해 db에서 5개의 칼럼 추출
        check_columns = ["세부", "대학", "전형", "학과", "년도"]
        async with db_engine.connect() as conn:
            select_stmt = text(f"SELECT {', '.join(check_columns)} FROM car_documents")
            result = await conn.execute(select_stmt)
            
            # DB 값과 비교 시 None/NaN 문제를 피하기 위해 문자열로 변환하여 비교
            existing_records = {
                tuple(str(val) for val in row) for row in result.fetchall()
            }
            
  
        # 새로운 데이터 중, 기존에 없는 데이터만 필터링
        unique_records = []
        for rec in new_records:
            # 새로운 데이터도 동일하게 정규화
            key = tuple(
                str(rec.get(col)) for col in check_columns
            )
            if key not in existing_records:
                unique_records.append(rec)

        print(f"중복 검사 완료 : {len(unique_records)}개의 새로운 데이터를 확인했습니다.")

        if not unique_records:
            print("추가할 새로운 문서가 없습니다.")
            return
        print(f"✅ 중복 검사 완료: 총 {len(new_records)}개 중 {len(unique_records)}개의 새로운 데이터를 발견했습니다.")
        

        async with db_engine.begin() as conn:
            await conn.execute(admission_stats_table.insert(), unique_records)
        print(f"✅ 'car_documents' 테이블에 {len(unique_records)}개의 데이터를 추가했습니다.")
                
    except Exception as e:
        print(f"❌ 중복 검사 중 오류 발생: {e}")
        return []


async def all_pipeline():
    """
    해당 파일의 전체 파이프라인을 실행합니다.
    """
    # 현재 스크립트 파일(prep_car.py)의 절대 경로를 가져옵니다.
    current_file_path = Path(__file__).resolve()
    
    # 루트 디렉토리 경로를 설정합니다. (현재 파일의 부모 폴더('utils')의 부모 폴더)
    ROOT_DIR = current_file_path.parent.parent
 
    base_dir = ROOT_DIR / 'data'        # data폴더에 통합 데이터 만들기
    STATS_CSV_PATH = base_dir / "car_all_sheet.csv" # 전처리 완료 통합 데이터
    DB_NAME = "hy_rag_db"
    
    db_engine = await get_or_create_async_db_engine(DB_NAME, install_pgvector=False)

    try:
        # CSV 파일 로드 및 전처리
        all_records = prep_csv(STATS_CSV_PATH)
        
        # 중복값 제거 후 db에 적재
        if all_records:
            await unique_filter_load(db_engine, all_records)

    finally:
        await db_engine.dispose()
        print("\n--- 모든 작업이 완료되었습니다. ---")

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(all_pipeline())