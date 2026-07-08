from sqlalchemy import Table, Column, Integer, String, DateTime, MetaData, Text
from zoneinfo import ZoneInfo
from datetime import datetime
import time

'''

사용자의 질의응답 데이터를 SQL DB에 저장하기 위한 스키마를 정의합니다.

'''

timezone = ZoneInfo("Asia/Seoul")

def qna_history_schema(
        table_name="qna_chat_history",
        metadata_obj = MetaData(),
        timezone = timezone
        ):
    
    """
    고객 사용 데이터를 저장하는 테이블의 스키마를 정의하는 함수입니다.
    table_name : 실제 테이블에 사용할 이름
    metadata_obj : 실제 사용할 메타데이터 객체
    timezone : 타임존 설정 (UTM으로 저장됨) - 출력시 별도 작업 필요

    """
    chat_history_table = Table(
        table_name,
        metadata_obj,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("session_id", String, index=True, nullable=False),
        Column("role", String, nullable=False), # 'human' 또는 'ai'
        Column("content", String, nullable=False),
        Column("created_at", DateTime(timezone=True),
            default=lambda: datetime.fromtimestamp(time.time(), tz=timezone),
            nullable=False,
        ), 
    )
    return metadata_obj, chat_history_table

def qna_summary_schema(
        table_name="qna_summary",
        metadata_obj = MetaData(),
        timezone = timezone
        ):
    """
    세션별 대화 요약 데이터를 저장하는 테이블의 스키마를 정의합니다.
    """
    summary_table = Table(
        table_name,
        metadata_obj,
        # session_id를 기본 키로 사용하여 세션당 하나의 요약만 저장되도록 보장
        Column("session_id", String, primary_key=True),
        Column("content", Text, nullable=False),
        Column("updated_at", DateTime(timezone=True),
            default=lambda: datetime.fromtimestamp(time.time(), tz=timezone),
            onupdate=lambda: datetime.fromtimestamp(time.time(), tz=timezone), # 업데이트 시 자동 갱신
            nullable=False,
        ),
    )
    return metadata_obj, summary_table