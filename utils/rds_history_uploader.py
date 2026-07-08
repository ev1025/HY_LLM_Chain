from sqlalchemy import Table, select
from sqlalchemy.ext.asyncio import AsyncEngine
from sqlalchemy.dialects.postgresql import insert

from typing import List, Union
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


async def get_history_from_db(db_engine: AsyncEngine, chat_history_table :str,  session_id: str) -> List[BaseMessage]:
    """
    세션 ID를 기반으로 DB에서 대화 기록을 가져와 LangChain 메시지 형식으로 변환합니다.
    """
    async with db_engine.connect() as conn:
        stmt = select(chat_history_table).where(
            chat_history_table.c.session_id == session_id
        ).order_by(chat_history_table.c.created_at)
        
        result = await conn.execute(stmt)
        history_rows = result.fetchall()

    # DB에서 가져온 데이터를 HumanMessage/AIMessage 객체로 변환
    messages = []
    for row in history_rows:
        if row.role == "human":
            messages.append(HumanMessage(content=row.content))
        elif row.role == "ai":
            messages.append(AIMessage(content=row.content))
    return messages


async def add_message_to_db(db_engine: AsyncEngine,chat_history_table :str, session_id: str, role: str, content: str):
    """
    새로운 대화 메시지를 DB에 저장합니다.
    """
    async with db_engine.begin() as conn:
        stmt = chat_history_table.insert().values(
            session_id=session_id,
            role=role,
            content=content
        )
        await conn.execute(stmt)

async def get_summary_from_db(db_engine: AsyncEngine, summary_table: Table, session_id: str) -> Union[str, None]:
    """
    세션 ID를 기반으로 DB에서 요약 내용을 가져옵니다.
    """
    async with db_engine.connect() as conn:
        stmt = select(summary_table.c.content).where(
            summary_table.c.session_id == session_id
        )
        result = await conn.execute(stmt)
        summary_row = result.scalar_one_or_none()
    return summary_row


async def upsert_summary_to_db(db_engine: AsyncEngine, summary_table: Table, session_id: str, content: str):
    """
    새로운 요약 내용을 DB에 저장하거나 이미 존재하면 업데이트합니다. (Upsert)
    """
    async with db_engine.begin() as conn:
        # PostgreSQL의 ON CONFLICT DO UPDATE 기능을 사용한 Upsert 구문
        stmt = insert(summary_table).values(
            session_id=session_id,
            content=content
        )
        upsert_stmt = stmt.on_conflict_do_update(
            index_elements=['session_id'], # 충돌을 감지할 컬럼 (기본 키)
            set_=dict(content=content)     # 충돌 시 업데이트할 내용
        )
        await conn.execute(upsert_stmt)