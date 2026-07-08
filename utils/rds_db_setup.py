import os
import asyncio 
import selectors 
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
ADMIN_DB = os.getenv("ADMIN_DB", "postgres")

async def get_or_create_async_db_engine(db_name: str, install_pgvector: bool = True) -> AsyncEngine:
    """
    DB를 확인/생성하고, SQLAlchemy AsyncEngine을 반환합니다.
    """
    # 관리자 DB('postgres')에 연결 - 생성 관리
    admin_engine_url = f"postgresql+asyncpg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{ADMIN_DB}"
    admin_engine = create_async_engine(admin_engine_url, isolation_level="AUTOCOMMIT")

    async with admin_engine.connect() as conn:
        result = await conn.execute(
            text("SELECT 1 FROM pg_database WHERE datname = :db_name"),
            {"db_name": db_name}
        )
        if not result.scalar_one_or_none():
            print(f"데이터베이스 '{db_name}'가 존재하지 않아 새로 생성합니다.")
            await conn.execute(text(f'CREATE DATABASE "{db_name}"'))
    
    # 관리자 엔진은 사용 후 연결을 종료합니다.
    await admin_engine.dispose()

    # 실제 사용할 DB('hy_rag_db')엔진 생성
    target_engine_url = f"postgresql+psycopg://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{db_name}"
    target_engine = create_async_engine(target_engine_url)

    # pgvector 확장 기능 활성화 (install_pgvector 값에 따라 조정) ---
    if install_pgvector:
        async with target_engine.connect() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.commit()
            print(f"'{db_name}' 데이터베이스에 'vector' 확장이 활성화되었습니다.")

    return target_engine



if __name__ == "__main__":
    if os.name == 'nt': # Windows 운영체제인 경우
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # asyncio.run()으로 비동기 테스트 함수를 실행합니다.
    asyncio.run(get_or_create_async_db_engine())

