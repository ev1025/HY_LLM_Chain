from dotenv import load_dotenv
load_dotenv()

import asyncio, yaml, uvicorn, json
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# 요약 
from utils.rds_history_uploader import add_message_to_db, get_summary_from_db
from utils.rds_db_setup import get_or_create_async_db_engine
from utils.rag_summary import (
    _refresh_summary_bg,
    _should_update_summary,
    summary_meta,
    summary_metadata,
    summary_store,
    summary_table,
)

# 히스토리
from utils.rds_schema_builder import qna_history_schema
from utils.rag_server_utils import (
    ensure_session_seeded_from_db,
    get_session_history,
    load_session_history_from_db,
    store,
)

# 모델
from rag_model import initialize_rag_pipeline

qna_metadata, qna_table_schema = qna_history_schema()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 앱 생명주기 훅. 서버 시작/종료 시 리소스를 준비/정리
    """
    print("🚀 API 서버 시작... RAG 체인 및 DB 연결을 초기화합니다...")
    with open("config/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    gpt_models = config["openai"]["models"]
    active_backend = config.get("active_backend", "weaviate")

    # 요약용/경량 LLM
    llm_summary = ChatOpenAI(
        model=gpt_models["kgs-summary"], temperature=0, max_retries=2
    )
    llm_answer = ChatOpenAI(
        model=gpt_models["kgs-answer"],
        temperature=0,
        streaming=True, 
        max_retries=2,
    )

    # RAG 기본 체인 (config에 따라 선택적으로 로드)
    base_rag_chain = initialize_rag_pipeline(
        config=config, 
        llm_answer=llm_answer,
        vdb_name=active_backend
    )

    # 히스토리 래퍼 적용
    chain_with_history = RunnableWithMessageHistory(
        base_rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    app.state.llm_summary = llm_summary
    app.state.chain = chain_with_history

    # DB 연결/테이블 준비
    app.state.db_engine = await get_or_create_async_db_engine("hy_rag_db")
    async with app.state.db_engine.begin() as conn:
        await conn.run_sync(qna_metadata.create_all)
        await conn.run_sync(summary_metadata.create_all)

    # 기존 세션 미리 캐시
    async with AsyncSession(app.state.db_engine) as session:
        result = await session.execute(
            text("SELECT DISTINCT session_id FROM qna_chat_history")
        )
        rows = result.fetchall()
        print(f"총 {len(rows)}건의 사용자 정보를 불러옵니다.")

    for (sid,) in rows:
        store[sid] = await load_session_history_from_db(
            app.state.db_engine, sid
        )
        s = await get_summary_from_db(app.state.db_engine, summary_table, sid)
        if s:
            summary_store[sid] = s
            summary_meta[sid] = len(store[sid].messages)

    print(
        f"💾 DB에서 총 {len(store)}개의 세션 및 {len(summary_store)}개의 요약 로드 완료."
    )

    print(f"✅ [{active_backend}] RAG 체인 + DB 연결 + 캐시 초기화 완료.")
    try:
        yield
    finally:
        await app.state.db_engine.dispose()
        print("👋 서버를 종료합니다.")


# =========================
# FastAPI app & CORS
# =========================
app = FastAPI(title="KGS Weaviate RAG 서버", lifespan=lifespan)
origins = [
    "https://hy-ai-homepage.vercel.app",
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8080",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# IO 모델
# =========================
class ChatRequest(BaseModel):
    session_id: str
    query: str

class ChatResponse(BaseModel):
    answer: str

# =========================
# 공통 의존성 함수
# =========================
async def get_common_deps(request: ChatRequest, fastapi_request: Request):
    """
    엔드포인트에서 중복되는 로직을 처리하는 공통 의존성 함수
    """
    # --- 1. 기본 변수 설정 ---
    chain = fastapi_request.app.state.chain
    db_engine = fastapi_request.app.state.db_engine
    llm_summary = fastapi_request.app.state.llm_summary
    session_id = request.session_id

    # --- 2. 세션 데이터 로드 (History & Summary) ---
    await ensure_session_seeded_from_db(db_engine, session_id)

    if session_id not in summary_store:
        s = await get_summary_from_db(db_engine, summary_table, session_id)
        if s:
            summary_store[session_id] = s
            summary_meta[session_id] = len(
                get_session_history(session_id).messages
            )
            
    cached_summary = summary_store.get(session_id, "이전 대화 없음.")
    
    # 필요한 모든 변수를 딕셔너리로 반환
    return {
        "chain": chain,
        "db_engine": db_engine,
        "llm_summary": llm_summary,
        "session_id": session_id,
        "query": request.query,
        "cached_summary": cached_summary,
    }

# =========================
# /chat (비-스트리밍 엔드포인트)
# =========================
@app.post("/chat")
async def chat_with_rag(
    deps: dict = Depends(get_common_deps)
) -> ChatResponse:
    """ 
    stream이 아닌 전체 응답을 받는 엔드포인트입니다.
    """
    chain = deps["chain"]
    db_engine = deps["db_engine"]
    llm_summary = deps["llm_summary"]
    session_id = deps["session_id"]
    query = deps["query"]
    cached_summary = deps["cached_summary"]

    # --- RAG 체인 실행 ---
    full_answer = await chain.ainvoke(
        {"input": query, "summary": cached_summary},
        config={"configurable": {"session_id": session_id}},
    )

    # --- 대화 기록 저장 및 요약 (순차 실행) ---
    try:
        await add_message_to_db(
            db_engine, qna_table_schema, session_id, "human", query
        )
        await add_message_to_db(
            db_engine, qna_table_schema, session_id, "ai", full_answer
        )
    except Exception as e:
        print(f"[{session_id}] DB 저장 중 오류 발생: {e}")

    memory_after = get_session_history(session_id)
    if _should_update_summary(session_id, memory_after.messages):
        asyncio.create_task(
            _refresh_summary_bg(
                session_id, memory_after.messages, llm_summary, db_engine
            )
        )

    return ChatResponse(answer=full_answer)


# =========================
# /chat-stream (스트리밍 엔드포인트)
# =========================
@app.post("/chat-stream")
async def chat_with_rag_stream(
    deps: dict = Depends(get_common_deps) 
) -> StreamingResponse:
    """ 
    stream 응답을 받는 엔드포인트입니다.
    """
    chain = deps["chain"]
    db_engine = deps["db_engine"]
    llm_summary = deps["llm_summary"]
    session_id = deps["session_id"]
    query = deps["query"]
    cached_summary = deps["cached_summary"]

    # --- 스트리밍 생성기 ---
    async def stream_generator() -> AsyncGenerator[str, None]:
        """
        스트리밍을 실행하고, 종료된 후 DB 저장 및 요약을 처리합니다.
        (외부 스코프의 chain, db_engine, session_id 등을 직접 사용)
        """
        full_answer = ""
        
        try:
            # RAG 체인 스트리밍 실행 (.astream 사용)
            async for chunk in chain.astream(
                {"input": query, "summary": cached_summary},
                config={"configurable": {"session_id": session_id}},
            ):
                full_answer += chunk
                # SSE (Server-Sent Events) 형식으로 데이터를 yield
                # yield f"data: {chunk}\n\n"
                yield json.dumps({"token": chunk}, ensure_ascii=False) + "\n"
            
            # 스트리밍이 정상적으로 끝났음을 알리는 신호
            yield "data: [DONE]\n\n"
            yield json.dumps({"status": "done"}, ensure_ascii=False) + "\n"

        except Exception as e:
            print(f"[{session_id}] 스트리밍 중 오류: {e}")
            # yield f"data: [ERROR] 스트리밍 중 오류가 발생했습니다: {e}\n\n"
            error_data = {"error": f"스트리밍 중 오류가 발생했습니다: {e}"}
            yield json.dumps(error_data, ensure_ascii=False) + "\n"
            return # 오류 발생 시 DB 저장 로직을 실행하지 않음

        # --- 스트리밍 종료 후, DB 저장 및 요약 (비동기) ---
        try:
            await add_message_to_db(
                db_engine, qna_table_schema, session_id, "human", query
            )
            await add_message_to_db(
                db_engine, qna_table_schema, session_id, "ai", full_answer
            )
        except Exception as e:
            print(f"[{session_id}] DB 저장 중 오류 발생: {e}")

        # 요약 태스크 생성
        memory_after = get_session_history(session_id)
        if _should_update_summary(session_id, memory_after.messages):
            asyncio.create_task(
                _refresh_summary_bg(
                    session_id, memory_after.messages, llm_summary, db_engine
                )
            )
            
    # return StreamingResponse(stream_generator(), media_type="text/event-stream")
    return StreamingResponse(stream_generator(), media_type="application/x-ndjson")

if __name__ == "__main__":
    uvicorn.run("wv_server:app", host="0.0.0.0", port=8000, reload=True)