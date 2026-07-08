from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn
import asyncio

from qna_model import initialize_rag_chain, get_rag_response
from utils.rds_history_uploader import get_history_from_db, add_message_to_db
from utils.rds_db_setup import get_or_create_async_db_engine
from utils.rds_schema_builder import qna_history_schema
from utils.rds_vdb_qna import all_pipeline

qna_metadata, qna_table_schema = qna_history_schema()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 API 서버 시작... RAG 체인 및 DB 연결을 초기화합니다...")
    await initialize_rag_chain()

    app.state.db_engine = await get_or_create_async_db_engine("hy_rag_db")
    async with app.state.db_engine.begin() as conn:
        await conn.run_sync(qna_metadata.create_all)

    # ✅ 벡터DB 적재 파이프라인을 백그라운드 태스크로 1회 실행
    # (서버 부팅 지연 방지, 로그는 rds_vdb_qna.py 쪽에서 그대로 출력)
    app.state.vdb_task = asyncio.create_task(all_pipeline())

    print("✅ RAG 체인 및 DB 연결/테이블 초기화 완료. 서버가 준비되었습니다.")
    try:
        yield
    finally:
        # 종료 시 백그라운드 태스크가 남아있으면 안전 종료
        task = getattr(app.state, "vdb_task", None)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        await app.state.db_engine.dispose()
        print("👋 서버를 종료합니다.")

# --- FastAPI 앱 초기화 ---
app = FastAPI(title="RAG Chatbot API", lifespan=lifespan)


origins = [
    "https://hy-ai-homepage.vercel.app/",
    "http://localhost",
    "http://localhost:3000", # React 개발 서버
    "http://localhost:8080", # Vue 개발 서버
    # 나중에 프론트엔드 배포 주소도 여기에 추가해야 합니다.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic 모델 정의 (API 입출력 형식) ---
class ChatRequest(BaseModel):
    session_id: str
    query: str

class ChatResponse(BaseModel):
    answer: str

# --- API 엔드포인트 정의 ---
@app.post("/chat", response_model=ChatResponse)
async def chat_with_rag(request: ChatRequest, fastapi_request: Request):
    """
    ## 입시 RAG 챗봇과 대화하고 답변을 받습니다.

    이 엔드포인트는 사용자의 질문을 받아 RAG 시스템을 통해 답변을 생성하고,\n
    멀티턴 대화를 위해 대화 내용을 DB에 기록합니다.

    ---

    ### **프론트엔드 개발자 안내 사항:**

    * **`session_id` (필수):**
        * **역할**: 멀티턴 대화를 위해 각 사용자를 식별하는 **고유한 ID**입니다.
        * **구현 방법**:
            1.  사용자가 웹사이트에서 챗봇 세션을 처음 시작할 때, 프론트엔드에서 이 ID를 **한 번만 생성**합니다. (예: UUID 라이브러리 사용)
            2.  생성된 `session_id`는 사용자의 대화가 끝날 때까지 브라우저의 **`localStorage`나 쿠키에 저장**해주세요.
            3.  사용자가 새로운 질문을 할 때마다, 저장된 `session_id`를 모든 `/chat` 요청에 **항상 동일하게** 담아서 보내주셔야 대화의 맥락이 유지됩니다.

    * **`query` (필수):**
        * 사용자가 채팅창에 입력한 현재 질문 메시지(문자열)입니다.

    ---

    * **응답 (`answer`):**
        * AI 챗봇이 생성한 답변 메시지(문자열)를 반환합니다.
    """
    # 매번 DB 엔진을 생성하는 대신, 앱 상태에 저장된 공유 엔진을 사용합니다.
    db_engine = fastapi_request.app.state.db_engine
    
    # DB에서 기록을 가져옴
    history = await get_history_from_db(db_engine, qna_table_schema, request.session_id)
    
    # rag_chain.py의 함수를 호출하여 답변 생성
    answer = await get_rag_response(request.query, history)
    
    # 대화 내용을 DB에 저장
    await add_message_to_db(db_engine, qna_table_schema, request.session_id, "human", request.query)
    await add_message_to_db(db_engine, qna_table_schema, request.session_id, "ai", answer)
    
    return ChatResponse(answer=answer)

# --- 서버 직접 실행을 위한 부분 ---
if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)

