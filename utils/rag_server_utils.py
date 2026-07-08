from typing import Dict, Literal, List
from langchain.schema import AIMessage, HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from utils.rag_filter_query import query_filter

# =========================
# History Management
# =========================

# 세션 캐시 (store)
store: Dict[str, InMemoryChatMessageHistory] = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    세션별 대화 히스토리 캐시를 가져오거나 생성합니다.
    """
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


async def load_session_history_from_db(
    engine, session_id: str
) -> InMemoryChatMessageHistory:
    """
    RDS에서 특정 세션의 과거 대화 로그를 읽어 히스토리 객체로 복원합니다.
    """
    history = InMemoryChatMessageHistory()
    async with AsyncSession(engine) as session:
        result = await session.execute(
            text(
                """
                SELECT role, content
                FROM qna_chat_history
                WHERE session_id = :sid
                ORDER BY created_at ASC
            """
            ),
            {"sid": session_id},
        )
        rows = result.fetchall()

    for row in rows:
        if row.role == "human":
            history.add_message(HumanMessage(content=row.content))
        elif row.role == "ai":
            history.add_message(AIMessage(content=row.content))
    return history


async def ensure_session_seeded_from_db(
    engine,
    session_id: str,
    mode: Literal["if_empty", "topup", "reload"] = "topup",
) -> None:
    """
    세션 캐시(store)를 DB 내용으로 보정한다.
    """
    db_hist: InMemoryChatMessageHistory = await load_session_history_from_db(
        engine, session_id
    )
    db_msgs = db_hist.messages

    cache_hist: InMemoryChatMessageHistory | None = store.get(session_id)

    if mode == "if_empty":
        if cache_hist is None or len(cache_hist.messages) == 0:
            store[session_id] = db_hist
        return

    if mode == "reload":
        store[session_id] = db_hist
        return

    # mode == "topup"
    if cache_hist is None:
        store[session_id] = db_hist
        return

    cache_len = len(cache_hist.messages)
    db_len = len(db_msgs)

    if db_len > cache_len:
        # 부족분만 뒤에서부터 보충
        missing = db_msgs[cache_len:]
        cache_hist.messages.extend(missing)


# =========================
# RAG Pipeline Helpers
# =========================
def log_and_pass_through(x, label=""):
    """
    CLI에서 문서 / 요약 / 최근 8건 대화를 확인하는 함수
    """
    print("\n" + "=" * 70)
    print(f"🔬 {label}")
    print("=" * 70)

    if isinstance(x, list) and all(isinstance(i, Document) for i in x):
        print(f"📄 검색된 문서 수: {len(x)}")
        for i, doc in enumerate(x[:5], start=1):
            score = doc.metadata.get("relevance_score", "N/A")
            content = (doc.page_content or "").replace("\n", " ").strip()
            preview = content[:300] + ("..." if len(content) > 300 else "")
            print(f"--- 문서 {i} (Score: {score}) ---")
            print(f"내용: {preview}")
        return x

    if isinstance(x, dict):
        summary = x.get("summary", "")
        hist = x.get("chat_history_8") or x.get("chat_history") or []

        print("\n🧾 [요약된 Chat History]")
        print((summary or "").strip() or "(요약 없음)")

        n = min(len(hist), 8)
        print(f"\n💬 최근 {n}개 대화:")
        for i, msg in enumerate(hist[-8:], start=max(1, len(hist) - 7)):
            role = getattr(msg, "type", "unknown").upper()
            content = getattr(msg, "content", "")
            print(
                f"  [{i:02d}] {role}: {content[:150]}{'...' if len(content)>150 else ''}"
            )
        return x
    print(x)
    return x


def prep_inputs(x: dict) -> dict:
    """
    요약 및 최근 8턴 대화 추출합니다.
    """
    chat_hist = x.get("chat_history") or []
    out = dict(x)  # 원본 훼손 방지
    out["summary"] = x.get("summary", "요약 없음")
    out["chat_history_8"] = chat_hist[-8:]
    out.pop("chat_history", None)  # 불필요 키 제거
    return out


async def run_preprocessing(x: dict) -> dict:
    """
    재작성된 user input과 추출된 메타데이터 리스트 반환합니다.
    """
    return await query_filter(
        query=x["original_input"], chat_history_8=x["chat_history_8"]
    )


def format_docs(docs: List[Document]) -> str:
    """
    LLM에게 제공되는 Documents 형태를 Markdown형식으로 변환합니다.
    """
    doc_strings = []
    for i, doc in enumerate(docs):
        doc_str = f"## 문서 {i+1}\n" f"{doc.page_content}"
        doc_strings.append(doc_str)
    return "\n\n---\n\n".join(doc_strings)


