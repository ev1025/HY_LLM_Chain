import asyncio, textwrap
from collections import defaultdict
from typing import Dict, List, Any # List 임포트 수정
import tiktoken

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import BaseMessage
from sqlalchemy.ext.asyncio import AsyncEngine

from utils.rds_history_uploader import upsert_summary_to_db
from utils.rds_schema_builder import qna_summary_schema

# =====================================
# 요약 관련 상태 변수 (전역)
# =====================================
summary_lock = defaultdict(asyncio.Lock)
store: Dict[str, InMemoryChatMessageHistory] = {}     # 히스토리 세션 초기화
summary_store: Dict[str, str] = {}                    # 요약 세션 초기화
summary_meta: Dict[str, int] = {}   # 마지막 요약에 반영된 메시지 개수

# =====================================
# 요약 관련 상수 및 토크나이저
# =====================================
enc = tiktoken.get_encoding("cl100k_base")

MAX_CONTEXT_TOKENS = 128_000      # gpt-4o-mini 최대 입력 토큰 수
SUMMARY_INPUT_FRACTION = 0.80     # 최대 입력 토큰 수의 80% 
SUMMARY_TOKEN_LIMIT = int(MAX_CONTEXT_TOKENS * SUMMARY_INPUT_FRACTION)  # 4o-mini 최대 입력 토큰 수의 80%
NOT_SUMMARIZED_LIMIT = 2000 # 8건 이후 요약 토큰 제한

# =====================================
# 요약용 DB 스키마 (서버 대신 여기서 관리)
# =====================================
summary_metadata, summary_table = qna_summary_schema() # RDS 요약 객체

# =====================================
# 요약 헬퍼 함수 (토큰화, 패킹)
# =====================================
def _tok(text: str) -> int:
    return len(enc.encode(text or ""))

def _pack_old_messages_by_tokens(old_msgs: List[BaseMessage], max_tokens: int) -> str:
    buf, toks = [], 0
    for m in old_msgs:
        line = f"{m.type.upper()}: {m.content}\n"
        need = _tok(line)
        if toks + need > max_tokens:
            break
        buf.append(line)
        toks += need
    return "".join(buf)

# =====================================
# 요약 프롬프트
# =====================================
initial_prompt = textwrap.dedent("""\
    당신은 대한민국의 대학교 입시 상담 요약가입니다. 아래 <대화기록>만을 근거로,
    '사용자 프로필'을 200토큰 이내로 작성하세요.

    출력은 아래 6줄 형식을 엄격히 지키세요(각 항목 한 문장, 불명확하면 '미상'):
    1) 관심/목표: [관심 대학·캠퍼스·전형/비교대상·희망전공]
    2) 자격 스냅샷: [해외이수연수·고교1년포함 여부·국적/복수국적·성적/시험(IB/SAT/어학) 보유/예정]
    3) 제약/리스크: [지원횟수·마감일·요건미충족 가능성·자료미비]
    4) 현재 질문/미해결: [사용자가 답을 원한 핵심 쟁점·확인 필요 항목]
    5) 결정 상태: [exploring | narrowing | deciding]
    6) 다음 행동: [다음에 할 일 1~3개(학교/전형/서류/일정 등)]

    규칙:
    - 고유명사(대학명/전형/학년도)는 원형 유지.
    - 대화에 없는 정보는 추측 금지, '미상' 표기.
    - ‘지식 설명/가이드’는 금지하고 상담 정보만 요약.
""")

update_prompt = textwrap.dedent("""\
    당신은 대한민국의 대학교 입시 상담 요약가입니다.
    <기존요약>과 <추가된 대화>를 통합해 최신 '사용자 프로필'로 갱신하세요(200토큰 이내).

    출력 형식(6줄, 각 항목 한 문장):
    1) 관심/목표: ...
    2) 자격 스냅샷: ...
    3) 제약/리스크: ...
    4) 현재 질문/미해결: ...
    5) 결정 상태: [exploring | narrowing | deciding]
    6) 다음 행동: ...

    통합 규칙:
    - <기존요약>의 핵심은 유지하되, <추가된 대화>의 ‘변경/추가’를 반영해 재작성.
    - 모호한 정보는 '미상', 추측 금지. 고유명사 원형 유지.
    - 중복 제거, 간결·명확하게. 모델의 조언/해설은 포함하지 말 것(상담 사실만).
""")

INITIAL_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(initial_prompt),
    HumanMessagePromptTemplate.from_template("<대화 기록>\n{chat_history_text}</대화기록>\n\n<요약>")
])
UPDATE_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(update_prompt),
    HumanMessagePromptTemplate.from_template(
        "<기존 요약>\n{existing_summary}</기존 요약>\n\n<추가된 대화>\n{chat_history_text}</추가된 대화>\n\n<최신 요약>")
])

# =====================================
# 요약 핵심 로직
# =====================================

async def build_summary_text(
    session_id: str,
    messages: List[BaseMessage], # type: list -> List
    llm_fast
) -> str:
    """
    세션 대화를 점진적으로 요약하고 캐시합니다. (v3: 리팩토링 적용)
    (이하 주석 동일)
    """
    # history에 대화가 없거나, 8건 미만이면 "이전 대화 없음" 반환
    if not messages or len(messages) < 8:
        return summary_store.get(session_id, "이전 대화 없음.")

    existing_summary = summary_store.get(session_id) # 기존 요약 불러오기
    last_idx = summary_meta.get(session_id, 0)       # 기존 요약 메세지 수
    
    res: Any = None # ★ 리팩토링: LLM 응답을 담을 변수

    # ===================
    # 기존 요약이 있을 때
    # ===================
    if existing_summary: # 요약할 게 없으면 기존 요약 반환
        if last_idx >= len(messages):
            return existing_summary

        not_summarized_msgs = messages[last_idx:] # 요약되지 않은 메세지들
        buf, toks = [], 0                         # 기본값 - (요약할 텍스트, 요약할 토큰 수)

        for m in not_summarized_msgs:
            line = f"{m.type.upper()}: {m.content}\n" # 각 메세지 구조화 (HUMAN: 내용 or AI: 내용 형태)
            t = _tok(line)                            # 각 메세지 토큰 수 계산
            if toks + t > NOT_SUMMARIZED_LIMIT:       # 토큰 제한 초과 시 중단
                break
            buf.append(line); toks += t

        added_text = "".join(buf).strip() # 요약할 텍스트
        
        if not added_text: # 요약할 텍스트 없으면 기존 요약 반환
            return existing_summary 

        # 요약 업데이트 프롬프트 구성
        update_messages = UPDATE_SUMMARY_PROMPT.format_messages(
            existing_summary=existing_summary,
            chat_history_text=added_text
        )
        res = await llm_fast.ainvoke(update_messages)

    # ===================
    # 최초 요약 생성 로직
    # ===================
    else:
        # 시스템 프롬프트와 기본 뼈대의 토큰 수를 계산합니다.
        instr_tokens = _tok(
            INITIAL_SUMMARY_PROMPT.messages[0].prompt.template  # 시스템 프롬프트
            + "<대화 기록>\n\n<요약>"                             # Human 프롬프트 템플릿의 뼈대
        ) 
        
        initial_summary_limit = max(0, SUMMARY_TOKEN_LIMIT - instr_tokens) # 첫 요약에 사용할 수 있는 토큰 양

        # 전체 메세지(messages)에서 제한 토큰 범위까지 메세지를 가져와서 압축
        chat_history_text = _pack_old_messages_by_tokens(messages, initial_summary_limit) 

        # 에러 방지용(요약할 텍스트가 없을 때 대응)
        if not chat_history_text.strip():
            chat_history_text = "\n".join(f"{m.type.upper()}: {m.content}" for m in messages[-10:])

        # 최초 요약 프롬프트 구성
        initial_messages = INITIAL_SUMMARY_PROMPT.format_messages(
            chat_history_text=chat_history_text
        )
        res = await llm_fast.ainvoke(initial_messages)
    
    # 요약 결과 처리
    summary_to_save = getattr(res, "content", str(res)).strip()
    
    # 에러방지 - 요약 결과가 없으면 기존 요약 반환
    if not summary_to_save:
        return existing_summary if existing_summary else "이전 대화 없음."

    summary_store[session_id] = summary_to_save  # 새로운 요약 갱신
    summary_meta[session_id] = len(messages)     # 전체 메세지 길이 갱신
    
    return summary_to_save

# =====================================
# 요약 트리거 로직
# =====================================

def _should_update_summary(session_id: str, msgs: List[BaseMessage]) -> bool:
    # 새로운 작업을 트리거하지 않습니다. (경합 조건 방지)
    if summary_lock[session_id].locked():
        return False

    MIN_WARMUP = 8               # 초기 8대화(사용자4, AI4) 요약 억제
    INTERVAL   = 4               # 4대화 (사용자2, AI2)
    TOKEN_CAP  = 2000            # 최근 너무 길면 즉시 요약

    n = len(msgs)
    if n <= MIN_WARMUP:
        return False

    last = summary_meta.get(session_id, 0)
    if (session_id not in summary_store) or (n - last >= INTERVAL):
        return True

    # 안전장치: 최근 2*INTERVAL 메시지 토큰이 임계치 넘으면 즉시 요약
    recent = msgs[-2*INTERVAL:]
    if sum(len(enc.encode(m.content or "")) for m in recent) > TOKEN_CAP:
        return True

    return False

async def _refresh_summary_bg(session_id: str, msgs: List[BaseMessage], llm_fast, db_engine: AsyncEngine):
    async with summary_lock[session_id]:   # ← 동시 실행 방지
        try:
            new_summary = await build_summary_text(session_id, msgs, llm_fast)
            await upsert_summary_to_db(db_engine, summary_table, session_id, new_summary)
            summary_store[session_id] = new_summary
            summary_meta[session_id] = len(msgs)
        except Exception as e:
            print(f"[summary] background update failed: {e}")