# SQLDatabase 꼭 써보기

from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_community.utilities.sql_database import SQLDatabase
from langchain.agents import create_sql_agent

# rds_db_setup.py에서 DB 연결 함수를 직접 가져와 사용합니다.
from utils.rds_db_setup import get_or_create_async_db_engine

# --- LLM 및 Agent 전역 변수 초기화 ---
llm = ChatOpenAI(temperature=0, model="gpt-5-mini")
sql_agent_executor = None
db_engine = None

# --- SQL Agent를 위한 시스템 프롬프트 정의 ---
# rds_db_car.py의 스키마를 기반으로 LLM에게 상세한 컨텍스트를 제공합니다.
AGENT_SYSTEM_PROMPT = """
당신은 대한민국 대학교 입시결과 데이터베이스를 조회하여 질문에 답변하는 친절한 AI 어시스턴트입니다.
사용자의 질문을 SQL 쿼리로 변환하여 'car_documents' 테이블에서 정보를 찾은 후, 그 결과를 바탕으로 자연스러운 한국어 문장으로 답변을 생성해야 합니다.

**매우 중요:**
- 테이블 이름은 `car_documents` 입니다.
- 테이블의 컬럼은 한글로 되어 있습니다. (예: "대학", "학과", "전형", "경쟁률", "모집인원", "년도", "입결0.5", "입결0.7")
- SQL 쿼리를 생성할 때, 컬럼 이름을 **반드시 쌍따옴표(`"`)로 감싸야 합니다.** 예: `SELECT "대학", "경쟁률" FROM car_documents`

**컬럼 설명:**
- **"세부"**: 대학교 이름 (예: '일반교과', '일반학종', '지역교과', '지역학종')
- **"대학"**: 대학교 이름 (예: '가천대', '가톨릭관동대', '가톨릭대')
- **"전형"**: 입시 전형 종류 (예: '학생부우수자', '일반', '지역균형', '일반학생[최저]')
- **"학과"**: 모집 단위 (예: '의예과', '의예', '의과대학', '의학과', '치의예')
- **"년도"**: 해당 입시 결과의 연도
- **"모집인원"**: 해당 학과에서 모집한 인원 수
- **"경쟁률"**: 해당 학과의 경쟁률
- **"충원율"**: 해당 학과의 충원율
- **"최저충족비율"**: 해당 학과의 최저충족비율
- **"충원+최저충족비율"**: 해당 학과의 충원율과 최저충족비율의 합
- **"50\%-70\% 컷차이"**: 해당 학과의 충원율과 최저충족비율의 합
- **"1-1차"**: 해당 학과의 충원율
- **"2-1차"**: 해당 학과의 충원율
- **"입결0.5"**: 50% 컷의 입시 결과 (내신 등급 등)
- **"입결0.7"**: 70% 컷의 입시 결과 (내신 등급 등)

**답변 가이드라인:**
1.  사용자의 질문 의도를 정확히 파악하여 필요한 정보를 조회하는 SQL 쿼리를 생성하세요.
2.  데이터베이스에서 조회한 결과를 바탕으로, 완전하고 친절한 한국어 문장으로 답변을 구성하세요.
3.  만약 질문에 대한 답을 데이터베이스에서 찾을 수 없다면, "데이터베이스에서 해당 정보를 찾을 수 없습니다."라고 솔직하게 답변하세요.
4.  복잡한 질문의 경우, 생각의 과정을 통해 어떤 쿼리를 만들고 실행했는지 보여줄 수 있습니다.
"""


async def initialize_sql_agent():
    """
    API 서버 시작 시 SQL Agent에 필요한 모든 객체를 초기화합니다.
    """
    global db_engine, sql_agent_executor
    
    # 1. DB 엔진 생성 (rds_db_setup.py의 함수 사용)
    # pgvector는 사용하지 않으므로 install_pgvector=False로 설정합니다.
    DB_NAME = "hy_rag_db"
    db_engine = await get_or_create_async_db_engine(DB_NAME, install_pgvector=False)
    
    # 2. LangChain SQLDatabase 객체 생성
    # LLM이 조회할 테이블을 'car_documents'로 명확히 지정합니다.
    db = SQLDatabase(engine=db_engine, include_tables=['car_documents'])

    # 3. SQL Agent 생성
    sql_agent_executor = create_sql_agent(
        llm=llm,
        db=db,
        agent_type="openai-tools",
        verbose=True,  # 개발 중에는 True로 설정하여 SQL 쿼리 생성 과정을 확인하세요.
        system_prompt=AGENT_SYSTEM_PROMPT,
    )
    
    print("✅ SQL Agent 초기화가 완료되었습니다.")


async def get_sql_response(query: str, chat_history: List[BaseMessage]) -> str:
    """
    사용자의 질문과 채팅 기록을 받아 SQL Agent를 실행하고 최종 답변(str)을 반환합니다.
    """
    if sql_agent_executor is None:
        raise RuntimeError("SQL Agent가 초기화되지 않았습니다. 먼저 initialize_sql_agent()를 호출해야 합니다.")
    
    # 채팅 기록을 포함하여 에이전트 호출 (멀티턴 대화 지원)
    # AIMessage, HumanMessage 객체로 변환하여 전달합니다.
    response = await sql_agent_executor.ainvoke({
        "input": query,
        "chat_history": chat_history
    })
    
    # 에이전트의 최종 답변은 'output' 키에 담겨 있습니다.
    return response.get("output", "죄송합니다, 답변을 생성하는 중에 문제가 발생했습니다.")