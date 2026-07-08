import json, yaml
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import Runnable
from pydantic import BaseModel, Field
import textwrap

UNI_FILE_PATH = 'config/university.json'

with open("config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# ================================
# reconstruction_chain
# ================================
class RewriteOutput(BaseModel):
    """
    LLM의 질문 재구조화 및 정보 추출 기능을 강제하기 위한 클래스
    """
    universities: List[str] = Field(
        ...,
        description="user_input에 포함된 대학교 이름 목록. 문맥정보가 필요한 경우 chat_history_8을 역순으로 참고. 없으면 빈 리스트 [].",
        json_schema_extra={"examples": [["연세대", "고려대"]]}
    )

    rewritten_query: str = Field(
        ...,
        description="user_input(대학교 이름 제외)을 재구성한 기본 검색 문장",
        json_schema_extra={"examples": ["3년 특례 전형이 뭔가요?"]}
    )

contextualize_q_system = textwrap.dedent("""\
당신은 user input을 RAG retriever용 '검색 질의'와 '메타데이터 필터용 대학 목록'으로 분리하는 AI입니다.
'user input'과 'chat_history_8'을 모두 고려하여 **2가지 정보**를 JSON으로 생성합니다.

## 금지

- 정답/설명/추론/근거/리스트/예시/이모지.
- '자세한','자세하게' 사용 금지

## 출력
- 오직 JSON 스키마(RewriteOutput)의 두 필드('universities', 'rewritten_query')만 허용됩니다.
1.  **universities**: 'user input'에서 추출한 **모든 대학교 이름**을 **담은 Python 리스트(List[str])**. 문맥이 필요한 경우 chat_history_8을 역순으로 참고,  없으면 빈 리스트 `[]` 반환.
2.  **rewritten_query**: 'user input'(대학교 이름 제외)의 의도만을 명확하게 재구성한 **'기본 검색 쿼리'**. 명확한 의도파악이 안 되는 경우 user input 반환
""").strip()

# 질문 재구조화 LLM
llm_context = ChatOpenAI(model=config['openai']['models']['kgs-filter'], 
                        timeout=30,
                        max_retries=2, 
                        max_tokens=128,
                        temperature=0.1,
                        frequency_penalty = 0,
                        presence_penalty = 0,
                        top_p= 0.1
                        )

def query_reconstruction_chain(llm_context: ChatOpenAI) -> Runnable:
    """ 
    User Input을 'universities' 리스트와 'rewritten_query'가 포함된 RewriteOutput 객체로 반환합니다.
    """
    reconstruction_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system),
            MessagesPlaceholder(variable_name="chat_history_8"),
            ("human", "{input}"),
        ])

    # with_structured_output는 JSON구조를 정의하고, strict=True는 그 구조를 반드시 지키도록하는 역할
    structured_llm = llm_context.with_structured_output(RewriteOutput, strict=True)

    reconstruct_chain = (
        {
            "chat_history_8": lambda x: x["chat_history_8"],
            "input": lambda x: x["input"],
        }
        | reconstruction_prompt 
        | structured_llm                      
    )
    return reconstruct_chain

# ================================
# 별칭을 표준명으로 매핑
# ================================
def load_uni_alias(filepath: str) -> Dict[str, str]:
    """
    사내 university 목록을 {별칭1: 표준명, 별칭2: 표준명...}으로 매핑하는 함수

    ex) {"연세대학교": "연세대", "연대" : "연세대" ...}
    """
    with open(filepath, 'r', encoding='utf-8') as f: 
        alias_map = json.load(f)
    
    reversed_map = {} 
    for standard_name, aliases in alias_map.items():
        reversed_map[standard_name] = standard_name 
        for alias in aliases:                       
            reversed_map[alias] = standard_name
    return reversed_map

def trans_uni_name(extracted_names: List[str],alias_map: Dict[str, str]) -> List[str]:
    """
    user input에서 추출한 university 리스트와 사내 university 딕셔너리를 매핑하여
    대학교 이름의 별칭을 표준명으로 변환합니다.

    ex) ["연세대학교", "이대"] → ["연세대", "이화여대"]
    """
    normalized_list = [alias_map.get(name, name) for name in extracted_names]
    return sorted(list(set(normalized_list)))

RECONSTRUCTION_CHAIN = query_reconstruction_chain(llm_context)

# =============
# 최종 쿼리 필터 로직
# =============
async def query_filter(query: str, 
                       chat_history_8: list, 
                       reconstruction_chain = RECONSTRUCTION_CHAIN) -> dict:
    """
    1) LLM이 'universities' (대학 리스트)와 'rewritten_query' (기본 쿼리)로 분리.
    2) 'universities' 리스트를 표준명으로 정규화.
    3) 'meta_filter'와 'rewritten_query'를 반환.

    Returns:
            dict: {
                "meta_filter": list[str],
                "rewritten_query": str
            }
    """

    # 1. LLM이 'universities' (list)와 'rewritten_query' (str)를 추출
    reconstruction_output = await reconstruction_chain.ainvoke({
        "input": query,
        "chat_history_8": chat_history_8,
    })
    
    llm_extracted_unis = reconstruction_output.universities
    rewritten_query = reconstruction_output.rewritten_query

    # 2. 대학 이름 표준화
    UNI_ALIAS_DICT = load_uni_alias(UNI_FILE_PATH) # 내부 지정 학교 이름
    
    # 3. LLM이 대학을 찾은 경우
    if llm_extracted_unis:
        standardized_names = trans_uni_name(llm_extracted_unis, UNI_ALIAS_DICT)
        standardized_names_set = {u.strip() for u in standardized_names if isinstance(u, str) and u.strip()}
        
        final_meta_set = standardized_names_set | {"공통"}
        
        return {
            "meta_filter": sorted(list(final_meta_set)),
            "rewritten_query": rewritten_query
        }

    # 4. LLM이 대학을 추출하지 못한 경우 (공통 검색)
    else:
        return {
            "meta_filter": ["공통"], 
            "rewritten_query": rewritten_query,
        }