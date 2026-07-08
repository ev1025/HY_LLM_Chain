from dotenv import load_dotenv
load_dotenv()

from typing import List
import langchain

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_postgres.vectorstores import PGVector
from langchain_core.messages import BaseMessage
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

from utils.rds_db_setup import get_or_create_async_db_engine



# --- 모델 정의 ---
# [최적화] 작업에 따라 2개의 다른 LLM 모델 정의
llm_fast = ChatOpenAI(temperature=0, model="gpt-4o-mini") # 질문 재구성을 위한 작고 빠른 모델
llm_quality = ChatOpenAI(temperature=0.5, model="gpt-5-mini")   # 최종 답변 생성을 위한 고품질 모델
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# --- 전역 변수 선언 ---
# API 서버가 시작될 때 initialize_rag_chain 함수에 의해 채워집니다.
retriever = None
rag_chain = None
contextualize_q_chain = None

# --- 공통 함수 및 프롬프트 정의 ---
def format_docs(docs):
    """검색된 Document 객체를 하나의 문자열로 포맷팅합니다."""
    return "\n\n".join(d.page_content for d in docs)

contextualize_q_system_prompt = """
채팅 기록과 사용자의 마지막 질문이 주어집니다. 
이 질문은 채팅 기록의 맥락을 참조할 수 있습니다. 
채팅 기록 없이도 이해할 수 있는 독립적인 질문으로 다시 만들어주세요. 
질문에 답하지는 말고, 필요하다면 질문을 다시 만들고, 그렇지 않다면 그대로 반환하세요."""
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

qa_system_prompt = """
당신은 대한민국 기업 (주)HY교육의 친절한 AI 어시스턴트입니다. 
제공된 컨텍스트 정보를 바탕으로 사용자의 대학 입시 질문에 답변하세요.
컨텍스트가 없는경우 최대한 출처 기반으로 답변하세요. \n\n{context}
"""
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

def contextualized_question(input: dict):
    """
    채팅 기록이 있을 경우에만 질문 재구성 체인을 실행합니다.
    """
    if input.get("chat_history"):
        return contextualize_q_chain
    else:
        return input["question"]

# --- RAG 시스템 초기화 및 실행 함수 (qna_server.py에서 호출) ---
async def initialize_rag_chain():
    """
    API 서버 시작 시 RAG 체인에 필요한 모든 객체를 초기화합니다.
    """
    global retriever, rag_chain, contextualize_q_chain
    
    DB_NAME = "hy_rag_db"
    QNA_COLLECTION_NAME = "qna_documents"
    
    # DB 엔진 및 VectorStore 연결
    db_engine = await get_or_create_async_db_engine(DB_NAME)
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=QNA_COLLECTION_NAME,
        connection=db_engine,
        use_jsonb=True,
    )

    # 1. 기본 Retriever를 MMR 검색 방식으로 설정
    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10, "fetch_k": 50, "lambda_mult": 0.5}
    )

    # 2. Cohere Rerank 압축기(Compressor) 설정
    compressor = CohereRerank(model="rerank-multilingual-v3.0", top_n=5)

    # 3. ContextualCompressionRetriever로 최종 Retriever 구성
    retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
    )
    
    langchain.debug = True
    
    # 질문 재구성 체인은 빠르고 저렴한 모델(llm_fast) 사용
    contextualize_q_chain = contextualize_q_prompt | llm_fast | StrOutputParser()
    
    # 최종 RAG 체인은 고품질 모델(llm_quality) 사용
    rag_chain = (
        RunnablePassthrough.assign(
            context=contextualized_question | retriever | format_docs
        )
        | qa_prompt
        | llm_quality
        | StrOutputParser()
    )

async def get_rag_response(query: str, chat_history: List[BaseMessage]) -> str:
    """
    질문과 채팅 기록을 받아 RAG 체인을 실행하고 최종 답변(str)을 반환합니다.
    """
    if rag_chain is None:
        raise RuntimeError("RAG chain is not initialized. Call initialize_rag_chain() first.")
        
    response = await rag_chain.ainvoke({
        "question": query, 
        "chat_history": chat_history
    })
    return response