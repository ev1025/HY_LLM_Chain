import weaviate, textwrap
from opensearchpy import OpenSearch
from elasticsearch import Elasticsearch

from langchain.retrievers import ContextualCompressionRetriever,EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain.schema.runnable import Runnable, RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI

from utils.rag_retriever import (
    OpenSearchTextRetriever,      # 오픈서치 bm25
    VectorStoreAdapterRetriever,  # os, es 벡터검색
    WeaviateHybridRetriever,      # Weaviate 하이브리드
    ElasticsearchTextRetriever,   # 엘라스틱서치 bm25
    create_cohere_rerank
)

from utils.rag_server_utils import (
    format_docs,
    prep_inputs,
    run_preprocessing,
)

from utils.rag_utils_wv import get_weaviate_client
from utils.rag_utils_open import get_opensearch_client, get_opensearch_vectorstore
from utils.rag_utils_es import get_es_client, get_es_vectorstore


qa_system_prompt = textwrap.dedent(
    """\
    You are HY AI, an expert in Korean university admissions.

    RULES
    - 가독성을 높이기 위해 이모지를 적극적으로 사용한다 (예: 😊, ✅, 📅).
    - <HYAI_INFO/>와 <SUMMARY/>이 존재한다면 사실 근거는 오직 이들에서만 가져온다.
    - 답변은 반드시 사용자의 user input에 한국어로 답변한다.
    - history는 용어 일관성·후속질문 파악에만 사용한다.
    - 사용자에게 “context” 대신 “HY AI 정보”라고 표현한다.

    <HYAI_INFO>
    {context}
    </HYAI_INFO>

    <SUMMARY>
    {summary}
    </SUMMARY>
    """
).strip()

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history_8"),
        ("human", "{input}"),
    ]
)



def weaviate_retriever(config: dict, client: weaviate.Client) -> BaseRetriever:
    """Weaviate용 리트리버 생성"""
    wv_config = config["weaviate"]
    retriever_config = config["retriever"]

    base_hybrid_retriever = WeaviateHybridRetriever(
        client=client,
        index_name=wv_config["kgs_index"],
        k=wv_config["k"],  # hybrid 검색이라서 다른 검색기 x 2
        alpha=wv_config["alpha"],
        use_question_field = wv_config['question'],  # question 사용 여부
        use_morph = wv_config['morph']               # kiwi 토크나이저 사용 여부

    )

    cohere_compressor = create_cohere_rerank(retriever_config["rerank"])

    final = ContextualCompressionRetriever(
        base_compressor=cohere_compressor,
        base_retriever=base_hybrid_retriever,
    )
    return final


def opensearch_retriever(config: dict, os_client: OpenSearch) -> BaseRetriever:
    """OpenSearch용 리트리버 생성"""
    retriever_config = config["retriever"]
    os_config = config["aws"]["opensearch"]

    # BM25
    bm25_config = retriever_config["bm25"]
    bm25_retriever = OpenSearchTextRetriever(
        os_client=os_client,
        index_name=os_config["kgs_index"],
        qtype=retriever_config["bm25"]["qtype"],
        k=retriever_config["bm25"]["k"],
        search_kwargs=bm25_config.get("search_kwargs", {}),
        use_question_field = os_config['question'],  # question 사용 여부
        use_morph = os_config['morph']               # kiwi 토크나이저 사용 여부
    )

    # VectorStore
    vector_store = get_opensearch_vectorstore(
        index_name=os_config["kgs_index"]
    )
    vcfg = retriever_config["vector_search"]
    vector_retriever = VectorStoreAdapterRetriever(
        vs=vector_store,
        k=vcfg["k"],
        base_kwargs=vcfg["parameters"].get(vcfg["search_type"], {}),
    )

    # Ensemble
    base_ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=retriever_config["ensemble"]["weights"],
    )

    cohere_compressor = create_cohere_rerank(retriever_config["rerank"])

    final = ContextualCompressionRetriever(
        base_compressor=cohere_compressor,
        base_retriever=base_ensemble,
    )
    return final

def elasticsearch_retriever(config: dict, es_client: Elasticsearch) -> BaseRetriever:
    """Elasticsearch용 리트리버 생성"""
    retriever_config = config["retriever"]
    es_cfg = config["elasticsearch"]

    # 1) BM25
    bm25_k = retriever_config["bm25"].get("k", 30)
    bm25_retriever = ElasticsearchTextRetriever(
        es_client=es_client, 
        index_name=es_cfg["kgs_index"], 
        k=bm25_k,
        use_question_field = es_cfg['question'],  # question 사용 여부
        use_morph = es_cfg['morph']               # kiwi 토크나이저 사용 여부
    )

    # 2) Vector 
    vector_store = get_es_vectorstore(index_name=es_cfg["kgs_index"])
    vcfg = retriever_config["vector_search"]
    
    vector_retriever = VectorStoreAdapterRetriever(
        vs=vector_store,
        k=vcfg.get("k", 10),
        base_kwargs=vcfg["parameters"].get(vcfg.get("search_type","similarity"), {})
    )

    # 3) Ensemble
    base_ensemble = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=es_cfg["weights"],
    )

    cohere_compressor = create_cohere_rerank(retriever_config["rerank"])
    
    final = ContextualCompressionRetriever(
        base_compressor=cohere_compressor,
        base_retriever=base_ensemble
    )

    return final


def initialize_rag_pipeline(
    config: dict, llm_answer: ChatOpenAI, vdb_name :str
) -> Runnable:
    """
    Config에 따라 Weaviate 또는 OpenSearch용 RAG 파이프라인을 생성합니다.
    """
    retriever = None

    if vdb_name == "weaviate":
        print("Weaviate 파이프라인을 초기화합니다...")
        wv_client = get_weaviate_client()
        retriever = weaviate_retriever(config, wv_client)
    elif vdb_name == "opensearch":
        print("OpenSearch 파이프라인을 초기화합니다...")
        os_client = get_opensearch_client()
        retriever = opensearch_retriever(config, os_client)
    elif vdb_name == "elasticsearch": 
        print("Elasticsearch 파이프라인을 초기화합니다...")
        es_client = get_es_client() 
        retriever = elasticsearch_retriever(config, es_client)
    else:
        raise ValueError(f"'{vdb_name}'는 알 수 없는 백엔드입니다.")

    # --- 공통 RAG 체인 로직 ---
    rag_chain = (
        RunnableLambda(prep_inputs).with_config(run_name="prep_once")
        | RunnablePassthrough.assign(original_input=lambda x: x["input"])
        | RunnablePassthrough.assign(
            qf=RunnableLambda(run_preprocessing).with_config(run_name="qf_once")
        )
        | RunnablePassthrough.assign(
            context=(
                RunnableLambda(
                    lambda x: {
                        **x["qf"],
                        "original_input": x["original_input"],
                    }
                )
                | retriever
                # | RunnableLambda(lambda docs: log_and_pass_through(docs, "After Retriever"))
                | format_docs
            )
        )
        | RunnableLambda(
            lambda x: {
                "input": x["original_input"],
                "chat_history_8": x["chat_history_8"],
                "summary": x["summary"],
                "context": x["context"],
                "rewritten_query": x["qf"].get("rewritten_query", ""),
            }
        )
        # | RunnableLambda(lambda x: log_and_pass_through(x, "Final QA Prompt Input"))
        | qa_prompt  
        | llm_answer
        | StrOutputParser()
    )
    return rag_chain