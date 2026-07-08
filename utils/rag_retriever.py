from __future__ import annotations
from typing import List, Any, Dict, Optional
import functools, asyncio, yaml
from abc import ABC, abstractmethod

import weaviate
from weaviate.classes.query import Filter, MetadataQuery

from pydantic import ConfigDict
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.vectorstores import VectorStore
from langchain_cohere import CohereRerank

from utils.rag_prep_upload import get_morph

with open('config/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# ========================= 공통 헬퍼 함수 =========================
def _get_valid_meta_list(meta_list: Optional[List[str]]) -> Optional[List[str]]:
    """ meta_list를 검사하고 유효한 리스트만 반환하는 헬퍼"""
    if not meta_list:
        return None
    valid_metas = [u for u in meta_list if u]
    if not valid_metas:
        return None
    return valid_metas

def meta_filter_oe(meta_filter_list: Optional[List[str]], field: str = "metadata.university"):
    valid_metas = _get_valid_meta_list(meta_filter_list)
    return {"terms": {field: valid_metas}} if valid_metas else None

def ensure_str_query(x: Any) -> str:
    """dict/None이 들어와도 안전하게 문자열 쿼리를 뽑는다. rewritten_query를 최우선으로 사용"""
    if isinstance(x, dict):
        return x.get("rewritten_query") or x.get("query") or x.get("input") or ""
    return "" if x is None else str(x)

# ==========================
# Base Multi-Search Retriever
# ==========================
class BaseMultiSearchRetriever(BaseRetriever, ABC):
    """
    공통 문서/대학 문서 분할 검색 로직을 처리하는 추상 베이스 클래스입니다.
    k 값은 초기 검색 풀 크기를 위해 Task 수에 따라 동적으로 증가합니다.
    """
    k: int = 10
    k_increment: int = 2  # (현 로직에서는 k_increment가 사용되지 않고 k로 고정됨)

    @abstractmethod
    async def _run_single_search(self, query: str, meta_filter_list: List[str] | None, k: int) -> List[Any]:
        """개별 백엔드 검색을 수행하는 비동기 추상 메소드"""
        pass

    @abstractmethod
    def _parse_results_to_documents(self, hits: List[Any]) -> List[Document]:
        """백엔드 검색 결과를 Document 리스트로 파싱하는 추상 메소드"""
        pass

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun):
        try:
            asyncio.get_running_loop()
            raise NotImplementedError(f"{type(self).__name__}는 async 환경에서만 사용해야 합니다. ainvoke()를 호출하세요.")
        except RuntimeError:
             return asyncio.run(self._aget_relevant_documents(query, run_manager=run_manager))

    async def _handle_simple_query(self, query: str) -> List[Document]:
        """단순 문자열 쿼리(필터 없음)를 처리합니다."""
        hits = await self._run_single_search(query, None, self.k)
        return self._parse_results_to_documents(hits)

    def _create_search_tasks(self, query_input: Dict[str, Any]) -> List[asyncio.Task]:
        """입력 사전을 기반으로 비동기 검색 작업을 생성합니다."""
        rewritten_query = query_input.get("rewritten_query") or ""
        if not rewritten_query: # 쿼리가 없으면 태스크 생성 불가
             return []

        meta_list = query_input.get("meta_filter") or []
        uni_only_list = [u for u in meta_list if u != "공통"]
        k_per_task = self.k # k_increment는 사용되지 않고 k로 고정되어 있었음

        tasks = []

        # Task 1: University Filter (N회 실행)
        if uni_only_list:
            for uni in uni_only_list:
                # 쿼리 조작(prefix) 로직
                prefixed_query = f"{uni} {rewritten_query}"
                tasks.append(self._run_single_search(prefixed_query, [uni], k_per_task))

        # Task 2: "공통" Filter (1회 실행)
        if "공통" in meta_list:
            tasks.append(self._run_single_search(rewritten_query, ["공통"], k_per_task))

        return tasks

    async def _parse_and_deduplicate(self, results_lists: List[List[Any]]) -> List[Document]:
        """결과 목록을 파싱하고 page_content 기준으로 중복을 제거합니다."""
        all_docs = []
        for hits_list in results_lists:
            all_docs.extend(self._parse_results_to_documents(hits_list))

        # page_content (안정적인 키)를 기반으로 중복 제거
        final_docs = {}
        for doc in all_docs:
            doc_key = hash(doc.page_content) # page_content를 키로 사용
            if doc_key not in final_docs:
                final_docs[doc_key] = doc

        return list(final_docs.values())

    async def _aget_relevant_documents(self, query_input: Any, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        """
        메인 비동기 실행 로직.
        입력이 dict고 'meta_filter'가 있으면 복합 쿼리, 아니면 단순 쿼리로 처리.
        """
        # 입력이 dict가 아니거나, 멀티 검색 조건(meta_filter)이 없는 경우
        if not isinstance(query_input, dict) or "meta_filter" not in query_input:
            query = ensure_str_query(query_input)
            return await self._handle_simple_query(query)

        # 입력이 dict고 meta_filter가 있는 복잡한 경우
        else:
            tasks = self._create_search_tasks(query_input)
            if not tasks:
                return []

            results_lists = await asyncio.gather(*tasks)
            return await self._parse_and_deduplicate(results_lists)


class BaseESOSRetriever(BaseMultiSearchRetriever):
    """
    Elasticsearch/OpenSearch의 'hits' 구조체 파싱 로직을 공유하는
    중간 베이스 클래스입니다.
    """
    @abstractmethod
    async def _run_single_search(self, query: str, meta_filter_list: List[str] | None, k: int) -> List[dict]:
        # _run_single_search는 여전히 하위 클래스에서 구현해야 하므로 abstract로 둡니다.
        pass

    def _parse_results_to_documents(self, hits: List[dict]) -> List[Document]:
        """
        OpenSearch와 Elasticsearch가 공유하는 공통 파싱 로직
        """
        docs: List[Document] = []
        for hit in hits:
            src = hit.get("_source", {}) or {}
            md = dict(src.get("metadata", {}) or {})
            md["_id"] = hit.get("_id") # 평가용 ID
            docs.append(
                Document(
                    page_content=src.get("text", "") or "",
                    metadata=md,
                )
            )
        return docs

class VectorStoreAdapterRetriever(BaseMultiSearchRetriever):
    """
    Langchain VectorStore (ES, OS 등)를
    BaseMultiSearchRetriever 로직과 연결하는 범용 어댑터.
    asimilarity_search가 없는 VectorStore를 위한 동기식 fallback을 지원합니다.

    """
    vs: VectorStore
    base_kwargs: Dict[str, Any] = {}
    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def _run_single_search(self, query: str, meta_list: list[str] | None, k: int) -> List[Document]:
        if not query:
            return []

        kw = {"k": k, **self.base_kwargs}

        # 통합된 필터 함수(meta_filter_oe) 사용
        f = meta_filter_oe(meta_list)
        if f:
            kw["filter"] = f  # search_kwargs에 filter 전달

        try:
            # 1. 비동기 함수(asimilarity_search) 우선 시도 (ES의 경우 여기 해당)
            if hasattr(self.vs, "asimilarity_search"):
                return await self.vs.asimilarity_search(query, **kw)

            # 2. 비동기 함수가 없으면, 동기 함수를 스레드에서 실행 (OS의 경우 여기 해당)
            else:
                # query와 **kw 인자를 함께 넘기기 위해 partial 사용
                partial_func = functools.partial(self.vs.similarity_search, query, **kw)
                loop = asyncio.get_running_loop()
                return await loop.run_in_executor(None, partial_func)

        except Exception as e:
            print(f"🚨 [VectorStoreAdapterRetriever Error] 쿼리 검색 실패: {e} (Query: {query})")
            return []

    def _parse_results_to_documents(self, hits: List[Document]) -> List[Document]:
        # VectorStore는 이미 Document 리스트를 반환하므로 그대로 반환
        return hits


# ==========================
# Elasticsearch Retriever
# ==========================
class ElasticsearchTextRetriever(BaseESOSRetriever):
    es_client: Any
    index_name: str
    use_morph: bool = True
    use_question_field: bool = True
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # BM25 필드 가중치
    text_boost: float = 1.0
    question_boost: float = 1.0
    text_morph_boost: float = 1.0
    question_morph_boost: float = 1.0

    async def _run_single_search(self, query: str, meta_filter_list: List[str] | None, k: int) -> List[dict]:
        if not query:
            return []

        es_filter = meta_filter_oe(meta_filter_list)
        filter_clauses = [es_filter] if es_filter else []
        
        # 1. kiwi 형태소분석 사용하는 경우
        if self.use_morph:
            search_query = get_morph(query)
            fields = [f"metadata.text_morph^{self.text_morph_boost}"]

            if self.use_question_field:
                fields.insert(0, f"metadata.question_morph^{self.question_morph_boost}")
        else:
            search_query = query
            fields = [f"text^{self.text_boost}"]

            if self.use_question_field:
                fields.insert(0, f"metadata.question^{self.question_boost}")

        must_clauses = [{
            "multi_match": {
                "query": search_query,
                "fields": fields,
                "type": "best_fields",
            }
        }]

        body = {
            "size": k,
            "track_total_hits": True,
            "query": {
                "bool": {
                    "must": must_clauses,
                    "filter": filter_clauses,
                }
            }
        }

        try:
            resp = await asyncio.to_thread(
                self.es_client.search,
                index=self.index_name,
                body=body
            )
            return resp.get("hits", {}).get("hits", [])
        except Exception as e:
            print(f"🚨 [ElasticsearchTextRetriever Error] 쿼리 검색 실패: {e} (Query: {query})")
            return []


# ==========================
# Rerank Cohere
# ==========================
class CohereThresholdCompressor(BaseDocumentCompressor):
    """
    CohereRerank 결과에 임계값 컷 및 Task 수에 따른 동적 Top-K 제한을 적용합니다.
    """
    base: Any
    threshold: float = 0.55
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @staticmethod
    def _extract_meta(q: Any) -> tuple[str, list[str]]:
        """Extracts text query and meta filter list from input."""
        if isinstance(q, dict):
            text_q = q.get("rewritten_query") or q.get("original_query")or ""
            meta_list = q.get("meta_filter") or []
            return text_q, meta_list
        return str(q) if q is not None else "", []

    def _process_results(self, reranked_docs: List[Document], meta_list: List[str]) -> List[Document]:
        """
        공통 후처리 로직: 임계값 컷 및 동적 K 적용
        """
        # 1. Threshold Cut
        th = float(self.threshold)
        threshold_docs = [d for d in reranked_docs if float(d.metadata.get("relevance_score", 0)) >= th]

        # 2. Dynamic Final K Calculation
        uni_only_list = [u for u in meta_list if u != "공통"]
        num_universities = len(uni_only_list)
        k_final = config['retriever']['rerank']['top_n'] + 2 * num_universities # (5, 2는 매직넘버지만 원본 유지)

        # 3. Top K Limit
        return threshold_docs[:k_final]


    def compress_documents(self, documents, query, **kwargs):
        text_q, meta_list = self._extract_meta(query)

        # 1. Base Rerank
        reranked_docs = self.base.compress_documents(documents, text_q, **kwargs)

        # 2. 공통 후처리
        return self._process_results(reranked_docs, meta_list)


    async def acompress_documents(self, documents, query, **kwargs):
        text_q, meta_list = self._extract_meta(query)

        # 1. Base Rerank (Async or To Thread)
        if hasattr(self.base, "acompress_documents"):
            reranked_docs = await self.base.acompress_documents(documents, text_q, **kwargs)
        else:
            reranked_docs = await asyncio.to_thread(self.base.compress_documents, documents, text_q, **kwargs)

        # 2. 공통 후처리
        return self._process_results(reranked_docs, meta_list)


def create_cohere_rerank(rerank_config: dict) -> CohereThresholdCompressor:
    """
    Cohere Reranker 및 Threshold Compressor를 생성하는 헬퍼 함수
    """
    cohere_base = CohereRerank(
        model=rerank_config["model"],
        top_n=rerank_config["top_n"],
    )
    threshold = rerank_config.get("threshold", 0.55)

    return CohereThresholdCompressor(base=cohere_base, threshold=threshold)