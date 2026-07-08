from dotenv import load_dotenv
load_dotenv()

import os, re ,json, time, base64, asyncio, requests, fitz, pymupdf4llm
from typing import List, Dict, Any, Optional
from llama_parse import LlamaParse
from openai import AsyncOpenAI

from .rag_prep_pdf import filename_to_metadata

MAX_CONCURRENT_REQUESTS = 5
OCR_TEXT_THRESHOLD = 50     # 페이지의 글자 수가 50자 미만이면, 이미지 OCR 처리

aclient = AsyncOpenAI()
sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# ======================================
# 공통 유틸 (PyMuPDF4LLM + Vision)
# ======================================
def _render_page_base64(doc: fitz.Document, page_idx: int, zoom: float = 2.0) -> str:
    """PyMuPDF로 특정 페이지를 이미지화하고, base64 문자열로 반환"""
    page = doc[page_idx] # 특정 페이지를 가져옵니다.
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom)) # page.get_pixmap : 픽셀기반 이미지 변환 / fitz.Matrix(zoom, zoom) : 이미지 배율
    return base64.b64encode(pix.tobytes("png")).decode("utf-8") # pix.tobytes("png") : 이미지를 바이트로 변환 / base64.b64encode() : 바이트를 base64로 인코딩 / decode("utf-8") : 바이트를 문자열로 디코딩

async def _async_ocr_gpt(base64_img: str) -> str:
    """gpt-4.1-mini으로 base64 문자열을 OCR 처리하여 markdown 반환"""
    if not base64_img or not aclient: return ""

    async with sem:
        resp = await aclient.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are the image pdf parser. "
                        "Write all text and tables in the image in markdown form "
                        "without any changes in content. Do not add explanations. "
                        "Clearly distinguish between columns and rows according to letter size and color."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_img}"},
                        }
                    ],
                },
            ],
            temperature=0.0,
        )
        md = resp.choices[0].message.content or ""

        return md


# ======================================
# 파서별 "페이지 단위 결과" 함수들 (page_number, text)만 반환
# ======================================
def _parse_upstage_pages(pdf_path: str) -> List[Dict[str, Any]]:
    """Upstage Document Parse: 100페이지씩 나눠서 처리, 페이지별 텍스트 리스트 반환"""
    
    api_key = os.getenv("UPSTAGE_API_KEY")
    name = os.path.basename(pdf_path)

    all_pages = []
    MAX_PAGES = 100

    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        print(f"[{name}] 총 {total_pages}페이지 처리 시작")

        base_pdf_bytes = None
        if total_pages <= MAX_PAGES:
            with open(pdf_path, "rb") as f:
                base_pdf_bytes = f.read()

        for start_page in range(0, total_pages, MAX_PAGES):
            end_page = min(start_page + MAX_PAGES, total_pages)
            
            if total_pages <= MAX_PAGES:
                pdf_bytes = base_pdf_bytes
            else:
                new_doc = fitz.open()
                new_doc.insert_pdf(doc, from_page=start_page, to_page=end_page - 1)
                pdf_bytes = new_doc.tobytes()
                new_doc.close()

            try:
                response = requests.post(
                    "https://api.upstage.ai/v1/document-ai/document-parse",
                    headers={"Authorization": f"Bearer {api_key}"},
                    files={"document": (name, pdf_bytes, "application/pdf")},
                    data={"ocr": "auto", "output_formats": "html", "coordinates": False},
                    timeout=300 # [지적사항2] Timeout 설정 추가 (필수)
                )
                response.raise_for_status()
                result = response.json() # 결과 변수 저장
            except Exception as e:
                print(f"Error ({start_page}~{end_page}): {e}")
                continue

            elements = result.get("elements", [])
            
            # Fallback 로직 추가 (데이터 누락 방지)
            if not elements:
                content = result.get("content") or {}
                # 딕셔너리나 문자열 형태 모두 처리
                html = content.get("html") if isinstance(content, dict) else content
                if html:
                    # 통짜 데이터라도 저장
                    all_pages.append({"page_number": start_page + 1, "text": html})
                continue # 다음 루프로

            # 정상 케이스: elements가 있을 때
            page_buffer = {}
            for elem in elements:
                real_page = elem.get("page", 1) + start_page
                
                content = elem.get("content") or {}
                if isinstance(content, dict):
                    text = content.get("html") or content.get("text")
                else:
                    text = elem.get("text")
                
                if not text:
                    text = elem.get("text", "") # 최후의 수단

                if text:
                    page_buffer.setdefault(real_page, []).append(text)

            for p_num, texts in page_buffer.items():
                all_pages.append({
                    "page_number": p_num,
                    "text": "\n".join(texts).strip()
                })

    except Exception as e:
        print(f"Critical Error: {e}")
    finally:
        if 'doc' in locals(): doc.close()

    return sorted(all_pages, key=lambda x: x["page_number"])


def _parse_llamaparse_pages(pdf_path: str) -> List[Dict[str, Any]]:
    """ LlamaParse로 페이지별 텍스트 리스트 반환 """
    name = os.path.basename(pdf_path)
    file_type = "image" if "image" in name else "text"

    print(f"[{name}] LlamaParse 실행 중... (Type: {file_type})")
    use_premium = file_type == "image"
    start = time.time()

    try:
        parser = LlamaParse(
            result_type="markdown", # 'text' or 'markdown'
            language="ko",
            verbose=True,
            premium_mode=use_premium,
            skip_diagonal_text=True,     # 워터마크 제거
            output_tables_as_HTML=False, # 표를 HTML 형식으로 추출
            preserve_layout_alignment_across_pages=False, # 페이지 바뀌면서 표가 어그러지는 거 막아줌
            merge_tables_across_pages_in_markdown=False,  # 표가 페이지 넘어갈 때(푸터 다 없어짐 조심)
            hide_headers=True,
            hide_footers=True,
            bounding_box = "0.1,0,0.1,0", # 헤더푸터 안 지워질 때 짤라버리기
            system_prompt_append=(
                "You are parsing Korean university admission guides and "
                "student records. Preserve tables exactly as they appear. "
                "Focus on admission quotas, schedules, eligibility, and evaluation criteria."
            ),
        )
        docs = parser.load_data(pdf_path)
    except Exception as e:
        print(f"🚨 [{name}] LlamaParse 파싱 실패: {e}")
        return []

    elapsed = time.time() - start

    pages: List[Dict[str, Any]] = []
    for i, doc in enumerate(docs):
        # 🔥 markdown 그대로, 앞뒤 공백만 제거
        page_text = (doc.text or "")
        if not page_text.strip():
            continue
        pages.append({"page_number": i + 1, "text": page_text})

    print(f"[{name}] LlamaParse 완료 ({elapsed:.2f}s), {len(pages)} 페이지")
    return pages


async def _parse_pymupdf4llm_pages(pdf_path: str) -> List[Dict[str, Any]]:
    """
    PyMuPDF4LLM(text) + gpt-4.1-mini(image) 사용해 페이지별 텍스트 리스트 반환
    """
    name = os.path.basename(pdf_path)
    print(f"🚀 [{name}] PyMuPDF4LLM 하이브리드 파싱 시작")

    pages = pymupdf4llm.to_markdown(
        pdf_path,
        table_strategy="lines_strict",
        page_chunks=True,
        write_images=False,
    )

    doc = fitz.open(pdf_path)

    md_chunks: List[str] = []
    full_jobs: List[tuple[int, asyncio.Task]] = []

    for page_idx, p in enumerate(pages):
        raw_md = p.get("text") or ""
        # 여기서는 text_only를 길이 판단용으로만 쓴다. (실제 결과에는 영향 X)
        text_only = re.sub(r'!\[.*?\]\(.*?\)', "", raw_md)
        text_only_len = len(text_only.strip())

        if text_only_len < OCR_TEXT_THRESHOLD:
            # FULL OCR 대상
            print(f"   - P.{page_idx+1}: text_only={text_only_len}자 → FULL OCR 대상")
            md_chunks.append("")
            if aclient:
                b64 = _render_page_base64(doc, page_idx, zoom=2.0)
                if b64:
                    task = asyncio.create_task(_async_ocr_gpt(b64))
                    full_jobs.append((page_idx, task))
            else:
                md_chunks[page_idx] = raw_md
        else:
            md_chunks.append(raw_md)

    # FULL OCR 결과 적용
    if full_jobs:
        print(f"   -> FULL OCR {len(full_jobs)}페이지 실행 중...")
        full_results = await asyncio.gather(*[t for _, t in full_jobs])
        for (page_idx, _), md in zip(full_jobs, full_results):
            # Vision이 준 markdown을 그대로 사용 (없으면 기존 텍스트)
            md_chunks[page_idx] = md or pages[page_idx].get("text", "")

    doc.close()

    result_pages: List[Dict[str, Any]] = []
    for page_idx, md_page in enumerate(md_chunks, start=1):
        raw_text = md_page or ""
        clean_text = raw_text.strip()
        if not clean_text:
            continue
        result_pages.append({"page_number": page_idx, "text": clean_text})

    print(f"[{name}] PyMuPDF4LLM 완료, {len(result_pages)} 페이지")
    return result_pages


# ======================================
# 메인 엔트리 – 한 번에 3개 파서 모두 돌리기
# ======================================
async def parse_pdf_with_all_parsers(
    pdf_path: str,
    file_name: str,
    drive_file_id: str,
    modified_time: str,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    PDF 한 개에 대해 Upstage / LlamaParse / PyMuPDF4LLM를 한 번에 실행하고
    각 파서별로 메타데이터가 붙은 row 리스트를 반환.
    text 필드는 각 파서가 만들어준 HTML/Markdown을 그대로 보존한다.
    """
    meta_from_name = filename_to_metadata(file_name)
    base_meta = {
        "file_name": file_name,
        "drive_file_id": drive_file_id,
        "modified_time": modified_time,
        "university": meta_from_name.get("university"),
        "year": meta_from_name.get("year"),
        "admission_type": meta_from_name.get("admission_type"),
        "document_type": meta_from_name.get("document_type"),
    }

    results: Dict[str, List[Dict[str, Any]]] = {}

    # ----- Upstage -----
    up_pages = _parse_upstage_pages(pdf_path)
    if up_pages:
        results["upstage"] = [
            {
                **base_meta,
                "parser": "upstage",
                "page_number": p["page_number"],
                "text": p["text"],  # HTML 그대로
            }
            for p in up_pages
        ]

    # ----- LlamaParse -----
    llama_pages = _parse_llamaparse_pages(pdf_path)
    if llama_pages:
        results["llamaparse"] = [
            {
                **base_meta,
                "parser": "llamaparse",
                "page_number": p["page_number"],
                "text": p["text"],  # markdown 그대로
            }
            for p in llama_pages
        ]

    # ----- PyMuPDF4LLM -----
    pymu_pages = await _parse_pymupdf4llm_pages(pdf_path)
    if pymu_pages:
        results["pymupdf4llm"] = [
            {
                **base_meta,
                "parser": "pymupdf4llm",
                "page_number": p["page_number"],
                "text": p["text"],  # markdown 그대로
            }
            for p in pymu_pages
        ]

    return results
