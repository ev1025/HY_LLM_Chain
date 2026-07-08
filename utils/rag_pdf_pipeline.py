from dotenv import load_dotenv
load_dotenv()

import os
import re
import json
import asyncio
import tempfile
import yaml
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from elasticsearch.helpers import bulk

from .rag_prep_pdf import (
    GoogleDriveManager,
    text_splitter,
    BASE_FOLDER_ID,
    LEVEL1_FOLDERS,
    LEVEL2_FOLDERS,
    LEVEL3_FOLDERS,
)
from .rag_pdf_parser import parse_pdf_with_all_parsers
from .s3_io import save_jsonl_to_s3, s3
from .rag_prep_upload import get_morph
from .rag_utils_es import (
    get_es_client,
    create_es_schema,
    build_elasticsearch_index,
    generate_chunk_id_es,
    EMBEDDING_MODEL,
    EMBEDDING_BATCH_SIZE,
)

# ======================================
# 설정 로드
# ======================================
with open("config/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

S3_BUCKET = config["aws"]["s3"]["bucket"]
S3_RAW_PREFIX = "pdf_parsed"

PARSER_SHORT = {
    "upstage": "u",
    "llamaparse": "l",
    "pymupdf4llm": "p",
}

# 제목으로 취급할 수 있는 최대 길이 (마크업 제거 후 기준)
HEADING_MAX_CHARS = 80

# 푸터(섹션명 + 페이지번호) 패턴: 예) "Ⅰ. 전형 요약 및 주요 사항  | 3"
FOOTER_LINE_RE = re.compile(
    r"^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\. .+?\s*\|\s*\d+\s*$"
)

# Upstage HTML에서 table 블록만 통째로 분리하기 위한 패턴
UPSTAGE_TABLE_PATTERN = re.compile(r"<table[\s\S]*?</table>", re.IGNORECASE)


# ======================================
# 1단계: Drive PDF → 3개 파서 → RAW JSONL(S3)
# ======================================

async def parse_single_drive_pdf_to_s3(
    drive_manager: GoogleDriveManager,
    pdf_info: Dict[str, Any],
) -> Dict[str, str]:
    """
    하나의 Drive PDF에 대해:
      - parse_pdf_with_all_parsers() 실행
      - 파서별 RAW row 리스트를 S3(JSONL)로 저장
    return:
      {"upstage": "s3://.../u_파일명.jsonl", ...}
    """
    file_id = pdf_info["id"]
    file_name = pdf_info["name"]
    modified_time = pdf_info.get("modifiedTime")

    print(f"\n=== [DRIVE] {file_name} ({file_id}) 처리 시작 ===")

    pdf_bytes = await drive_manager.download_file(file_id)
    if not pdf_bytes:
        print(f"  ❌ {file_name} 다운로드 실패, 스킵")
        return {}

    # 임시 파일에 PDF 저장
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    base_name, _ = os.path.splitext(file_name)

    # 3개 파서 한 번에 실행
    rows_by_parser: Dict[str, List[Dict[str, Any]]] = await parse_pdf_with_all_parsers(
        tmp_path,
        file_name=file_name,
        drive_file_id=file_id,
        modified_time=modified_time,
    )

    all_s3_uris: Dict[str, str] = {}
    for parser_name, rows in rows_by_parser.items():
        if not rows:
            continue
        short = PARSER_SHORT.get(parser_name, parser_name[0].lower())
        s3_uri = save_jsonl_to_s3(
            db=S3_RAW_PREFIX,
            rows=rows,
            bucket=S3_BUCKET,
            s3_file_path=f"{short}_{base_name}",
        )
        all_s3_uris[parser_name] = s3_uri

    os.remove(tmp_path)
    print(f"=== [DRIVE] {file_name} 처리 완료, S3 URIs: {all_s3_uris} ===")
    return all_s3_uris


async def run_parsers_for_all_drive_pdfs() -> Dict[str, List[str]]:
    """
    rag_prep_pdf 폴더 구조를 따라 모든 PDF에 대해
    3개 파서를 실행하고 RAW JSONL(S3) 경로를 모읍니다.
    return:
      {"upstage": [s3://...], "llamaparse": [...], "pymupdf4llm": [...]}
    """
    drive_manager = GoogleDriveManager()
    if not drive_manager.service:
        print("❌ Google Drive 서비스가 초기화되지 않았습니다.")
        return {}

    collected_uris = {"upstage": [], "llamaparse": [], "pymupdf4llm": []}

    for l1_name in LEVEL1_FOLDERS:
        l1_id = await drive_manager.find_folder_id(l1_name, BASE_FOLDER_ID)
        if not l1_id:
            continue
        for l2_name in LEVEL2_FOLDERS:
            l2_id = await drive_manager.find_folder_id(l2_name, l1_id)
            if not l2_id:
                continue
            for l3_name in LEVEL3_FOLDERS:
                l3_id = await drive_manager.find_folder_id(l3_name, l2_id)
                if not l3_id:
                    continue

                current_path = f"HY AI 데이터 > {l1_name} > {l2_name} > {l3_name}"
                print(f"\n--- {current_path} 탐색 중... ---")
                pdfs = await drive_manager.list_pdfs_in_folder(l3_id)
                if not pdfs:
                    print("  > 이 폴더에 PDF가 없습니다.")
                    continue

                for pdf_info in pdfs:
                    uris = await parse_single_drive_pdf_to_s3(drive_manager, pdf_info)
                    for k in collected_uris.keys():
                        if k in uris:
                            collected_uris[k].append(uris[k])

    return collected_uris


# ======================================
# RAW JSONL 읽기
# ======================================

def _read_raw_jsonl_from_s3(s3_uri: str) -> List[Dict[str, Any]]:
    """
    parse_pdf_with_all_parsers가 만든 RAW JSONL을 읽어서 row 리스트 반환.
    각 라인은 base_meta + parser + page_number + text 구조.
    """
    if not s3_uri.startswith("s3://"):
        raise ValueError("s3_uri는 s3:// 형태여야 합니다.")

    _, bucket_key = s3_uri.split("s3://", 1)
    bucket, key = bucket_key.split("/", 1)

    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read().decode("utf-8")

    rows: List[Dict[str, Any]] = []
    for line in body.splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows


# ======================================
# 섹션 / 헤더 / 푸터 / 타입 헬퍼
# ======================================

def _is_footer_like_line(line: str) -> bool:
    """푸터(섹션명 + 페이지 번호) 형태인지 검사."""
    s = line.strip()
    if not s:
        return False
    if FOOTER_LINE_RE.match(s):
        return True
    return False


def _strip_markup_for_morph(text: str, parser: Optional[str] = None) -> str:
    """
    형태소 분석 / 섹션 판별용 텍스트 정규화:
    - Upstage(HTML): 태그 제거 후 줄 유지
    - 공통: markdown / 각종 문법 기호 단순 정리
    - 끝에서 푸터(섹션명 + 페이지 번호) 라인은 제거
    """
    t = text or ""

    # 1) HTML → plain text (Upstage)
    if parser == "upstage":
        try:
            soup = BeautifulSoup(t, "html.parser")
            # <br> 등은 줄바꿈으로 유지
            for br in soup.find_all("br"):
                br.replace_with("\n")
            t = soup.get_text("\n")
        except Exception:
            t = re.sub(r"<[^>]+>", " ", t)

    # 2) 자잘한 HTML attribute 노이즈 제거
    t = re.sub(
        r'\b(rowspan|colspan|style|class|id)\s*=\s*"[^"]*"',
        " ",
        t,
        flags=re.IGNORECASE,
    )

    # 3) 푸터 라인 제거
    lines = []
    for line in t.splitlines():
        if _is_footer_like_line(line):
            continue
        lines.append(line)
    t = "\n".join(lines)

    # 4) Markdown / 문법 기호 정리 (문단 기호는 그대로 둠)
    #    개행은 남긴다 (첫 줄/여러 줄 구조 보려고)
    t = re.sub(r"[#*`\[\]\{\}_>]", " ", t)  # |, -, 숫자, ￭ 등은 남겨둠
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s*\n+", "\n", t)

    return t.strip()


def _normalize_section_title(raw: str) -> str:
    """
    섹션 제목 정규화:
    - 뒤에 붙은 ' 5' 같은 페이지 번호는 제거 (앞 부분에 다른 숫자가 없을 때만)
    """
    if not raw:
        return raw
    s = raw.strip()

    m = re.match(r"(.+?)\s+([0-9]{1,2})$", s)
    if m:
        prefix, num = m.groups()
        # prefix 안에 다른 숫자가 없으면 페이지 번호로 보고 제거
        if not re.search(r"\d", prefix):
            s = prefix.strip()
    return s


def _is_main_heading(first: str) -> bool:
    """대제목(SECTION_MAIN)에 해당하는 형태인지."""
    # 로마 숫자: Ⅰ. Ⅱ. ...
    if re.match(r"^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]+\.", first):
        return True
    # 제1장, 제 2장 ...
    if re.match(r"^제\s*\d+장", first):
        return True
    # 1. ..., 2. ... (맨 앞 번호)
    if re.match(r"^\d+\.\s+", first):
        return True
    # 영문 로마자 I. II. ...
    if re.match(r"^[IVXLC]+\.\s+", first):
        return True
    return False


def _is_sub_heading(line: str) -> bool:
    """
    소제목(SECTION_SUB) 후보 판정:
    - ￭ / ● / • 로 시작하는 짧은 문장
    - 가. / 나. / 다. ...
    - (1), (가), 1) 등 번호형
    ※ '-', '※' 는 노트/일반 불릿으로 취급하고 제외
    """
    if not line:
        return False

    s = line.lstrip()

    # 1) 불릿 기호 (※ 는 제외)
    if s.startswith(("￭", "●", "•")):
        return True

    # 2) 가. 나. 다. ...
    if re.match(r"^[가-힣]\.\s*", s):
        return True

    # 3) (가), (나), (1) ...
    if re.match(r"^\([가-힣0-9]+\)\s*", s):
        return True

    # 4) 1), 2) ...
    if re.match(r"^\d+\)\s*", s):
        return True

    # 5) (선택) 1. 2. 를 서브로도 보고 싶으면 여기 추가
    # if re.match(r"^\d+\.\s+", s):
    #     return True

    return False



def _infer_section_levels(plain_chunk: str) -> Tuple[Optional[str], Optional[str]]:
    """
    plain 텍스트(마크업 제거된 상태)를 보고 (section_main, section_sub) 추론.
    - 여러 줄을 훑으면서 main / sub 후보를 찾는다.
    - main: 위쪽 1~2줄에서만 찾고 (Ⅱ. 원서접수 및 전형 일정, 제1장 ...)
    - sub : 이후 라인들에서 ￭ / ● / • / (가) / 1) 등의 패턴을 찾는다.
    """
    if not plain_chunk:
        return (None, None)

    # 줄별로 쪼개서 양쪽 공백 제거
    raw_lines = [l.strip() for l in plain_chunk.splitlines() if l.strip()]
    if not raw_lines:
        return (None, None)

    main: Optional[str] = None
    sub: Optional[str] = None

    for idx, line in enumerate(raw_lines):
        # HTML 태그/마크다운 기호 최소 정리
        s = re.sub(r"<[^>]+>", " ", line)
        s = re.sub(r"[#*`\[\]\{\}]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        if not s:
            continue

        # 너무 길면 제목/소제목일 확률 낮으니 스킵
        if len(s) > HEADING_MAX_CHARS:
            continue

        # 1) main 후보: 위쪽 0~1번째 줄에서만 찾는다 (타이틀은 보통 위에 있음)
        if main is None and idx <= 1:
            if _is_main_heading(s):
                main = _normalize_section_title(s)
                # 계속 돌아서 sub도 같이 찾는다
                continue

        # 2) sub 후보: 아직 sub가 없을 때, 어디서든 찾을 수 있음
        if sub is None and _is_sub_heading(s):
            # 불릿/번호 제거 후 저장
            cleaned = re.sub(r"^[￭●•\s]+", "", s)
            cleaned = re.sub(r"^(\d+[\.\)]|\([^)]+\))\s*", "", cleaned)
            sub = _normalize_section_title(cleaned or s)
            # 첫 소제목만 사용 (그 뒤는 무시)
            continue

    return (main, sub)



def _infer_type(chunk: str, parser: Optional[str] = None) -> str:
    """텍스트/테이블/이미지 타입 추론."""
    text_lower = chunk.lower()

    if parser == "upstage":
        if any(tag in text_lower for tag in ("<table", "<tr", "<td", "<th")) \
           or 'rowspan="' in text_lower or 'colspan="' in text_lower:
            return "table"
        if "<img" in text_lower or "<figure" in text_lower:
            return "image"
        return "text"

    # llama / pymupdf4llm (markdown) 기준
    # 헤더 줄 + 구분선 있는 경우만 table로 본다
    if chunk.count("|") >= 4 and re.search(r"\|\s*[-:]+\s*\|", chunk):
        return "table"
    return "text"


def _is_heading_chunk(chunk_html: str, parser: Optional[str]) -> bool:
    """
    chunk_html가 '헤더 한 줄만' 있는 청크인지 판정.
    (길이/줄 수 기반 + 문단 패턴 기반)
    """
    plain = _strip_markup_for_morph(chunk_html, parser)
    if not plain:
        return False

    lines = [l.strip() for l in plain.splitlines() if l.strip()]
    if not lines:
        return False

    # 여러 줄이면 본문 섞였다고 보고 스킵
    if len(lines) > 1:
        return False

    first = lines[0]
    if len(first) > HEADING_MAX_CHARS:
        return False

    return _is_main_heading(first) or _is_sub_heading(first)


def _merge_heading_chunks(chunks: List[str], parser: Optional[str]) -> List[str]:
    merged: List[str] = []
    pending_header: Optional[str] = None

    for ch in chunks:
        if not ch.strip():
            continue

        if _is_heading_chunk(ch, parser):
            if pending_header is None:
                pending_header = ch
            else:
                pending_header = pending_header + "\n" + ch
        else:
            if pending_header:
                merged.append(pending_header + "\n" + ch)
                pending_header = None
            else:
                merged.append(ch)

    if pending_header:
        merged.append(pending_header)

    return merged

def _is_pure_heading_for_merge(plain_chunk: str) -> bool:
    plain = (plain_chunk or "").strip()
    if not plain:
        return False

    lines = [l.strip() for l in plain.splitlines() if l.strip()]
    if not lines:
        return False

    # 줄 수는 1~2줄 정도까지 허용
    if len(lines) > 2:
        return False

    # 한 줄이라도 너무 길면 제목으로 보기 힘듦
    if any(len(l) > HEADING_MAX_CHARS for l in lines):
        return False

    has_any = False
    for i, line in enumerate(lines):
        # 각 줄이 main 또는 sub 패턴이어야 "깨끗한 헤더 뭉치"로 인정
        if _is_main_heading(line) or _is_sub_heading(line):
            has_any = True
        else:
            return False

    return has_any



def _split_upstage_html_into_blocks(page_html: str):
    blocks = []
    last_end = 0
    table_pattern = re.compile(r"<table[\s\S]*?</table>", re.IGNORECASE)

    for m in table_pattern.finditer(page_html):
        start, end = m.start(), m.end()
        before = page_html[last_end:start]
        table_html = m.group(0)

        if before.strip():
            plain = _strip_markup_for_morph(before, parser="upstage")
            if _is_pure_heading_for_merge(plain):
                # 👉 헤더 덩어리로만 구성된 경우: 테이블과 합친다
                blocks.append(("table", before + table_html))
            else:
                blocks.append(("html", before))
                blocks.append(("table", table_html))
        else:
            blocks.append(("table", table_html))

        last_end = end

    tail = page_html[last_end:]
    if tail.strip():
        blocks.append(("html", tail))

    return blocks



# ======================================
# 2단계: RAW JSONL(S3) → 청킹 → ES 업로드용 Document
# ======================================

def _chunk_raw_rows_to_chunk_docs(raw_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped = defaultdict(list)
    for r in raw_rows:
        key = (r["drive_file_id"], r["file_name"])
        grouped[key].append(r)

    docs_to_upsert: List[Dict[str, Any]] = []

    for (file_id, file_name), rows in grouped.items():
        rows.sort(key=lambda x: x.get("page_number", 0))

        current_main: Optional[str] = None
        current_sub: Optional[str] = None

        # 🔥 파일 단위 pending 헤더 버퍼
        pending_header_html: Optional[str] = None
        pending_header_plain: Optional[str] = None

        for row in rows:
            text = (row.get("text") or "").strip()
            if not text:
                continue

            page_num = row.get("page_number")
            parser_name = row.get("parser")

            meta_base = {
                "file_name": file_name,
                "drive_file_id": file_id,
                "modified_time": row.get("modified_time"),
                "university": row.get("university"),
                "year": row.get("year"),
                "admission_type": row.get("admission_type"),
                "document_type": row.get("document_type"),
                "parser": parser_name,
                "page_number": page_num,
            }

            if parser_name == "upstage":
                blocks = _split_upstage_html_into_blocks(text)
            else:
                blocks = [("html", text)]

            chunk_idx_in_page = 0

            for block_type, block_html in blocks:
                if not block_html.strip():
                    continue

                if block_type == "table":
                    block_chunks = [block_html]
                else:
                    raw_chunks = text_splitter.split_text(block_html)
                    block_chunks = _merge_heading_chunks(raw_chunks, parser_name)

                for chunk_html in block_chunks:
                    if not chunk_html.strip():
                        continue

                    # 형태소/섹션 판정용 plain 텍스트
                    plain_for_logic = _strip_markup_for_morph(chunk_html, parser_name)
                    if not plain_for_logic:
                        continue

                    # 1️⃣ 먼저 타입 판정 (테이블이면 헤더로 쓰지 않음)
                    if block_type == "table":
                        ctype = "table"
                    else:
                        ctype = _infer_type(chunk_html, parser_name)

                    # 2️⃣ 순수 헤더만 있는 청크인지 체크
                    if ctype != "table" and _is_pure_heading_for_merge(plain_for_logic):
                        # 섹션 컨텍스트는 여기서 갱신
                        main_cand, sub_cand = _infer_section_levels(plain_for_logic)
                        if main_cand:
                            current_main = main_cand
                            current_sub = None
                        if sub_cand:
                            current_sub = sub_cand

                        # 그리고 문서는 만들지 않고, 다음 청크와 합치기 위해 버퍼에 저장
                        pending_header_html = chunk_html
                        pending_header_plain = plain_for_logic
                        continue  # ❗ 이 청크는 여기서 끝

                    # 3️⃣ 여기까지 왔다는 건 실제 내용 청크라는 뜻
                    #    만약 직전에 헤더만 있었으면 지금 청크와 합쳐서 하나로 만들기
                    if pending_header_html:
                        merged_html = pending_header_html + "\n" + chunk_html
                        merged_plain = pending_header_plain + "\n" + plain_for_logic
                        pending_header_html = None
                        pending_header_plain = None

                        chunk_text_for_doc = merged_html
                        plain_for_logic_effective = merged_plain
                    else:
                        chunk_text_for_doc = chunk_html
                        plain_for_logic_effective = plain_for_logic

                    # 4️⃣ 이 시점에 섹션 레벨을 한 번 더 보고(본문 청크가 새로운 헤더를 가질 수도 있으니까)
                    main_cand, sub_cand = _infer_section_levels(plain_for_logic_effective)
                    if main_cand:
                        current_main = main_cand
                        current_sub = None
                    if sub_cand:
                        current_sub = sub_cand

                    # 타입은 헤더가 붙었더라도 다시 한 번 안전하게 판정
                    if block_type == "table":
                        ctype_final = "table"
                    else:
                        ctype_final = _infer_type(chunk_text_for_doc, parser_name)

                    # 형태소 텍스트
                    text_morph = get_morph(plain_for_logic_effective) if plain_for_logic_effective else None

                    chunk_idx_in_page += 1
                    meta = {
                        **meta_base,
                        "chunk_number": chunk_idx_in_page,
                        "section_main": current_main,
                        "section_sub": current_sub,
                        "type": ctype_final,
                        "text_morph": text_morph,
                    }

                    doc = Document(page_content=chunk_text_for_doc, metadata=meta)
                    doc_id = generate_chunk_id_es(file_id, f"{page_num}_{chunk_idx_in_page}")
                    docs_to_upsert.append({"id": doc_id, "document": doc})

        print(f"✅ chunk docs 생성: {len(docs_to_upsert)}개")
    return docs_to_upsert



def upload_chunk_docs_to_es(
    client,
    index_name: str,
    docs_to_upsert: List[Dict[str, Any]],
):
    """
    docs_to_upsert (id + Document) 리스트를
    바로 Elasticsearch index에 임베딩 + bulk 인덱싱.
    """
    if not docs_to_upsert:
        print("❌ 업로드할 문서(chunk)가 없습니다.")
        return

    print(f"✅ Elasticsearch 업로드 시작: index={index_name}, docs={len(docs_to_upsert)}")

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # 임베딩 생성 (배치)
    texts = [row["document"].page_content for row in docs_to_upsert]
    all_vectors: List[List[float]] = []

    for i in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[i: i + EMBEDDING_BATCH_SIZE]
        vecs = embeddings.embed_documents(batch)
        all_vectors.extend(vecs)

    assert len(all_vectors) == len(docs_to_upsert)

    # bulk action 생성
    def action_generator():
        for row, vec in zip(docs_to_upsert, all_vectors):
            doc: Document = row["document"]
            _id = row["id"]
            yield {
                "_op_type": "index",
                "_index": index_name,
                "_id": _id,
                "_source": {
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "vector_field": vec,
                },
            }

    success, errors = bulk(
        client,
        actions=action_generator(),
        chunk_size=400,
        max_chunk_bytes=9 * 1024 * 1024,
        raise_on_error=False,
    )

    print("\n" + "=" * 50)
    print("🎉 데이터 적재 완료!")
    print(f"  - 성공: {success} 건")
    print(f"  - 실패: {len(errors)} 건")
    if errors:
        print("실패 내역 (최대 5개):", errors[:5])
    print("=" * 50)


# ======================================
# 3단계: ES 인덱스 생성 + RAW JSONL 기반 인덱싱
# ======================================

def create_parser_index(client, index_name: str):
    """
    parser별 전용 인덱스 스키마 생성.
    section 필드는 없애고 section_main / section_sub만 사용.
    """
    metadata_mapping = {
        "file_name": {"type": "keyword"},
        "drive_file_id": {"type": "keyword"},
        "modified_time": {"type": "date"},
        "university": {"type": "keyword"},
        "year": {"type": "integer"},
        "admission_type": {"type": "keyword"},
        "document_type": {"type": "keyword"},
        "section_main": {"type": "text", "analyzer": "standard"},
        "section_sub": {"type": "text", "analyzer": "standard"},
        "type": {"type": "keyword"},             # "text" / "table" / "image"
        "page_number": {"type": "integer"},
        "chunk_number": {"type": "integer"},
        "text_morph": {"type": "text", "analyzer": "kiwi_ws"},
        "parser": {"type": "keyword"},
    }

    schema_body = create_es_schema(metadata=metadata_mapping)
    build_elasticsearch_index(client, index_name, schema_body)


def index_all_parsers_to_es_from_raw_s3(raw_s3_uris: Dict[str, List[str]]):
    """
    RAW JSONL(S3)만을 활용해서:
      - 메모리에서 청킹
      - 바로 ES 인덱스에 업로드
    chunk JSONL은 생성/저장하지 않는다.
    """
    client = get_es_client()

    for parser_name, uris in raw_s3_uris.items():
        if parser_name == "upstage":
            index_name = "kgs_pdf_upstage"
        elif parser_name == "llamaparse":
            index_name = "kgs_pdf_llamaparse"
        elif parser_name == "pymupdf4llm":
            index_name = "kgs_pdf_pymupdf4llm"
        else:
            continue

        create_parser_index(client, index_name)

        for raw_uri in uris:
            print(f"\n=== {parser_name} → {index_name} 업로드 (RAW: {raw_uri}) ===")
            raw_rows = _read_raw_jsonl_from_s3(raw_uri)
            docs_to_upsert = _chunk_raw_rows_to_chunk_docs(raw_rows)
            upload_chunk_docs_to_es(client, index_name, docs_to_upsert)


# ======================================
# 전체 파이프라인 원샷 실행 (옵션)
# ======================================

async def run_full_pdf_ingestion_pipeline():
    """
    1) Drive → 3개 파서 → RAW JSONL(S3)
    2) RAW JSONL(S3) → 메모리 청킹 → ES 인덱스 업로드
    """
    # 1. 파서 실행 + RAW JSONL 저장
    raw_s3_uris = await run_parsers_for_all_drive_pdfs()

    # 2. RAW 기반으로 바로 ES 인덱싱
    index_all_parsers_to_es_from_raw_s3(raw_s3_uris)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="PDF → 파서 → S3(JSONL) → (옵션) 청킹+Elasticsearch 인덱싱 파이프라인"
    )
    parser.add_argument(
        "--step",
        choices=["parse-only", "full"],
        default="parse-only",
        help=(
            "parse-only: Google Drive에서 PDF 읽어서 파서 JSONL만 S3에 저장\n"
            "full: parse-only + 청킹해서 Elasticsearch 인덱싱까지 실행"
        ),
    )

    args = parser.parse_args()

    if args.step == "parse-only":
        asyncio.run(run_parsers_for_all_drive_pdfs())
    # else:
    #     asyncio.run(run_full_pdf_ingestion_pipeline())
