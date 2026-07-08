import os, asyncio, yaml
from pathlib import Path
from datetime import datetime
from .rag_pdf_parser import parse_pdf_with_all_parsers
from .rag_pdf_pipeline import _chunk_raw_rows_to_chunk_docs
from .s3_io import save_jsonl_to_s3

ROOT_DIR = Path(__file__).resolve().parents[1]
TEST_PDF = ROOT_DIR / "samples" / "고려대(서울)_2026_수시_모집요강.pdf"

with open(ROOT_DIR / "config" / "config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

S3_BUCKET = config["aws"]["s3"]["bucket"]
S3_RAW_PREFIX = "pdf_parsed"

async def main():
    file_name = TEST_PDF.name

    rows_by_parser = await parse_pdf_with_all_parsers(
        pdf_path=str(TEST_PDF),
        file_name=file_name,
        drive_file_id="dummy",
        modified_time="2025-01-01T00:00:00Z",
    )

    # === 여기 추가 ===
    date_str = datetime.now().strftime("%Y%m%d")
    short_map = {
        "upstage": "u",
        "llamaparse": "l",
        "pymupdf4llm": "p",
    }

    for parser_name, short in short_map.items():
        rows = rows_by_parser.get(parser_name, [])
        if not rows:
            continue
        s3_uri = save_jsonl_to_s3(
            db=S3_RAW_PREFIX,
            rows=rows,
            bucket=S3_BUCKET,
            s3_file_path=f"{short}_{file_name}_localtest_{date_str}",
        )
        print(f"{parser_name} RAW jsonl S3 URI:", s3_uri)

    # 아래는 청킹 테스트용 (원래 있던 코드)
    for name, rows in rows_by_parser.items():
        print(f"\n[{name}] raw rows:", len(rows))
        if not rows:
            continue
        docs = _chunk_raw_rows_to_chunk_docs(rows)
        print(f" -> chunks: {len(docs)}")
        if docs:
            print("    첫 chunk text[:200]:", docs[0]["document"].page_content[:200])
            print("    metadata:", docs[0]["document"].metadata)


if __name__ == "__main__":
    asyncio.run(main())