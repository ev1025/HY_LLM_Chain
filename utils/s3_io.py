import os, json, tempfile, boto3
from typing import  Iterable, Any, Dict, Tuple, List
from langchain_core.documents import Document
from datetime import datetime, timezone



s3 = boto3.client("s3", region_name=os.getenv("AWS_REGION","ap-northeast-2"))

def save_jsonl_to_s3(db : str, rows: Iterable[Dict[str, Any]], bucket: str, s3_file_path:str) -> str:
    """
    dict형태로 정리된 Document를 JSONL 형식으로 임시 파일에 기록한 뒤,
    지정한 S3 버킷/키로 업로드하고 최종 S3 URI를 반환합니다.

    Args:
        rows (Iterable[Dict[str, Any]]): dict형태로 정리된 Document
        bucket (str): 사용할 S3 버킷 이름 (예: "hy-rag")
        key (str): S3에 저장할 파일 경로 (예: "susi/pdf_embedding.jsonl")

    Returns:
        str: 업로드된 객체의 S3 URI(예: "s3://hy-rag/susi/pdf_embedding.jsonl").

    """
    # dict파일을 임시파일을 jsonl로 생성 
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as f:
        tmp = f.name
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    s3_key = f"{db}/{s3_file_path}_{today}.jsonl"  # 실행날짜로 S3에 jsonl 파일 생성


    # 임시로 생성한 파일 S3로 업로드 후 임시파일 삭제
    s3.upload_file(tmp, bucket, s3_key)
    os.remove(tmp)
    
    return f"s3://{bucket}/{s3_key}"



def load_jsonl_from_s3(s3_uri: str) -> Tuple[List[Document], List[str]]:
    """
    s3://bucket/key 에 있는 JSONL을 읽어 [Document], [id] 반환
    각 라인은 {"id": "...", "page_content": "...", "metadata": {...}} 형태여야 함.
    """
    if not s3_uri.startswith("s3://"):
        raise ValueError("INPUT_FILE_PATH는 s3:// 형태의 S3 URI여야 합니다.")

    _, bucket_key = s3_uri.split("s3://", 1)
    bucket, key = bucket_key.split("/", 1)

    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read().decode("utf-8")

    docs: List[Document] = []
    ids: List[str] = []
    for line in body.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        docs.append(Document(page_content=row["page_content"], metadata=row["metadata"]))
        ids.append(row["id"])
    return docs, ids


def chunk_to_jsonl(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    생성된 chunk 쌍을 List 안에 dict형태로 넣습니다.
    ID, 페이지 내용(page_content), 메타데이터(metadata)만 추출하여 
    직렬화 가능한(serializable) 리스트 형태로 변환합니다.

    Args:
        all_docs: 각 항목이 'id'와 'document' (Document 객체)를 포함하는 리스트.

    Returns:
        ID, page_content, metadata만 포함하는 직렬화된 딕셔너리 리스트.
    """
    serializable_docs = [
        {
            "id": row["id"],
            "page_content": row["document"].page_content,
            "metadata": row["document"].metadata
        }
        for row in data
    ]
    return serializable_docs