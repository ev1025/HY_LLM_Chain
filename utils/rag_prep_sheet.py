from dotenv import load_dotenv # langsmith 사용을 위해 가장 위에 삽입
load_dotenv()

import pandas as pd
import yaml, os, json
from typing import List, Dict, Set, Any, Tuple, Callable, Optional
import datetime

import gspread
from google.oauth2.service_account import Credentials
from langchain_core.documents import Document

with open('config/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 구글시트 URL 및 Sheet name 추출
SPREADSHEET_URL = config['google_drive']['kgs_sheet']
SHEET_NAME = "특례 질문정리"

GCP_API_KEY = os.environ.get('GCP_API_KEY')
SCOPES = config['google_drive']['scopes']

if GCP_API_KEY:
    key_info = json.loads(GCP_API_KEY)
    creds = Credentials.from_service_account_info(key_info, scopes=SCOPES)
else:
    creds = Credentials.from_service_account_file('hygoogle-service-key.json', scopes=SCOPES)

def get_gsheet_data() -> pd.DataFrame:
    """
    Google Sheets에서 데이터를 읽어 DataFrame으로 반환합니다.
    """
    client = None
    try:
        client = gspread.authorize(creds)
        spreadsheet = client.open_by_url(SPREADSHEET_URL)
        worksheet = spreadsheet.worksheet(SHEET_NAME)
        data = worksheet.get_all_records()
        df = pd.DataFrame(data)
        df = df.fillna('') # NaN 값을 빈 문자열로 변환
        
        print(f"✅ Google Sheets에 총 {len(df)}개의 데이터가 보관되어 있습니다.")
        return df
    except Exception as e:
        print(f"❌ Google Sheets 데이터 로딩 중 오류 발생: {e}")
        return pd.DataFrame()

    finally:
        # 클라이언트 연결 종료
        if client and hasattr(client, 'session') and hasattr(client.session, 'close'):
            client.session.close()

# ===================================================
# 공통 Sheet 처리 파이프라인
# ===================================================
def prepare_sheet_data(
    get_existing_doc_ids_func: Callable[[], Set[str]],
    generate_sheet_id_func: Callable[[str], str],
    get_morph_func: Optional[Callable[[str], str]] = None
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Google Sheets 데이터를 가져와 DB 데이터와 비교 후
    업로드 및 삭제 대상을 결정하여 반환합니다.
    """
    gsheet_df = get_gsheet_data()

    # 1. DB별 함수 호출
    existing_doc_ids = get_existing_doc_ids_func()

    docs_to_upsert: List[Dict[str, Any]] = []
    sheet_ids: Set[str] = set() # 현재 GSheet에 존재하는 문서 ID 집합

    # 구글시트 'GPT답변 판정'열을 기준으로 데이터 추출
    for _, row in gsheet_df.iterrows():
        if row.get('GPT답변 판정') not in ['정확', '최종'] and not row.get('HY AI 답변'):
            continue
        
        if row.get('GPT답변 판정') != '최종':
            if row.get('GPT답변 판정') == '정확':
                page_content = str(row.get('GPT 답변 (무료 버전 기준 - 심층보고 X)', ''))
            else:
                page_content = str(row.get('HY AI 답변', '')).strip()
            parts = ["### 전문가 답변", page_content]
            extra = str(row.get('게시물 답변내용', '')).strip()
            if extra:
                parts += ["", "### 기타 첨언", extra]
            page_content = "\n".join(parts)
        else:
            page_content = str(row.get('HY AI 답변', '')).strip()
        
        page_content = f"{page_content}".strip()
        question = str(row.get('질문', '')).strip()
        year_str = str(row.get('연도', ''))
        try:
            year = int(year_str)
        except ValueError:
            current_year = datetime.datetime.now().year
            year = current_year+1
        university = str(row.get('학교', ''))
        category1 = str(row.get('분류1', ''))
        category2 = str(row.get('분류2', ''))
        category3 = str(row.get('분류3', ''))

        # ID 생성을 위한 원본 문자열 조합
        id_string = (
            f"{question}|"
            f"{page_content}|"
            f"{year}|{university}|{category1}|{category2}|{category3}"
        )

        # 2. DB별 ID 생성기(해시 함수) 호출
        doc_id = generate_sheet_id_func(id_string)
        sheet_ids.add(doc_id)
        
        # 신규 ID인 경우에만 추가
        if doc_id not in existing_doc_ids:
            _metadata = {
                "year": year, 
                "university": university,
                "category1": category1,
                "category2": category2, 
                "category3": category3,
                "question": question,
            }

        # 3. DB별 형태소 분석기를 호출하여 필드를 *분리* 생성
            if get_morph_func:
                if question:
                    _metadata["question_morph"] = get_morph_func(question)
                
                if page_content:
                    _metadata["text_morph"] = get_morph_func(page_content) 
            else:
                if question:
                    _metadata["question_morph"] = question
                if page_content:
                    _metadata["text_morph"] = page_content


            docs_to_upsert.append({
                "id": doc_id, 
                "document": Document(page_content=page_content, metadata=_metadata)
            })

    # 삭제 대상 ID 계산
    delete_ids = list(existing_doc_ids - sheet_ids)

    if len(docs_to_upsert)+len(delete_ids)>0:
        print(f"✅ 추가/업데이트 대상 {len(docs_to_upsert)}건, 삭제 대상 {len(delete_ids)}건 확인.")
    else:
        print("❌ 추가/업데이트할 문서가 없습니다.")

    return docs_to_upsert, delete_ids