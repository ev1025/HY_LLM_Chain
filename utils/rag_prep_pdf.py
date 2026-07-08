from dotenv import load_dotenv # langsmith 사용을 위해 가장 위에 삽입
load_dotenv()

import os, yaml,json
from typing import Tuple, Optional, Set , Dict, Callable, Any
import asyncio
import fitz  # PyMuPDF
import time
import http.client

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ===================================================
# config에서 구글드라이브의 폴더, 파일 정보 받아오기
# ===================================================
with open('config/config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# 기준 폴더의 고유번호 및 각 폴더 이름 추출
BASE_FOLDER_ID = config['google_drive']['base_folder_id']
LEVEL1_FOLDERS = config['google_drive']['folder_hierarchy']['level1']
LEVEL2_FOLDERS = config['google_drive']['folder_hierarchy']['level2']
LEVEL3_FOLDERS = config['google_drive']['folder_hierarchy']['level3']

# 구글드라이브 연동 정보
SCOPES = config['google_drive']['scopes']
GCP_API_KEY = os.environ.get('GCP_API_KEY')

if GCP_API_KEY:
    gcp_key_info = json.loads(GCP_API_KEY)
    creds = Credentials.from_service_account_info(gcp_key_info, scopes=SCOPES)
else:
    creds = Credentials.from_service_account_file('hygoogle-service-key.json', scopes=SCOPES)

# ===================================================
# Google Drive 관리 클래스
# ===================================================
class GoogleDriveManager:
    def __init__(self):
        """클래스 초기화 시 서비스 계정으로 Google Drive API에 연결합니다."""
        try:
            self.service = build('drive', 'v3', credentials=creds)
            print("✅ Google Drive 서비스 연결 성공")
        except Exception as e:
            self.service = None
            print(f"❌ 오류: Google Drive 서비스 연결 실패. ({e})")

    async def _execute_paged_list(self, **kwargs):
        """
        [구글드라이브 권한 확장]
        Drive files.list를 페이지네이션하며 모든(Shared) 드라이브까지 포함해 결과를 모읍니다.
        """
        if not self.service: return []
        
        kwargs.setdefault('supportsAllDrives', True)
        kwargs.setdefault('includeItemsFromAllDrives', True)
        kwargs.setdefault('corpora', 'allDrives')
        
        loop = asyncio.get_running_loop()
        files, token = [], None
        while True:
            if token: kwargs['pageToken'] = token
            resp = await loop.run_in_executor(None, lambda: self.service.files().list(**kwargs).execute())
            files.extend(resp.get('files', []))
            token = resp.get('nextPageToken')
            if not token: break
        return files

    async def find_folder_id(self, folder_name: str, parent_id: str) -> Optional[str]:
        """
        [구글드라이브 폴더 탐색]
        부모 폴더(parent_id) 아래에서 지정한 이름(folder_name)의 하위 폴더 ID를 찾습니다.
        """
        try:
            safe_name = folder_name.replace("'", "\\'")
            query = f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and name='{safe_name}' and trashed=false"
            items = await self._execute_paged_list(q=query, fields='files(id)')
            return items[0]['id'] if items else None
        except HttpError as e:
            print(f"오류: 폴더 '{folder_name}'를 찾는 중 오류 발생: {e}")
            return None

    async def list_pdfs_in_folder(self, folder_id: str) -> list:
        """
        [구글드라이브 내 파일 정보 수집(modifiedTime)]
        특정 폴더 ID 하위의 PDF 파일들을 메타데이터 포함하여 나열합니다.
        """
        if not folder_id: return []
        try:
            query = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
            fields = "nextPageToken, files(id, name, modifiedTime)"
            return await self._execute_paged_list(q=query, orderBy="modifiedTime desc", fields=fields)
        except HttpError as e:
            print(f"오류: PDF 목록을 가져오는 중 오류 발생: {e}")
            return []

    async def download_file(self, file_id: str) -> Optional[bytes]:
            """파일 ID를 사용하여 파일을 다운로드하고 바이트(bytes)를 반환합니다."""
            if not self.service: return None
            
            # IncompleteRead 오류에 대비한 재시도 로직
            for i in range(3):
                try:
                    request = self.service.files().get_media(fileId=file_id)
                    loop = asyncio.get_running_loop()
                    return await loop.run_in_executor(None, request.execute)
                except http.client.IncompleteRead as e:
                    wait_time = 2 ** i
                    print(f"  > 경고: '{file_id}' 다운로드 중 IncompleteRead 오류. {wait_time}초 후 재시도... ({i+1}/3)")
                    time.sleep(wait_time)
                except HttpError as e:
                    print(f"  > 오류: 파일(id: {file_id}) 다운로드 중 HttpError 발생: {e}")
                    return None
            
            print(f"  > 오류: 파일(id: {file_id}) 다운로드에 3번 실패했습니다.")
            return None


# ===================================================
# 파일이름을 파싱해 메타데이터 생성
# ===================================================
def filename_to_metadata(filename: str) -> dict:
    """
    '대학교_연도_전형_세부사항.pdf' 파일명을 파싱해 메타데이터 딕셔너리를 생성합니다.
    """
    base_name = filename.replace('.pdf', '')
    parts = base_name.split('_')
    if len(parts) < 4: return {"source": filename}
    
    uni, year_str, adm_type, doc_type = parts[0], parts[1], parts[2], parts[3]
    try:
        year = int(year_str)
    except ValueError:
        year = 0
    return {"source": filename, "university": uni, "year": year, "admission_type": adm_type, "document_type": doc_type}



text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=1600, 
  chunk_overlap=200, 
  is_separator_regex=True,
  add_start_index=True,
  separators=[r"\n#{1,3}\s", r"\n-{3,}\n", r"\n[Ⅰ-ⅩV]+\.\s?", r"\n\d+\.\s",
              r"\n▪|\n◦|\n•|\n- ", r"\n{2,}", r"(?<=[\.!?])\s+\n", r"\n", r"\s"],
  
)
async def process_pdf_to_docs(
    drive_manager: GoogleDriveManager, 
    pdf_info: dict,
    generate_chunk_id_func: Callable[[str, int], str], # DB별 ID 생성기
    get_morph_func: Optional[Callable[[str], str]] = None # DB별 형태소 분석기 (선택적)
) -> list:
    """
    Google Drive에서 PDF를 다운로드해 텍스트 추출 후,
    DB별 전략 함수를 사용하여 Chunk 분할 및 LangChain Document 목록으로 반환합니다.
    """
    file_id, file_name, modified_time = pdf_info['id'], pdf_info['name'], pdf_info.get('modifiedTime')
    
    pdf_bytes = await drive_manager.download_file(file_id)
    if not pdf_bytes:
        print(f"  ❌ 오류: '{file_name}' 다운로드 실패. 건너뜁니다.")
        return []

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            full_text = "\n".join(page.get_text("text") or "" for page in doc)
        
        chunks = text_splitter.split_text(full_text)
        docs = []
        parsed_metadata = filename_to_metadata(file_name)
        
        for i, chunk in enumerate(chunks):
            # DB별 ID 생성기 호출
            doc_id = generate_chunk_id_func(file_id, i + 1)
            
            final_metadata = {
                "chunk_number": i + 1,
                "drive_file_id": file_id,
                "modified_time": modified_time,
                **parsed_metadata,
            }
            
            # 형태소 분석기(예: Weaviate)가 제공된 경우에만 실행
            if get_morph_func:
                final_metadata["text_morph"] = get_morph_func(chunk)
                
            docs.append({"id": doc_id, "document": Document(page_content=chunk, metadata=final_metadata)})

        print(f"  ✅ '{file_name}'에서 {len(docs)}개의 Document 생성")
        return docs
    except Exception as e:
        print(f"  ❌ 오류: '{file_name}' 처리 중 문제 발생: {e}")
        return []

# ===================================================
# pdf chunking 파이프라인
# ===================================================
async def prepare_pdf_data(
    get_ids_func: Callable[[], Tuple[Set[str], Dict[str, Any]]],
    need_update_func: Callable[[str, Any], bool],
    generate_chunk_id_func: Callable[[str, int], str],
    get_morph_func: Optional[Callable[[str], str]] = None
) -> Tuple[list, list]:
    """
    [파이프라인 실행 main함수]
    DB 전략 함수들을 주입받아 전체 폴더 계층을 순회하며
    신규/변경된 파일을 처리하고 삭제 대상을 식별합니다.
    """
    
    # 1. DB별 함수 호출
    existing_ids, id_to_mtime = get_ids_func()
    
    drive_manager = GoogleDriveManager()
    if not drive_manager.service:
        return [], []

    docs_to_upsert = []
    drive_ids = set()

    for l1_name in LEVEL1_FOLDERS:
        l1_id = await drive_manager.find_folder_id(l1_name, BASE_FOLDER_ID)
        if not l1_id: continue
        for l2_name in LEVEL2_FOLDERS:
            l2_id = await drive_manager.find_folder_id(l2_name, l1_id)
            if not l2_id: continue
            for l3_name in LEVEL3_FOLDERS:
                l3_id = await drive_manager.find_folder_id(l3_name, l2_id)
                if not l3_id: continue
                
                current_path = f"HY AI 데이터 > {l1_name} > {l2_name} > {l3_name}"
                print(f"\n--- {current_path} 탐색 중... ---")
                
                pdfs = await drive_manager.list_pdfs_in_folder(l3_id)
                if not pdfs:
                    print("  > 이 폴더에 PDF 파일이 없습니다.")
                    continue
                
                to_process = []
                for p in pdfs:
                    drive_ids.add(p['id'])
                    # 2. DB별 함수 호출
                    if p['id'] not in existing_ids or need_update_func(p.get('modifiedTime'), id_to_mtime.get(p['id'])):
                        to_process.append(p)
                        
                if not to_process:
                    print(f"❌ 처리할 신규/변경 파일이 없습니다. (총 {len(pdfs)}개 파일 최신)")
                else:
                    print(f"✅ 총 {len(to_process)}건의 신규/변경 파일 처리 시작")
                    for pdf in to_process:
                        # 3. DB별 함수 전달
                        doc_list = await process_pdf_to_docs(
                            drive_manager, 
                            pdf,
                            generate_chunk_id_func,
                            get_morph_func
                        )
                        docs_to_upsert.extend(doc_list)

    delete_ids = list(existing_ids - drive_ids)
    
    if len(docs_to_upsert) > 0:
        print(f"\n총 {len(docs_to_upsert)}개의 Document(청크)를 생성했습니다.")
    else:
        print("\n❌ 추가/업데이트할 문서가 없습니다.")


    return docs_to_upsert, delete_ids