# app.py (TTFT 및 전체 시간 측정)
# streamlit run app.py
import os, uuid, json, requests, streamlit as st
import time  # ⬅️ 시간 측정을 위해 추가

API_URL = os.getenv("RAG_API_URL", "http://localhost:8000/chat")

st.set_page_config(page_title="HY RAG Chat", page_icon="💬", layout="centered")
st.title("💬 HY RAG Chat")

# 세션 상태
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 메시지 렌더
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

def stream_response(prompt: str):
    full = ""
    with requests.post(
        API_URL,
        json={"session_id": st.session_state.session_id, "query": prompt},
        stream=True,
        timeout=300,
    ) as r:
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        r.encoding = "utf-8"

        # 1) SSE 우선
        if "text/event-stream" in ctype:
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.startswith("data:"):
                    delta = line[5:]
                    full += delta
                    yield full
            return

        # 2) 그 외의 스트림(plain/json 청크)도 최대한 실시간으로
        buff = ""
        for chunk in r.iter_content(chunk_size=1, decode_unicode=True):
            if not chunk:
                continue
            buff += chunk
            full += chunk
            yield full
        # 단건 JSON 응답 fallback
        try:
            obj = json.loads(buff)
            if isinstance(obj, dict):
                full = obj.get("answer") or obj.get("delta") or full
        except Exception:
            pass
    yield full

# 입력창
if prompt := st.chat_input("질문을 입력하세요…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        assembled = ""
        
        start_time = time.time()      # ⬅️ 1. API 요청 시작 시간
        first_token_time = None     # ⬅️ 2. 첫 토큰 수신 시간을 기록할 변수
        
        try:
            # 스트리밍 응답 표시
            for partial in stream_response(prompt):
                
                # ⬇️ 3. 루프가 처음 실행된 순간이 첫 토큰을 받은 시간
                if first_token_time is None:
                    first_token_time = time.time()
                    
                assembled = partial
                placeholder.markdown(assembled + "▌")
            
            end_time = time.time()  # ⬅️ 4. 응답 완료 시간 측정
            
            # 5. 시간 계산
            duration = end_time - start_time
            ttft = (first_token_time - start_time) if first_token_time else 0.0
            
            # ⬅️ 최종 응답에 TTFT와 전체 시간 포함
            final_response = f"{assembled}\n\n---\n<small>*(첫 응답: {ttft:.2f}초 / 전체: {duration:.2f}초)*</small>"
            placeholder.markdown(final_response, unsafe_allow_html=True)
        
        except requests.exceptions.RequestException as e:
            end_time = time.time()  # ⬅️ 오류 발생 시간 측정
            duration = end_time - start_time
            
            # 오류 발생 시에도 TTFT가 기록되었다면 함께 표시
            ttft_msg = ""
            if first_token_time:
                ttft = first_token_time - start_time
                ttft_msg = f"(첫 응답: {ttft:.2f}초)"
                
            error_message = f"서버 연결 오류: {e}\n\n(시도 시간: {duration:.2f}초 {ttft_msg})"
            placeholder.error(error_message)
            assembled = "답변 수신 실패"

    st.session_state.messages.append({"role": "assistant", "content": assembled})