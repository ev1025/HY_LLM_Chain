from locust import HttpUser, task, between
import time

class ChatbotUser(HttpUser):
    wait_time = between(0.2, 0.5) # 0.2 ~ 0.5초 랜덤 대기 후 재요청 

    def on_start(self):
        self.common_headers = {
            "accept": "text/event-stream",
            "content-type": "application/json",
        }

    @task
    def chat_stream(self):
        start_time = time.time()
        first_chunk_received = False
        total_bytes = 0

        try:
            with self.client.post(
                "/chat-stream",
                json={"session_id": "test", 
                    "query": "이대 12년 특례 알려주세요"},
                headers=self.common_headers,
                stream=True,           # 중요
                catch_response=True,   # 수동 리포팅
                timeout=120,
            ) as resp:

                if resp.status_code != 200:
                    now_ms = (time.time() - start_time) * 1000.0
                    exc = Exception(f"HTTP {resp.status_code}")
                    # 첫 토큰 실패
                    self.environment.events.request.fire(
                        request_type="POST", name="stream",
                        response_time=now_ms, response_length=0, exception=exc
                    )
                    # 전체 실패
                    self.environment.events.request.fire(
                        request_type="POST", name="non_stream",
                        response_time=now_ms, response_length=0, exception=exc
                    )
                    return

                # === 핵심: SSE vs 비-SSE를 분기 ===
                content_type = (resp.headers.get("Content-Type") or "").lower()

                if content_type.startswith("text/event-stream"):
                    # SSE: 한 줄이 곧 한 이벤트. 줄단위, 최소 청크로 즉시 읽기
                    iterator = resp.iter_lines(chunk_size=1, decode_unicode=False)
                else:
                    # 일반 청크 스트림: 첫 바이트를 바로 반환하도록 chunk_size=1
                    iterator = resp.iter_content(chunk_size=1, decode_unicode=False)

                for chunk in iterator:
                    if not chunk:
                        continue

                    # 누적 바이트
                    total_bytes += len(chunk)

                    # 첫 토큰(TTFT)
                    if not first_chunk_received:
                        first_chunk_received = True
                        ttft_ms = (time.time() - start_time) * 1000.0
                        self.environment.events.request.fire(
                            request_type="POST",
                            name="stream",                 # ✅ TTFT
                            response_time=ttft_ms,
                            response_length=len(chunk),
                            exception=None,
                        )

                # 전체 완료(E2E)
                total_ms = (time.time() - start_time) * 1000.0
                self.environment.events.request.fire(
                    request_type="POST",
                    name="non_stream",                 # ✅ 전체 완료
                    response_time=total_ms,
                    response_length=total_bytes,
                    exception=None,
                )

        except Exception as e:
            now_ms = (time.time() - start_time) * 1000.0
            if not first_chunk_received:
                # 첫 토큰 전에 터지면 stream도 실패로
                self.environment.events.request.fire(
                    request_type="POST", name="stream",
                    response_time=now_ms, response_length=0, exception=e
                )
            # non_stream은 항상 기록
            self.environment.events.request.fire(
                request_type="POST", name="non_stream",
                response_time=now_ms, response_length=0, exception=e
            )
