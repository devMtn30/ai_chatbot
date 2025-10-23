# 프로젝트 소개

이 문서는 LangChain 에이전트/툴 스택으로 재구성된 ShotGrid 챗봇의 구조를 개발자 관점에서 정리합니다. FastAPI, LangChain, ShotGrid REST API, 선택형 LLM(예: OpenAI, Ollama)을 결합해 제작팀 질문에 데이터를 기반으로 응답하거나 후속 액션을 안내하는 것이 목표입니다.

## 시스템 개요
- **엔드포인트**: `main.py`는 `/chat` POST 엔드포인트를 노출하고 JSON 페이로드를 수신합니다. 요청마다 UUID 기반 `thread_id`를 생성해 실행 로그에 활용합니다.
- **서비스 계층**: `app/services/chat_service.py`의 `run_chat` 함수가 단일 진입점 역할을 하며, LangChain `AgentExecutor`를 호출하고 도구 실행 로그를 구조화합니다.
- **에이전트**: `app/agents/chat_agent.py`는 시스템 프롬프트와 툴 세트를 이용해 LangChain `create_tool_calling_agent`로 에이전트를 초기화합니다. 초기화된 executor는 LRU 캐시에 저장되어 요청마다 재사용됩니다.
- **LLM 어댑터**: `app/llm/__init__.py`는 환경 변수(`LLM_PROVIDER`, `LLM_MODEL` 등)에 따라 OpenAI(`ChatOpenAI`) 또는 Ollama(`ChatOllama`)를 생성합니다. 향후 다른 공급자를 추가하고 싶다면 이 모듈에 분기 로직을 더하면 됩니다.
- **툴**: `app/tools/shotgrid.py`는 ShotGrid 관련 API를 LangChain 구조화 툴로 노출합니다. 각 툴은 Pydantic 스키마로 인자를 검증하고, httpx 기반 비동기 래퍼(`shotgrid_client.py`)를 호출한 뒤 JSON 문자열을 반환합니다.
- **ShotGrid 클라이언트**: `shotgrid_client.py`는 GET 엔드포인트 호출과 쿼리스트링 구성, 이메일 정규화를 담당합니다.

## 요청 처리 흐름
1. 클라이언트가 `{ "chatInput": "<질문>" }` 형태로 요청을 전송합니다.
2. `main.py`는 `run_chat`을 호출하고 thread 태그를 LangChain 실행 설정으로 전달합니다.
3. 에이전트는 프롬프트를 기반으로 답변을 계획하고 필요 시 ShotGrid 툴을 호출합니다. 툴 반환값은 JSON 문자열이며, 서비스 계층이 다시 파싱해 구조화합니다.
4. 최종 응답에는 자연어 `reply`, 툴 실행 로그(`tool_calls`), 에러 문자열(없으면 `None`)이 포함됩니다. 호환성을 위해 `api_response`, `intent`, `suggest_json`도 반환하지만 현재는 각각 도구 실행 내역, `None`, `None`을 제공합니다.

## 툴 목록
- `get_versions_feedback`: 업로더별 최신 버전 피드백 상태 조회.
- `get_versions_by_status`: 버전 상태 코드 기준 목록 조회 (`sckOnly` 필터 지원).
- `get_versions_by_shot_status`: 샷 상태 기준 버전 목록 조회.
- `get_projects_progress`: 프로젝트 진행률 요약.
- `get_team_workload`: 프로젝트/스텝별 팀 작업량 조회 (`team` 별칭 허용).
- `get_most_urgent_task`: 사용자에게 가장 긴급한 ShotGrid 태스크 조회.
- `get_version_raw`: 버전 ID 기반 원본 데이터 조회.

각 툴은 실행 시간과 HTTP 상태를 로깅하며, 실패 시 `{"error": "..."}`
형태의 JSON을 반환해 에이전트가 적절한 안내를 할 수 있도록 합니다.

## 환경 설정
- `.env`에 `LLM_PROVIDER`, `LLM_MODEL`, `LLM_TEMPERATURE`, `OPENAI_API_KEY`(OpenAI 사용 시), `OLLAMA_BASE_URL`(Ollama 사용 시) 등을 정의합니다.
- ShotGrid 접근 정보는 `SG_BASE_URL`, `DEFAULT_EMAIL_DOMAIN`으로 지정합니다.
- 의존성 설치는 `pip install -r requirements.txt`, 개발 서버는 `uvicorn main:app --reload`로 실행합니다.

## 확장 전략
- **새 툴 추가**: `shotgrid_client.py`에서 HTTP 래퍼를 구현한 뒤 `app/tools/shotgrid.py`에 Pydantic 스키마와 LangChain 툴을 등록합니다. 에이전트 프롬프트가 새 기능을 인지하도록 필요 시 `SYSTEM_PROMPT`를 업데이트합니다.
- **추가 LLM 지원**: `app/llm/__init__.py`에 분기를 추가하고, 필요한 경우 requirements.txt에 공급자별 LangChain 패키지를 명시하십시오.
- **기능 테스트**: LangChain의 `FakeListChatModel`을 통해 LLM 응답을 고정하면 에이전트 로직을 결정론적으로 검증할 수 있습니다. ShotGrid 호출은 httpx의 MockTransport 또는 VCR 스타일 녹화를 활용하세요.

## 향후 과제 아이디어
- 에이전트 실행 결과를 Observability 스택(Prometheus, OpenTelemetry 등)에 전송해 툴 호출 빈도와 실패율을 모니터링합니다.
- 자주 쓰는 질문 패턴을 캐싱하거나, LLM 호출 전 룰 기반 필터를 적용해 비용을 절감합니다.
- 다중 턴 지원이 필요할 경우 `run_chat`에 대화 히스토리를 주입하고, LangChain 메모리 컴포넌트를 활용해 세션 상태를 보존하는 방안을 검토합니다.
