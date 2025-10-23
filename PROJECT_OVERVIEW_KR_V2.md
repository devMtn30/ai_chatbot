# 프로젝트 리팩터링 보고서 (LangGraph → LangChain)

이 문서는 기존 LangGraph 기반 ShotGrid 챗봇을 LangChain Agent/Tool 스택으로 전환한 후의 구조와 변화를 개발자 관점에서 상세히 설명합니다. 변경 배경, 설계 의도, 주요 코드 차이, 기대 효과를 전부 포함해 추후 확장 및 유지보수 시 참고 자료로 사용할 수 있도록 구성했습니다.

---

## 1. 변경 배경과 목표

### 1.1 기존 구조(LangGraph)의 한계
- **노드 기반 제어 흐름의 복잡성**: `graph.py`에서 상태 타입(TypedDict), 노드 정의, 조건부 라우팅을 모두 수작업으로 유지해야 했습니다. 액션이 늘어날수록 LangGraph 그래프 구조도 함께 수정해야 하므로 관리 비용이 증가했습니다.
- **LLM 호출 중복**: 의도 분류(`intents.py`)와 답변 생성 노드가 별도의 OpenAI 호출을 수행했습니다. 지원되지 않는 요청의 경우 미지원 안내를 위해 추가 LLM 호출이 발생해 비용·레이턴시 부담이 있었습니다.
- **모델 교체의 어려움**: OpenAI Chat Completions를 전제로 작성돼 있어 향후 Ollama나 사내 모델을 도입할 때 코드 변경 범위가 크고 프롬프트 재사용성이 낮았습니다.
- **확장성 제약**: LangGraph는 상태 머신을 커스터마이징하기 좋지만, API 호출을 도구처럼 등록하고 조합하려면 추가적인 래퍼 코드를 작성해야 했습니다. 새 ShotGrid 기능을 붙일 때마다 분기 로직과 상태 업데이트를 직접 다뤄야 했습니다.

### 1.2 리팩터링 목표
1. **LangChain Agent/Tool 스택 도입**으로 도구 호출과 LLM 연계를 일관된 추상화 위에서 처리.
2. **LLM 공급자 분리**로 OpenAI, Ollama 등 다양한 모델을 교체해도 서비스 코드는 수정하지 않도록 모듈화.
3. **코드 가독성과 유지보수성 향상**: 서비스, 에이전트, 도구, LLM 설정을 명확히 분리하고, Pydantic 스키마로 툴 인자를 검증.
4. **중복 제거 및 비용 절감**: 단일 에이전트 흐름으로 LLM 호출 횟수를 최소화하고, 필요 시 툴 호출만 수행.

---

## 2. 전체 아키텍처 비교

| 항목 | 리팩터링 이전 | 리팩터링 이후 |
| --- | --- | --- |
| LLM 흐름 | 의도 분류 → 지원 여부 조건 분기 → API 호출 → 응답 합성 | LangChain Agent가 질문을 해석하고 필요한 툴을 호출 |
| 상태 관리 | LangGraph `StateGraph`, TypedDict 기반 상태 객체 | LangChain `AgentExecutor`와 `intermediate_steps`를 이용한 자연스러운 흐름 |
| LLM 추상화 | OpenAI Chat Completions에 직접 의존 | `app/llm` 모듈에서 공급자별로 Chat 모델 생성 (OpenAI, Ollama 등) |
| ShotGrid 연동 | LangGraph 노드 내부에서 httpx 호출 | LangChain 도구(`@tool`)로 정의, 에이전트가 자동 호출 |
| FastAPI 응답 | `intent`, `api_response`, `suggest_json` 중심 | `reply`, `tool_calls` 중심 (기존 필드는 하위 호환으로 유지 값은 `None`) |
| 파일 구조 | 주로 루트에 그래프/의도 파일 집중 | `app/` 패키지로 서비스·에이전트·도구·LLM 구성 요소 분리 |

---

## 3. 핵심 변경 사항 요약

1. **LangGraph 제거**: `graph.py`, `intents.py` 삭제. 그래프 노드 기반 라우팅 대신 LangChain 에이전트가 모든 요청을 처리합니다.
2. **패키지 구조 도입**: `app/` 디렉토리 신설 (`agents`, `services`, `tools`, `llm`). 역할별 모듈로 코드 정리.
3. **LangChain 도구 작성**: 기존 ShotGrid API 호출을 LangChain 툴로 변환하고, Pydantic 스키마를 이용해 입력 검증 및 문서화를 동시에 수행.
4. **LLM 추상화 레이어**: 환경 변수(`LLM_PROVIDER`, `LLM_MODEL`, `LLM_TEMPERATURE`, `OPENAI_API_KEY`, `OLLAMA_BASE_URL`)를 바탕으로 Chat 모델을 선택.
5. **서비스 계층 정리**: FastAPI는 이제 `run_chat` 서비스만 호출하며, 반환값은 에이전트 실행 결과를 그대로 활용한 구조체.
6. **문서 갱신**: 컨트리뷰터 가이드(`AGENTS.md`)와 프로젝트 소개(`PROJECT_OVERVIEW_KR.md`)를 LangChain 구조에 맞게 업데이트. 새 보고서(`PROJECT_OVERVIEW_KR_V2.md`)로 변경 배경과 세부 사항을 기록.

---

## 4. 주요 코드 차이

### 4.1 FastAPI 엔드포인트 (`main.py`)

#### 변경 전
```python
from graph import app_graph

@app.post("/chat")
def chat(req: ChatReq):
    state = {...}
    final = app_graph.invoke(state, config={...})
    return {
        "reply": final.get("reply_text"),
        "intent": final.get("intent"),
        "api_response": final.get("api_response"),
        "suggest_json": final.get("suggest_json")
    }
```

#### 변경 후
```python
from app.services import run_chat

@app.post("/chat")
def chat(req: ChatReq):
    result = run_chat(req.chatInput, thread_id=thread_id)
    return {
        "reply": result.get("reply"),
        "tool_calls": result.get("tool_calls"),
        "api_response": result.get("tool_calls"),
        "error": result.get("error"),
        "intent": None,
        "suggest_json": None,
    }
```

**차이점**: LangChain 서비스 계층으로 책임을 이관하여 FastAPI 엔드포인트는 단순히 서비스 함수 호출과 로깅만 수행합니다. `api_response`는 하위 호환을 위해 동일 키를 유지하되 내부적으로 `tool_calls`를 재활용합니다.

### 4.2 서비스 계층 (`app/services/chat_service.py`)

```python
def run_chat(message: str, *, thread_id: str) -> Dict[str, Any]:
    executor = get_agent_executor()
    result = executor.invoke({"input": message}, config={"tags": [thread_id]})
    steps = result.get("intermediate_steps") or []
    tool_calls = [_serialise_tool_step(step) for step in steps]
    return {
        "reply": result.get("output", ""),
        "tool_calls": tool_calls,
        "error": None,
    }
```

- `AgentExecutor`는 LangChain이 제공하는 고수준 API로, 에이전트 실행과 도구 호출을 관리합니다.
- `return_intermediate_steps=True` 옵션을 켜둬 툴 호출 로그를 FastAPI 응답에 전달합니다.
- 오류 발생 시 공통 예외 처리를 수행해 사용자에게 친절한 메시지를 반환합니다.

### 4.3 에이전트 정의 (`app/agents/chat_agent.py`)

```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
])
agent = create_tool_calling_agent(llm=llm, tools=SHOTGRID_TOOLS, prompt=prompt)

return AgentExecutor(
    agent=agent,
    tools=SHOTGRID_TOOLS,
    return_intermediate_steps=True,
    max_iterations=settings.agent_max_iterations,
)
```

- 시스템 프롬프트는 한국어 응답, 데이터 기반 요약, 후속 액션 제시를 명시합니다.
- LangChain이 툴 호출 루프를 관리하므로, 더 이상 의도 분기나 조건 라우팅을 직접 구현할 필요가 없습니다.
- `lru_cache`를 이용해 에이전트 초기화를 1회만 수행하므로 반복 호출에 비용이 들지 않습니다.

### 4.4 툴 정의 (`app/tools/shotgrid.py`)

```python
class FeedbackParams(BaseModel):
    userEmail: str = Field(..., description="사용자 이메일 또는 로컬 파트")
    status: str = Field(default="vsupck", description="ShotGrid 버전 상태 코드")
    limit: int = Field(default=20, ge=1, le=100, description="조회할 최대 항목 수")

@tool("get_versions_feedback", args_schema=FeedbackParams)
def tool_get_versions_feedback(params: FeedbackParams) -> str:
    response = _run_async(get_versions_feedback(...))
    payload = _normalize_response(response)
    return json.dumps(payload, ensure_ascii=False)
```

- LangChain `@tool` 데코레이터와 Pydantic 스키마를 활용해 입력 검증과 문서를 동시에 처리합니다.
- httpx 비동기 호출은 `asyncio.run`으로 감싸 동기 환경에서도 실행 가능합니다.
- 응답은 JSON 문자열로 반환해 에이전트가 쉽게 파싱하고 후속 대화를 이어갈 수 있습니다.

### 4.5 LLM 추상화 (`app/llm/__init__.py`)

```python
@lru_cache(maxsize=1)
def get_chat_model() -> BaseChatModel:
    provider = settings.llm_provider.lower()
    if provider == "openai":
        return ChatOpenAI(model=settings.llm_model, ...)
    if provider == "ollama":
        return ChatOllama(model=settings.llm_model, ...)
    raise LLMConfigurationError(...)
```

- OpenAI, Ollama 양쪽을 지원하며 필요 시 다른 공급자를 쉽게 추가할 수 있도록 구조화했습니다.
- 필수 패키지가 설치되지 않았거나 API 키가 없는 경우 명확한 예외 메시지를 제공합니다.

---

## 5. 리팩터링 사유 및 기대 효과

### 5.1 리팩터링 사유
1. **추가 액션 대응 속도 향상**: 더 이상 액션별 의도 분류, 조건 라우팅, 상태 업데이트를 수작업으로 맞출 필요가 없습니다. 툴만 추가하면 에이전트가 자동으로 활용합니다.
2. **LLM 교체 용이성 확보**: 로컬 모델(Ollama) 도입이나 클라우드 모델 전환 시 `LLM_PROVIDER` 값만 변경하면 됩니다. 추후 Azure OpenAI, Anthropic 등을 붙이는 것도 동일 패턴으로 가능.
3. **코드 베이스 정리**: 각 책임 범위를 명확히 나눈 패키지 구조로 가독성과 테스트 편의성을 높였습니다.
4. **비용 및 성능 최적화 기반 마련**: LLM 호출이 한 번으로 줄어들고, 툴 호출은 필요한 경우에만 이루어져 불필요한 비용을 줄일 수 있습니다.

### 5.2 기대 효과
- **유지보수 효율**: 기능 추가/변경 시 영향 범위가 명확합니다. 예를 들어 툴 로직 수정은 `app/tools/shotgrid.py` 한 파일에서 대부분 해결됩니다.
- **테스트 용이성**: LangChain의 테스트 도구(`FakeListChatModel`, `AgentExecutor` 모킹 등)를 활용해 에이전트와 툴을 개별적으로 검증할 수 있습니다.
- **확장성 & 재사용성**: `app/llm` 모듈을 통해 다른 마이크로서비스에서도 동일한 LLM 초기화 로직을 재사용할 수 있습니다. `SHOTGRID_TOOLS` 목록은 향후 LangChain 체인이나 다른 에이전트에서도 그대로 활용 가능합니다.
- **관측성**: `intermediate_steps`에 포함된 로그를 그대로 FastAPI 응답에 전달하기 때문에, 클라이언트에서 어떤 툴이 언제 호출됐는지 추적이 쉬워집니다. 추후 Observability 스택에 전송해도 좋습니다.

---

## 6. 변경 후 주의사항 및 권장 작업

1. **환경 변수 관리**: `.env` 또는 배포 환경에 `LLM_PROVIDER`, `LLM_MODEL`, `OPENAI_API_KEY`/`OLLAMA_BASE_URL`, `AGENT_MAX_ITERATIONS` 등을 반드시 설정해야 합니다. 기본값은 OpenAI 기반이지만 키가 없으면 초기화 단계에서 실패합니다.
2. **의존성 설치**: `langchain-openai`, `langchain-community` 등 LangChain 관련 패키지가 추가되었습니다. `pip install -r requirements.txt`로 최신 종속성을 확보하세요.
3. **테스트 시나리오 구성**: 에이전트 실행이 결정론적이지 않을 수 있으므로, 핵심 툴은 단위 테스트로, 전체 흐름은 통합 테스트/샌드박스 환경으로 나누어 확인하는 전략을 권장합니다.
4. **문서 업데이트**: 개발자 온보딩 문서(본 보고서 포함)를 리포지토리 위키나 PR 템플릿에 링크해 구조 변경 내용을 공유하세요.
5. **추가 자동화 고려**: `run_chat` 결과의 툴 로그를 기반으로 모니터링 시스템에 데이터를 적재하거나, 실패 케이스를 수집해 LLM 프롬프트/툴을 개선하는 파이프라인을 구축할 수 있습니다.

---

## 7. 부록: 전체 파일 구조 (리팩터링 이후)

```
mnt_chatbot/
├── app/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   └── chat_agent.py
│   ├── llm/
│   │   └── __init__.py
│   ├── services/
│   │   ├── __init__.py
│   │   └── chat_service.py
│   └── tools/
│       ├── __init__.py
│       └── shotgrid.py
├── config.py
├── main.py
├── shotgrid_client.py
├── requirements.txt
├── AGENTS.md
├── PROJECT_OVERVIEW_KR.md
└── PROJECT_OVERVIEW_KR_V2.md  ← 현재 문서
```

---

## 8. 마무리 요약

이번 리팩터링은 LangGraph 기반 상태 머신을 LangChain Agent/Tool 구조로 전환하여 코드 간결성과 확장성을 확보하는 데 목적이 있습니다. 핵심 강조점은 다음과 같습니다.

1. **구조 분리**: FastAPI ↔ 서비스 ↔ 에이전트 ↔ 툴 ↔ LLM 계층이 명확해졌습니다.
2. **모델 교체 유연성**: 오픈소스/로컬 모델 및 다양한 API 공급자로 쉽게 전환할 수 있는 구조를 마련했습니다.
3. **빠른 기능 추가**: ShotGrid API가 추가되어도 툴로 등록하고 프롬프트를 조정하는 수준으로 대응 가능합니다.
4. **운영 용이성**: 툴 호출 로그와 에러 정보를 일원화해 추적과 디버깅이 쉬워졌습니다.

추가 아이디어나 피드백이 있다면 문서 말미에 내용을 덧붙여 지속적으로 업데이트해 주시기 바랍니다. 이 문서는 개발자 발표와 레퍼런스용으로 활용할 계획입니다.***
