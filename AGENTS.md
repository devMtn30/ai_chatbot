# Repository Guidelines

## Project Structure & Module Organization
The FastAPI entry point lives in `main.py` and now delegates chat handling to the LangChain stack inside the `app/` package. `app/services/chat_service.py` defines the orchestration surface that FastAPI calls, `app/agents/chat_agent.py` wires the agent executor and prompt, and `app/tools/shotgrid.py` exposes ShotGrid integrations as structured tools. LLM configuration is isolated in `app/llm/__init__.py`, while environment secrets remain in `config.py`. ShotGrid HTTP wrappers stay in `shotgrid_client.py`; keep new API clients colocated there.

## Build, Test, and Development Commands
Install dependencies with `pip install -r requirements.txt`. Run the API locally using `uvicorn main:app --reload` to iterate quickly. You can smoke-test the LangChain agent without HTTP by opening a Python REPL and invoking `from app.services import run_chat; run_chat("...", thread_id="cli")`. When you add automated coverage, standardise on `python -m pytest` from the repository root so CI can mirror the flow.

## Coding Style & Naming Conventions
Use four-space indentation, snake_case for functions and variables, and PascalCase for Pydantic schemas. Tool signatures should declare explicit `args_schema` models, while shared helpers belong in `app/` subpackages to avoid circular imports. Keep agent prompts and constants close to the agent module, and document non-obvious prompt behaviour with concise comments. When touching HTTP utilities, continue returning serialisable dicts so tool outputs stay JSON-friendly for the agent.

## Testing Guidelines
Prefer pytest with `pytest-asyncio` for exercising the asynchronous ShotGrid wrappers. For agent behaviour, you can unit-test the tool functions directly and add contract-style tests that call `run_chat` with a mocked LLM (LangChain’s `FakeListChatModel` works well). Store tests under `tests/` with filenames like `test_chat_service.py`, and name functions `test_<behavior>_<condition>` to capture intent clearly. Keep `test_main.http` updated for manual regression of the FastAPI endpoint.

## Commit & Pull Request Guidelines
Adopt Conventional Commits (`feat:`, `fix:`, `chore:`) to describe changes succinctly. Break work into reviewable chunks—LLM prompt updates, new tools, and config changes each deserve dedicated commits. Pull requests should outline user-facing effects, mention any new environment variables, and include before/after transcripts or cURL snippets when modifying the HTTP surface. Add follow-up tasks as checklist items if additional tools or prompts are planned.

## Environment & Secrets
Configuration is loaded via `.env` using `python-dotenv`. Set `LLM_PROVIDER`, `LLM_MODEL`, `LLM_TEMPERATURE`, and provider-specific values such as `OPENAI_API_KEY` or `OLLAMA_BASE_URL` without hardcoding them in source. `SG_BASE_URL` and `DEFAULT_EMAIL_DOMAIN` still govern ShotGrid access; avoid modifying code defaults and share sample values through `.env.example` when necessary.
