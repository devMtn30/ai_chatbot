from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, Optional, Type, TypeVar

from langchain_core.tools import tool
from pydantic import BaseModel, Field, ValidationError

from shotgrid_client import (
    get_most_urgent_task,
    get_projects_progress,
    get_team_workload,
    get_version_raw,
    get_versions_by_shot_status,
    get_versions_by_status,
    get_versions_feedback,
    normalize_email,
)

logger = logging.getLogger("chat.tools.shotgrid")

ModelT = TypeVar("ModelT", bound=BaseModel)


def _run_async(coro):
    """Run an async coroutine from a synchronous context."""
    return asyncio.run(coro)


def _normalize_response(response) -> dict:
    try:
        data = response.json()
    except Exception:  # pragma: no cover - fallback for non-json payloads
        data = {"text": getattr(response, "text", "")}
    return {"status": getattr(response, "status_code", None), "data": data}


def _wrap_result(action: str, payload: dict, duration: float) -> str:
    logger.info(
        "shotgrid_tool action=%s duration=%.3fs status=%s",
        action,
        duration,
        payload.get("status") or payload.get("error"),
    )
    return json.dumps(payload, ensure_ascii=False)


def _coerce_params(params: Any, model_cls: Type[ModelT]) -> ModelT:
    """Normalise tool inputs into the expected Pydantic schema."""
    if isinstance(params, model_cls):
        return params
    if isinstance(params, str):
        try:
            parsed: Dict[str, Any] = json.loads(params)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON decode 오류: {exc}") from exc
    elif isinstance(params, dict):
        parsed = params
    else:
        raise TypeError(f"지원하지 않는 입력 타입: {type(params)!r}")

    try:
        return model_cls.model_validate(parsed)
    except ValidationError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"입력 검증 오류: {exc}") from exc


class FeedbackParams(BaseModel):
    userEmail: str = Field(..., description="사용자 이메일 또는 로컬 파트")
    status: str = Field(
        default="vsupck",
        description="ShotGrid 버전 상태 코드 (기본값 vsupck)",
    )
    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="조회할 최대 항목 수 (1~100)",
    )


@tool("get_versions_feedback", args_schema=FeedbackParams)
def tool_get_versions_feedback(params: FeedbackParams) -> str:
    """조회한 사용자의 최신 ShotGrid 버전 피드백 상태를 반환합니다."""
    start = time.perf_counter()
    try:
        params_obj = _coerce_params(params, FeedbackParams)
        email = normalize_email(params_obj.userEmail)
        response = _run_async(
            get_versions_feedback(
                email, status=params_obj.status, limit=params_obj.limit
            )
        )
        payload = _normalize_response(response)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("tool_get_versions_feedback failed")
        payload = {"error": str(exc)}
    return _wrap_result("get_versions_feedback", payload, time.perf_counter() - start)


class VersionsByStatusParams(BaseModel):
    userEmail: str = Field(..., description="사용자 이메일 또는 로컬 파트")
    status: str = Field(..., description="ShotGrid 버전 상태 코드")
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="조회할 최대 항목 수 (1~100)",
    )
    sckOnly: Optional[bool] = Field(
        default=None,
        description="SCK 결과만 필터링할지 여부",
    )


@tool("get_versions_by_status", args_schema=VersionsByStatusParams)
def tool_get_versions_by_status(params: VersionsByStatusParams) -> str:
    """사용자 이메일과 상태 코드로 ShotGrid 버전 목록을 조회합니다."""
    start = time.perf_counter()
    try:
        params_obj = _coerce_params(params, VersionsByStatusParams)
        email = normalize_email(params_obj.userEmail)
        response = _run_async(
            get_versions_by_status(
                email,
                status=params_obj.status,
                limit=params_obj.limit,
                sck_only=params_obj.sckOnly,
            )
        )
        payload = _normalize_response(response)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("tool_get_versions_by_status failed")
        payload = {"error": str(exc)}
    return _wrap_result("get_versions_by_status", payload, time.perf_counter() - start)


class VersionsByShotStatusParams(BaseModel):
    userEmail: str = Field(..., description="사용자 이메일 또는 로컬 파트")
    shotStatus: str = Field(
        default="Active",
        description="ShotGrid 샷 상태 (기본값 Active)",
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="조회할 최대 항목 수 (1~100)",
    )


@tool("get_versions_by_shot_status", args_schema=VersionsByShotStatusParams)
def tool_get_versions_by_shot_status(params: VersionsByShotStatusParams) -> str:
    """샷 상태 기준으로 사용자의 ShotGrid 버전 목록을 조회합니다."""
    start = time.perf_counter()
    try:
        params_obj = _coerce_params(params, VersionsByShotStatusParams)
        email = normalize_email(params_obj.userEmail)
        response = _run_async(
            get_versions_by_shot_status(
                email,
                shot_status=params_obj.shotStatus,
                limit=params_obj.limit,
            )
        )
        payload = _normalize_response(response)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("tool_get_versions_by_shot_status failed")
        payload = {"error": str(exc)}
    return _wrap_result(
        "get_versions_by_shot_status", payload, time.perf_counter() - start
    )


@tool("get_projects_progress")
def tool_get_projects_progress() -> str:
    """ShotGrid 프로젝트 진행률 요약 정보를 반환합니다."""
    start = time.perf_counter()
    try:
        response = _run_async(get_projects_progress())
        payload = _normalize_response(response)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("tool_get_projects_progress failed")
        payload = {"error": str(exc)}
    return _wrap_result("get_projects_progress", payload, time.perf_counter() - start)


class TeamWorkloadParams(BaseModel):
    projectId: str = Field(..., description="ShotGrid 프로젝트 ID")
    step: Optional[str] = Field(
        default=None,
        description="ShotGrid 파이프라인 스텝 코드",
    )
    team: Optional[str] = Field(
        default=None,
        description="스텝과 동일하게 사용할 수 있는 팀 약어",
    )


@tool("get_team_workload", args_schema=TeamWorkloadParams)
def tool_get_team_workload(params: TeamWorkloadParams) -> str:
    """프로젝트와 파이프라인 스텝별 팀 작업량을 조회합니다."""
    start = time.perf_counter()
    try:
        params_obj = _coerce_params(params, TeamWorkloadParams)
        step_value = (params_obj.step or params_obj.team or "COMP").upper()
        response = _run_async(
            get_team_workload(project_id=params_obj.projectId, step=step_value)
        )
        payload = _normalize_response(response)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("tool_get_team_workload failed")
        payload = {"error": str(exc)}
    return _wrap_result("get_team_workload", payload, time.perf_counter() - start)


class UrgentTaskParams(BaseModel):
    userEmail: str = Field(..., description="사용자 이메일 또는 로컬 파트")


@tool("get_most_urgent_task", args_schema=UrgentTaskParams)
def tool_get_most_urgent_task(params: UrgentTaskParams) -> str:
    """사용자에게 할당된 가장 긴급 ShotGrid 태스크를 조회합니다."""
    start = time.perf_counter()
    try:
        params_obj = _coerce_params(params, UrgentTaskParams)
        email = normalize_email(params_obj.userEmail)
        response = _run_async(get_most_urgent_task(email))
        payload = _normalize_response(response)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("tool_get_most_urgent_task failed")
        payload = {"error": str(exc)}
    return _wrap_result("get_most_urgent_task", payload, time.perf_counter() - start)


class VersionRawParams(BaseModel):
    versionId: str = Field(..., description="ShotGrid 버전 ID")


@tool("get_version_raw", args_schema=VersionRawParams)
def tool_get_version_raw(params: VersionRawParams) -> str:
    """버전 ID로 ShotGrid Raw 버전 데이터를 반환합니다."""
    start = time.perf_counter()
    try:
        params_obj = _coerce_params(params, VersionRawParams)
        response = _run_async(get_version_raw(params_obj.versionId))
        payload = _normalize_response(response)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.exception("tool_get_version_raw failed")
        payload = {"error": str(exc)}
    return _wrap_result("get_version_raw", payload, time.perf_counter() - start)


SHOTGRID_TOOLS = [
    tool_get_versions_feedback,
    tool_get_versions_by_status,
    tool_get_versions_by_shot_status,
    tool_get_projects_progress,
    tool_get_team_workload,
    tool_get_most_urgent_task,
    tool_get_version_raw,
]
