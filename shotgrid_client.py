import httpx
from typing import Optional, Dict, Any
from config import settings

BASE = settings.sg_base_url

def _qs(params: Dict[str, Any]) -> str:
    q = "&".join(
        f"{k}={httpx.QueryParams({k:v})[k]}"
        for k, v in params.items()
        if v not in (None, "", False)
    )
    return f"?{q}" if q else ""

def normalize_email(val: Optional[str]) -> str:
    if not val:
        return ""
    s = str(val).strip()
    if "@" in s:
        return s
    return f"{s}@{settings.default_email_domain}"

# ====== 스펙에 존재하는 엔드포인트들 ======

async def get_versions_feedback(user_email: str, status: str = "vsupck", limit: int = 20):
    url = f"{BASE}/api/shotgrid/versions/feedback" + _qs({
        "userEmail": user_email, "status": status, "limit": limit
    })
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(url)
        return r

async def get_my_sck_versions(user_email: str, limit: int = 10):
    url = f"{BASE}/api/shotgrid/versions/my-sck" + _qs({
        "userEmail": user_email, "limit": limit
    })
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(url)
        return r

async def get_versions_by_status(user_email: str, status: str, limit: int = 10, sck_only: Optional[bool] = None):
    url = f"{BASE}/api/shotgrid/versions/status/{status}" + _qs({
        "userEmail": user_email, "limit": limit, "sckOnly": sck_only
    })
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(url)
        return r

async def get_versions_by_shot_status(user_email: str, shot_status: str, limit: int = 10):
    url = f"{BASE}/api/shotgrid/versions/shot-status/{shot_status}" + _qs({
        "userEmail": user_email, "limit": limit
    })
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(url)
        return r

async def get_projects_progress():
    url = f"{BASE}/api/shotgrid/projects/progress"
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(url)
        return r

async def get_team_workload(project_id: str, step: str):
    url = f"{BASE}/api/shotgrid/projects/{project_id}/team-workload" + _qs({
        "step": step
    })
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(url)
        return r

async def get_most_urgent_task(user_email: str):
    url = f"{BASE}/api/shotgrid/tasks/urgent" + _qs({
        "userEmail": user_email
    })
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(url)
        return r

async def get_version_raw(version_id: str):
    url = f"{BASE}/api/shotgrid/versions/{version_id}/raw"
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.get(url)
        return r
