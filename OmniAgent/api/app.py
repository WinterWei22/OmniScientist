from __future__ import annotations

import hmac
import os
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, Field

from .run_manager import RunManager


app = FastAPI(title="OmniAgent Run API", version="1.0")
manager = RunManager()


class RunCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str | None = Field(default=None, min_length=1)
    index: int | None = Field(default=None, ge=1)
    model: str | None = Field(default=None, min_length=1)


class RunStopRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(default="user requested stop", min_length=1, max_length=500)


def authorize(request: Request) -> None:
    configured = os.getenv("OMNIAGENT_API_AUTH_TOKEN", "")
    if not configured:
        return
    supplied = request.headers.get("authorization", "")
    expected = f"Bearer {configured}"
    if not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")


def manager_error(exc: Exception) -> HTTPException:
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail="Run or artifact not found")
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail="Artifact not found")
    return HTTPException(status_code=400, detail=str(exc))


@app.get("/healthz")
async def healthz(request: Request) -> dict[str, Any]:
    authorize(request)
    return {"status": "ok", "service": "omniagent-run-api", "version": app.version}


@app.get("/v1/cases")
async def list_cases(
    request: Request,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=100),
) -> dict[str, Any]:
    authorize(request)
    try:
        return manager.list_tasks(offset=offset, limit=limit)
    except Exception as exc:
        raise manager_error(exc) from exc


@app.get("/v1/cases/{task_id}")
async def get_case(task_id: str, request: Request) -> dict[str, Any]:
    authorize(request)
    try:
        return manager.resolve_task(task_id=task_id)
    except Exception as exc:
        raise manager_error(exc) from exc


@app.post("/v1/runs", status_code=202)
async def create_run(
    body: RunCreateRequest,
    request: Request,
    x_dashscope_api_key: str | None = Header(default=None, alias="X-DashScope-Api-Key"),
) -> dict[str, Any]:
    authorize(request)
    try:
        record = await manager.start(
            task_id=body.task_id,
            index=body.index,
            model=body.model,
            api_key=x_dashscope_api_key,
        )
    except Exception as exc:
        raise manager_error(exc) from exc
    return {
        "run_id": record.run_id,
        "harness_run_id": record.harness_run_id,
        "status": record.status,
        "task_id": record.task_id,
        "case": record.task,
        "events_url": f"/v1/runs/{record.run_id}/events",
        "result_url": f"/v1/runs/{record.run_id}/result",
    }


@app.get("/v1/runs/{run_id}")
async def get_run(run_id: str, request: Request) -> dict[str, Any]:
    authorize(request)
    try:
        return manager.status(run_id)
    except Exception as exc:
        raise manager_error(exc) from exc


@app.post("/v1/runs/{run_id}/stop")
async def stop_run(
    run_id: str, body: RunStopRequest, request: Request
) -> dict[str, Any]:
    authorize(request)
    try:
        return await manager.stop(run_id, body.reason)
    except Exception as exc:
        raise manager_error(exc) from exc


@app.get("/v1/runs/{run_id}/events")
async def get_events(
    run_id: str,
    request: Request,
    cursor: int = Query(default=0, ge=0),
    limit: int = Query(default=100, ge=1, le=200),
) -> dict[str, Any]:
    authorize(request)
    try:
        return manager.events(run_id, cursor, limit)
    except Exception as exc:
        raise manager_error(exc) from exc


@app.get("/v1/runs/{run_id}/result")
async def get_result(run_id: str, request: Request) -> dict[str, Any]:
    authorize(request)
    try:
        return manager.result(run_id)
    except Exception as exc:
        raise manager_error(exc) from exc


@app.get("/v1/runs/{run_id}/result/{name}")
async def get_artifact(run_id: str, name: str, request: Request) -> Any:
    authorize(request)
    try:
        return manager.artifact(run_id, name)
    except Exception as exc:
        raise manager_error(exc) from exc
