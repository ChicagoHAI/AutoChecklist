"""Settings management endpoints."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from app.services.settings_service import save_settings, get_masked_settings, get_provider_kwargs
from app.services.storage import DATA_DIR

router = APIRouter()


class SettingsUpdate(BaseModel):
    default_provider: Optional[str] = None
    default_model: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    vllm_base_url: Optional[str] = None


class ConnectionTestRequest(BaseModel):
    provider: str
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None


@router.get("")
async def get_settings():
    """Get settings with masked API keys."""
    return get_masked_settings()


@router.put("")
async def update_settings(updates: SettingsUpdate):
    """Update settings."""
    update_dict = {k: v for k, v in updates.model_dump().items() if v is not None}
    return save_settings(update_dict)


@router.post("/test")
async def test_connection(request: ConnectionTestRequest):
    """Test provider connection with a simple API call."""
    try:
        from autochecklist import get_client

        # Start from saved settings, then overlay any explicit overrides
        kwargs = get_provider_kwargs(
            provider_override=request.provider,
            model_override=request.model,
        )
        if request.api_key:
            kwargs["api_key"] = request.api_key
        if request.base_url:
            kwargs["base_url"] = request.base_url

        client = get_client(**kwargs)

        # Quick test: just try to complete a simple prompt
        test_model = request.model or "openai/gpt-4o-mini"
        test_messages = [{"role": "user", "content": "Say 'ok'"}]

        if hasattr(client, "chat_completion_async"):
            await client.chat_completion_async(
                model=test_model, messages=test_messages
            )
        else:
            executor = ThreadPoolExecutor(max_workers=1)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                executor,
                lambda: client.chat_completion(
                    model=test_model, messages=test_messages,
                ),
            )

        return {"success": True, "message": "Connection successful"}
    except Exception as e:
        return {"success": False, "message": str(e)}


@router.delete("/data")
async def clear_all_data():
    """Clear all stored data (evaluations, checklists, batches)."""
    cleared = []
    for subdir in ["evaluations", "checklists", "batches"]:
        dir_path = DATA_DIR / subdir
        if dir_path.exists():
            # Remove all files in the directory
            count = 0
            for f in dir_path.iterdir():
                if f.is_file():
                    f.unlink()
                    count += 1
            cleared.append({"directory": subdir, "files_deleted": count})
        else:
            cleared.append({"directory": subdir, "files_deleted": 0})

    return {"status": "cleared", "details": cleared}
