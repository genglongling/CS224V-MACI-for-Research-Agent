"""
FastAPI app for interactive multi-agent debate on user-provided topics.

Usage (from MAD-main directory):

  export OPENAI_API_KEY="your_key"
  uvicorn app_interactive:app --reload --port 8001

Then open: http://127.0.0.1:8001/interactive
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.runners.interactive_debate import run_interactive_debate


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="MAD Interactive Debate")

# CORS for local front-end usage
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class InteractiveDebateRequest(BaseModel):
    topic: str = Field(..., description="User-provided debate topic.")
    max_agents: int = Field(5, ge=2, le=8, description="Maximum number of agents/viewpoints.")
    rounds: int = Field(3, ge=1, le=6, description="Number of debate rounds.")


@app.get("/interactive")
async def serve_interactive_page():
    """
    Serve the simple front-end page.
    """
    html_path = STATIC_DIR / "interactive.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="interactive.html not found")
    return FileResponse(str(html_path))


@app.post("/api/interactive_debate")
async def api_interactive_debate(payload: InteractiveDebateRequest) -> Dict[str, Any]:
    """
    Run an interactive debate for a user topic and return the full result JSON.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set on server.")

    try:
        result = run_interactive_debate(
            topic=payload.topic,
            max_agents=payload.max_agents,
            num_rounds=payload.rounds,
            models_cfg_path=str(BASE_DIR / "configs" / "models.yaml"),
            pairing="qwen_qwen",
            output_dir=BASE_DIR / "results" / "interactive",
        )
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Interactive debate failed: {e}")


@app.get("/")
async def root():
    return {"message": "MAD Interactive Debate API. Visit /interactive for the UI."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8001)


