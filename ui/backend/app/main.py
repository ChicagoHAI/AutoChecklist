"""FastAPI application entry point with CORS configuration."""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.routers import batch, checklist_builder, checklists, evaluate, pipelines, prompt_templates, registry_info, settings
from app.services.storage import ensure_dirs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ui")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: ensure data directories exist."""
    ensure_dirs()
    logger.info("Data directories ready")
    logger.info("Auto-Checklists UI backend started")
    yield


app = FastAPI(
    title="Auto-Checklists API",
    description="API for automatically evaluating outputs using LLM-based checklist methods",
    version="0.1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log request details and timing."""
    start = time.time()
    response = await call_next(request)
    elapsed = (time.time() - start) * 1000
    # Skip noisy OPTIONS preflight and static asset requests
    if request.method != "OPTIONS" and not request.url.path.startswith("/_next"):
        level = logging.WARNING if response.status_code >= 400 else logging.INFO
        logger.log(
            level,
            "%s %s → %d (%.0fms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
    return response

# CORS configuration - allow all origins for local/network access
# In production, you should restrict this to specific domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # Must be False when allow_origins is "*"
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(evaluate.router, prefix="/api/evaluate", tags=["evaluate"])
app.include_router(settings.router, prefix="/api/settings", tags=["settings"])
app.include_router(registry_info.router, prefix="/api/registry", tags=["registry"])
app.include_router(batch.router, prefix="/api/batch", tags=["batch"])
app.include_router(checklists.router, prefix="/api/checklists", tags=["checklists"])
app.include_router(prompt_templates.router, prefix="/api/prompt-templates", tags=["prompt-templates"])
app.include_router(pipelines.router, prefix="/api/pipelines", tags=["pipelines"])
app.include_router(checklist_builder.router, prefix="/api/checklist-builder", tags=["checklist-builder"])


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/api/examples")
async def get_examples():
    """Return pre-defined examples for the form."""
    return [
        {
            "name": "Haiku Writing",
            "input": "Write a haiku about autumn leaves falling.",
            "target": "Crimson leaves descend\nDancing in the autumn breeze\nNature's last farewell",
            "reference": "Golden leaves drift down\nWhispering secrets of change\nAutumn's soft goodbye",
        },
        {
            "name": "Explain to Child",
            "input": "Explain photosynthesis to a 10-year-old.",
            "target": "Plants are like little chefs! They use sunlight as their oven, water from the soil as one ingredient, and air as another. They mix these together to cook up their own food called glucose, which gives them energy to grow. And the best part? They release oxygen as a byproduct, which is what we breathe!",
            "reference": None,
        },
        {
            "name": "Code Review",
            "input": "Review this Python function for issues:\n\ndef add(a, b):\n    return a + b",
            "target": "The function is simple and correct for basic addition. However, it lacks type hints, docstring, and input validation. Consider: `def add(a: int, b: int) -> int:`",
            "reference": "This function correctly adds two numbers. Suggestions: 1) Add type hints for clarity. 2) Add a docstring. 3) Consider handling edge cases like None inputs.",
        },
    ]
