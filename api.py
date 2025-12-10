"""
Optimizer Agent API
FastAPI application with training and scoring endpoints.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional

from utils import setup_logging, ensure_directories, get_service_info, HealthCheck
from train import train_model
from inference import score_structures, is_model_loaded

# Setup logging
setup_logging("INFO")
logger = logging.getLogger(__name__)


# =============================================================================
# LIFESPAN HANDLER
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting Optimizer Agent...")
    ensure_directories()
    
    # Check if model exists
    if is_model_loaded():
        logger.info("Trained model found and ready")
    else:
        logger.warning("No trained model found. Call POST /train to create one.")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Optimizer Agent...")


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="Optimizer Agent",
    description="Preference-based reward model for prompt structure optimization",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class StructureScore(BaseModel):
    """Single structure score response."""
    structure_id: int
    predicted_success_score: float
    top_positive_attributes: List[str]
    top_negative_attributes: List[str]
    current_outlier_count: float
    current_z_score: float
    renderer: Optional[str] = None
    skeleton_preview: Optional[str] = None
    warning: Optional[str] = None


class ScoreResponse(BaseModel):
    """Score structures response."""
    structures: List[StructureScore]
    global_preference_vector: Dict[str, float]
    model_metadata: Optional[Dict[str, Any]]
    warning: Optional[str] = None


class TrainResponse(BaseModel):
    """Training response."""
    status: str
    training_samples: Optional[int] = None
    positive_samples: Optional[int] = None
    negative_samples: Optional[int] = None
    auc_score: Optional[float] = None
    trained_at: Optional[str] = None
    model_path: Optional[str] = None
    reason: Optional[str] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    service: str
    version: str
    model_loaded: bool
    last_trained: Optional[str] = None
    cache_stats: Dict[str, int]
    checks: Optional[Dict[str, Any]] = None


# =============================================================================
# ENDPOINTS
# =============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "service": "optimizer-agent",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check(detailed: bool = False):
    """
    Health check endpoint.
    
    Args:
        detailed: If true, run full connectivity checks
    """
    info = get_service_info()
    
    response = HealthResponse(
        status="healthy" if info["model_loaded"] else "degraded",
        service=info["service"],
        version=info["version"],
        model_loaded=info["model_loaded"],
        last_trained=info.get("last_trained"),
        cache_stats=info["cache_stats"]
    )
    
    if detailed:
        response.checks = HealthCheck.full_check()
    
    return response


@app.post("/train", response_model=TrainResponse)
async def trigger_training(background_tasks: BackgroundTasks, async_mode: bool = False):
    """
    Trigger model training.
    
    Fetches all data from Airtable, extracts features, trains XGBoost model,
    and saves results.
    
    Args:
        async_mode: If true, run training in background and return immediately
        
    Note: Training can take 5-10+ minutes depending on data size.
    """
    logger.info("Training triggered via API")
    
    if async_mode:
        # Run in background
        background_tasks.add_task(train_model)
        return TrainResponse(
            status="started",
            reason="Training started in background. Check /health for completion."
        )
    
    # Run synchronously
    result = train_model()
    
    return TrainResponse(**result)


@app.post("/score_structures", response_model=ScoreResponse)
async def get_structure_scores():
    """
    Score all active structures.
    
    Returns predicted success scores for each active structure based on
    the trained preference model.
    
    Returns uniform scores (0.5) if no model has been trained yet.
    """
    logger.info("Scoring structures via API")
    
    try:
        result = score_structures()
        
        return ScoreResponse(
            structures=[StructureScore(**s) for s in result["structures"]],
            global_preference_vector=result["global_preference_vector"],
            model_metadata=result.get("model_metadata"),
            warning=result.get("warning")
        )
        
    except Exception as e:
        logger.error(f"Error scoring structures: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
async def get_model_info():
    """
    Get information about the trained model.
    
    Returns training metadata, metrics, and feature importance.
    """
    from inference import get_model_metadata, is_model_loaded
    
    if not is_model_loaded():
        return {
            "status": "not_trained",
            "message": "No model found. Call POST /train to create one."
        }
    
    metadata = get_model_metadata()
    
    return {
        "status": "trained",
        "trained_at": metadata.get("trained_at"),
        "metrics": metadata.get("metrics", {}),
        "feature_importance": metadata.get("feature_importance", [])[:10]
    }


@app.get("/preferences")
async def get_global_preferences():
    """
    Get the global preference vector.
    
    Returns averaged fashion attributes from all liked (outlier) images.
    """
    from config import GLOBAL_PREF_PATH
    from utils import load_json_file
    
    prefs = load_json_file(GLOBAL_PREF_PATH)
    
    if prefs is None:
        return {
            "status": "not_computed",
            "message": "Global preferences not computed. Call POST /train first."
        }
    
    # Return top 30 preferences
    top_prefs = dict(list(prefs.items())[:30])
    
    return {
        "status": "computed",
        "preferences": top_prefs,
        "total_attributes": len(prefs)
    }


@app.get("/prompts/top")
async def get_top_prompts(limit: int = 20, min_samples: int = 2):
    """
    Get prompts with highest success rates.
    
    Args:
        limit: Maximum number of prompts to return (default 20)
        min_samples: Minimum sample count to include (default 2)
        
    Returns:
        Top performing prompts sorted by success rate
    """
    from config import PROMPT_STATS_PATH
    from utils import load_json_file
    
    stats = load_json_file(PROMPT_STATS_PATH)
    
    if stats is None:
        return {
            "status": "not_computed",
            "message": "Prompt stats not computed. Call POST /train first."
        }
    
    # Filter by min_samples and limit
    filtered = {
        k: v for k, v in stats.items() 
        if v.get("sample_count", 0) >= min_samples
    }
    
    top_prompts = dict(list(filtered.items())[:limit])
    
    # Also get bottom performers for contrast
    bottom_prompts = dict(list(filtered.items())[-5:]) if len(filtered) > 5 else {}
    
    return {
        "status": "computed",
        "top_prompts": top_prompts,
        "bottom_prompts": bottom_prompts,
        "total_unique_prompts": len(stats),
        "prompts_with_min_samples": len(filtered)
    }


@app.get("/prompts/{prompt_hash}")
async def get_prompt_stats(prompt_hash: str):
    """
    Get stats for a specific prompt by hash.
    
    Args:
        prompt_hash: 12-character MD5 hash of prompt text
    """
    from config import PROMPT_STATS_PATH
    from utils import load_json_file
    
    stats = load_json_file(PROMPT_STATS_PATH)
    
    if stats is None:
        return {
            "status": "not_computed",
            "message": "Prompt stats not computed. Call POST /train first."
        }
    
    if prompt_hash not in stats:
        return {
            "status": "not_found",
            "message": f"Prompt hash {prompt_hash} not found"
        }
    
    return {
        "status": "found",
        "prompt_hash": prompt_hash,
        "stats": stats[prompt_hash]
    }


@app.get("/structure_prompt_insights")
async def get_structure_prompt_insights():
    """
    Get prompt success insights grouped by structure.
    
    Returns top-performing prompts for each structure.
    """
    from config import STRUCTURE_PROMPT_STATS_PATH
    from utils import load_json_file
    
    stats = load_json_file(STRUCTURE_PROMPT_STATS_PATH)
    
    if stats is None:
        return {
            "status": "not_computed",
            "message": "Structure prompt insights not computed. Call POST /train first."
        }
    
    return {
        "status": "computed",
        "structure_count": len(stats),
        "insights": stats
    }


@app.delete("/cache")
async def clear_cache(cache_type: str = "all"):
    """
    Clear feature cache.
    
    Args:
        cache_type: "clip", "attributes", or "all"
    """
    import os
    import shutil
    from config import CLIP_CACHE_DIR, ATTRIBUTES_CACHE_DIR
    
    cleared = []
    
    if cache_type in ["clip", "all"]:
        if os.path.exists(CLIP_CACHE_DIR):
            shutil.rmtree(CLIP_CACHE_DIR)
            os.makedirs(CLIP_CACHE_DIR)
            cleared.append("clip")
    
    if cache_type in ["attributes", "all"]:
        if os.path.exists(ATTRIBUTES_CACHE_DIR):
            shutil.rmtree(ATTRIBUTES_CACHE_DIR)
            os.makedirs(ATTRIBUTES_CACHE_DIR)
            cleared.append("attributes")
    
    return {
        "status": "cleared",
        "caches_cleared": cleared
    }


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
