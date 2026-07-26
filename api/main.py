"""
TuneKit API
===========
FastAPI backend for the TuneKit pipeline.
"""

import os
import sys
import shutil
import uuid
import json
import requests
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tunekit import (
    TuneKitState,
    ingest_data,
    validate_quality,
    analyze_dataset,
    generate_package,
    recommend_model,
    enrich_dataset,
)
from tunekit.training import (
    generate_training_notebook,
    save_notebook,
)

# ============================================================================
# APP SETUP
# ============================================================================

app = FastAPI(
    title="TuneKit API",
    description="Automated SLM Fine-Tuning Pipeline",
    version="0.1.0",
)

allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000,http://127.0.0.1:8000")
allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

sessions: dict[str, dict] = {}
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
PACKAGE_MAPPING_FILE = "output/.package_mapping.json"

# Session management for memory efficiency (tuned for 1GB RAM)
MAX_SESSIONS = 25  # Maximum concurrent sessions
SESSION_EXPIRY_MINUTES = 30  # Sessions expire after 30 minutes


def cleanup_expired_sessions():
    """Remove expired sessions to free memory."""
    if not sessions:
        return
    
    import gc
    from datetime import datetime, timedelta
    now = datetime.now()
    expired = []
    
    for session_id, session in sessions.items():
        created_str = session.get("created_at", "")
        try:
            created = datetime.fromisoformat(created_str)
            if now - created > timedelta(minutes=SESSION_EXPIRY_MINUTES):
                expired.append(session_id)
        except (ValueError, TypeError):
            pass
    
    for session_id in expired:
        # Clean up uploaded file
        file_path = sessions[session_id].get("file_path")
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass
        del sessions[session_id]
    
    # Force garbage collection if we cleaned up sessions
    if expired:
        gc.collect()


def enforce_session_limit():
    """Remove oldest sessions if we exceed the limit."""
    cleanup_expired_sessions()
    
    if len(sessions) >= MAX_SESSIONS:
        # Sort by creation time and remove oldest
        sorted_sessions = sorted(
            sessions.items(),
            key=lambda x: x[1].get("created_at", "")
        )
        # Remove oldest 10 sessions
        for session_id, session in sorted_sessions[:10]:
            file_path = session.get("file_path")
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass
            del sessions[session_id]


def reload_raw_data_if_needed(session: dict) -> bool:
    """Reload raw_data from file if it's None. Returns True if successful."""
    state = session.get("state")
    if not state:
        return False
    
    if state.get("raw_data") is not None:
        return True  # Already loaded
    
    file_path = session.get("file_path")
    if not file_path or not os.path.exists(file_path):
        return False
    
    # Re-ingest the data
    temp_state = create_empty_state(file_path, "")
    result = ingest_data(temp_state)
    
    if result.get("error_msg"):
        return False
    
    state["raw_data"] = result.get("raw_data")
    return True

def save_package_mapping(session_id: str, package_path: str):
    """Save session_id -> package_path mapping to persistent file."""
    os.makedirs("output", exist_ok=True)
    mapping = load_package_mapping()
    mapping[session_id] = package_path
    with open(PACKAGE_MAPPING_FILE, "w") as f:
        json.dump(mapping, f)

def load_package_mapping() -> dict:
    """Load package mappings from persistent file."""
    if os.path.exists(PACKAGE_MAPPING_FILE):
        try:
            with open(PACKAGE_MAPPING_FILE, "r") as f:
                return json.load(f)
        except:
            return {}
    return {}

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ============================================================================
# STARTUP & MIDDLEWARE
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Clean up any leftover files on startup."""
    import gc
    # Clean uploads folder on startup
    if os.path.exists(UPLOAD_DIR):
        for f in os.listdir(UPLOAD_DIR):
            try:
                os.remove(os.path.join(UPLOAD_DIR, f))
            except OSError:
                pass
    gc.collect()


@app.middleware("http")
async def cleanup_middleware(request, call_next):
    """Run cleanup on every request to prevent memory buildup."""
    cleanup_expired_sessions()
    response = await call_next(request)
    return response


# ============================================================================
# MODELS
# ============================================================================

class AnalyzeRequest(BaseModel):
    session_id: str
    user_description: str


class RecommendRequest(BaseModel):
    session_id: str
    user_task: str  # classify, qa, conversation, generation, extraction
    deployment_target: str = 'not_sure'  # cloud_api, mobile_app, edge_device, web_browser, desktop_app, not_sure


class PlanRequest(BaseModel):
    session_id: str
    user_task: Optional[str] = None  # classify, qa, conversation, generation, extraction
    deployment_target: Optional[str] = None  # cloud_api, mobile_app, etc.
    selected_model_id: Optional[str] = None  # If user chose an alternative model


class GenerateRequest(BaseModel):
    session_id: str


class EnrichRequest(BaseModel):
    session_id: str
    top_n: Optional[int] = None
    balance: bool = True


class SessionResponse(BaseModel):
    session_id: str
    status: str
    message: str
    rows: Optional[int] = 0
    stats: Optional[dict] = None


class AnalyzeResponse(BaseModel):
    session_id: str
    num_rows: int
    columns: list[str]
    quality_score: float
    quality_issues: list[str]
    inferred_task_type: str
    task_confidence: float
    num_classes: Optional[int]
    column_candidates: dict
    sample_rows: Optional[list[dict]] = None  # First 5 rows for preview


class PlanResponse(BaseModel):
    session_id: str
    final_task_type: str
    base_model: str
    reasoning: str
    training_config: dict


class GenerateResponse(BaseModel):
    session_id: str
    package_path: str
    download_url: str


class ColabRequest(BaseModel):
    session_id: str


class ColabResponse(BaseModel):
    session_id: str
    notebook_url: str
    message: str
    status: str
    colab_url: Optional[str] = None  # Direct Colab URL (via Gist)
    gist_url: Optional[str] = None   # GitHub Gist URL
    files: Optional[list] = None
    download_endpoint: Optional[str] = None
    error: Optional[str] = None


def upload_to_gist(notebook_content: str, filename: str = "tunekit_training.ipynb") -> dict:
    """
    Upload notebook to GitHub Gist and return URLs.

    Returns:
        dict with 'gist_url', 'colab_url', or 'error'
    """
    github_token = os.getenv("GITHUB_TOKEN")

    if not github_token:
        return {"error": "GITHUB_TOKEN not configured"}

    try:
        response = requests.post(
            "https://api.github.com/gists",
            headers={
                "Authorization": f"token {github_token}",
                "Accept": "application/vnd.github.v3+json",
            },
            json={
                "description": "TuneKit Training Notebook - Fine-tune your model on Google Colab",
                "public": False,  # Secret gist (unlisted but accessible via URL)
                "files": {
                    filename: {
                        "content": notebook_content
                    }
                }
            },
            timeout=30,
        )

        if response.status_code == 201:
            data = response.json()
            gist_id = data["id"]
            owner = data["owner"]["login"]
            gist_url = data["html_url"]

            colab_url = f"https://colab.research.google.com/gist/{owner}/{gist_id}/{filename}"

            return {
                "gist_url": gist_url,
                "colab_url": colab_url,
                "gist_id": gist_id,
            }
        else:
            error_msg = response.json().get("message", "Unknown error")
            return {"error": f"GitHub API error: {error_msg}"}

    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_empty_state(file_path: str, user_description: str = "") -> TuneKitState:
    """Create an empty TuneKitState."""
    return {
        "file_path": file_path,
        "user_description": user_description,
        "raw_data": None, "columns": None, "sample_rows": None, "num_rows": None,
        "quality_score": None, "quality_issues": None,
        "inferred_task_type": None, "task_confidence": None, "num_classes": None,
        "label_list": None, "column_analysis": None, "column_candidates": None, "dataset_stats": None,
        "final_task_type": None, "base_model": None, "training_config": None, "planning_reasoning": None,
        "est_cost": None, "est_time_minutes": None, "user_approved": None,
        "job_id": None, "job_status": None, "current_epoch": None, "metrics": None,
        "error_msg": None, "retry_count": None,
        "final_report": None, "next_steps": None, "model_download_url": None,
    }


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the landing page."""
    landing_path = os.path.join(FRONTEND_DIR, "landing.html")
    with open(landing_path, "r") as f:
        return f.read()


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the main dashboard/app."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    with open(index_path, "r") as f:
        return f.read()


@app.get("/favicon.ico")
async def favicon():
    """Serve the favicon."""
    favicon_path = os.path.join(FRONTEND_DIR, "favicon.svg")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path, media_type="image/svg+xml")
    logo_path = os.path.join(FRONTEND_DIR, "logo.png")
    if os.path.exists(logo_path):
        return FileResponse(logo_path, media_type="image/png")
    raise HTTPException(status_code=404)


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "message": "TuneKit API is running"}


@app.post("/upload", response_model=SessionResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload a dataset file and return a session_id for subsequent requests."""
    # Enforce session limits to prevent memory exhaustion
    enforce_session_limit()
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    from pathlib import Path
    filename = os.path.basename(file.filename)
    filename = "".join(c for c in filename if c.isalnum() or c in "._-")
    if not filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    ext = os.path.splitext(filename)[1].lower()
    if ext != ".jsonl":
        raise HTTPException(status_code=400, detail="Only JSONL format is supported. Please upload a .jsonl file.")
    
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB max
    session_id = str(uuid.uuid4())[:8]
    file_path = os.path.join(UPLOAD_DIR, f"{session_id}_{filename}")
    
    resolved_path = Path(file_path).resolve()
    resolved_upload_dir = Path(UPLOAD_DIR).resolve()
    try:
        resolved_path.relative_to(resolved_upload_dir)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid file path")
    
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB")
    
    with open(file_path, "wb") as f:
        f.write(content)
    
    sessions[session_id] = {
        "file_path": file_path,
        "filename": file.filename,
        "original_filename": file.filename,
        "state": None,
        "created_at": datetime.now().isoformat(),
    }
    
    try:
        state = create_empty_state(file_path, "")
        result = ingest_data(state)
        state.update(result)
        
        if state.get("error_msg"):
            if os.path.exists(file_path):
                os.remove(file_path)
            del sessions[session_id]
            raise HTTPException(status_code=400, detail=state["error_msg"])
        
        num_rows = state.get("num_rows")
        stats = state.get("stats", {})
        
        if not num_rows or num_rows == 0:
            if stats and stats.get("total_examples"):
                num_rows = stats.get("total_examples")
        
        num_rows = int(num_rows) if num_rows else 0
        
        # Memory optimization: clear raw_data after computing stats
        # We can reload from file when needed
        state["raw_data"] = None
        sessions[session_id]["state"] = state
        
        return {
            "session_id": session_id,
            "status": "uploaded",
            "message": f"File '{file.filename}' uploaded successfully",
            "rows": num_rows,
            "stats": stats,
        }
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        if session_id in sessions:
            del sessions[session_id]
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")


@app.post("/recommend")
async def recommend(request: RecommendRequest):
    """Analyze dataset and recommend the best model based on user task and deployment target."""
    session_id = request.session_id
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Upload a file first.")
    
    session = sessions[session_id]
    state = session.get("state")
    
    if not state:
        raise HTTPException(status_code=400, detail="No data found. Please re-upload your file.")
    
    # Reload raw_data from file if needed (memory optimization)
    if not reload_raw_data_if_needed(session):
        raise HTTPException(status_code=400, detail="Could not load data. Please re-upload your file.")
    
    if state.get("quality_score") is None:
        result = validate_quality(state)
        state.update(result)
    
    if state.get("conversation_characteristics") is None:
        result = analyze_dataset(state)
        state.update(result)
    
    conversation_chars = state.get("conversation_characteristics", {})
    num_examples = state.get("num_rows", 0)
    
    recommendation = recommend_model(
        user_task=request.user_task,
        conversation_characteristics=conversation_chars,
        num_examples=num_examples,
        deployment_target=request.deployment_target
    )
    
    state["user_task"] = request.user_task
    state["deployment_target"] = request.deployment_target
    state["recommendation"] = recommendation
    
    # Clear raw_data after use to save memory
    state["raw_data"] = None
    sessions[session_id]["state"] = state
    
    return {
        "session_id": session_id,
        "recommendation": recommendation,
        "analysis": {
            "stats": state.get("stats", {}),
            "quality_score": state.get("quality_score", 0),
            "quality_issues": state.get("quality_issues", []),
            "conversation_characteristics": conversation_chars,
        }
    }


class AddSystemPromptRequest(BaseModel):
    session_id: str
    system_prompt: str


@app.post("/add-system-prompt")
async def add_system_prompt(request: AddSystemPromptRequest):
    """Add a system prompt to all conversations in the dataset."""
    session_id = request.session_id
    system_prompt = request.system_prompt.strip()
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not system_prompt:
        raise HTTPException(status_code=400, detail="System prompt cannot be empty")
    
    session = sessions[session_id]
    state = session.get("state")
    
    if not state:
        raise HTTPException(status_code=400, detail="No data found. Please re-upload your file.")
    
    # Reload raw_data from file if needed (memory optimization)
    if not reload_raw_data_if_needed(session):
        raise HTTPException(status_code=400, detail="Could not load data. Please re-upload your file.")
    
    raw_data = state["raw_data"]
    modified_count = 0
    
    for entry in raw_data:
        messages = entry.get("messages", [])
        has_system = any(msg.get("role") == "system" for msg in messages)
        
        if not has_system:
            messages.insert(0, {
                "role": "system",
                "content": system_prompt
            })
            modified_count += 1
    
    file_path = session["file_path"]
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for entry in raw_data:
                f.write(json.dumps(entry) + '\n')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save modified data: {str(e)}")
    
    state["file_path"] = file_path
    result = ingest_data(state)
    state.update(result)
    
    if state.get("error_msg"):
        raise HTTPException(status_code=400, detail=state["error_msg"])
    
    # Clear raw_data after use to save memory
    state["raw_data"] = None
    sessions[session_id]["state"] = state
    
    return {
        "session_id": session_id,
        "status": "success",
        "message": f"Added system prompt to {modified_count} conversations",
        "modified_count": modified_count,
        "stats": state.get("stats", {})
    }


@app.get("/download-dataset/{session_id}")
async def download_dataset(session_id: str):
    """Download the dataset file (with any modifications like system prompts)."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    file_path = session.get("file_path")
    
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    original_filename = session.get("original_filename", "dataset.jsonl")
    if not original_filename.endswith('.jsonl'):
        original_filename = original_filename.rsplit('.', 1)[0] + '.jsonl'
    
    if session.get("state", {}).get("stats", {}).get("has_system_prompts"):
        name, ext = os.path.splitext(original_filename)
        original_filename = f"{name}_with_system_prompt{ext}"
    
    return FileResponse(
        path=file_path,
        filename=original_filename,
        media_type='application/jsonl'
    )


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    """
    Legacy endpoint - Run ingestion, validation, and analysis on the uploaded file.
    """
    session_id = request.session_id
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Upload a file first.")
    
    session = sessions[session_id]
    state = session.get("state")
    
    if not state:
        file_path = session["file_path"]
    state = create_empty_state(file_path, request.user_description)
    result = ingest_data(state)
    state.update(result)
    
    if state.get("error_msg"):
        raise HTTPException(status_code=400, detail=state["error_msg"])
    
    result = validate_quality(state)
    state.update(result)
    
    result = analyze_dataset(state)
    state.update(result)
    
    sessions[session_id]["state"] = state
    
    return {
        "session_id": session_id,
        "num_rows": state.get("num_rows", 0),
        "quality_score": state.get("quality_score", 0),
        "quality_issues": state.get("quality_issues", []),
        "conversation_characteristics": state.get("conversation_characteristics", {}),
        "stats": state.get("stats", {}),
    }


@app.get("/sample-data/{session_id}")
async def get_sample_data(session_id: str, limit: int = 100):
    """Get sample rows from the uploaded dataset."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    # Reload raw_data from file if needed (memory optimization)
    if not reload_raw_data_if_needed(session):
        raise HTTPException(status_code=400, detail="Could not load data. Please re-upload your file.")
    
    raw_data = session["state"]["raw_data"]
    result = {
        "rows": raw_data[:limit],
        "total_rows": len(raw_data),
        "returned_rows": min(limit, len(raw_data))
    }
    
    # Clear raw_data after use to save memory
    session["state"]["raw_data"] = None
    
    return result


@app.post("/plan")
async def plan(request: PlanRequest):
    """Create training plan based on recommendation."""
    session_id = request.session_id
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    state = session.get("state")
    
    if not state:
        raise HTTPException(status_code=400, detail="Run /recommend first")
    
    recommendation = state.get("recommendation", {})
    primary_rec = recommendation.get("primary_recommendation", {})
    
    if not primary_rec:
        raise HTTPException(status_code=400, detail="No recommendation found. Run /recommend first.")
    
    user_task = request.user_task or state.get("user_task", "conversation")
    deployment_target = request.deployment_target or state.get("deployment_target", "not_sure")
    selected_model_id = request.selected_model_id
    selected_model_name = primary_rec.get('model_name', 'Phi-4 Mini')
    
    if selected_model_id:
        model_id = selected_model_id
        alternatives = recommendation.get("alternatives", [])
        for alt in alternatives:
            if alt.get("model_id") == selected_model_id:
                selected_model_name = alt.get("model_name", selected_model_id)
                break
    else:
        model_id = primary_rec.get("model_id", "microsoft/Phi-4-mini-instruct")
    
    state["final_task_type"] = user_task
    state["base_model"] = model_id
    state["model_name"] = selected_model_name
    state["planning_reasoning"] = f"Based on your {user_task} task and {deployment_target} deployment target, we recommend {selected_model_name}."
    
    state["training_config"] = {
        "model_name": model_id,
        "task_type": "chat",
        "training_args": {
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "fp16": True,
            "logging_steps": 10,
            "save_strategy": "epoch",
            "warmup_ratio": 0.1,
        },
    }
    
    sessions[session_id]["state"] = state
    
    return {
        "session_id": session_id,
        "final_task_type": state["final_task_type"],
        "base_model": state["base_model"],
        "reasoning": state["planning_reasoning"],
        "training_config": state["training_config"],
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate the training package."""
    session_id = request.session_id
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    state = session.get("state")
    
    if not state or not state.get("final_task_type"):
        raise HTTPException(status_code=400, detail="Run /plan first")
    
    result = generate_package(state)
    state.update(result)
    sessions[session_id]["state"] = state
    
    package_path = state["package_path"]
    save_package_mapping(session_id, package_path)
    
    return GenerateResponse(
        session_id=session_id,
        package_path=package_path,
        download_url=f"/download/{session_id}",
    )


@app.post("/enrich")
async def enrich(request: EnrichRequest):
    """Enrich the dataset with metrics, prioritization and balancing."""
    session_id = request.session_id
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    state = session.get("state")
    
    if not state:
        raise HTTPException(status_code=400, detail="No data found")
        
    # Reload raw_data if needed
    if not reload_raw_data_if_needed(session):
        raise HTTPException(status_code=400, detail="Could not load data")
        
    # Set config
    state["enrich_config"] = {
        "top_n": request.top_n,
        "balance": request.balance
    }
    
    result = enrich_dataset(state)
    
    # Update state with enriched data
    # IMPORTANT: We replace raw_data so downstream tools use the improved version
    if result.get("enriched_data"):
        state["raw_data"] = result["enriched_data"]
        state["num_rows"] = len(result["enriched_data"])
        
        # Update file on disk with enriched data?
        # Maybe we should save a new file version.
        # For now, let's just update memory state and maybe save to a temp file if we want to download it.
        
        # Save enriched data to a new file for persistence/download
        original_path = session["file_path"]
        name, ext = os.path.splitext(original_path)
        enriched_path = f"{name}_enriched{ext}"
        
        try:
            with open(enriched_path, 'w', encoding='utf-8') as f:
                for entry in result["enriched_data"]:
                    f.write(json.dumps(entry) + '\n')
            
            # Update session to point to new file
            session["file_path"] = enriched_path
            state["file_path"] = enriched_path
            
        except Exception as e:
            print(f"Warning: Failed to save enriched file: {e}")
    
    state.update(result)
    
    # Re-run validation/analysis on new data
    val_res = validate_quality(state)
    state.update(val_res)
    
    # Clear raw_data to save memory
    state["raw_data"] = None
    sessions[session_id]["state"] = state
    
    return {
        "session_id": session_id,
        "status": "success",
        "stats": result.get("enrichment_stats", {}),
        "quality_score": state.get("quality_score"),
        "quality_issues": state.get("quality_issues")
    }


@app.get("/download/{session_id}")
async def download(session_id: str):
    """Download the generated training package as a ZIP file."""
    package_path = None
    
    if session_id in sessions:
        session = sessions[session_id]
        state = session.get("state")
        if state:
            package_path = state.get("package_path")
    
    if not package_path:
        mapping = load_package_mapping()
        package_path = mapping.get(session_id)
    
    if not package_path:
        raise HTTPException(status_code=404, detail="Package not found. Please regenerate the package.")
    
    if not os.path.exists(package_path):
        raise HTTPException(status_code=404, detail="Package folder not found on disk")
    
    zip_path = f"{package_path}.zip"
    if not os.path.exists(zip_path):
        shutil.make_archive(package_path, "zip", package_path)
    
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"tunekit_package_{session_id}.zip",
    )


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get the current state of a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    state = session.get("state", {})
    
    return {
        "session_id": session_id,
        "filename": session.get("filename"),
        "created_at": session.get("created_at"),
        "has_state": state is not None,
        "current_step": _get_current_step(state),
    }


def _get_current_step(state: dict) -> str:
    """Determine which step the session is at."""
    if not state:
        return "uploaded"
    if state.get("job_status") == "completed":
        return "trained"
    if state.get("job_id"):
        return "training"
    if state.get("package_path"):
        return "generated"
    if state.get("final_task_type"):
        return "planned"
    if state.get("inferred_task_type"):
        return "analyzed"
    if state.get("raw_data"):
        return "ingested"
    return "uploaded"


# ============================================================================
# COLAB NOTEBOOK GENERATION
# ============================================================================

@app.post("/generate-colab", response_model=ColabResponse)
async def generate_colab_notebook(request: ColabRequest):
    """Generate a Google Colab notebook for training. Requires /plan to be completed first."""
    session_id = request.session_id
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    state = session.get("state")
    
    if not state or not state.get("training_config"):
        raise HTTPException(status_code=400, detail="Run /plan first to configure training")
    
    try:
        data_path = state.get("file_path")
        if not data_path or not os.path.exists(data_path):
            raise HTTPException(status_code=404, detail="Dataset file not found")
        
        with open(data_path, "r") as f:
            dataset_jsonl = f.read()
        
        model_id = state["training_config"].get("model_name", "microsoft/Phi-4-mini-instruct")
        model_name = state.get("model_name", "Phi-4 Mini")
        
        analysis = {
            "num_examples": state.get("num_rows", 0),
            "task_type": state.get("final_task_type", "chat"),
            "model_size": state.get("model_size", "3B"),
        }
        
        notebook_content = generate_training_notebook(
            dataset_jsonl=dataset_jsonl,
            model_id=model_id,
            model_name=model_name,
            analysis=analysis,
        )
        
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)
        notebook_filename = f"tunekit_train_{session_id[:8]}.ipynb"
        notebook_path = os.path.join(output_dir, notebook_filename)
        
        save_notebook(notebook_content, notebook_path)
        save_package_mapping(session_id, notebook_path)
        gist_result = upload_to_gist(notebook_content, notebook_filename)

        colab_url = gist_result.get("colab_url")
        gist_url = gist_result.get("gist_url")

        return ColabResponse(
            session_id=session_id,
            notebook_url=f"/download-notebook/{session_id}",
            message="Notebook generated successfully!",
            status="success",
            colab_url=colab_url,
            gist_url=gist_url,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate notebook: {str(e)}")


@app.get("/download-notebook/{session_id}")
async def download_notebook(session_id: str):
    """Download the generated Colab notebook."""
    mapping = load_package_mapping()
    notebook_path = mapping.get(session_id)
    
    if not notebook_path and session_id in sessions:
        state = sessions[session_id].get("state", {})
        notebook_path = state.get("notebook_path")
    
    if not notebook_path or not os.path.exists(notebook_path):
        raise HTTPException(status_code=404, detail="Notebook not found. Please generate it first.")
    
    return FileResponse(
        path=notebook_path,
        media_type="application/x-ipynb+json",
        filename=os.path.basename(notebook_path),
    )


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
