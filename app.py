#!/usr/bin/env python3
"""
FastAPI application for TikTok Data Processing Pipeline.
Provides REST API endpoints to trigger and monitor pipeline execution.
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Path as FastAPIPath
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# Import your existing PipelineExecutor
from main import PipelineExecutor

# Pydantic Models
class PipelineRequest(BaseModel):
    config_path: Optional[str] = Field(default="config/main.yaml", description="Path to pipeline configuration file")
    execution_name: Optional[str] = Field(default=None, description="Custom name for this execution")
    override_config: Optional[Dict[str, Any]] = Field(default=None, description="Configuration overrides")

class ReviewValidationRequest(BaseModel):
    text: str = Field(..., description="Review text content")
    business_name: Optional[str] = Field(None, description="Business name")
    author_name: Optional[str] = Field(None, description="Review author name")
    rating: Optional[float] = Field(None, description="Review rating (1-5)")
    time: Optional[str] = Field(None, description="Review time/date")
    additional_data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional review metadata")

class ReviewValidationResponse(BaseModel):
    is_valid: bool = Field(..., description="Whether the review is predicted to be valid")
    confidence: float = Field(..., description="Model confidence (0-1)")
    probability_valid: float = Field(..., description="Probability that review is valid")
    probability_invalid: float = Field(..., description="Probability that review is invalid")
    stage_used: Optional[int] = Field(None, description="Which ML stage was used for prediction")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    model_version: str = Field(..., description="Version of model used")

class PipelineResponse(BaseModel):
    execution_id: str
    status: str
    message: str
    started_at: datetime
    config_path: str

class ExecutionStatus(BaseModel):
    execution_id: str
    status: str  # "running", "completed", "failed", "not_found"
    started_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    output_directory: Optional[str] = None
    log_file: Optional[str] = None

class ExecutionList(BaseModel):
    executions: List[ExecutionStatus]
    total_count: int

# Global storage for execution tracking
execution_tracker: Dict[str, ExecutionStatus] = {}
executor_pool = ThreadPoolExecutor(max_workers=3)

# Global model cache for review validation
_cached_pipeline = None
_pipeline_cache_time = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown events."""
    # Startup
    logging.info("FastAPI TikTok Pipeline Service starting up...")
    
    # Create necessary directories
    Path('logs').mkdir(exist_ok=True)
    Path('data').mkdir(exist_ok=True)
    Path('config').mkdir(exist_ok=True)
    
    yield
    
    # Shutdown
    logging.info("FastAPI TikTok Pipeline Service shutting down...")
    executor_pool.shutdown(wait=True)

# Initialize FastAPI app
app = FastAPI(
    title="TikTok Data Processing Pipeline API",
    description="REST API for executing and monitoring TikTok data processing pipelines",
    version="1.0.0",
    lifespan=lifespan
)

def run_pipeline_sync(execution_id: str, config_path: str, override_config: Optional[Dict[str, Any]] = None):
    """Run pipeline synchronously in background thread."""
    try:
        # Update status to running
        if execution_id in execution_tracker:
            execution_tracker[execution_id].status = "running"
        
        # Create custom pipeline executor
        executor = PipelineExecutor(config_path)
        executor.execution_id = execution_id
        
        # Apply configuration overrides if provided
        if override_config:
            # Merge override config with existing config
            def deep_merge(base_dict: dict, override_dict: dict) -> dict:
                result = base_dict.copy()
                for key, value in override_dict.items():
                    if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                        result[key] = deep_merge(result[key], value)
                    else:
                        result[key] = value
                return result
            
            executor.config = deep_merge(executor.config, override_config)
        
        # Setup logging for this execution
        executor._setup_logging()
        
        # Run the pipeline
        executor.run()
        
        # Update status to completed
        if execution_id in execution_tracker:
            execution_tracker[execution_id].status = "completed"
            execution_tracker[execution_id].completed_at = datetime.now()
            execution_tracker[execution_id].output_directory = executor.execution_output_dir
            execution_tracker[execution_id].log_file = f"logs/pipeline_{execution_id}.log"
        
        logging.info(f"Pipeline execution {execution_id} completed successfully")
        
    except Exception as e:
        error_msg = str(e)
        logging.error(f"Pipeline execution {execution_id} failed: {error_msg}")
        
        # Update status to failed
        if execution_id in execution_tracker:
            execution_tracker[execution_id].status = "failed"
            execution_tracker[execution_id].completed_at = datetime.now()
            execution_tracker[execution_id].error_message = error_msg

async def get_or_load_pipeline():
    """Get cached pipeline or load it from models directory"""
    global _cached_pipeline, _pipeline_cache_time
    
    # Check if we need to load/reload the pipeline
    models_dir = Path("models")
    if not models_dir.exists():
        raise HTTPException(status_code=503, detail="Models directory not found. Please train the pipeline first.")
    
    # Check for required model files
    required_files = ['calibrated_model.pkl', 'text_encoder.pkl', 'text_features.pkl', 'context_features.pkl', 'scaler.pkl']
    missing_files = [f for f in required_files if not (models_dir / f).exists()]
    
    if missing_files:
        raise HTTPException(
            status_code=503, 
            detail=f"Missing model files: {missing_files}. Please train the pipeline first."
        )
    
    # Check if models were updated
    latest_model_time = max((models_dir / f).stat().st_mtime for f in required_files)
    
    if _cached_pipeline is None or _pipeline_cache_time is None or latest_model_time > _pipeline_cache_time:
        logging.info("Loading/reloading ML pipeline...")
        try:
            # Import the rigorous ML pipeline
            from src.03_model_processing_v3 import RigorousMLPipeline
            
            _cached_pipeline = RigorousMLPipeline.load_models(models_dir)
            _pipeline_cache_time = latest_model_time
            logging.info("✅ ML pipeline loaded successfully")
            
        except Exception as e:
            logging.error(f"❌ Failed to load ML pipeline: {e}")
            raise HTTPException(status_code=503, detail=f"Failed to load ML pipeline: {str(e)}")
    
    return _cached_pipeline

async def validate_single_review(review_data: ReviewValidationRequest) -> ReviewValidationResponse:
    """Validate a single review using the ML pipeline"""
    import time
    import pandas as pd
    import numpy as np
    
    start_time = time.time()
    
    logging.info("="*60)
    logging.info("🤖 STARTING AI REVIEW VALIDATION")
    logging.info(f"📝 Review Text: '{review_data.text[:100]}...'")
    logging.info(f"🏢 Business: {review_data.business_name}")
    logging.info(f"👤 Author: {review_data.author_name}")
    logging.info(f"⭐ Rating: {review_data.rating}")
    
    # Get the ML pipeline
    logging.info("🔄 Loading ML pipeline...")
    pipeline = await get_or_load_pipeline()
    logging.info(f"✅ Pipeline loaded: {type(pipeline).__name__}")
    
    # Convert review data to DataFrame format expected by pipeline
    logging.info("🔧 Preparing features for ML pipeline...")
    review_df = pd.DataFrame([{
        'text': review_data.text,
        'business_name': review_data.business_name or 'Unknown Business',
        'author_name': review_data.author_name or 'Anonymous',
        'rating': review_data.rating or 3.0,
        'time': review_data.time or '',
        'avg_rating': review_data.additional_data.get('avg_rating', 4.0),
        'num_of_reviews': review_data.additional_data.get('num_of_reviews', 100),
        'pics': review_data.additional_data.get('pics', 0),
        'category': review_data.additional_data.get('category', 'Restaurant'),
        'state': review_data.additional_data.get('state', 'unknown'),
        'price': review_data.additional_data.get('price', 'unknown'),
        'latitude': review_data.additional_data.get('latitude', 0.0),
        'longitude': review_data.additional_data.get('longitude', 0.0),
        'has_business_response': review_data.additional_data.get('has_business_response', False),
        'response_length': review_data.additional_data.get('response_length', 0),
        'text_length': len(review_data.text),
        'word_count': len(review_data.text.split()),
        'exclamation_count': review_data.text.count('!'),
        'caps_ratio': sum(1 for c in review_data.text if c.isupper()) / max(len(review_data.text), 1)
    }])
    
    logging.info(f"📊 DataFrame shape: {review_df.shape}")
    logging.info(f"📊 DataFrame columns: {list(review_df.columns)}")
    
    try:
        # Run ML inference
        logging.info("🧠 Running ML inference...")
        inference_start = time.time()
        
        probabilities = pipeline.predict_proba(review_df)
        predictions = pipeline.predict(review_df)
        
        inference_time = (time.time() - inference_start) * 1000
        logging.info(f"⚡ ML inference completed in {inference_time:.2f}ms")
        
        # Extract results
        prob_invalid = float(probabilities[0])  # Probability of being invalid/junk
        prob_valid = 1.0 - prob_invalid
        is_valid = bool(predictions[0] == 0)  # 0 = valid, 1 = invalid
        confidence = float(abs(prob_invalid - 0.5) * 2)  # 0 = uncertain, 1 = very confident
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        logging.info("📊 ML PREDICTION RESULTS:")
        logging.info(f"   🎯 Prediction: {'VALID' if is_valid else 'INVALID'}")
        logging.info(f"   📈 Probability Valid: {prob_valid:.3f} ({prob_valid*100:.1f}%)")
        logging.info(f"   📉 Probability Invalid: {prob_invalid:.3f} ({prob_invalid*100:.1f}%)")
        logging.info(f"   🎚️ Confidence: {confidence:.3f} ({confidence*100:.1f}%)")
        logging.info(f"   ⏱️ Total Processing Time: {processing_time:.2f}ms")
        logging.info("="*60)
        
        return ReviewValidationResponse(
            is_valid=is_valid,
            confidence=confidence,
            probability_valid=prob_valid,
            probability_invalid=prob_invalid,
            stage_used=None,  # Would need to modify pipeline to track this
            processing_time_ms=processing_time,
            model_version="rigorous_ml_v3"
        )
        
    except Exception as e:
        logging.error(f"❌ Error during review validation: {e}")
        logging.error(f"❌ Exception type: {type(e).__name__}")
        import traceback
        logging.error(f"❌ Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@app.post("/review/validate", response_model=ReviewValidationResponse)
async def validate_review(request: ReviewValidationRequest):
    """Validate a single review using the trained ML pipeline."""
    try:
        return await validate_single_review(request)
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"❌ Unexpected error in review validation: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/model/status")
async def get_model_status():
    """Get the current status of loaded ML models."""
    import os
    from pathlib import Path
    
    models_dir = Path("models")
    model_info = {
        "models_directory_exists": models_dir.exists(),
        "cached_pipeline_loaded": _cached_pipeline is not None,
        "cache_time": _pipeline_cache_time,
        "available_models": {},
        "missing_models": []
    }
    
    if models_dir.exists():
        required_files = ['calibrated_model.pkl', 'text_encoder.pkl', 'text_features.pkl', 'context_features.pkl', 'scaler.pkl']
        optional_files = ['ensemble_model.pkl', 'training_metrics.json', 'feature_importance.pkl', 'pipeline_state.json']
        
        for file in required_files + optional_files:
            file_path = models_dir / file
            if file_path.exists():
                stat = file_path.stat()
                model_info["available_models"][file] = {
                    "size_bytes": stat.st_size,
                    "size_mb": round(stat.st_size / 1024 / 1024, 2),
                    "modified_time": stat.st_mtime,
                    "required": file in required_files
                }
            else:
                if file in required_files:
                    model_info["missing_models"].append(file)
    
    # Check if pipeline is ready
    model_info["pipeline_ready"] = (
        model_info["models_directory_exists"] and 
        len(model_info["missing_models"]) == 0
    )
    
    if _cached_pipeline is not None:
        model_info["pipeline_type"] = type(_cached_pipeline).__name__
        model_info["pipeline_methods"] = [method for method in dir(_cached_pipeline) if not method.startswith('_')]
    
    return model_info

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "TikTok Data Processing Pipeline API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "validate_review": "POST /review/validate",
            "model_status": "GET /model/status",
            "start_pipeline": "POST /pipeline/start",
            "get_status": "GET /pipeline/status/{execution_id}",
            "list_executions": "GET /pipeline/executions",
            "get_logs": "GET /pipeline/logs/{execution_id}",
            "health": "GET /health"
        }
    }

@app.post("/pipeline/start", response_model=PipelineResponse)
async def start_pipeline(
    request: PipelineRequest,
    background_tasks: BackgroundTasks
):
    """Start a new pipeline execution."""
    try:
        # Validate config file exists
        if not os.path.exists(request.config_path):
            raise HTTPException(
                status_code=400, 
                detail=f"Configuration file not found: {request.config_path}"
            )
        
        # Generate execution ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        execution_id = f"{request.execution_name}_{timestamp}" if request.execution_name else timestamp
        
        # Create execution status record
        execution_status = ExecutionStatus(
            execution_id=execution_id,
            status="queued",
            started_at=datetime.now()
        )
        
        execution_tracker[execution_id] = execution_status
        
        # Start pipeline in background
        background_tasks.add_task(
            run_pipeline_sync,
            execution_id,
            request.config_path,
            request.override_config
        )
        
        return PipelineResponse(
            execution_id=execution_id,
            status="queued",
            message="Pipeline execution started",
            started_at=execution_status.started_at,
            config_path=request.config_path
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start pipeline: {str(e)}")

@app.get("/pipeline/status/{execution_id}", response_model=ExecutionStatus)
async def get_pipeline_status(execution_id: str = FastAPIPath(..., description="Pipeline execution ID")):
    """Get status of a specific pipeline execution."""
    if execution_id not in execution_tracker:
        raise HTTPException(status_code=404, detail=f"Execution {execution_id} not found")
    
    return execution_tracker[execution_id]

@app.get("/pipeline/executions", response_model=ExecutionList)
async def list_executions(
    limit: int = Query(default=10, ge=1, le=100, description="Maximum number of executions to return"),
    offset: int = Query(default=0, ge=0, description="Number of executions to skip"),
    status: Optional[str] = Query(default=None, description="Filter by status")
):
    """List all pipeline executions with optional filtering."""
    executions = list(execution_tracker.values())
    
    # Filter by status if provided
    if status:
        executions = [e for e in executions if e.status == status]
    
    # Sort by start time (most recent first)
    executions.sort(key=lambda x: x.started_at, reverse=True)
    
    # Apply pagination
    total_count = len(executions)
    paginated_executions = executions[offset:offset + limit]
    
    return ExecutionList(
        executions=paginated_executions,
        total_count=total_count
    )

@app.get("/pipeline/logs/{execution_id}")
async def get_pipeline_logs(execution_id: str = FastAPIPath(..., description="Pipeline execution ID")):
    """Get logs for a specific pipeline execution."""
    if execution_id not in execution_tracker:
        raise HTTPException(status_code=404, detail=f"Execution {execution_id} not found")
    
    log_file = f"logs/pipeline_{execution_id}.log"
    
    if not os.path.exists(log_file):
        raise HTTPException(status_code=404, detail=f"Log file not found for execution {execution_id}")
    
    return FileResponse(
        path=log_file,
        media_type='text/plain',
        filename=f"pipeline_{execution_id}.log"
    )

@app.get("/pipeline/output/{execution_id}")
async def list_output_files(execution_id: str = FastAPIPath(..., description="Pipeline execution ID")):
    """List output files for a specific pipeline execution."""
    if execution_id not in execution_tracker:
        raise HTTPException(status_code=404, detail=f"Execution {execution_id} not found")
    
    execution_status = execution_tracker[execution_id]
    
    if not execution_status.output_directory or not os.path.exists(execution_status.output_directory):
        raise HTTPException(status_code=404, detail=f"Output directory not found for execution {execution_id}")
    
    output_dir = Path(execution_status.output_directory)
    files = []
    
    for file_path in output_dir.rglob("*"):
        if file_path.is_file():
            relative_path = file_path.relative_to(output_dir)
            files.append({
                "filename": file_path.name,
                "path": str(relative_path),
                "size": file_path.stat().st_size,
                "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            })
    
    return {
        "execution_id": execution_id,
        "output_directory": str(output_dir),
        "files": files,
        "total_files": len(files)
    }

@app.delete("/pipeline/execution/{execution_id}")
async def delete_execution(execution_id: str = FastAPIPath(..., description="Pipeline execution ID")):
    """Delete an execution record (does not stop running executions)."""
    if execution_id not in execution_tracker:
        raise HTTPException(status_code=404, detail=f"Execution {execution_id} not found")
    
    execution_status = execution_tracker[execution_id]
    
    if execution_status.status == "running":
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot delete running execution {execution_id}"
        )
    
    del execution_tracker[execution_id]
    
    return {"message": f"Execution {execution_id} deleted successfully"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_executions": len([e for e in execution_tracker.values() if e.status == "running"]),
        "total_executions": len(execution_tracker)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
