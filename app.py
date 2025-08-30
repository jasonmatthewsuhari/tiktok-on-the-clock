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

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
        CORSMiddleware,
    allow_origins=["*"],   # Or restrict to your frontend origin
    allow_methods=["*"],
    allow_headers=["*"],
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

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "TikTok Data Processing Pipeline API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
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
