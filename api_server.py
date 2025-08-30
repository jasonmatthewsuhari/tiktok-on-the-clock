"""
FastAPI server for single review processing through the pipeline.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import pandas as pd
import tempfile
import os
import yaml
import logging
from pathlib import Path
import importlib
import sys
from datetime import datetime
import uuid

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

app = FastAPI(title="Review Pipeline API", version="1.0.0")

# Enable CORS for browser extension
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ReviewInput(BaseModel):
    """Input model for single review processing"""
    user_id: Optional[str] = None
    author_name: str
    time: Optional[int] = None
    rating: float
    text: str
    pics: Optional[str] = ""
    resp: Optional[str] = ""
    gmap_id: Optional[str] = None
    business_name: str
    address: Optional[str] = ""
    description: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    category: Optional[str] = ""
    avg_rating: Optional[float] = None
    num_of_reviews: Optional[int] = None
    price: Optional[str] = ""
    hours: Optional[str] = ""
    state: Optional[str] = ""
    url: Optional[str] = ""

class PipelineResult(BaseModel):
    """Result model for pipeline processing"""
    success: bool
    processing_id: str
    input_data: Dict[str, Any]
    stage_results: Dict[str, Any]
    final_predictions: Dict[str, Any]
    processing_time: float
    error_message: Optional[str] = None

class SingleRowPipelineExecutor:
    """Execute pipeline stages for a single row of data"""
    
    def __init__(self, config_path: str = "config/main.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.processing_id = str(uuid.uuid4())[:8]
        self._setup_logging()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            return config
        except Exception as e:
            logging.error(f"Failed to load config: {e}")
            raise
    
    def _setup_logging(self):
        """Setup logging for API processing."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(level=logging.INFO, format=log_format)
        
    def _create_temp_csv(self, review_data: ReviewInput) -> str:
        """Create temporary CSV file with single review."""
        # Convert Pydantic model to dictionary
        data_dict = review_data.dict()
        
        # Add current timestamp if not provided
        if not data_dict.get('time'):
            data_dict['time'] = int(datetime.now().timestamp() * 1000)
        
        # Create DataFrame with single row
        df = pd.DataFrame([data_dict])
        
        # Create temporary file
        temp_dir = Path("temp_api")
        temp_dir.mkdir(exist_ok=True)
        
        temp_file = temp_dir / f"single_review_{self.processing_id}.csv"
        df.to_csv(temp_file, index=False, encoding='utf-8')
        
        logging.info(f"Created temporary CSV: {temp_file}")
        return str(temp_file)
    
    def _execute_stage(self, stage: Dict[str, Any], input_file: str, output_file: str) -> bool:
        """Execute a single pipeline stage."""
        stage_name = stage.get('name', 'Unknown')
        module_name = stage.get('module', '')
        function_name = stage.get('function', 'run')
        stage_config = stage.get('config', {})
        
        if not stage.get('enabled', True):
            logging.info(f"Stage '{stage_name}' is disabled, skipping...")
            return True
        
        logging.info(f"Executing stage: {stage_name}")
        
        try:
            # Import the module dynamically
            module = importlib.import_module(module_name)
            
            # Get the function to execute
            if hasattr(module, function_name):
                func = getattr(module, function_name)
                
                if callable(func):
                    # Prepare stage configuration
                    enhanced_config = stage_config.copy()
                    enhanced_config['input_file'] = input_file
                    enhanced_config['output_file'] = output_file
                    enhanced_config['execution_id'] = self.processing_id
                    # Set execution_output_dir to avoid file search issues
                    enhanced_config['execution_output_dir'] = str(Path("temp_api"))
                    
                    logging.info(f"Executing {stage_name} with config: {enhanced_config}")
                    
                    # Execute the function
                    result = func(enhanced_config)
                    logging.info(f"Stage '{stage_name}' completed with result: {result}")
                    return result
                else:
                    error_msg = f"'{function_name}' in module '{module_name}' is not callable"
                    logging.error(error_msg)
                    raise ValueError(error_msg)
            else:
                error_msg = f"Function '{function_name}' not found in module '{module_name}'"
                logging.error(error_msg)
                raise AttributeError(error_msg)
                
        except ImportError as e:
            error_msg = f"Failed to import module '{module_name}': {e}"
            logging.error(error_msg)
            raise ImportError(error_msg)
        except Exception as e:
            error_msg = f"Error executing stage '{stage_name}': {e}"
            logging.error(error_msg)
            raise Exception(error_msg)
    
    def process_single_review(self, review_data: ReviewInput) -> Dict[str, Any]:
        """Process a single review through the entire pipeline."""
        start_time = datetime.now()
        
        try:
            # Create temporary CSV with single review
            input_file = self._create_temp_csv(review_data)
            
            # Get pipeline stages
            stages = self.config.get('pipeline', {}).get('stages', [])
            if not stages:
                raise ValueError("No stages defined in pipeline configuration")
            
            logging.info(f"Processing single review through {len(stages)} stages")
            
            stage_results = {}
            current_file = input_file
            
            # Execute each stage
            for i, stage in enumerate(stages, 1):
                stage_name = stage.get('name', f'stage_{i}')
                
                # Determine output file for this stage
                temp_dir = Path("temp_api")
                
                # For stage 1, copy input to expected name
                if i == 1:
                    expected_input = temp_dir / "stage1_output.csv"
                    if current_file != str(expected_input):
                        import shutil
                        shutil.copy2(current_file, expected_input)
                        current_file = str(expected_input)
                
                # Set standard output file name
                output_file = str(temp_dir / f"stage{i}_output.csv")
                
                logging.info(f"Processing stage {i}/{len(stages)}: {stage_name}")
                
                # Execute stage
                try:
                    success = self._execute_stage(stage, current_file, output_file)
                    
                    if not success:
                        raise RuntimeError(f"Stage '{stage_name}' returned False")
                        
                except Exception as stage_error:
                    logging.error(f"Stage '{stage_name}' failed with error: {stage_error}")
                    raise RuntimeError(f"Stage '{stage_name}' failed: {str(stage_error)}")
                
                # Read stage output for results
                if os.path.exists(output_file):
                    stage_df = pd.read_csv(output_file, encoding='utf-8', sep=';')
                    stage_results[stage_name] = {
                        'row_count': len(stage_df),
                        'columns': list(stage_df.columns),
                        'sample_data': stage_df.iloc[0].to_dict() if len(stage_df) > 0 else {}
                    }
                    current_file = output_file
                else:
                    stage_results[stage_name] = {'error': 'No output file generated'}
            
            # Read final results
            final_df = pd.read_csv(current_file, encoding='utf-8', sep=';')
            final_predictions = final_df.iloc[0].to_dict() if len(final_df) > 0 else {}
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Cleanup temporary files
            self._cleanup_temp_files()
            
            return {
                'success': True,
                'processing_id': self.processing_id,
                'input_data': review_data.dict(),
                'stage_results': stage_results,
                'final_predictions': final_predictions,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logging.error(f"Pipeline processing failed: {e}")
            self._cleanup_temp_files()
            
            return {
                'success': False,
                'processing_id': self.processing_id,
                'input_data': review_data.dict(),
                'stage_results': {},
                'final_predictions': {},
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'error_message': str(e)
            }
    
    def _cleanup_temp_files(self):
        """Clean up temporary files created during processing."""
        temp_dir = Path("temp_api")
        if temp_dir.exists():
            for file in temp_dir.glob(f"*{self.processing_id}*"):
                try:
                    file.unlink()
                    logging.info(f"Cleaned up temporary file: {file}")
                except Exception as e:
                    logging.warning(f"Failed to cleanup {file}: {e}")

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"message": "Review Pipeline API is running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    """Detailed health check with pipeline status."""
    try:
        # Test config loading
        with open("config/main.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        stages = config.get('pipeline', {}).get('stages', [])
        
        return {
            "status": "healthy",
            "pipeline_stages": len(stages),
            "config_loaded": True,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/process_review", response_model=PipelineResult)
async def process_review(review: ReviewInput):
    """Process a single review through the entire pipeline."""
    try:
        logging.info(f"Received review processing request for: {review.business_name}")
        
        # Create pipeline executor
        executor = SingleRowPipelineExecutor()
        
        # Process the review
        result = executor.process_single_review(review)
        
        return PipelineResult(**result)
        
    except Exception as e:
        logging.error(f"API processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pipeline_info")
async def get_pipeline_info():
    """Get information about the pipeline configuration."""
    try:
        with open("config/main.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        stages = config.get('pipeline', {}).get('stages', [])
        stage_info = []
        
        for stage in stages:
            stage_info.append({
                'name': stage.get('name'),
                'module': stage.get('module'),
                'enabled': stage.get('enabled', True),
                'description': stage.get('config', {}).get('description', '')
            })
        
        return {
            "pipeline_name": config.get('pipeline', {}).get('name', 'Unknown'),
            "description": config.get('pipeline', {}).get('description', ''),
            "total_stages": len(stages),
            "stages": stage_info
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load pipeline info: {e}")

if __name__ == "__main__":
    import uvicorn
    
    # Ensure temp directory exists
    Path("temp_api").mkdir(exist_ok=True)
    
    print("🚀 Starting Review Pipeline API Server...")
    print("📚 API Documentation available at: http://localhost:8000/docs")
    print("🔍 Health check available at: http://localhost:8000/health")
    
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True)
