#!/usr/bin/env python3
"""
Main entry point for the TikTok Data Processing Pipeline.
Loads configuration from YAML and executes pipeline stages.
"""

import os
import sys
import yaml
import importlib
import logging
from pathlib import Path
from typing import Dict, Any, List
import traceback
from datetime import datetime

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class PipelineExecutor:
    """Executes pipeline stages based on YAML configuration."""
    
    def __init__(self, config_path: str = "config/main.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.execution_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._setup_logging()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            logging.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            print("Something went wrong: ", e)
            sys.exit(1)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level = self.config.get('pipeline', {}).get('config', {}).get('log_level', 'INFO')
        
        # Create logs directory if it doesn't exist
        logs_dir = Path('logs')
        logs_dir.mkdir(exist_ok=True)
        
        # Create timestamped log file
        log_file = logs_dir / f"pipeline_{self.execution_id}.log"
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file)
            ]
        )
        
        logging.info(f"Pipeline execution started with ID: {self.execution_id}")
        logging.info(f"Log file: {log_file}")
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        config = self.config.get('pipeline', {}).get('config', {})
        data_dir = config.get('data_dir', 'data/')
        output_dir = config.get('output_dir', 'data/output/')
        
        # Create execution-specific output directory with timestamp
        execution_output_dir = Path(output_dir) / self.execution_id
        
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        execution_output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info(f"Created directories: {data_dir}, {output_dir}")
        logging.info(f"Execution output directory: {execution_output_dir}")
        
        # Store execution output directory for stages to use
        self.execution_output_dir = str(execution_output_dir)
    
    def _execute_stage(self, stage: Dict[str, Any]) -> bool:
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
                
                # Execute the function with stage configuration
                if callable(func):
                    # Add execution directory to stage config
                    enhanced_config = stage_config.copy()
                    enhanced_config['execution_output_dir'] = self.execution_output_dir
                    enhanced_config['execution_id'] = self.execution_id
                    
                    result = func(enhanced_config)
                    logging.info(f"Stage '{stage_name}' completed successfully")
                    return True
                else:
                    logging.error(f"'{function_name}' in module '{module_name}' is not callable")
                    return False
            else:
                logging.error(f"Function '{function_name}' not found in module '{module_name}'")
                return False
                
        except ImportError as e:
            logging.error(f"Failed to import module '{module_name}': {e}")
            return False
        except Exception as e:
            logging.error(f"Error executing stage '{stage_name}': {e}")
            logging.error(traceback.format_exc())
            return False
        
    # Add this method to your PipelineExecutor class
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get a summary of the pipeline execution."""
        return {
            "execution_id": self.execution_id,
            "config_path": self.config_path,
            "output_directory": getattr(self, 'execution_output_dir', None),
            "pipeline_name": self.config.get('pipeline', {}).get('name', 'Unknown'),
            "stages_count": len(self.config.get('pipeline', {}).get('stages', [])),
            "timestamp": datetime.now().isoformat()
        }

    
    def run(self):
        """Run the complete pipeline."""
        pipeline_config = self.config.get('pipeline', {})
        pipeline_name = pipeline_config.get('name', 'Unknown Pipeline')
        
        logging.info(f"Starting pipeline: {pipeline_name}")
        logging.info(f"Pipeline description: {pipeline_config.get('description', 'No description')}")
        logging.info(f"Execution ID: {self.execution_id}")
        
        # Create necessary directories
        self._create_directories()
        
        # Get execution configuration
        exec_config = pipeline_config.get('execution', {})
        stop_on_error = exec_config.get('stop_on_error', True)
        max_retries = exec_config.get('max_retries', 3)
        
        # Execute stages
        stages = pipeline_config.get('stages', [])
        if not stages:
            logging.warning("No stages defined in pipeline configuration")
            return
        
        logging.info(f"Found {len(stages)} pipeline stages")
        
        for i, stage in enumerate(stages, 1):
            logging.info(f"Processing stage {i}/{len(stages)}: {stage.get('name', 'Unknown')}")
            
            success = False
            retry_count = 0
            
            while not success and retry_count < max_retries:
                if retry_count > 0:
                    logging.info(f"Retrying stage '{stage.get('name')}' (attempt {retry_count + 1}/{max_retries})")
                
                success = self._execute_stage(stage)
                
                if not success:
                    retry_count += 1
                    if retry_count >= max_retries:
                        logging.error(f"Stage '{stage.get('name')}' failed after {max_retries} attempts")
                        if stop_on_error:
                            logging.error("Pipeline execution stopped due to stage failure")
                            return
                    else:
                        logging.warning(f"Stage '{stage.get('name')}' failed, will retry...")
        
        logging.info("Pipeline execution completed successfully!")
        logging.info(f"Execution ID: {self.execution_id}")
        logging.info(f"Output directory: {self.execution_output_dir}")

def main():
    try:
        config_path = "config/main.yaml"
        executor = PipelineExecutor(config_path)
        executor.run()
    except KeyboardInterrupt:
        logging.info("Pipeline execution interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
