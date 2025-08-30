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
import argparse
from pathlib import Path
from typing import Dict, Any, List
import traceback
from datetime import datetime

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class PipelineExecutor:
    """Executes pipeline stages based on YAML configuration."""
    
    def __init__(self, config_path: str = "config/main.yaml", skip_stages: List[str] = None):
        self.config_path = config_path
        self.config = self._load_config()
        self.execution_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.skip_stages = skip_stages or []
        self._setup_logging()
        
        if self.skip_stages:
            logging.info(f"Skipping stages: {', '.join(self.skip_stages)}")
        
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
        """Setup logging configuration with immediate file flushing."""
        log_level = self.config.get('pipeline', {}).get('config', {}).get('log_level', 'INFO')
        
        # Create logs directory if it doesn't exist
        logs_dir = Path('logs')
        logs_dir.mkdir(exist_ok=True)
        
        # Create timestamped log file
        log_file = logs_dir / f"pipeline_{self.execution_id}.log"
        
        # Create file handler with immediate flushing
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level))
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        
        # Create stream handler for console output with UTF-8 encoding
        import io
        utf8_stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        stream_handler = logging.StreamHandler(utf8_stdout)
        stream_handler.setLevel(getattr(logging, log_level))
        stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level))
        
        # Clear any existing handlers
        root_logger.handlers.clear()
        
        # Add our handlers
        root_logger.addHandler(file_handler)
        root_logger.addHandler(stream_handler)
        
        # Create a custom logging function that forces immediate flush
        original_handle = logging.Handler.handle
        
        def handle_with_flush(self, record):
            original_handle(self, record)
            if hasattr(self, 'flush'):
                self.flush()
        
        logging.Handler.handle = handle_with_flush
        
        logging.info(f"Pipeline execution started with ID: {self.execution_id}")
        logging.info(f"Live logging enabled - log file: {log_file}")
    
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
                    # Add execution directory and previous stage to config
                    enhanced_config = stage_config.copy()
                    enhanced_config['execution_output_dir'] = self.execution_output_dir
                    enhanced_config['execution_id'] = self.execution_id
                    enhanced_config['previous_stage'] = stage.get('previous_stage')
                    
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
        stages = self.config.get('pipeline', {}).get('stages', [])
        if not stages:
            logging.warning("No stages defined in pipeline configuration")
            return False
        
        # Filter out skipped stages
        filtered_stages = []
        for stage in stages:
            stage_name = stage.get('name', 'Unknown')
            if stage_name in self.skip_stages:
                display_name = stage.get('display_name', stage_name)
                logging.info(f"Skipping stage: {display_name}")
                continue
            filtered_stages.append(stage)
        
        logging.info(f"Starting pipeline with {len(filtered_stages)} stages (skipped {len(stages) - len(filtered_stages)})")
        
        previous_stage_name = None
        for i, stage in enumerate(filtered_stages, 1):
            stage_name = stage.get('name', 'Unknown')
            display_name = stage.get('display_name', stage_name)
            logging.info(f"\nProcessing stage {i}/{len(filtered_stages)}: {display_name}")
            
            # Add previous stage info to the stage config
            enhanced_stage = stage.copy()
            if previous_stage_name:
                enhanced_stage['previous_stage'] = previous_stage_name
                logging.info(f"Previous executed stage: {previous_stage_name}")
            
            success = False
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    if self._execute_stage(enhanced_stage):
                        success = True
                        previous_stage_name = stage_name  # Update for next iteration
                        break
                    else:
                        retry_count += 1
                        if retry_count < max_retries:
                            logging.info(f"Retrying {display_name} (attempt {retry_count + 1}/{max_retries})")
                        else:
                            logging.error(f"{display_name} failed after {max_retries} attempts")
                            break
                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        logging.warning(f"{display_name} failed, will retry...")
                        logging.warning(f"Error: {e}")
                    else:
                        logging.error(f"{display_name} failed after {max_retries} attempts")
                        logging.error(f"Error: {e}")
                        break
            
            if not success:
                logging.error("Pipeline execution stopped due to stage failure")
                return False
        
        logging.info("Pipeline execution completed successfully!")
        logging.info(f"Execution ID: {self.execution_id}")
        logging.info(f"Output directory: {self.execution_output_dir}")
        return True

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="TikTok Data Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Run all stages
  python main.py --skip 01_take_input_csv          # Skip stage 1
  python main.py --skip 03_model_processing 05_evaluation  # Skip stages 3 and 5
  python main.py --config custom_config.yaml       # Use custom config file
        """
    )
    
    parser.add_argument(
        '--skip', 
        nargs='*', 
        default=[],
        metavar='STAGE',
        help='List of stage names to skip (e.g., 01_take_input_csv 03_model_processing)'
    )
    
    parser.add_argument(
        '--config',
        default='config/main.yaml',
        metavar='FILE',
        help='Path to configuration YAML file (default: config/main.yaml)'
    )
    
    parser.add_argument(
        '--list-stages',
        action='store_true',
        help='List all available stages and exit'
    )
    
    return parser.parse_args()

def list_available_stages(config_path: str):
    """List all available stages from the configuration."""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        stages = config.get('pipeline', {}).get('stages', [])
        print("Available pipeline stages:")
        print("=" * 50)
        
        for i, stage in enumerate(stages, 1):
            stage_name = stage.get('name', 'Unknown')
            display_name = stage.get('display_name', stage_name)
            print(f"{i:2d}. {stage_name:<25} - {display_name}")
        
        print(f"\nTotal stages: {len(stages)}")
        print("\nUsage examples:")
        if stages:
            print(f"  python main.py --skip {stages[0].get('name', '')}")
            if len(stages) > 2:
                print(f"  python main.py --skip {stages[0].get('name', '')} {stages[2].get('name', '')}")
        
    except Exception as e:
        print(f"Error reading configuration: {e}")
        sys.exit(1)

if __name__ == "__main__":
    args = parse_arguments()
    
    # List stages if requested
    if args.list_stages:
        list_available_stages(args.config)
        sys.exit(0)
    
    # Validate skip stages
    if args.skip:
        print(f"Starting pipeline with skipped stages: {', '.join(args.skip)}")
    
    try:
        executor = PipelineExecutor(config_path=args.config, skip_stages=args.skip)
        success = executor.run()
        
        if success:
            logging.info("Pipeline completed successfully!")
        else:
            logging.error("Pipeline failed!")
            
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logging.info("Pipeline execution interrupted by user")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        logging.error(traceback.format_exc())
        sys.exit(1)
