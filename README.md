# TikTok On The Clock - Data Processing Pipeline

A flexible and configurable data processing pipeline system for TikTok data analysis.

## Overview

This project implements a modular pipeline system that processes data through multiple stages, each defined in separate Python modules. The pipeline is configured via YAML files and can be easily extended with new processing stages.

## Pipeline Architecture

The pipeline system consists of:

- **Configuration**: YAML files that define pipeline stages and parameters
- **Pipeline Executor**: Main orchestration logic that loads and runs stages
- **Pipeline Stages**: Individual Python modules that perform specific data processing tasks
- **Utilities**: Common functions and helpers for pipeline operations

## Directory Structure

```
├── main.py                          # Main pipeline executor
├── config/
│   └── main.yaml                    # Pipeline configuration
|   data/                            # Data input/output directory
│   └── extract_1000_random_rows.py  # Extracting random rows
|   └── combine_csv.py               # Combining csv files
|   └── label_rows.py                # Labeling rows
├── extension/
│   └── images
│   │   ├── generate-icons.js      
│   │   ├── icon.html
│   │   ├── icon16.png
│   │   ├── icon48.png
│   │   ├── icon128.png
│   │   ├── package.json            
│   ├── background.js                
│   ├── content.js                 
│   ├── manifest.json               
│   ├── popup.css
│   ├── popup.html
│   ├── popup.js
│   ├── popup_enhanced.css
│   ├── popup_enhanced.html
│   ├── popup_enhanced.js
├── logs/                            # Pipeline execution logs
│   ├── pipeline_20250831_014556.txt              
│   ├── pipeline_20250831_014908.txt             
│   ├── pipeline_20250831_020449.txt               
│   ├── pipeline_20250831_020838.txt
│   ├── pipeline_20250831_021451.txt
│   ├── pipeline_20250831_021939.txt
│   ├── pipeline_20250831_022232.txt
│   ├── pipeline_20250831_022303.txt
│   ├── pipeline_20250831_023250.txt
│   ├── pipeline_20250831_023341.txt
├──models/                            
│   ├── models_san.ipynb            
│   ├── relevance_metadata_ccbfefaa4768.json           
│   ├── relevance_scores_ccbfefaa4768.pkl                     
├── src/
│   └── __pycache__
│   │   ├── __init__.cpython-312
│   │   ├── 01_take_input_csv.cpython-312
│   │   ├── 02_rule_based_filtering.cpython-312
│   │   ├── 03_model_processing.cpython-312
│   │   ├── 04_relevance_check.cpython-312
│   │   ├── 05_evaluation.cpython-312
│   ├── 01_take_input_csv.py         # Stage 1: CSV input processing
│   ├── 02_rule_based_filtering.py   # Stage 2: Data filtering
│   ├── 03_model_processing          # Stage 3: Model Filtering
│   ├── 04_relevance_check.py        # Stage 4: Relevance Checking
│   ├── 05_evaluation.py             # Stage 5: Evaluating
│   ├── utils/
│   │   ├── __init__.py
│   │   └── pipeline_utils.py     # Common utilities
│   └── __init__.py      
├── .gitignore
├── api_server.py
├── app.py
├── main.py
├── README.py
├── requirements.in
├── setup.bash
├── setup.cmd.exe
├── requirements.txt       # Python dependencies
└── test_pipeline.py      # Test suite
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test the Pipeline

```bash
python test_pipeline.py
```

### 3. Run the Pipeline

```bash
python main.py
```

## Configuration

The pipeline is configured via `config/main.yaml`. Each stage defines:

- **name**: Human-readable stage name
- **module**: Python module path to import
- **function**: Function name to call (default: `run`)
- **enabled**: Whether the stage should run
- **config**: Stage-specific configuration parameters

### Example Configuration

```yaml
pipeline:
  name: "TikTok Data Processing Pipeline"
  stages:
    - name: "01_take_input_csv"
      module: "src.01_take_input_csv"
      function: "run"
      enabled: true
      config:
        input_file: "data/input.csv"
        output_file: "data/stage1_output.csv"
        # Output will be saved as: stage1_output_YYYYMMDD_HHMMSS.csv
```

## Timestamp-Based File Naming

The pipeline automatically adds timestamps to all output files to prevent overwriting and track execution history:

- **Stage 1 Output**: `stage1_output_20241201_143022.csv`
- **Stage 2 Output**: `stage2_output_20241201_143025.csv`
- **Log Files**: `pipeline_20241201_143020.log`

### Execution Tracking

Each pipeline run gets a unique execution ID (e.g., `20241201_143020`) and creates:
- Timestamped output files
- Execution-specific log files
- Organized output directories

## Adding New Pipeline Stages

To add a new pipeline stage:

1. **Create the module** in the `src/` directory
2. **Implement the `run` function** that accepts a config dictionary
3. **Add the stage** to `config/main.yaml`
4. **Test the module** using `test_pipeline.py`

### Example Stage Module

```python
def run(config: Dict[str, Any]) -> bool:
    """
    Execute the pipeline stage.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Your processing logic here
        return True
    except Exception as e:
        logging.error(f"Stage failed: {e}")
        return False
```

## Pipeline Features

- **Dynamic Module Loading**: Stages are loaded and executed dynamically
- **Error Handling**: Comprehensive error handling with retry logic
- **Logging**: Detailed logging of pipeline execution with timestamped files
- **Configuration**: Flexible YAML-based configuration
- **Extensibility**: Easy to add new stages and modify existing ones
- **Validation**: Data validation and metadata tracking
- **Execution Tracking**: Unique execution IDs and organized output structure

## Data Flow

1. **Stage 1**: Reads CSV input, performs basic validation and cleaning
   - Output: `stage1_output_YYYYMMDD_HHMMSS.csv`
2. **Stage 2**: Applies rule-based filtering using configurable rules
   - Automatically finds the most recent stage 1 output
   - Output: `stage2_output_YYYYMMDD_HHMMSS.csv`
3. **Output**: Processed data is saved to timestamped files in execution directories

## Logging

The pipeline generates detailed logs:
- Console output for real-time monitoring
- Timestamped log files in `logs/` directory
- Stage-specific metadata files
- Execution tracking with unique IDs

## Testing

Run the test suite to verify pipeline setup:

```bash
python test_pipeline.py
```

Tests cover:
- Module imports
- YAML configuration loading
- Directory structure
- Pipeline executor instantiation

## Dependencies

- **pandas**: Data manipulation and analysis
- **pyyaml**: YAML configuration parsing
- **pathlib2**: Path operations (Python < 3.4 compatibility)
- **typing-extensions**: Type hints support

## Troubleshooting

### Common Issues

1. **Module Import Errors**: Ensure all dependencies are installed
2. **Configuration Errors**: Check YAML syntax and file paths
3. **Permission Errors**: Verify write access to data and logs directories

### Debug Mode

Enable debug logging by modifying the log level in `config/main.yaml`:

```yaml
pipeline:
  config:
    log_level: "DEBUG"
```

### File Organization

After running the pipeline, you'll find:
- **Output files**: `data/output/execution_YYYYMMDD_HHMMSS/`
- **Log files**: `logs/pipeline_YYYYMMDD_HHMMSS.log`
- **Input files**: `data/input.csv` (if provided)

## Contributing

1. Follow the existing module structure
2. Implement proper error handling
3. Add comprehensive logging
4. Test your changes with `test_pipeline.py`
5. Update documentation as needed

## License

[Add your license information here]
