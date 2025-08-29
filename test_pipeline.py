#!/usr/bin/env python3
"""
Test script for the TikTok Data Processing Pipeline.
This script tests the pipeline execution without running the full pipeline.
"""

import os
import sys
import importlib
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_module_imports():
    """Test that all pipeline modules can be imported."""
    print("Testing module imports...")
    
    modules_to_test = [
        'src.01_take_input_csv',
        'src.02_rule_based_filtering',
        'src.utils.pipeline_utils'
    ]
    
    for module_name in modules_to_test:
        try:
            module = importlib.import_module(module_name)
            print(f"✓ Successfully imported {module_name}")
            
            # Test if the module has a 'run' function (for pipeline stages)
            if hasattr(module, 'run'):
                print(f"  ✓ Module {module_name} has 'run' function")
            else:
                print(f"  ⚠ Module {module_name} does not have 'run' function")
                
        except ImportError as e:
            print(f"✗ Failed to import {module_name}: {e}")
            return False
    
    return True

def test_yaml_config():
    """Test that the YAML configuration can be loaded."""
    print("\nTesting YAML configuration...")
    
    try:
        import yaml
        
        config_path = "config/main.yaml"
        if not os.path.exists(config_path):
            print(f"✗ Configuration file not found: {config_path}")
            return False
        
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        print("✓ Successfully loaded YAML configuration")
        
        # Check pipeline structure
        if 'pipeline' in config:
            pipeline = config['pipeline']
            print(f"✓ Pipeline name: {pipeline.get('name', 'Unknown')}")
            print(f"✓ Pipeline stages: {len(pipeline.get('stages', []))}")
            
            for i, stage in enumerate(pipeline.get('stages', []), 1):
                print(f"  Stage {i}: {stage.get('name', 'Unknown')}")
                print(f"    Module: {stage.get('module', 'Unknown')}")
                print(f"    Function: {stage.get('function', 'Unknown')}")
                print(f"    Enabled: {stage.get('enabled', True)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to load YAML configuration: {e}")
        return False

def test_directory_structure():
    """Test that the required directory structure exists."""
    print("\nTesting directory structure...")
    
    required_dirs = [
        'config',
        'src',
        'src/utils',
        'data'
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ Directory exists: {dir_path}")
        else:
            print(f"✗ Directory missing: {dir_path}")
            # Try to create it
            try:
                os.makedirs(dir_path, exist_ok=True)
                print(f"  ✓ Created directory: {dir_path}")
            except Exception as e:
                print(f"  ✗ Failed to create directory: {e}")
                return False
    
    return True

def test_pipeline_executor():
    """Test that the pipeline executor can be instantiated."""
    print("\nTesting pipeline executor...")
    
    try:
        # Import the main module
        sys.path.insert(0, os.path.dirname(__file__))
        from main import PipelineExecutor
        
        # Try to create an instance
        executor = PipelineExecutor("config/main.yaml")
        print("✓ Successfully created PipelineExecutor instance")
        
        # Check if configuration was loaded
        if hasattr(executor, 'config') and executor.config:
            print("✓ Configuration loaded successfully")
            return True
        else:
            print("✗ Configuration not loaded")
            return False
            
    except Exception as e:
        print(f"✗ Failed to create PipelineExecutor: {e}")
        return False

def main():
    """Run all tests."""
    print("=== TikTok Data Processing Pipeline Test Suite ===\n")
    
    tests = [
        ("Module Imports", test_module_imports),
        ("YAML Configuration", test_yaml_config),
        ("Directory Structure", test_directory_structure),
        ("Pipeline Executor", test_pipeline_executor)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Running test: {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED\n")
            else:
                print(f"✗ {test_name} FAILED\n")
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}\n")
    
    print("=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The pipeline is ready to run.")
        print("\nTo run the pipeline, execute:")
        print("  python main.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
