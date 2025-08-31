import requests
import json
import time
from pathlib import Path
import sys

def check_server_availability(base_url, timeout=30):
    """Check if the FastAPI server is running"""
    print(f"🔍 Checking if server is available at {base_url}...")
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                print("✅ Server is running and accessible!")
                return True
        except requests.exceptions.ConnectionError:
            print("⏳ Server not available yet, waiting...")
            time.sleep(2)
        except Exception as e:
            print(f"🔍 Checking server: {e}")
            time.sleep(2)
    
    print(f"❌ Server not available after {timeout} seconds")
    return False

def run_complete_pipeline_workflow():
    """Complete workflow with proper error handling"""
    
    base_url = "http://localhost:8000"
    
    # Check if server is running
    if not check_server_availability(base_url):
        print("\n🚨 FastAPI server is not running!")
        print("Please start the server first:")
        print("1. Open a terminal/command prompt")
        print("2. Navigate to your project directory")
        print("3. Activate your virtual environment: venv\\Scripts\\activate")
        print("4. Run: python app.py")
        print("   OR: uvicorn app:app --host 0.0.0.0 --port 8000 --reload")
        return None
    
    try:
        # 1. Start pipeline
        start_payload = {
            "config_path": "config/main.yaml",
            "execution_name": "data_analysis",
            "override_config": {
                "processing": {
                    "batch_size": 1000,
                    "enable_sentiment_analysis": True
                }
            }
        }
        
        print("🚀 Starting pipeline...")
        response = requests.post(f"{base_url}/pipeline/start", json=start_payload, timeout=10)
        response.raise_for_status()
        result = response.json()
        execution_id = result["execution_id"]
        
        print(f"✅ Started pipeline: {execution_id}")
        print(f"📊 Status: {result['status']}")
        
        # 2. Monitor execution
        print("\n📈 Monitoring execution...")
        max_wait_time = 1800  # 30 minutes
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            try:
                status_response = requests.get(f"{base_url}/pipeline/status/{execution_id}", timeout=10)
                status_response.raise_for_status()
                status = status_response.json()
                
                print(f"Status: {status['status']} | Started: {status['started_at']}")
                
                if status["status"] == "completed":
                    print("✅ Pipeline completed successfully!")
                    break
                elif status["status"] == "failed":
                    print(f"❌ Pipeline failed: {status.get('error_message', 'Unknown error')}")
                    return None
                
                time.sleep(10)  # Wait 10 seconds before checking again
                
            except requests.exceptions.RequestException as e:
                print(f"⚠️ Error checking status: {e}")
                time.sleep(5)
                continue
        
        else:
            print("⏰ Pipeline execution timed out")
            return None
        
        # 3. Retrieve results
        print("\n📁 Retrieving output files...")
        try:
            output_response = requests.get(f"{base_url}/pipeline/output/{execution_id}", timeout=10)
            output_response.raise_for_status()
            output_data = output_response.json()
            
            print(f"Output directory: {output_data['output_directory']}")
            print(f"Total files generated: {output_data['total_files']}")
            
            for file_info in output_data['files']:
                print(f"  📄 {file_info['filename']} ({file_info['size']} bytes)")
        
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Could not retrieve output files: {e}")
            output_data = None
        
        # 4. Download logs
        print("\n📋 Downloading logs...")
        try:
            log_response = requests.get(f"{base_url}/pipeline/logs/{execution_id}", timeout=10)
            log_response.raise_for_status()
            log_filename = f"pipeline_{execution_id}.log"
            
            with open(log_filename, 'w', encoding='utf-8') as f:
                f.write(log_response.text)
            
            print(f"Logs saved to: {log_filename}")
        
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Could not download logs: {e}")
            log_filename = None
        
        # 5. Summary
        print(f"\n📊 Execution Summary:")
        print(f"Execution ID: {execution_id}")
        print(f"Status: {status['status']}")
        print(f"Started: {status['started_at']}")
        print(f"Completed: {status.get('completed_at', 'N/A')}")
        if output_data:
            print(f"Output Directory: {output_data['output_directory']}")
        if log_filename:
            print(f"Log File: {log_filename}")
        
        return {
            "execution_id": execution_id,
            "status": status,
            "output_files": output_data,
            "log_file": log_filename
        }
    
    except requests.exceptions.RequestException as e:
        print(f"🚨 Request failed: {e}")
        return None
    except Exception as e:
        print(f"🚨 Unexpected error: {e}")
        return None

# Run the workflow
if __name__ == "__main__":
    results = run_complete_pipeline_workflow()
    if results:
        print("\n🎉 Workflow completed successfully!")
    else:
        print("\n💥 Workflow failed!")
