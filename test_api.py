#!/usr/bin/env python3
"""
Test script for the FastAPI pipeline integration
"""

import requests
import json
import time

API_BASE_URL = 'http://localhost:8000'

def test_health_endpoint():
    """Test the health check endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['status']}")
            print(f"   Pipeline stages: {data.get('pipeline_stages', 'Unknown')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection failed: {e}")
        return False

def test_pipeline_info():
    """Test the pipeline info endpoint"""
    print("\n📋 Testing pipeline info endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/pipeline_info", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Pipeline info retrieved")
            print(f"   Name: {data.get('pipeline_name', 'Unknown')}")
            print(f"   Stages: {data.get('total_stages', 0)}")
            for stage in data.get('stages', []):
                status = "✓" if stage.get('enabled') else "✗"
                print(f"   {status} {stage.get('name', 'Unknown')}")
            return True
        else:
            print(f"❌ Pipeline info failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection failed: {e}")
        return False

def test_single_review_processing():
    """Test processing a single review"""
    print("\n🚀 Testing single review processing...")
    
    # Sample review data
    test_review = {
        "business_name": "Joe's Pizza Place",
        "author_name": "John Doe",
        "rating": 4.0,
        "text": "Great pizza and excellent service! The staff was friendly and the atmosphere was cozy. Would definitely come back again.",
        "description": "Family-owned Italian restaurant serving authentic pizza and pasta dishes in downtown.",
        "avg_rating": 4.2,
        "num_of_reviews": 150,
        "category": "Restaurant",
        "state": "CA",
        "address": "123 Main St, Downtown"
    }
    
    print(f"   Review: \"{test_review['text'][:50]}...\"")
    print(f"   Business: {test_review['business_name']}")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/process_review",
            json=test_review,
            timeout=120  # Allow up to 2 minutes for processing
        )
        processing_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Processing completed in {processing_time:.1f}s")
            print(f"   Processing ID: {result.get('processing_id')}")
            print(f"   Success: {result.get('success')}")
            
            if result.get('success'):
                stages = result.get('stage_results', {})
                print(f"   Stages completed: {len(stages)}")
                
                final_predictions = result.get('final_predictions', {})
                if 'model_prediction' in final_predictions:
                    validity = "Valid" if final_predictions['model_prediction'] == 0 else "Invalid"
                    confidence = final_predictions.get('model_confidence', 0) * 100
                    print(f"   AI Prediction: {validity} ({confidence:.1f}% confidence)")
                
                if 'relevance_score' in final_predictions:
                    relevance = "Relevant" if final_predictions.get('is_relevant') == 1 else "Not Relevant"
                    score = final_predictions.get('relevance_score', 0) * 100
                    print(f"   Relevance: {relevance} ({score:.1f}% score)")
                
                print(f"   Generated features: {len([k for k in final_predictions.keys() if k not in ['model_prediction', 'model_confidence', 'relevance_score', 'is_relevant']])}")
            else:
                print(f"   Error: {result.get('error_message', 'Unknown error')}")
            
            return result.get('success', False)
        else:
            print(f"❌ Processing failed: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('detail', 'Unknown error')}")
            except:
                print(f"   Raw response: {response.text[:200]}")
            return False
            
    except requests.exceptions.Timeout:
        print(f"❌ Processing timed out after {processing_time:.1f}s")
        return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing FastAPI Pipeline Integration")
    print("=" * 50)
    
    # Test basic connectivity
    if not test_health_endpoint():
        print("\n❌ Basic connectivity failed. Make sure the API server is running:")
        print("   python api_server.py")
        return False
    
    # Test pipeline configuration
    if not test_pipeline_info():
        print("\n❌ Pipeline configuration test failed.")
        return False
    
    # Test actual processing
    if not test_single_review_processing():
        print("\n❌ Review processing test failed.")
        return False
    
    print("\n🎉 All tests passed! The FastAPI backend is working correctly.")
    print("\n📝 Next steps:")
    print("   1. Load the enhanced extension in Chrome (Developer Mode)")
    print("   2. Open popup_enhanced.html as the extension popup")
    print("   3. Test manual review input or auto-extraction")
    print("   4. Verify end-to-end pipeline processing")
    
    return True

if __name__ == "__main__":
    main()
