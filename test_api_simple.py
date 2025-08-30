#!/usr/bin/env python3
"""
Simple test script for the FastAPI pipeline integration - focusing on Stages 1-2
"""

import requests
import json

API_BASE_URL = 'http://localhost:8000'

def test_simple_review():
    """Test processing a simple review through stages 1-2"""
    print("🚀 Testing simple review processing (Stages 1-2)...")
    
    # Simple review data
    test_review = {
        "business_name": "Joe's Pizza",
        "author_name": "John",
        "rating": 4.0,
        "text": "Good pizza.",
        "description": "Pizza restaurant.",
        "pics": "no"
    }
    
    print(f"   Review: \"{test_review['text']}\"")
    print(f"   Business: {test_review['business_name']}")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/process_review",
            json=test_review,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Processing completed")
            print(f"   Success: {result.get('success')}")
            print(f"   Processing ID: {result.get('processing_id')}")
            
            if result.get('success'):
                stages = result.get('stage_results', {})
                print(f"   Stages completed: {len(stages)}")
                
                for stage_name, stage_info in stages.items():
                    print(f"   - {stage_name}: {stage_info.get('row_count', 0)} rows")
            else:
                print(f"   Error: {result.get('error_message', 'Unknown error')}")
            
            return result
        else:
            print(f"❌ Processing failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None

if __name__ == "__main__":
    result = test_simple_review()
    if result and result.get('success'):
        print("\n🎉 Simple pipeline test passed!")
    else:
        print("\n❌ Simple pipeline test failed.")
