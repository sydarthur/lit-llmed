#!/usr/bin/env python3
"""
Test Zotero API connection and permissions
"""

import requests
import json

def test_zotero_api(api_key, user_id):
    """Test Zotero API connection and permissions."""
    print("🔍 Testing Zotero API Connection...")
    print(f"User ID: {user_id}")
    print(f"API Key: {api_key[:10]}...")
    
    headers = {
        'Zotero-API-Key': api_key,
        'User-Agent': 'Literature Fetcher/1.0'
    }
    
    base_url = f"https://api.zotero.org/users/{user_id}"
    
    # Test 1: Check library access
    print("\n1️⃣ Testing library read access...")
    try:
        response = requests.get(f"{base_url}/items?limit=1", headers=headers, timeout=10)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ Library read access: OK")
            items = response.json()
            print(f"   Found {len(items)} items in library")
        elif response.status_code == 403:
            print("   ❌ Library read access: FORBIDDEN")
            print("   Check: API key needs 'Allow library access' permission")
        else:
            print(f"   ❌ Unexpected response: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Check write permissions
    print("\n2️⃣ Testing write permissions...")
    try:
        # Try to get existing collections first
        response = requests.get(f"{base_url}/collections", headers=headers, timeout=10)
        print(f"   Collections access status: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ Collections read access: OK")
            collections = response.json()
            print(f"   Found {len(collections)} collections")
            
            # Test creating a test collection
            test_collection = [{
                "name": "API Test Collection (Safe to Delete)",
                "parentCollection": False
            }]
            
            response = requests.post(
                f"{base_url}/collections", 
                headers={**headers, 'Content-Type': 'application/json'},
                json=test_collection,
                timeout=10
            )
            
            if response.status_code == 201:
                print("   ✅ Collection write access: OK")
                # Clean up test collection
                try:
                    result = response.json()
                    if 'successful' in result and '0' in result['successful']:
                        collection_key = result['successful']['0']['key']
                        delete_response = requests.delete(
                            f"{base_url}/collections/{collection_key}",
                            headers=headers,
                            timeout=10
                        )
                        if delete_response.status_code == 204:
                            print("   🧹 Cleaned up test collection")
                except:
                    print("   ⚠️  Test collection created but couldn't clean up")
                    
            elif response.status_code == 403:
                print("   ❌ Collection write access: FORBIDDEN")
                print("   Check: API key needs 'Allow write access' permission")
            else:
                print(f"   ❌ Unexpected write response: {response.status_code}")
                
        elif response.status_code == 403:
            print("   ❌ Collections read access: FORBIDDEN")
            
    except Exception as e:
        print(f"   ❌ Error testing write permissions: {e}")
    
    # Test 3: Check API key details
    print("\n3️⃣ Checking API key details...")
    try:
        response = requests.get(f"{base_url}/keys/{api_key}", headers=headers, timeout=10)
        if response.status_code == 200:
            key_info = response.json()
            print("   ✅ API key is valid")
            print(f"   Name: {key_info.get('name', 'Unknown')}")
            print(f"   Library access: {key_info.get('access', {}).get('library', False)}")
            print(f"   Write access: {key_info.get('access', {}).get('write', False)}")
        else:
            print(f"   ❌ Cannot get key details: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error checking key details: {e}")
    
    print("\n" + "="*50)
    print("💡 TROUBLESHOOTING GUIDE:")
    print("1. Go to https://www.zotero.org/settings/keys")
    print("2. Create a new private key with these permissions:")
    print("   ✅ Allow library access")
    print("   ✅ Allow write access")
    print("3. Copy the new API key (starts with uppercase letters)")
    print("4. Your User ID should be numbers only")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python test_zotero_connection.py <api_key> <user_id>")
        sys.exit(1)
    
    api_key = sys.argv[1]
    user_id = sys.argv[2]
    
    test_zotero_api(api_key, user_id)