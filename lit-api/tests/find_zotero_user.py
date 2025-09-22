#!/usr/bin/env python3
"""
Find Zotero user ID from username or test different approaches
"""

import requests
import json

def find_zotero_user_id(api_key, username=None):
    """Try to find the correct user ID for Zotero API."""
    print("🔍 Finding Zotero User Information...")
    
    headers = {
        'Zotero-API-Key': api_key,
        'User-Agent': 'Literature Fetcher/1.0'
    }
    
    # Method 1: Try the username directly in the API
    if username:
        print(f"\n1️⃣ Testing username: {username}")
        try:
            # Try with /users/username format
            response = requests.get(f"https://api.zotero.org/users/{username}", headers=headers, timeout=10)
            print(f"   Status: {response.status_code}")
            
            if response.status_code == 200:
                print("   ✅ Username works as user ID!")
                return username
            elif response.status_code == 404:
                print("   ❌ User not found with this username")
            else:
                print(f"   ❌ Error: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Method 2: Try to get user info from key validation endpoint
    print(f"\n2️⃣ Checking API key validity...")
    try:
        # This endpoint might give us user info
        response = requests.get("https://api.zotero.org/keys/current", headers=headers, timeout=10)
        print(f"   Status: {response.status_code}")
        
        if response.status_code == 200:
            key_info = response.json()
            print("   ✅ API key is valid!")
            print(f"   Key info: {json.dumps(key_info, indent=2)}")
            
            # Look for user ID in the response
            if 'userID' in key_info:
                user_id = key_info['userID']
                print(f"   🎯 Found User ID: {user_id}")
                return str(user_id)
            
        elif response.status_code == 403:
            print("   ❌ API key lacks permissions")
        else:
            print(f"   ❌ Unexpected response: {response.status_code}")
            if response.text:
                print(f"   Response: {response.text[:200]}")
                
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Method 3: Try some common user ID patterns
    print(f"\n3️⃣ Testing common patterns...")
    if username:
        test_patterns = [
            username,
            username.lower(),
            username.upper(),
            f"user_{username}",
            username.replace('_', ''),
        ]
        
        for pattern in test_patterns:
            try:
                response = requests.get(f"https://api.zotero.org/users/{pattern}/items?limit=1", headers=headers, timeout=5)
                if response.status_code == 200:
                    print(f"   ✅ Pattern works: {pattern}")
                    return pattern
                elif response.status_code != 404:
                    print(f"   Pattern {pattern}: {response.status_code}")
            except:
                continue
    
    print("\n❌ Could not determine correct user ID")
    return None

def test_with_user_id(api_key, user_id):
    """Test the API with a specific user ID."""
    print(f"\n🧪 Testing API with User ID: {user_id}")
    
    headers = {
        'Zotero-API-Key': api_key,
        'User-Agent': 'Literature Fetcher/1.0'
    }
    
    try:
        # Test basic library access
        response = requests.get(f"https://api.zotero.org/users/{user_id}/items?limit=1", headers=headers, timeout=10)
        print(f"   Library access: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ Library access successful!")
            items = response.json()
            print(f"   Library contains {len(items)} items (showing first 1)")
            return True
        else:
            print(f"   ❌ Library access failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python find_zotero_user.py <api_key> [username]")
        sys.exit(1)
    
    api_key = sys.argv[1]
    username = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Try to find the correct user ID
    user_id = find_zotero_user_id(api_key, username)
    
    if user_id:
        print(f"\n🎯 Recommended User ID: {user_id}")
        
        # Test it
        if test_with_user_id(api_key, user_id):
            print(f"\n✅ Success! Use these credentials:")
            print(f"   API Key: {api_key}")
            print(f"   User ID: {user_id}")
        else:
            print(f"\n❌ User ID found but API test failed")
    else:
        print(f"\n💡 Next steps:")
        print(f"1. Go to https://www.zotero.org/settings/keys")
        print(f"2. Look for 'Your userID for use in API calls' (should be numbers)")
        print(f"3. Or create a new API key with proper permissions")
        print(f"4. Make sure the key has 'Allow library access' and 'Allow write access'")