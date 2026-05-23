import requests
import json
import sys

URL = "https://shakti-agent-865544319222.asia-south1.run.app"

def run_tests():
    print("=" * 60)
    print("RUNNING REMOTE CLOUD RUN SMOKE TESTS")
    print(f"Target Service URL: {URL}")
    print("=" * 60)

    # 1. Test GET /
    print("\n--- Test 1: Frontend GET / ---")
    try:
        r = requests.get(f"{URL}/")
        print(f"Status Code: {r.status_code}")
        print(f"Response Size: {len(r.text)} bytes")
        if r.status_code == 200 and "<html" in r.text.lower():
            print("[PASS] GET / passed (served HTML)")
        else:
            print("[FAIL] GET / failed")
    except Exception as e:
        print(f"[FAIL] GET / failed with exception: {e}")

    # 2. Test GET /health
    print("\n--- Test 2: Health Check GET /health ---")
    try:
        r = requests.get(f"{URL}/health")
        print(f"Status Code: {r.status_code}")
        print(f"Response: {r.json()}")
        if r.status_code == 200 and r.json().get("status") == "ok":
            print("[PASS] GET /health passed")
        else:
            print("[FAIL] GET /health failed")
    except Exception as e:
        print(f"[FAIL] GET /health failed with exception: {e}")

    # 3. Test POST /chat HEALTH Pillar
    print("\n--- Test 3: POST /chat (HEALTH Pillar, Age: 25-40) ---")
    body = {
        "user_id": "demo1",
        "age_band": "25-40",
        "query": "I am planning my first pregnancy in 6 months, what should I prepare?",
        "session_id": "s1"
    }
    try:
        r = requests.post(f"{URL}/chat", json=body)
        print(f"Status Code: {r.status_code}")
        resp = r.json()
        print(f"Sub-agent assigned: {resp.get('sub_agent')}")
        print(f"Pillar: {resp.get('pillar')}")
        print(f"Citations found: {len(resp.get('citations', []))}")
        print(f"Response Preview: {resp.get('response', '')[:150]}...")
        if r.status_code == 200 and resp.get("sub_agent") == "HEALTH":
            print("[PASS] HEALTH Pillar POST /chat passed")
        else:
            print("[FAIL] HEALTH Pillar POST /chat failed")
    except Exception as e:
        print(f"[FAIL] HEALTH Pillar POST /chat failed with exception: {e}")

    # 4. Test POST /chat FINANCE Pillar
    print("\n--- Test 4: POST /chat (FINANCE Pillar, Age: 25-40) ---")
    body = {
        "user_id": "demo2",
        "age_band": "25-40",
        "query": "How should I start investing in mutual funds?",
        "session_id": "s2"
    }
    try:
        r = requests.post(f"{URL}/chat", json=body)
        print(f"Status Code: {r.status_code}")
        resp = r.json()
        print(f"Sub-agent assigned: {resp.get('sub_agent')}")
        print(f"Pillar: {resp.get('pillar')}")
        print(f"Citations found: {len(resp.get('citations', []))}")
        print(f"Response Preview: {resp.get('response', '')[:150]}...")
        if r.status_code == 200 and resp.get("sub_agent") == "FINANCE":
            print("[PASS] FINANCE Pillar POST /chat passed")
        else:
            print("[FAIL] FINANCE Pillar POST /chat failed")
    except Exception as e:
        print(f"[FAIL] FINANCE Pillar POST /chat failed with exception: {e}")

    # 5. Test POST /chat CAREER Pillar
    print("\n--- Test 5: POST /chat (CAREER Pillar, Age: 41+) ---")
    body = {
        "user_id": "demo3",
        "age_band": "41+",
        "query": "I want to start consulting after retirement, how do I begin?",
        "session_id": "s3"
    }
    try:
        r = requests.post(f"{URL}/chat", json=body)
        print(f"Status Code: {r.status_code}")
        resp = r.json()
        print(f"Sub-agent assigned: {resp.get('sub_agent')}")
        print(f"Pillar: {resp.get('pillar')}")
        print(f"Citations found: {len(resp.get('citations', []))}")
        print(f"Response Preview: {resp.get('response', '')[:150]}...")
        if r.status_code == 200 and resp.get("sub_agent") == "CAREER":
            print("[PASS] CAREER Pillar POST /chat passed")
        else:
            print("[FAIL] CAREER Pillar POST /chat failed")
    except Exception as e:
        print(f"[FAIL] CAREER Pillar POST /chat failed with exception: {e}")

if __name__ == "__main__":
    run_tests()
