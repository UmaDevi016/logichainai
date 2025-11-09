import streamlit as st
import json
import requests
import sys

def main():
    try:
        s = dict(st.secrets)
    except Exception as e:
        print("Could not read st.secrets:", e)
        sys.exit(1)

    print("Streamlit secrets (partial):")
    # print only keys, not full values for safety
    print(json.dumps({k: ("***" + str(v)[-4:]) for k, v in s.items()}, indent=2))

    key = s.get("JINA_API_KEY")
    url = s.get("JINA_API_URL")
    if not key or not url:
        print("Missing JINA_API_KEY or JINA_API_URL in secrets.")
        sys.exit(1)

    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    sample = {"prompt": "Test summary. Reply OK", "max_tokens": 10}
    try:
        r = requests.post(url, json=sample, headers=headers, timeout=12)
        print("HTTP status:", r.status_code)
        try:
            print("Response JSON:")
            print(json.dumps(r.json(), indent=2))
        except Exception:
            print("Response text:")
            print(r.text)
    except Exception as e:
        print("HTTP call failed:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
