import requests
import time
import json

def test_latency():
    url = "http://localhost:8000/chat/stream"
    payload = {
        "message": "Hello, how are you?",
        "chat_mode": "normal",
        "file_ids": []
    }
    
    start_time = time.time()
    print(f"Sending request to {url}...")
    
    try:
        with requests.post(url, json=payload, stream=True) as response:
            print(f"Response status: {response.status_code}")
            if response.status_code != 200:
                print("Error: ", response.text)
                return
            
            first_byte_time = None
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    if first_byte_time is None:
                        first_byte_time = time.time()
                        latency = first_byte_time - start_time
                        print(f"Latency to first byte: {latency:.4f} seconds")
                    # print(chunk)
            
            total_time = time.time() - start_time
            print(f"Total stream time: {total_time:.4f} seconds")
                
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_latency()
