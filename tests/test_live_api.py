import httpx
import asyncio
import json
import os

async def main():
    url = "http://127.0.0.1:8080/chat/stream"
    payload = {
        "message": "Write a long answer about AI in 10 points",
        "chat_mode": "agent",
        "user_id": "test",
        "session_id": "s1"
    }
    
    output_file = os.path.join(os.path.dirname(__file__), "agent.md")
    full_content = []
    
    print(f"Sending request to {url}...")
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", url, json=payload) as response:
                if response.status_code != 200:
                    print(f"Error: {response.status_code}")
                    print(await response.aread())
                    return
                
                print("Streaming response and extracting content...")
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break
                        
                        try:
                            data_json = json.loads(data_str)
                            content = data_json.get("content", "")
                            msg_type = data_json.get("type", "")
                            
                            if msg_type not in ["info", "done", "error"]:
                                if isinstance(content, str):
                                    print(content, end="", flush=True)
                                    full_content.append(content)
                                elif isinstance(content, dict):
                                    # If it's a dict, it might be a structured block
                                    # We'll just skip it for now or try to extract 'text' if it exists
                                    if "text" in content:
                                        text = content["text"]
                                        print(text, end="", flush=True)
                                        full_content.append(text)
                                    else:
                                        # Fallback to string representation for debugging
                                        s = json.dumps(content)
                                        print(f"\n[DEBUG DICT]: {s}\n")
                                        # full_content.append(s) # Don't append structured data to MD yet
                        except json.JSONDecodeError:
                            print(f"\n[Error decoding JSON]: {data_str}")
                
        # Save to markdown
        content_text = "".join([str(c) for c in full_content])
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content_text)
        
        print(f"\n\n[SUCCESS] Content saved to {output_file}")
        
    except Exception as e:
        print(f"\n[Connection error]: {e}")

if __name__ == "__main__":
    asyncio.run(main())
