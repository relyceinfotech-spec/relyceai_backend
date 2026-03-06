import asyncio
from app.agent.tool_executor import _tool_calculate, _tool_read_file, _tool_retrieve_knowledge, _tool_search_web, can_call_search

async def test_tools():
    print("=== Testing calculate ===")
    res_good = _tool_calculate("523 * 8.5")
    print("Good Calc:", res_good)
    res_bad = _tool_calculate("import os; os.system('del *')")
    print("Bad Calc:", res_bad)

    print("\n=== Testing read_file ===")
    # Good
    with open("data/test.txt", "w") as f:
        f.write("A" * 5000)
    res_good = await _tool_read_file("data/test.txt")
    
    # Needs to be extracted because execute_tool now does the payload truncation in the standard pipeline
    # We will test execute_tool directly instead of raw tools.
    from app.agent.tool_executor import execute_tool, ToolCall
    tc = ToolCall(name="read_file", args="data/test.txt")
    res_trunc = await execute_tool(tc)
    
    print("Truncation Worked:", len(res_trunc.data) <= 4050, "Length:", len(res_trunc.data))
    
    # Bad
    res_bad = await _tool_read_file("../../.env")
    print("Bad Read:", res_bad)

    print("\n=== Testing retrieve_knowledge ===")
    res_know = await _tool_retrieve_knowledge("quantum mechanics")
    print("Knowledge:", res_know)

    print("\n=== Testing search_web latency ===")
    res_search_1 = await _tool_search_web("weather", session_id="user_1")
    print("Search 1:", "Success" if res_search_1["status"] == "success" else res_search_1)
    res_search_2 = await _tool_search_web("weather", session_id="user_1")
    print("Search 2:", res_search_2)

if __name__ == "__main__":
    import os
    if not os.path.exists("data"):
        os.makedirs("data")
    asyncio.run(test_tools())
