from app.agent.tool_policy import get_allowed_tools_for_mode, filter_tools_by_mode


def test_normal_mode_blocks_privileged_tools():
    allowed = get_allowed_tools_for_mode("normal")
    assert "memory_delete" not in allowed
    assert "code_exec_sandbox" not in allowed
    assert "search_web" in allowed


def test_coding_mode_allows_code_tools():
    allowed = get_allowed_tools_for_mode("coding")
    assert "code_exec_sandbox" in allowed
    assert "validate_code" in allowed


def test_filter_tools_by_mode_keeps_only_allowlisted():
    selected = {"search_web", "memory_delete", "code_exec_sandbox"}
    filtered = filter_tools_by_mode("normal", selected)
    assert filtered == {"search_web"}
