# Phase 2 test additions for strategy_engine.py

def test_research_task_intent():
    from app.agent.strategy_engine import detect_intent
    intent = detect_intent("how to use React hooks")
    assert intent.is_research_task is True
    # React is a tech entity, but 'how to use' does not trigger code generation
    assert intent.has_tech_domain is True
    assert intent.is_code_generation is False

def test_research_plan_priority():
    from app.agent.strategy_engine import IntentSignals, select_planning_mode
    # Code takes priority over research
    intent = IntentSignals(is_code_generation=True, is_research_task=True)
    assert select_planning_mode(intent) == "ADAPTIVE_CODE_PLAN"
    
    # Research takes priority over tools
    intent2 = IntentSignals(is_research_task=True, requires_tools=True)
    assert select_planning_mode(intent2) == "RESEARCH_PLAN"

def test_detect_knowledge_gap():
    from app.agent.strategy_engine import detect_knowledge_gap
    # API DOC
    assert detect_knowledge_gap({"query": "API documentation for Stripe"}) == "API_DOC"
    # LIBRARY USAGE
    assert detect_knowledge_gap({"error": "how to use pip install requests"}) == "LIBRARY_USAGE"
    # ERROR FIX
    assert detect_knowledge_gap({"last_error": "SyntaxError: invalid syntax not working"}) == "ERROR_FIX"
    # NONE
    assert detect_knowledge_gap({"query": "hello world"}) is None

def test_decide_research():
    from app.agent.strategy_engine import decide_research
    # API
    res = decide_research("API_DOC")
    assert res is not None
    assert res["mode"] == "DOC_SEARCH"
    assert res["tool"] == "search_web"
    
    # NONE
    assert decide_research(None) is None
    
def test_hybrid_advice_research():
    from app.agent.hybrid_controller import generate_strategy_advice
    advice = generate_strategy_advice("how to use fastAPI", {"query": "how to use fastAPI"})
    # "how to use" sets is_research_task. "fastAPI" sets tech_domain. 
    # Research task > Tech Domain -> RESEARCH_PLAN
    assert advice.planning_mode == "RESEARCH_PLAN"
    # But research_needed should be populated because "how to use" triggers LIBRARY_USAGE
    assert advice.research_needed is not None
    assert advice.research_needed["mode"] == "CODE_EXAMPLE"
