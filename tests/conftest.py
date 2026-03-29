from __future__ import annotations

import asyncio
import inspect

_SESSION_LOOP = None


def _get_session_loop():
    global _SESSION_LOOP
    if _SESSION_LOOP is None or _SESSION_LOOP.is_closed():
        _SESSION_LOOP = asyncio.new_event_loop()
        asyncio.set_event_loop(_SESSION_LOOP)
    return _SESSION_LOOP


def pytest_pyfunc_call(pyfuncitem):
    test_func = pyfuncitem.obj
    if not inspect.iscoroutinefunction(test_func):
        return None
    kwargs = {
        name: pyfuncitem.funcargs[name]
        for name in pyfuncitem._fixtureinfo.argnames
        if name in pyfuncitem.funcargs
    }
    loop = _get_session_loop()
    loop.run_until_complete(test_func(**kwargs))
    return True


def pytest_sessionfinish(session, exitstatus):
    global _SESSION_LOOP
    if _SESSION_LOOP is not None and not _SESSION_LOOP.is_closed():
        _SESSION_LOOP.close()
