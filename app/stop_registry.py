from __future__ import annotations

import threading
import uuid

from typing import Dict


_stop_events: Dict[str, threading.Event] = {}
_stop_events_lock = threading.Lock()


def new_stop_token() -> str:
    return uuid.uuid4().hex


def get_stop_event(token: str) -> threading.Event:
    with _stop_events_lock:
        event = _stop_events.get(token)
        if event is None:
            event = threading.Event()
            _stop_events[token] = event
        return event


def request_stop(token: str) -> None:
    get_stop_event(token).set()


def clear_stop(token: str) -> None:
    get_stop_event(token).clear()
