"""Shared helpers for identifiers, timestamps, and JSON."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_id(prefix: str = "id") -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)


def loads(text: str | None, default: Any = None) -> Any:
    if not text:
        return default if default is not None else {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default if default is not None else {}
