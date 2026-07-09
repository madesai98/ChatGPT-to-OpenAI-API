#!/usr/bin/env python3
"""
OpenAI-compatible API server backed by a logged-in ChatGPT browser session.

Features:
- OpenAI-shaped inspection endpoints for common harness surfaces:
  /v1/responses
  /v1/chat/completions
  /v1/models
  /v1/embeddings
  /v1/completions
  /v1/moderations
  /v1/audio/*
  /v1/images/*
  /v1/files
  /v1/assistants, /v1/threads, /v1/threads/*/runs
- Catch-all /v1/{path:path} fallback so unusual harness calls are still logged.
- Web dashboard at /dashboard showing formatted request and response data.
- JSONL disk log at ./openai_mock_logs/requests.jsonl.
- In-memory log APIs at /__events and /__logs.

Run:
  pip install -r requirements.txt
  python run.py --host 127.0.0.1 --port 8000

Point harnesses/SDKs at:
  OPENAI_BASE_URL=http://127.0.0.1:8000/v1
  OPENAI_API_KEY=anything

Dashboard:
  http://127.0.0.1:8000/dashboard

Optional:
  MOCK_OPENAI_MAX_CAPTURE_BYTES=0 python run.py --port 8000
  MOCK_OPENAI_LOG_DIR=./logs python run.py --port 8000
  python run.py --log-level debug
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import struct
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse

from browser import BrowserError
from openai_bridge import BridgeResult, BrowserChatGPTBridge
from app_logging import VERBOSE


# -----------------------------------------------------------------------------
# App and global state
# -----------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    await BRIDGE.close()


app = FastAPI(title="WebLLM2API", version="1.0.0", lifespan=lifespan)
LOGGER = logging.getLogger("webllm2api.server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LOG_DIR = Path(os.getenv("MOCK_OPENAI_LOG_DIR", "./openai_mock_logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
REQUEST_LOG = LOG_DIR / "requests.jsonl"

GENERATED_TEXT = "DUMMY RESPONSE"
STREAM_GENERATED_TEXT = " ".join([GENERATED_TEXT] * 10)
EMBEDDING_DIM = int(os.getenv("MOCK_OPENAI_EMBEDDING_DIM", "16"))

MAX_CAPTURE_BYTES = int(os.getenv("MOCK_OPENAI_MAX_CAPTURE_BYTES", str(10 * 1024 * 1024)))
IN_MEMORY_LOG_LIMIT = int(os.getenv("MOCK_OPENAI_IN_MEMORY_LOG_LIMIT", "500"))
REQUEST_EVENTS: deque[dict[str, Any]] = deque(maxlen=IN_MEMORY_LOG_LIMIT)

IGNORED_LOG_PATH_PREFIXES = (
    "/dashboard",
    "/__events",
    "/__logs",
    "/favicon.ico",
)

STORE: dict[str, dict[str, Any]] = {
    "responses": {},
    "chat_completions": {},
    "assistants": {},
    "threads": {},
    "messages": {},
    "runs": {},
    "files": {},
    "batches": {},
    "vector_stores": {},
}
CONVERSATION_URLS: dict[str, str] = {}
RESPONSE_INPUTS: dict[str, list[Any]] = {}
BRIDGE = BrowserChatGPTBridge()


@app.exception_handler(BrowserError)
async def browser_error_handler(_: Request, exc: BrowserError) -> JSONResponse:
    LOGGER.warning("ChatGPT browser request failed: %s", exc)
    return JSONResponse(
        {
            "error": {
                "message": str(exc),
                "type": "browser_error",
                "param": None,
                "code": "browser_error",
            }
        },
        status_code=502,
    )


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def now() -> int:
    return int(time.time())


def rid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


def redact_headers(headers: dict[str, str]) -> dict[str, str]:
    redacted = {}
    for k, v in headers.items():
        if k.lower() in {"authorization", "openai-api-key", "api-key", "x-api-key"}:
            redacted[k] = "<redacted>"
        else:
            redacted[k] = v
    return redacted


def estimate_tokens(value: Any) -> int:
    try:
        text = json.dumps(value, ensure_ascii=False)
    except Exception:
        text = str(value)
    return max(1, len(text) // 4)


def json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


async def read_json(request: Request) -> dict[str, Any]:
    raw = await request.body()
    if not raw:
        return {}

    try:
        value = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_json",
                "message": str(exc),
                "body_preview": raw[:200].decode("utf-8", errors="replace"),
            },
        ) from exc

    if not isinstance(value, dict):
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_json",
                "message": "Expected a JSON object request body.",
                "parsed_type": type(value).__name__,
            },
        )

    return value


def model_from(payload: dict[str, Any], default: str = "") -> str:
    return str(payload.get("model") or default)


def text_from_chat_messages(messages: Any) -> str:
    if not isinstance(messages, list):
        return ""

    for msg in reversed(messages):
        if not isinstance(msg, dict):
            continue
        if msg.get("role") != "user":
            continue

        content = msg.get("content", "")
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    if "text" in part:
                        parts.append(str(part["text"]))
                    elif part.get("type") == "input_text":
                        parts.append(str(part.get("text", "")))
                    elif part.get("type") == "text":
                        text_value = part.get("text", "")
                        if isinstance(text_value, dict):
                            parts.append(str(text_value.get("value", "")))
                        else:
                            parts.append(str(text_value))
            return "\n".join(parts)

    return ""


def first_tool_call(payload: dict[str, Any]) -> dict[str, Any] | None:
    tools = payload.get("tools") or payload.get("functions") or []
    tool_choice = payload.get("tool_choice") or payload.get("function_call")

    required = tool_choice == "required" or isinstance(tool_choice, dict)
    if not required or not isinstance(tools, list) or not tools:
        return None

    first = tools[0]
    if not isinstance(first, dict):
        return None

    if first.get("type") == "function":
        fn = first.get("function") or {}
        name = fn.get("name")
    else:
        name = first.get("name")

    if not name:
        return None

    return {
        "id": rid("call"),
        "type": "function",
        "function": {
            "name": str(name),
            "arguments": "{}",
        },
    }


def usage_for(payload: dict[str, Any], output_text: str) -> dict[str, int]:
    input_tokens = estimate_tokens(payload)
    output_tokens = estimate_tokens(output_text)
    return {
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def response_usage_for(payload: dict[str, Any], output_text: str) -> dict[str, int]:
    input_tokens = estimate_tokens(payload)
    output_tokens = estimate_tokens(output_text)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


async def sleep_briefly() -> None:
    import asyncio
    await asyncio.sleep(0.02)


async def sse(events: list[dict[str, Any]], done: bool = True) -> AsyncIterator[bytes]:
    for event in events:
        await sleep_briefly()
        yield b"data: " + json_bytes(event) + b"\n\n"
    if done:
        yield b"data: [DONE]\n\n"


def text_deltas(text: str) -> list[str]:
    if not text:
        return []
    if text == STREAM_GENERATED_TEXT:
        return [
            GENERATED_TEXT + (" " if index < 9 else "")
            for index in range(10)
        ]
    return [text]


def request_timeout(payload: dict[str, Any]) -> float | None:
    value = payload.get("timeout")
    if value is None:
        return None
    try:
        return max(1.0, float(value))
    except (TypeError, ValueError):
        return None


async def browser_generation(
    payload: dict[str, Any],
    *,
    surface: str,
) -> BridgeResult:
    conversation_url = None
    previous_id = payload.get("previous_response_id")
    if previous_id:
        conversation_url = CONVERSATION_URLS.get(str(previous_id))
        if conversation_url is None:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": {
                        "message": f"Previous response {previous_id!r} was not found.",
                        "type": "invalid_request_error",
                        "param": "previous_response_id",
                        "code": "previous_response_not_found",
                    }
                },
            )
    return await BRIDGE.generate(
        payload,
        surface=surface,
        conversation_url=conversation_url,
        timeout=request_timeout(payload),
    )


def deterministic_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values = []
    for i in range(dim):
        b = digest[i % len(digest)]
        values.append(round((b / 255.0) * 2.0 - 1.0, 6))
    return values


def embedding_to_base64(vec: list[float]) -> str:
    return base64.b64encode(struct.pack(f"{len(vec)}f", *vec)).decode("ascii")


def tiny_png_base64() -> str:
    return ""


def maybe_plain_text_response(format_name: str | None, payload: dict[str, Any]) -> Response:
    text = payload.get("text", "")
    if format_name in {"text", "srt", "vtt"}:
        return PlainTextResponse(str(text))
    return JSONResponse(payload)


def parse_sse_events(text: str) -> list[Any]:
    events: list[Any] = []

    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue

        data_lines = []
        event_name = None

        for line in block.splitlines():
            if line.startswith("event:"):
                event_name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())

        if not data_lines:
            continue

        data_text = "\n".join(data_lines)

        if data_text == "[DONE]":
            events.append({"event": event_name, "data": "[DONE]"})
            continue

        try:
            parsed = json.loads(data_text)
        except Exception:
            parsed = data_text

        if event_name:
            events.append({"event": event_name, "data": parsed})
        else:
            events.append(parsed)

    return events


def capture_payload(raw: bytes, content_type: str | None = None) -> dict[str, Any]:
    """
    Captures request/response bytes in a dashboard-friendly shape.

    Set MOCK_OPENAI_MAX_CAPTURE_BYTES=0 for unlimited capture.
    """
    content_type = content_type or ""
    original_size = len(raw)

    if MAX_CAPTURE_BYTES > 0 and original_size > MAX_CAPTURE_BYTES:
        captured = raw[:MAX_CAPTURE_BYTES]
        truncated = True
    else:
        captured = raw
        truncated = False

    text = captured.decode("utf-8", errors="replace")

    parsed_json = None
    sse_events = None

    stripped = text.lstrip()
    looks_json = (
        "application/json" in content_type
        or content_type.endswith("+json")
        or stripped.startswith("{")
        or stripped.startswith("[")
    )

    if looks_json and text:
        try:
            parsed_json = json.loads(text)
        except Exception:
            parsed_json = None

    if "text/event-stream" in content_type and text:
        sse_events = parse_sse_events(text)

    return {
        "content_type": content_type,
        "size_bytes": original_size,
        "captured_bytes": len(captured),
        "truncated": truncated,
        "text": text,
        "json": parsed_json,
        "sse_events": sse_events,
    }


def persist_event(record: dict[str, Any]) -> None:
    REQUEST_EVENTS.append(record)

    with REQUEST_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    response = record.get("response") or {}
    LOGGER.debug(
        "%s %s -> %s in %sms",
        record.get("method"),
        record.get("path"),
        response.get("status_code"),
        record.get("duration_ms"),
    )
    LOGGER.log(
        VERBOSE,
        "Request/response payload: %s",
        json.dumps(record, ensure_ascii=False),
    )


# -----------------------------------------------------------------------------
# Request/response logging middleware
# -----------------------------------------------------------------------------

@app.middleware("http")
async def log_requests(request: Request, call_next):
    path = request.url.path

    if path.startswith(IGNORED_LOG_PATH_PREFIXES):
        return await call_next(request)

    request_id = rid("req")
    started_at = time.time()

    raw_request_body = await request.body()
    request_headers = dict(request.headers)
    request_content_type = request_headers.get("content-type", "")

    async def receive():
        return {"type": "http.request", "body": raw_request_body, "more_body": False}

    request_for_downstream = Request(request.scope, receive)

    base_record = {
        "id": request_id,
        "ts": started_at,
        "iso_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started_at)),
        "method": request.method,
        "path": path,
        "query": str(request.url.query),
        "client": request.client.host if request.client else None,
        "request": {
            "headers": redact_headers(request_headers),
            "body": capture_payload(raw_request_body, request_content_type),
        },
    }

    try:
        response = await call_next(request_for_downstream)
    except Exception as exc:
        LOGGER.exception(
            "Unhandled error while processing %s %s",
            request.method,
            path,
        )
        record = {
            **base_record,
            "duration_ms": round((time.time() - started_at) * 1000, 2),
            "response": {
                "status_code": 500,
                "headers": {},
                "body": {
                    "content_type": "text/plain",
                    "size_bytes": 0,
                    "captured_bytes": 0,
                    "truncated": False,
                    "text": "",
                    "json": None,
                    "sse_events": None,
                },
                "error": repr(exc),
            },
        }
        persist_event(record)
        raise

    response_headers = dict(response.headers)
    response_content_type = response_headers.get("content-type", response.media_type or "")
    captured_response_chunks: list[bytes] = []
    response_size_bytes = 0

    async def tee_response_body():
        nonlocal response_size_bytes

        try:
            async for chunk in response.body_iterator:
                if isinstance(chunk, str):
                    chunk_bytes = chunk.encode("utf-8")
                else:
                    chunk_bytes = bytes(chunk)

                response_size_bytes += len(chunk_bytes)

                if MAX_CAPTURE_BYTES <= 0:
                    captured_response_chunks.append(chunk_bytes)
                else:
                    already_captured = sum(len(c) for c in captured_response_chunks)
                    remaining = MAX_CAPTURE_BYTES - already_captured
                    if remaining > 0:
                        captured_response_chunks.append(chunk_bytes[:remaining])

                yield chunk
        finally:
            captured_response_body = b"".join(captured_response_chunks)

            record = {
                **base_record,
                "duration_ms": round((time.time() - started_at) * 1000, 2),
                "response": {
                    "status_code": response.status_code,
                    "headers": redact_headers(response_headers),
                    "body": capture_payload(captured_response_body, response_content_type),
                    "actual_size_bytes": response_size_bytes,
                },
            }

            persist_event(record)

    return StreamingResponse(
        tee_response_body(),
        status_code=response.status_code,
        headers=response_headers,
        media_type=response.media_type,
        background=response.background,
    )


# -----------------------------------------------------------------------------
# Dashboard and log APIs
# -----------------------------------------------------------------------------

DASHBOARD_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>WebLLM2API Inspector</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    :root {
      --bg: #0f1115;
      --panel: #171a21;
      --panel2: #20242d;
      --text: #f2f5f8;
      --muted: #9aa4b2;
      --border: #313846;
      --accent: #7dd3fc;
      --ok: #86efac;
      --warn: #fde68a;
      --bad: #fca5a5;
      --code: #0b0d11;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    header {
      position: sticky;
      top: 0;
      z-index: 10;
      background: rgba(15, 17, 21, 0.96);
      border-bottom: 1px solid var(--border);
      padding: 16px 20px;
      backdrop-filter: blur(10px);
    }

    h1 {
      margin: 0 0 12px;
      font-size: 20px;
    }

    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }

    input, select, button {
      background: var(--panel2);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px 10px;
      font-size: 14px;
    }

    button {
      cursor: pointer;
    }

    button:hover {
      border-color: var(--accent);
    }

    main {
      padding: 20px;
      display: grid;
      gap: 16px;
    }

    .event {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      overflow: hidden;
    }

    .event-header {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      justify-content: space-between;
      padding: 14px 16px;
      border-bottom: 1px solid var(--border);
      background: var(--panel2);
    }

    .left, .right {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }

    .badge {
      border: 1px solid var(--border);
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 12px;
      color: var(--muted);
    }

    .method {
      color: var(--accent);
      font-weight: 700;
    }

    .path {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 13px;
    }

    .status-ok { color: var(--ok); }
    .status-warn { color: var(--warn); }
    .status-bad { color: var(--bad); }

    details {
      border-top: 1px solid var(--border);
    }

    summary {
      cursor: pointer;
      padding: 12px 16px;
      color: var(--accent);
      font-weight: 600;
    }

    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      padding: 0 16px 16px;
    }

    @media (max-width: 900px) {
      .grid {
        grid-template-columns: 1fr;
      }
    }

    h3 {
      margin: 10px 0;
      font-size: 14px;
      color: var(--muted);
    }

    pre {
      margin: 0;
      background: var(--code);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 12px;
      overflow: auto;
      max-height: 520px;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      line-height: 1.45;
    }

    .empty {
      color: var(--muted);
      text-align: center;
      padding: 50px;
      border: 1px dashed var(--border);
      border-radius: 14px;
    }

    .small {
      color: var(--muted);
      font-size: 12px;
    }
  </style>
</head>
<body>
  <header>
    <h1>WebLLM2API Inspector</h1>
    <div class="controls">
      <input id="search" placeholder="Search path, method, body..." size="34" />
      <select id="endpoint">
        <option value="">All endpoints</option>
      </select>
      <select id="limit">
        <option value="50">Last 50</option>
        <option value="100" selected>Last 100</option>
        <option value="250">Last 250</option>
        <option value="500">Last 500</option>
      </select>
      <label class="small">
        <input id="autorefresh" type="checkbox" checked />
        auto-refresh
      </label>
      <button id="refresh">Refresh</button>
      <button id="clear">Clear</button>
      <span id="summary" class="small"></span>
    </div>
  </header>

  <main id="events"></main>

  <script>
    const state = {
      events: [],
      timer: null,
    };

    const $ = (id) => document.getElementById(id);

    function statusClass(code) {
      if (code >= 200 && code < 300) return "status-ok";
      if (code >= 300 && code < 500) return "status-warn";
      return "status-bad";
    }

    function pretty(value) {
      if (value === undefined || value === null) return "";
      if (typeof value === "string") return value;
      return JSON.stringify(value, null, 2);
    }

    function bestBodyView(body) {
      if (!body) return "";
      if (body.sse_events) return body.sse_events;
      if (body.json !== null && body.json !== undefined) return body.json;
      return body.text || "";
    }

    function wholeRecordSearchText(event) {
      try {
        return JSON.stringify(event).toLowerCase();
      } catch {
        return "";
      }
    }

    function escapeHtml(text) {
      return String(text)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }

    function updateEndpointFilter(events) {
      const current = $("endpoint").value;
      const paths = [...new Set(events.map(e => e.path))].sort();

      $("endpoint").innerHTML =
        '<option value="">All endpoints</option>' +
        paths.map(p => `<option value="${escapeHtml(p)}">${escapeHtml(p)}</option>`).join("");

      if (paths.includes(current)) {
        $("endpoint").value = current;
      }
    }

    function eventCardHtml(event) {
      const reqBody = bestBodyView(event.request?.body);
      const resBody = bestBodyView(event.response?.body);
      const status = event.response?.status_code ?? 0;
      const duration = event.duration_ms ?? 0;
      const eventId = event.id || `${event.method || ""}:${event.path || ""}:${event.ts || ""}`;

      return `
        <section class="event" data-event-id="${escapeHtml(eventId)}">
          <div class="event-header">
            <div class="left">
              <span class="badge method">${escapeHtml(event.method)}</span>
              <span class="path">${escapeHtml(event.path)}${event.query ? "?" + escapeHtml(event.query) : ""}</span>
              <span class="badge ${statusClass(status)}">${status}</span>
              <span class="badge">${duration} ms</span>
            </div>
            <div class="right">
              <span class="badge">${escapeHtml(event.iso_time || "")}</span>
              <span class="badge">${escapeHtml(event.id || "")}</span>
            </div>
          </div>

          <details open>
            <summary>Request / Response Bodies</summary>
            <div class="grid">
              <div>
                <h3>Request body</h3>
                <pre>${escapeHtml(pretty(reqBody))}</pre>
              </div>
              <div>
                <h3>Response body</h3>
                <pre>${escapeHtml(pretty(resBody))}</pre>
              </div>
            </div>
          </details>

          <details>
            <summary>Headers</summary>
            <div class="grid">
              <div>
                <h3>Request headers</h3>
                <pre>${escapeHtml(pretty(event.request?.headers || {}))}</pre>
              </div>
              <div>
                <h3>Response headers</h3>
                <pre>${escapeHtml(pretty(event.response?.headers || {}))}</pre>
              </div>
            </div>
          </details>

          <details>
            <summary>Full captured event JSON</summary>
            <div style="padding: 0 16px 16px;">
              <pre>${escapeHtml(pretty(event))}</pre>
            </div>
          </details>
        </section>
      `;
    }

    function eventHash(event) {
      return JSON.stringify([
        event.method,
        event.path,
        event.query,
        event.duration_ms,
        event.response?.status_code,
        event.request,
        event.response,
      ]);
    }

    function findRenderedEvent(container, eventId) {
      return Array.from(container.querySelectorAll(".event"))
        .find(card => card.dataset.eventId === eventId);
    }

    function cardFromHtml(html) {
      const template = document.createElement("template");
      template.innerHTML = html.trim();
      return template.content.firstElementChild;
    }

    function restoreDetailsState(card, openStates) {
      card.querySelectorAll("details").forEach((details, index) => {
        if (openStates[index] !== undefined) {
          details.open = openStates[index];
        }
      });
    }

    function render() {
      const search = $("search").value.trim().toLowerCase();
      const endpoint = $("endpoint").value;
      const container = $("events");

      let events = state.events;

      if (endpoint) {
        events = events.filter(e => e.path === endpoint);
      }

      if (search) {
        events = events.filter(e => wholeRecordSearchText(e).includes(search));
      }

      $("summary").textContent = `${events.length} shown / ${state.events.length} loaded`;

      if (!events.length) {
        if (!container.querySelector(".empty")) {
          container.innerHTML = '<div class="empty">No matching requests yet. Send a harness request to any /v1 endpoint.</div>';
        }
        return;
      }

      container.querySelector(".empty")?.remove();

      const visibleIds = new Set(events.map(event => event.id || `${event.method || ""}:${event.path || ""}:${event.ts || ""}`));
      for (const card of Array.from(container.querySelectorAll(".event"))) {
        if (!visibleIds.has(card.dataset.eventId)) {
          card.remove();
        }
      }

      let previousCard = null;
      for (const event of events) {
        const eventId = event.id || `${event.method || ""}:${event.path || ""}:${event.ts || ""}`;
        const hash = eventHash(event);
        let card = findRenderedEvent(container, eventId);

        if (!card) {
          card = cardFromHtml(eventCardHtml(event));
        } else if (card.dataset.hash !== hash) {
          const openStates = Array.from(card.querySelectorAll("details")).map(details => details.open);
          const nextCard = cardFromHtml(eventCardHtml(event));
          restoreDetailsState(nextCard, openStates);
          card.replaceWith(nextCard);
          card = nextCard;
        }

        card.dataset.hash = hash;

        const expectedNext = previousCard ? previousCard.nextSibling : container.firstChild;
        if (card !== expectedNext) {
          container.insertBefore(card, expectedNext);
        }
        previousCard = card;
      }
    }

    async function loadEvents() {
      const limit = $("limit").value;
      const res = await fetch(`/__events?limit=${encodeURIComponent(limit)}`);
      const json = await res.json();
      state.events = json.data || [];
      updateEndpointFilter(state.events);
      render();
    }

    async function clearEvents() {
      await fetch("/__logs", { method: "DELETE" });
      await loadEvents();
    }

    $("refresh").addEventListener("click", loadEvents);
    $("clear").addEventListener("click", clearEvents);
    $("search").addEventListener("input", render);
    $("endpoint").addEventListener("change", render);
    $("limit").addEventListener("change", loadEvents);

    $("autorefresh").addEventListener("change", () => {
      if (state.timer) {
        clearInterval(state.timer);
        state.timer = null;
      }

      if ($("autorefresh").checked) {
        state.timer = setInterval(loadEvents, 1500);
      }
    });

    state.timer = setInterval(loadEvents, 1500);
    loadEvents();
  </script>
</body>
</html>
"""


@app.get("/", include_in_schema=False)
@app.get("/health")
@app.get("/v1/health")
async def health():
    return {
        "ok": True,
        "server": "webllm2api",
        "version": "1.0.0",
        "browser_running": BRIDGE.running,
        "browser_backed_endpoints": [
            "/v1/responses",
            "/v1/chat/completions",
            "/v1/completions",
        ],
        "compatibility_stubs": [
            "embeddings",
            "moderations",
            "audio",
            "images",
            "files",
            "assistants",
            "batches",
            "vector_stores",
        ],
        "log_file": str(REQUEST_LOG),
        "base_url": "/v1",
        "dashboard": "/dashboard",
        "file": str(Path(__file__).resolve()),
    }


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(DASHBOARD_HTML)


@app.get("/__events")
async def get_events(limit: int = 100):
    data = list(REQUEST_EVENTS)[-limit:]
    data.reverse()
    return {"data": data}


@app.get("/__logs")
async def get_logs(limit: int = 100):
    """
    Returns recent events from memory. The JSONL file is still written to disk
    at openai_mock_logs/requests.jsonl.
    """
    data = list(REQUEST_EVENTS)[-limit:]
    data.reverse()
    return {"data": data}


@app.delete("/__logs")
async def clear_logs():
    REQUEST_EVENTS.clear()
    REQUEST_LOG.write_text("", encoding="utf-8")
    return {"ok": True}


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

@app.get("/v1/models")
async def list_models():
    created = 1735689600
    models = [
        "gpt-5.5",
        "gpt-5.5-thinking",
        "gpt-5-5-thinking",
    ]
    return {
        "object": "list",
        "data": [
            {"id": m, "object": "model", "created": created, "owned_by": ""}
            for m in models
        ],
    }


@app.get("/v1/models/{model_id}")
async def get_model(model_id: str):
    return {
        "id": model_id,
        "object": "model",
        "created": 1735689600,
        "owned_by": "",
    }


# -----------------------------------------------------------------------------
# Chat Completions
# -----------------------------------------------------------------------------


def make_chat_completion_object(
    payload: dict[str, Any],
    result: BridgeResult,
    completion_id: str,
    created: int,
    model: str,
) -> dict[str, Any]:
    tool_calls = [
        {
            "id": rid("call"),
            "type": "function",
            "function": {
                "name": call.name,
                "arguments": call.arguments,
            },
        }
        for call in result.tool_calls
    ]
    if tool_calls:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": tool_calls,
        }
        finish_reason = "tool_calls"
    else:
        message = {"role": "assistant", "content": result.text}
        finish_reason = "stop"
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "system_fingerprint": None,
        "choices": [
            {
                "index": 0,
                "message": message,
                "logprobs": None,
                "finish_reason": finish_reason,
            }
        ],
        "usage": usage_for(payload, result.text),
    }


def chat_stream_chunks(
    payload: dict[str, Any],
    obj: dict[str, Any],
) -> list[dict[str, Any]]:
    choice = obj["choices"][0]
    message = choice["message"]
    chunks: list[dict[str, Any]] = []

    for index, tool_call in enumerate(message.get("tool_calls") or []):
        chunks.append(
            {
                "id": obj["id"],
                "object": "chat.completion.chunk",
                "created": obj["created"],
                "model": obj["model"],
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [{**tool_call, "index": index}]
                        },
                        "finish_reason": None,
                    }
                ],
                "usage": None,
            }
        )

    if message.get("content"):
        for delta in text_deltas(message["content"]):
            chunks.append(
                {
                    "id": obj["id"],
                    "object": "chat.completion.chunk",
                    "created": obj["created"],
                    "model": obj["model"],
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": delta},
                            "finish_reason": None,
                        }
                    ],
                    "usage": None,
                }
            )

    chunks.append(
        {
            "id": obj["id"],
            "object": "chat.completion.chunk",
            "created": obj["created"],
            "model": obj["model"],
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": choice["finish_reason"],
                }
            ],
            "usage": None,
        }
    )
    if (payload.get("stream_options") or {}).get("include_usage"):
        chunks.append(
            {
                "id": obj["id"],
                "object": "chat.completion.chunk",
                "created": obj["created"],
                "model": obj["model"],
                "choices": [],
                "usage": obj["usage"],
            }
        )
    return chunks


@app.post("/v1/chat/completions")
async def create_chat_completion(request: Request):
    payload = await read_json(request)
    model = model_from(payload, "gpt-5.5")
    created = now()
    completion_id = rid("chatcmpl")

    if payload.get("stream"):
        async def stream_chat() -> AsyncIterator[bytes]:
            initial = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
                "usage": None,
            }
            async for chunk in sse([initial], done=False):
                yield chunk

            try:
                result = await browser_generation(payload, surface="chat")
            except BrowserError as exc:
                async for chunk in sse(
                    [{"error": {"message": str(exc), "type": "browser_error"}}]
                ):
                    yield chunk
                return

            resolved_model = model_from(payload, result.model or model)
            obj = make_chat_completion_object(
                payload,
                result,
                completion_id,
                created,
                resolved_model,
            )
            if payload.get("store", False):
                STORE["chat_completions"][completion_id] = obj
            async for chunk in sse(chat_stream_chunks(payload, obj)):
                yield chunk

        return StreamingResponse(
            stream_chat(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    result = await browser_generation(payload, surface="chat")
    obj = make_chat_completion_object(
        payload,
        result,
        completion_id,
        created,
        model_from(payload, result.model or model),
    )
    if payload.get("store", False):
        STORE["chat_completions"][completion_id] = obj
    return obj


@app.get("/v1/chat/completions")
async def list_chat_completions():
    return {"object": "list", "data": list(STORE["chat_completions"].values())}


@app.get("/v1/chat/completions/{completion_id}")
async def get_chat_completion(completion_id: str):
    return STORE["chat_completions"].get(
        completion_id,
        {
            "id": completion_id,
            "object": "chat.completion",
            "created": now(),
            "model": "",
            "choices": [],
        },
    )


@app.post("/v1/chat/completions/{completion_id}")
async def update_chat_completion(completion_id: str, request: Request):
    payload = await read_json(request)
    obj = STORE["chat_completions"].setdefault(
        completion_id,
        {"id": completion_id, "object": "chat.completion", "created": now()},
    )
    obj.update({"metadata": payload.get("metadata", payload)})
    return obj


@app.delete("/v1/chat/completions/{completion_id}")
async def delete_chat_completion(completion_id: str):
    STORE["chat_completions"].pop(completion_id, None)
    return {"id": completion_id, "object": "chat.completion.deleted", "deleted": True}


@app.get("/v1/chat/completions/{completion_id}/messages")
async def get_chat_completion_messages(completion_id: str):
    obj = STORE["chat_completions"].get(completion_id)
    message = None
    if obj and obj.get("choices"):
        message = obj["choices"][0].get("message")
    return {"object": "list", "data": [message] if message else []}


# -----------------------------------------------------------------------------
# Responses API
# -----------------------------------------------------------------------------

def make_response_object(
    payload: dict[str, Any],
    result: BridgeResult,
    response_id: str | None = None,
) -> dict[str, Any]:
    model = model_from(payload, result.model or "gpt-5.5")
    response_id = response_id or rid("resp")
    created = now()
    output: list[dict[str, Any]] = []

    if result.reasoning:
        output.append(
            {
                "id": rid("rs"),
                "type": "reasoning",
                "summary": [
                    {"type": "summary_text", "text": result.reasoning}
                ],
            }
        )

    if result.tool_calls:
        for call in result.tool_calls:
            output.append(
                {
                    "id": rid("fc"),
                    "type": "function_call",
                    "status": "completed",
                    "call_id": rid("call"),
                    "name": call.name,
                    "arguments": call.arguments,
                }
            )
    else:
        output.append(
            {
                "id": rid("msg"),
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": result.text,
                        "annotations": [],
                    }
                ],
            }
        )

    return {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "status": "completed",
        "background": bool(payload.get("background", False)),
        "error": None,
        "incomplete_details": None,
        "instructions": payload.get("instructions"),
        "max_output_tokens": payload.get("max_output_tokens"),
        "model": model,
        "output": output,
        "output_text": result.text,
        "parallel_tool_calls": payload.get("parallel_tool_calls", True),
        "previous_response_id": payload.get("previous_response_id"),
        "reasoning": payload.get("reasoning"),
        "store": payload.get("store", True),
        "temperature": payload.get("temperature"),
        "text": payload.get("text"),
        "tool_choice": payload.get("tool_choice", "auto"),
        "tools": payload.get("tools", []),
        "top_p": payload.get("top_p"),
        "truncation": payload.get("truncation", "disabled"),
        "usage": response_usage_for(payload, result.text),
        "user": payload.get("user"),
        "metadata": payload.get("metadata"),
    }


def response_stream_events(obj: dict[str, Any]) -> list[dict[str, Any]]:
    sequence = 0

    def event(event_type: str, **values: Any) -> dict[str, Any]:
        nonlocal sequence
        sequence += 1
        return {
            "type": event_type,
            **values,
            "sequence_number": sequence,
        }

    pending = {
        **obj,
        "status": "in_progress",
        "output": [],
        "output_text": "",
        "usage": None,
    }
    events = [
        event("response.created", response=pending),
        event("response.in_progress", response=pending),
    ]

    for output_index, item in enumerate(obj["output"]):
        item_type = item["type"]
        if item_type == "message":
            added_item = {**item, "status": "in_progress", "content": []}
        elif item_type == "reasoning":
            added_item = {**item, "summary": []}
        elif item_type == "function_call":
            added_item = {**item, "status": "in_progress", "arguments": ""}
        else:
            added_item = item

        events.append(
            event(
                "response.output_item.added",
                response_id=obj["id"],
                output_index=output_index,
                item=added_item,
            )
        )

        if item_type == "message":
            content = item["content"][0]
            events.append(
                event(
                    "response.content_part.added",
                    response_id=obj["id"],
                    item_id=item["id"],
                    output_index=output_index,
                    content_index=0,
                    part={"type": "output_text", "text": "", "annotations": []},
                )
            )
            for delta in text_deltas(content["text"]):
                events.append(
                    event(
                        "response.output_text.delta",
                        response_id=obj["id"],
                        item_id=item["id"],
                        output_index=output_index,
                        content_index=0,
                        delta=delta,
                    )
                )
            events.extend(
                [
                    event(
                        "response.output_text.done",
                        response_id=obj["id"],
                        item_id=item["id"],
                        output_index=output_index,
                        content_index=0,
                        text=content["text"],
                    ),
                    event(
                        "response.content_part.done",
                        response_id=obj["id"],
                        item_id=item["id"],
                        output_index=output_index,
                        content_index=0,
                        part=content,
                    ),
                ]
            )
        elif item_type == "reasoning":
            summary = item["summary"][0]["text"] if item["summary"] else ""
            if summary:
                events.extend(
                    [
                        event(
                            "response.reasoning_summary_part.added",
                            response_id=obj["id"],
                            item_id=item["id"],
                            output_index=output_index,
                            summary_index=0,
                            part={"type": "summary_text", "text": ""},
                        ),
                        event(
                            "response.reasoning_summary_text.delta",
                            response_id=obj["id"],
                            item_id=item["id"],
                            output_index=output_index,
                            summary_index=0,
                            delta=summary,
                        ),
                        event(
                            "response.reasoning_summary_text.done",
                            response_id=obj["id"],
                            item_id=item["id"],
                            output_index=output_index,
                            summary_index=0,
                            text=summary,
                        ),
                        event(
                            "response.reasoning_summary_part.done",
                            response_id=obj["id"],
                            item_id=item["id"],
                            output_index=output_index,
                            summary_index=0,
                            part=item["summary"][0],
                        ),
                    ]
                )
        elif item_type == "function_call":
            events.extend(
                [
                    event(
                        "response.function_call_arguments.delta",
                        response_id=obj["id"],
                        item_id=item["id"],
                        output_index=output_index,
                        delta=item["arguments"],
                    ),
                    event(
                        "response.function_call_arguments.done",
                        response_id=obj["id"],
                        item_id=item["id"],
                        output_index=output_index,
                        name=item["name"],
                        arguments=item["arguments"],
                    ),
                ]
            )

        events.append(
            event(
                "response.output_item.done",
                response_id=obj["id"],
                output_index=output_index,
                item=item,
            )
        )

    events.append(event("response.completed", response=obj))
    return events


def response_input_items(payload: dict[str, Any]) -> list[Any]:
    input_value = payload.get("input", "")
    if isinstance(input_value, str):
        return [
            {
                "id": rid("msg"),
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": input_value}],
            }
        ]
    if not isinstance(input_value, list):
        return [input_value]
    items: list[Any] = []
    for value in input_value:
        if isinstance(value, dict) and "id" not in value:
            value = {"id": rid("item"), **value}
        items.append(value)
    return items


@app.post("/v1/responses")
async def create_response(request: Request):
    payload = await read_json(request)
    response_id = rid("resp")

    if payload.get("stream"):
        async def stream_response() -> AsyncIterator[bytes]:
            seed = make_response_object(
                payload,
                BridgeResult("", "", None, None, ""),
                response_id=response_id,
            )
            async for chunk in sse(
                response_stream_events(seed)[:2],
                done=False,
            ):
                yield chunk

            try:
                result = await browser_generation(payload, surface="responses")
            except (BrowserError, HTTPException) as exc:
                message = (
                    str(exc.detail)
                    if isinstance(exc, HTTPException)
                    else str(exc)
                )
                async for chunk in sse(
                    [
                        {
                            "type": "error",
                            "sequence_number": 3,
                            "code": "browser_error",
                            "message": message,
                            "param": None,
                        }
                    ]
                ):
                    yield chunk
                return

            obj = make_response_object(
                payload,
                result,
                response_id=response_id,
            )
            if payload.get("store", True):
                STORE["responses"][obj["id"]] = obj
                CONVERSATION_URLS[obj["id"]] = result.conversation_url
                RESPONSE_INPUTS[obj["id"]] = response_input_items(payload)

            async for chunk in sse(response_stream_events(obj)[2:]):
                yield chunk

        return StreamingResponse(
            stream_response(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    result = await browser_generation(payload, surface="responses")
    obj = make_response_object(payload, result, response_id=response_id)
    if payload.get("store", True):
        STORE["responses"][obj["id"]] = obj
        CONVERSATION_URLS[obj["id"]] = result.conversation_url
        RESPONSE_INPUTS[obj["id"]] = response_input_items(payload)
    return obj


@app.get("/v1/responses/{response_id}")
async def get_response(response_id: str):
    obj = STORE["responses"].get(response_id)
    if obj is None:
        raise HTTPException(status_code=404, detail="Response not found.")
    return obj


@app.delete("/v1/responses/{response_id}")
async def delete_response(response_id: str):
    STORE["responses"].pop(response_id, None)
    CONVERSATION_URLS.pop(response_id, None)
    RESPONSE_INPUTS.pop(response_id, None)
    return {"id": response_id, "object": "response.deleted", "deleted": True}


@app.post("/v1/responses/{response_id}/cancel")
async def cancel_response(response_id: str):
    obj = STORE["responses"].get(response_id)
    if obj is None:
        raise HTTPException(status_code=404, detail="Response not found.")
    await BRIDGE.stop()
    obj["status"] = "cancelled"
    return obj


@app.get("/v1/responses/{response_id}/input_items")
async def list_response_input_items(response_id: str):
    if response_id not in STORE["responses"]:
        raise HTTPException(status_code=404, detail="Response not found.")
    items = RESPONSE_INPUTS.get(response_id, [])
    first_id = items[0].get("id") if items and isinstance(items[0], dict) else None
    last_id = items[-1].get("id") if items and isinstance(items[-1], dict) else None
    return {
        "object": "list",
        "data": items,
        "first_id": first_id,
        "last_id": last_id,
        "has_more": False,
    }


@app.post("/v1/responses/count_input_tokens")
async def count_input_tokens(request: Request):
    payload = await read_json(request)
    return {"object": "response.input_tokens", "input_tokens": estimate_tokens(payload)}


# -----------------------------------------------------------------------------
# Embeddings
# -----------------------------------------------------------------------------

@app.post("/v1/embeddings")
async def create_embeddings(request: Request):
    payload = await read_json(request)
    model = model_from(payload)
    input_value = payload.get("input", "")

    if isinstance(input_value, list):
        items = input_value
    else:
        items = [input_value]

    encoding_format = payload.get("encoding_format", "float")
    data = []
    for i, item in enumerate(items):
        text = json.dumps(item, ensure_ascii=False) if not isinstance(item, str) else item
        vec = deterministic_embedding(text)
        embedding: list[float] | str
        if encoding_format == "base64":
            embedding = embedding_to_base64(vec)
        else:
            embedding = vec
        data.append({"object": "embedding", "embedding": embedding, "index": i})

    return {
        "object": "list",
        "data": data,
        "model": model,
        "usage": {
            "prompt_tokens": estimate_tokens(input_value),
            "total_tokens": estimate_tokens(input_value),
        },
    }


# -----------------------------------------------------------------------------
# Legacy Completions
# -----------------------------------------------------------------------------

@app.post("/v1/completions")
async def create_completion(request: Request):
    payload = await read_json(request)
    result = await browser_generation(payload, surface="completion")
    model = model_from(payload, result.model or "gpt-5.5")
    completion_id = rid("cmpl")
    text = result.text

    if payload.get("stream"):
        chunks = []
        for delta in text_deltas(text):
            chunks.append(
                {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": now(),
                    "model": model,
                    "choices": [
                        {
                            "text": delta,
                            "index": 0,
                            "logprobs": None,
                            "finish_reason": None,
                        }
                    ],
                }
            )
        chunks.append(
            {
                "id": completion_id,
                "object": "text_completion",
                "created": now(),
                "model": model,
                "choices": [
                    {"text": "", "index": 0, "logprobs": None, "finish_reason": "stop"}
                ],
            }
        )
        return StreamingResponse(sse(chunks), media_type="text/event-stream")

    return {
        "id": completion_id,
        "object": "text_completion",
        "created": now(),
        "model": model,
        "choices": [{"text": text, "index": 0, "logprobs": None, "finish_reason": "stop"}],
        "usage": usage_for(payload, text),
    }


# -----------------------------------------------------------------------------
# Moderations
# -----------------------------------------------------------------------------

@app.post("/v1/moderations")
async def create_moderation(request: Request):
    payload = await read_json(request)
    categories = {
        "harassment": False,
        "harassment/threatening": False,
        "hate": False,
        "hate/threatening": False,
        "illicit": False,
        "illicit/violent": False,
        "self-harm": False,
        "self-harm/intent": False,
        "self-harm/instructions": False,
        "sexual": False,
        "sexual/minors": False,
        "violence": False,
        "violence/graphic": False,
    }
    return {
        "id": rid("modr"),
        "model": payload.get("model") or "",
        "results": [
            {
                "flagged": False,
                "categories": categories,
                "category_scores": {k: 0.0 for k in categories},
                "category_applied_input_types": {k: ["text"] for k in categories},
            }
        ],
    }


# -----------------------------------------------------------------------------
# Audio
# -----------------------------------------------------------------------------

@app.post("/v1/audio/transcriptions")
async def create_transcription(request: Request):
    form = await request.form()
    response_format = str(form.get("response_format") or "json")
    stream = str(form.get("stream") or "false").lower() == "true"

    if stream:
        events = [
            {
                "type": "transcript.text.delta",
                "delta": STREAM_GENERATED_TEXT,
                "logprobs": [],
            },
            {
                "type": "transcript.text.done",
                "text": STREAM_GENERATED_TEXT,
            },
        ]
        return StreamingResponse(sse(events), media_type="text/event-stream")

    payload = {
        "text": GENERATED_TEXT,
        "usage": {
            "type": "duration",
            "seconds": 1,
        },
    }

    if response_format == "verbose_json":
        payload.update(
            {
                "task": "transcribe",
                "language": "english",
                "duration": 1.0,
                "segments": [
                    {
                        "id": 0,
                        "seek": 0,
                        "start": 0.0,
                        "end": 1.0,
                        "text": payload["text"],
                        "tokens": [],
                        "temperature": 0.0,
                        "avg_logprob": 0.0,
                        "compression_ratio": 0.0,
                        "no_speech_prob": 0.0,
                    }
                ],
            }
        )

    return maybe_plain_text_response(response_format, payload)


@app.post("/v1/audio/translations")
async def create_translation(request: Request):
    form = await request.form()
    response_format = str(form.get("response_format") or "json")
    payload = {"text": GENERATED_TEXT}
    return maybe_plain_text_response(response_format, payload)


@app.post("/v1/audio/speech")
async def create_speech(request: Request):
    return Response(content=b"", media_type="audio/mpeg")


# -----------------------------------------------------------------------------
# Images
# -----------------------------------------------------------------------------

@app.post("/v1/images/generations")
@app.post("/v1/images/edits")
@app.post("/v1/images/variations")
async def create_image(request: Request):
    return {
        "created": now(),
        "data": [
            {
                "b64_json": tiny_png_base64(),
                "revised_prompt": GENERATED_TEXT,
            }
        ],
    }


# -----------------------------------------------------------------------------
# Files
# -----------------------------------------------------------------------------

@app.post("/v1/files")
async def upload_file(request: Request):
    form = await request.form()
    file_id = rid("file")
    filename = "unknown"
    size = 0
    purpose = str(form.get("purpose") or "assistants")

    for _, value in form.multi_items():
        if hasattr(value, "filename"):
            filename = value.filename or filename
            content = await value.read()
            size = len(content)
            break

    obj = {
        "id": file_id,
        "object": "file",
        "bytes": size,
        "created_at": now(),
        "filename": filename,
        "purpose": purpose,
        "status": "processed",
    }
    STORE["files"][file_id] = obj
    return obj


@app.get("/v1/files")
async def list_files():
    return {"object": "list", "data": list(STORE["files"].values())}


@app.get("/v1/files/{file_id}")
async def retrieve_file(file_id: str):
    return STORE["files"].get(
        file_id,
        {
            "id": file_id,
            "object": "file",
            "bytes": 0,
            "created_at": now(),
            "filename": "",
            "purpose": "assistants",
            "status": "processed",
        },
    )


@app.delete("/v1/files/{file_id}")
async def delete_file(file_id: str):
    STORE["files"].pop(file_id, None)
    return {"id": file_id, "object": "file", "deleted": True}


@app.get("/v1/files/{file_id}/content")
async def retrieve_file_content(file_id: str):
    return PlainTextResponse("")


# -----------------------------------------------------------------------------
# Minimal Assistants / Threads / Runs compatibility for older harnesses
# -----------------------------------------------------------------------------

@app.post("/v1/assistants")
async def create_assistant(request: Request):
    payload = await read_json(request)
    assistant_id = rid("asst")
    obj = {
        "id": assistant_id,
        "object": "assistant",
        "created_at": now(),
        "name": payload.get("name"),
        "description": payload.get("description"),
        "model": payload.get("model") or "",
        "instructions": payload.get("instructions"),
        "tools": payload.get("tools", []),
        "metadata": payload.get("metadata", {}),
    }
    STORE["assistants"][assistant_id] = obj
    return obj


@app.get("/v1/assistants")
async def list_assistants():
    return {"object": "list", "data": list(STORE["assistants"].values()), "has_more": False}


@app.get("/v1/assistants/{assistant_id}")
async def retrieve_assistant(assistant_id: str):
    return STORE["assistants"].get(
        assistant_id,
        {
            "id": assistant_id,
            "object": "assistant",
            "created_at": now(),
            "model": "",
            "tools": [],
        },
    )


@app.post("/v1/assistants/{assistant_id}")
async def modify_assistant(assistant_id: str, request: Request):
    payload = await read_json(request)
    obj = STORE["assistants"].setdefault(
        assistant_id,
        {"id": assistant_id, "object": "assistant", "created_at": now()},
    )
    obj.update(payload)
    return obj


@app.delete("/v1/assistants/{assistant_id}")
async def delete_assistant(assistant_id: str):
    STORE["assistants"].pop(assistant_id, None)
    return {"id": assistant_id, "object": "assistant.deleted", "deleted": True}


@app.post("/v1/threads")
async def create_thread(request: Request):
    payload = await read_json(request)
    thread_id = rid("thread")
    obj = {
        "id": thread_id,
        "object": "thread",
        "created_at": now(),
        "metadata": payload.get("metadata", {}),
        "tool_resources": payload.get("tool_resources"),
    }
    STORE["threads"][thread_id] = obj
    STORE["messages"].setdefault(thread_id, [])
    return obj


@app.get("/v1/threads/{thread_id}")
async def retrieve_thread(thread_id: str):
    return STORE["threads"].get(
        thread_id,
        {"id": thread_id, "object": "thread", "created_at": now(), "metadata": {}},
    )


@app.post("/v1/threads/{thread_id}")
async def modify_thread(thread_id: str, request: Request):
    payload = await read_json(request)
    obj = STORE["threads"].setdefault(
        thread_id,
        {"id": thread_id, "object": "thread", "created_at": now()},
    )
    obj.update(payload)
    return obj


@app.delete("/v1/threads/{thread_id}")
async def delete_thread(thread_id: str):
    STORE["threads"].pop(thread_id, None)
    STORE["messages"].pop(thread_id, None)
    return {"id": thread_id, "object": "thread.deleted", "deleted": True}


@app.post("/v1/threads/{thread_id}/messages")
async def create_thread_message(thread_id: str, request: Request):
    payload = await read_json(request)
    message_id = rid("msg")
    obj = {
        "id": message_id,
        "object": "thread.message",
        "created_at": now(),
        "thread_id": thread_id,
        "status": "completed",
        "role": payload.get("role", "user"),
        "content": payload.get("content", []),
        "assistant_id": None,
        "run_id": None,
        "attachments": payload.get("attachments", []),
        "metadata": payload.get("metadata", {}),
    }
    STORE["messages"].setdefault(thread_id, []).append(obj)
    return obj


@app.get("/v1/threads/{thread_id}/messages")
async def list_thread_messages(thread_id: str):
    messages = STORE["messages"].get(thread_id, [])
    return {
        "object": "list",
        "data": list(reversed(messages)),
        "first_id": messages[-1]["id"] if messages else None,
        "last_id": messages[0]["id"] if messages else None,
        "has_more": False,
    }


@app.get("/v1/threads/{thread_id}/messages/{message_id}")
async def retrieve_thread_message(thread_id: str, message_id: str):
    for msg in STORE["messages"].get(thread_id, []):
        if msg["id"] == message_id:
            return msg
    return {
        "id": message_id,
        "object": "thread.message",
        "created_at": now(),
        "thread_id": thread_id,
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "text", "text": {"value": GENERATED_TEXT, "annotations": []}}],
    }


def make_run(thread_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    run_id = rid("run")
    prompt_tokens = estimate_tokens(payload)
    completion_tokens = estimate_tokens(GENERATED_TEXT)
    return {
        "id": run_id,
        "object": "thread.run",
        "created_at": now(),
        "thread_id": thread_id,
        "assistant_id": payload.get("assistant_id", rid("asst")),
        "status": "completed",
        "required_action": None,
        "last_error": None,
        "model": payload.get("model") or "",
        "instructions": payload.get("instructions"),
        "tools": payload.get("tools", []),
        "metadata": payload.get("metadata", {}),
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


@app.post("/v1/threads/{thread_id}/runs")
async def create_run(thread_id: str, request: Request):
    payload = await read_json(request)
    obj = make_run(thread_id, payload)
    STORE["runs"][obj["id"]] = obj

    assistant_message = {
        "id": rid("msg"),
        "object": "thread.message",
        "created_at": now(),
        "thread_id": thread_id,
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "text", "text": {"value": GENERATED_TEXT, "annotations": []}}],
        "assistant_id": obj["assistant_id"],
        "run_id": obj["id"],
        "attachments": [],
        "metadata": {},
    }
    STORE["messages"].setdefault(thread_id, []).append(assistant_message)
    return obj


@app.post("/v1/threads/runs")
async def create_thread_and_run(request: Request):
    payload = await read_json(request)
    thread_id = rid("thread")
    STORE["threads"][thread_id] = {
        "id": thread_id,
        "object": "thread",
        "created_at": now(),
        "metadata": {},
    }
    STORE["messages"].setdefault(thread_id, [])
    obj = make_run(thread_id, payload)
    STORE["runs"][obj["id"]] = obj
    return obj


@app.get("/v1/threads/{thread_id}/runs")
async def list_runs(thread_id: str):
    runs = [r for r in STORE["runs"].values() if r.get("thread_id") == thread_id]
    return {"object": "list", "data": runs, "has_more": False}


@app.get("/v1/threads/{thread_id}/runs/{run_id}")
async def retrieve_run(thread_id: str, run_id: str):
    return STORE["runs"].get(run_id, make_run(thread_id, {"assistant_id": rid("asst")}))


@app.post("/v1/threads/{thread_id}/runs/{run_id}")
async def modify_run(thread_id: str, run_id: str, request: Request):
    payload = await read_json(request)
    obj = STORE["runs"].setdefault(run_id, make_run(thread_id, payload))
    obj.update({"metadata": payload.get("metadata", payload)})
    return obj


@app.post("/v1/threads/{thread_id}/runs/{run_id}/cancel")
async def cancel_run(thread_id: str, run_id: str):
    obj = STORE["runs"].get(run_id, make_run(thread_id, {}))
    obj["status"] = "cancelled"
    STORE["runs"][run_id] = obj
    return obj


@app.post("/v1/threads/{thread_id}/runs/{run_id}/submit_tool_outputs")
async def submit_tool_outputs(thread_id: str, run_id: str, request: Request):
    payload = await read_json(request)
    obj = STORE["runs"].get(run_id, make_run(thread_id, payload))
    obj["status"] = "completed"
    obj["required_action"] = None
    STORE["runs"][run_id] = obj
    return obj


@app.get("/v1/threads/{thread_id}/runs/{run_id}/steps")
async def list_run_steps(thread_id: str, run_id: str):
    return {
        "object": "list",
        "data": [
            {
                "id": rid("step"),
                "object": "thread.run.step",
                "created_at": now(),
                "run_id": run_id,
                "assistant_id": rid("asst"),
                "thread_id": thread_id,
                "type": "message_creation",
                "status": "completed",
                "step_details": {"type": "message_creation", "message_creation": {"message_id": rid("msg")}},
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ],
        "has_more": False,
    }


# -----------------------------------------------------------------------------
# Minimal Batches compatibility
# -----------------------------------------------------------------------------

@app.post("/v1/batches")
async def create_batch(request: Request):
    payload = await read_json(request)
    batch_id = rid("batch")
    obj = {
        "id": batch_id,
        "object": "batch",
        "endpoint": payload.get("endpoint", "/v1/responses"),
        "errors": None,
        "input_file_id": payload.get("input_file_id"),
        "completion_window": payload.get("completion_window", "24h"),
        "status": "completed",
        "output_file_id": rid("file"),
        "error_file_id": None,
        "created_at": now(),
        "in_progress_at": now(),
        "expires_at": now() + 86400,
        "finalizing_at": now(),
        "completed_at": now(),
        "failed_at": None,
        "expired_at": None,
        "cancelling_at": None,
        "cancelled_at": None,
        "request_counts": {"total": 1, "completed": 1, "failed": 0},
        "metadata": payload.get("metadata"),
    }
    STORE["batches"][batch_id] = obj
    return obj


@app.get("/v1/batches")
async def list_batches():
    return {"object": "list", "data": list(STORE["batches"].values()), "has_more": False}


@app.get("/v1/batches/{batch_id}")
async def retrieve_batch(batch_id: str):
    return STORE["batches"].get(
        batch_id,
        {
            "id": batch_id,
            "object": "batch",
            "endpoint": "/v1/responses",
            "status": "completed",
            "created_at": now(),
            "request_counts": {"total": 0, "completed": 0, "failed": 0},
        },
    )


@app.post("/v1/batches/{batch_id}/cancel")
async def cancel_batch(batch_id: str):
    obj = STORE["batches"].get(batch_id) or {
        "id": batch_id,
        "object": "batch",
        "endpoint": "/v1/responses",
        "created_at": now(),
    }
    obj["status"] = "cancelled"
    obj["cancelled_at"] = now()
    STORE["batches"][batch_id] = obj
    return obj


# -----------------------------------------------------------------------------
# Minimal Vector Stores compatibility
# -----------------------------------------------------------------------------

@app.post("/v1/vector_stores")
async def create_vector_store(request: Request):
    payload = await read_json(request)
    vector_store_id = rid("vs")
    obj = {
        "id": vector_store_id,
        "object": "vector_store",
        "created_at": now(),
        "name": payload.get("name"),
        "usage_bytes": 0,
        "file_counts": {"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0},
        "status": "completed",
        "expires_after": payload.get("expires_after"),
        "expires_at": None,
        "last_active_at": now(),
        "metadata": payload.get("metadata", {}),
    }
    STORE["vector_stores"][vector_store_id] = obj
    return obj


@app.get("/v1/vector_stores")
async def list_vector_stores():
    return {"object": "list", "data": list(STORE["vector_stores"].values()), "has_more": False}


@app.get("/v1/vector_stores/{vector_store_id}")
async def retrieve_vector_store(vector_store_id: str):
    return STORE["vector_stores"].get(
        vector_store_id,
        {
            "id": vector_store_id,
            "object": "vector_store",
            "created_at": now(),
            "usage_bytes": 0,
            "file_counts": {"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0},
            "status": "completed",
        },
    )


@app.post("/v1/vector_stores/{vector_store_id}")
async def modify_vector_store(vector_store_id: str, request: Request):
    payload = await read_json(request)
    obj = STORE["vector_stores"].setdefault(
        vector_store_id,
        {
            "id": vector_store_id,
            "object": "vector_store",
            "created_at": now(),
            "usage_bytes": 0,
            "status": "completed",
        },
    )
    obj.update(payload)
    return obj


@app.delete("/v1/vector_stores/{vector_store_id}")
async def delete_vector_store(vector_store_id: str):
    STORE["vector_stores"].pop(vector_store_id, None)
    return {"id": vector_store_id, "object": "vector_store.deleted", "deleted": True}


@app.post("/v1/vector_stores/{vector_store_id}/files")
async def create_vector_store_file(vector_store_id: str, request: Request):
    payload = await read_json(request)
    return {
        "id": payload.get("file_id", rid("file")),
        "object": "vector_store.file",
        "usage_bytes": 0,
        "created_at": now(),
        "vector_store_id": vector_store_id,
        "status": "completed",
        "last_error": None,
    }


@app.get("/v1/vector_stores/{vector_store_id}/files")
async def list_vector_store_files(vector_store_id: str):
    return {"object": "list", "data": [], "has_more": False}


# -----------------------------------------------------------------------------
# Catch-all for observing unsupported harness calls without breaking immediately
# -----------------------------------------------------------------------------

@app.api_route(
    "/v1/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
)
async def openai_fallback(path: str, request: Request):
    payload = await read_json(request)
    if request.method == "OPTIONS":
        return Response(status_code=204)

    return JSONResponse(
        {
            "object": "unimplemented",
            "method": request.method,
            "path": f"/v1/{path}",
            "received_json": payload,
        },
        status_code=200,
    )


