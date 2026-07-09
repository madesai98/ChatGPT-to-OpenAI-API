"""ChatGPT website adapter built on the generic Zendriver controller."""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Literal
from urllib.parse import urlparse

from browser import BrowserController, BrowserError


CHATGPT_URL = "https://chatgpt.com/"
COMPOSER_SELECTOR = "#prompt-textarea"
SEND_SELECTOR = 'button[data-testid="send-button"]'
STOP_SELECTOR = (
    'button[data-testid="stop-button"],button[aria-label="Stop generating"]'
)
TARGET_MODEL_SLUG = "gpt-5-5-thinking"

StreamKind = Literal["reasoning", "output", "status"]


@dataclass(frozen=True)
class StreamEvent:
    kind: StreamKind
    delta: str = ""
    full_text: str = ""
    model: str | None = None
    source: str = "websocket"


@dataclass(frozen=True)
class ChatGPTResponse:
    reasoning: str
    output: str
    model: str | None
    effort: str | None
    conversation_url: str


StreamCallback = Callable[[StreamEvent], None | Awaitable[None]]


class ChatGPTStreamDecoder:
    """Decode ChatGPT's WebSocket envelope and nested SSE delta stream."""

    _REASONING_CONTENT_TYPES = {
        "thoughts",
        "reasoning",
        "reasoning_recap",
        "reasoning_summary",
    }

    def __init__(self) -> None:
        self.phase: Literal["reasoning", "output"] = "reasoning"
        self.path = ""
        self.operation = ""
        self.message_kind: Literal["reasoning", "output"] | None = None
        self.reasoning = ""
        self.output = ""
        self.model: str | None = None
        self.effort: str | None = None
        self.complete = False
        self.last_token = False
        self._seen_stream_items: set[str] = set()

    def feed_websocket(self, payload: str) -> list[StreamEvent]:
        try:
            outer = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return []

        events: list[StreamEvent] = []
        for stream_item_id, encoded_item in self._encoded_items(outer):
            if stream_item_id and stream_item_id in self._seen_stream_items:
                continue
            if stream_item_id:
                self._seen_stream_items.add(stream_item_id)
            for data in self._sse_data(encoded_item):
                events.extend(self._feed_data(data))
        return events

    def _encoded_items(self, value: Any) -> list[tuple[str | None, str]]:
        found: list[tuple[str | None, str]] = []

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                encoded = node.get("encoded_item")
                if isinstance(encoded, str):
                    item_id = node.get("stream_item_id")
                    found.append(
                        (str(item_id) if item_id is not None else None, encoded)
                    )
                for child in node.values():
                    walk(child)
            elif isinstance(node, list):
                for child in node:
                    walk(child)

        walk(value)
        return found

    @staticmethod
    def _sse_data(encoded_item: str) -> list[Any]:
        decoded: list[Any] = []
        for block in re.split(r"\r?\n\r?\n+", encoded_item.strip()):
            data_lines = [
                line[5:].lstrip()
                for line in block.splitlines()
                if line.startswith("data:")
            ]
            if not data_lines:
                continue
            raw = "\n".join(data_lines)
            try:
                decoded.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
        return decoded

    def _feed_data(self, data: Any) -> list[StreamEvent]:
        if not isinstance(data, dict):
            return []

        data_type = data.get("type")
        if data_type == "message_marker":
            marker = data.get("marker")
            if marker == "cot_token":
                self.phase = "reasoning"
                self.message_kind = "reasoning"
            elif marker == "final_channel_token":
                self.phase = "output"
                self.message_kind = "output"
            elif marker == "last_token":
                self.last_token = True
            return []

        if data_type == "message_stream_complete":
            self.complete = True
            return []

        if data_type == "server_ste_metadata":
            metadata = data.get("metadata") or {}
            self.model = metadata.get("model_slug") or self.model
            self.effort = metadata.get("thinking_effort") or self.effort
            return []

        if data_type == "input_message":
            return []

        return self._feed_delta(data)

    def _feed_delta(self, delta: dict[str, Any]) -> list[StreamEvent]:
        value = delta.get("v")
        if isinstance(value, dict):
            message = value.get("message")
            if isinstance(message, dict):
                return self._start_message(message)

        if delta.get("o") == "patch" and isinstance(value, list):
            events: list[StreamEvent] = []
            for patch in value:
                if isinstance(patch, dict):
                    events.extend(self._feed_delta(patch))
            return events

        if isinstance(delta.get("p"), str):
            self.path = delta["p"]
        if isinstance(delta.get("o"), str):
            self.operation = delta["o"]

        if self.operation != "append" or not isinstance(value, str):
            return []
        if not self._is_text_path(self.path):
            return []
        return [self._text_event(self.message_kind or self.phase, value)]

    def _start_message(self, message: dict[str, Any]) -> list[StreamEvent]:
        author = message.get("author") or {}
        if author.get("role") != "assistant":
            self.message_kind = None
            return []

        metadata = message.get("metadata") or {}
        self.model = (
            metadata.get("resolved_model_slug")
            or metadata.get("model_slug")
            or self.model
        )
        self.effort = metadata.get("thinking_effort") or self.effort

        content = message.get("content") or {}
        content_type = str(content.get("content_type") or "")
        channel = str(message.get("channel") or "")
        if (
            content_type in self._REASONING_CONTENT_TYPES
            or channel in {"analysis", "commentary"}
        ):
            self.message_kind = "reasoning"
        elif channel == "final":
            self.message_kind = "output"
        else:
            self.message_kind = self.phase

        initial_text = self._content_text(content)
        if not initial_text:
            return []
        return [self._text_event(self.message_kind, initial_text)]

    @staticmethod
    def _content_text(content: dict[str, Any]) -> str:
        parts = content.get("parts")
        if isinstance(parts, list):
            return "".join(part for part in parts if isinstance(part, str))
        for key in ("text", "summary", "content"):
            value = content.get(key)
            if isinstance(value, str):
                return value
        return ""

    @staticmethod
    def _is_text_path(path: str) -> bool:
        return bool(
            re.search(
                r"/(?:parts/\d+|text|summary|content)$",
                path,
            )
        )

    def _text_event(
        self,
        kind: Literal["reasoning", "output"],
        delta: str,
    ) -> StreamEvent:
        if kind == "reasoning":
            self.reasoning += delta
            full_text = self.reasoning
        else:
            self.output += delta
            full_text = self.output
        return StreamEvent(
            kind=kind,
            delta=delta,
            full_text=full_text,
            model=self.model,
            source="websocket",
        )


class ChatGPT:
    """Selectors and behavior specific to chatgpt.com."""

    def __init__(self, browser: BrowserController) -> None:
        self.browser = browser

    async def open(self) -> str:
        return await self.browser.goto(CHATGPT_URL)

    async def new_chat(self) -> str:
        """Open an empty ChatGPT conversation."""
        await self.open()
        await self.browser.wait_for(COMPOSER_SELECTOR, timeout=30)
        if (await self.composer_text()).strip():
            await self.clear_text()
        return await self.browser.url()

    async def open_chat(self, conversation: str) -> str:
        """Open a ChatGPT conversation URL or conversation ID."""
        value = conversation.strip()
        if not value:
            raise BrowserError("Conversation URL or ID is required.")
        if "://" not in value:
            value = f"https://chatgpt.com/c/{value.strip('/')}"
        parsed = urlparse(value)
        if parsed.scheme != "https" or parsed.netloc != "chatgpt.com":
            raise BrowserError("Conversation must be on https://chatgpt.com/.")
        await self.browser.goto(value)
        await self.browser.wait_for(COMPOSER_SELECTOR, timeout=30)
        return await self.browser.url()

    async def ensure_open(self) -> None:
        current = await self.browser.url()
        if not current.startswith("https://chatgpt.com/"):
            await self.open()
        await self.browser.wait_for(COMPOSER_SELECTOR, timeout=30)

    async def add_text(self, text: str) -> None:
        """Append text to composer using CDP's paste-like insertText event."""
        if not text:
            raise BrowserError("add_text requires non-empty text.")
        await self.ensure_open()
        composer = await self.browser.wait_for(COMPOSER_SELECTOR, timeout=30)
        await composer.focus()
        try:
            from zendriver import cdp

            await self.browser.page.send(cdp.input_.insert_text(text))
        except Exception:
            await composer.send_keys(text)

    async def composer_text(self) -> str:
        await self.ensure_open()
        return str(
            await self.browser.evaluate(
                """(() => {
                    const el = document.querySelector("#prompt-textarea");
                    return el ? (el.innerText || el.textContent || "") : "";
                })()"""
            )
        )

    async def clear_text(self) -> None:
        await self.ensure_open()
        composer = await self.browser.wait_for(COMPOSER_SELECTOR, timeout=30)
        await composer.apply(
            """(el) => {
                el.focus();
                el.innerHTML = "<p><br></p>";
                el.dispatchEvent(new InputEvent("input", {
                    bubbles: true,
                    inputType: "deleteContentBackward"
                }));
            }"""
        )

    async def ask(
        self,
        text: str,
        callback: StreamCallback | None = None,
        *,
        conversation: str | None = None,
        new_chat: bool = False,
        timeout: float = 900.0,
    ) -> ChatGPTResponse:
        """Atomically choose a conversation, replace composer text, and send."""
        if not text:
            raise BrowserError("ask requires non-empty text.")
        if conversation and new_chat:
            raise BrowserError("conversation and new_chat are mutually exclusive.")
        if conversation:
            await self.open_chat(conversation)
        elif new_chat:
            await self.new_chat()
        else:
            await self.ensure_open()
        await self.clear_text()
        await self.add_text(text)
        return await self.send(callback, timeout=timeout)

    async def effort(self) -> str | None:
        value = await self.browser.evaluate(
            """(() => {
                const pills = [...document.querySelectorAll(
                    'button.__composer-pill[aria-haspopup="menu"]'
                )];
                const pill = pills.find(
                    e => ["High", "Standard", "Extended", "Low"].includes(
                        (e.innerText || "").trim()
                    )
                );
                return pill ? (pill.innerText || "").trim() : null;
            })()"""
        )
        return str(value) if value else None

    async def ensure_high(self) -> None:
        """Select High reasoning effort when selector is available."""
        await self.ensure_open()
        if await self.effort() == "High":
            return

        pills = await self.browser.page.query_selector_all(
            'button.__composer-pill[aria-haspopup="menu"]'
        )
        effort_pill = next(
            (
                pill
                for pill in pills
                if (pill.text_all or "").strip()
                in {"Standard", "Extended", "Low", "High"}
            ),
            None,
        )
        if effort_pill is None:
            # Current ChatGPT builds can hide this control while retaining the
            # account's persisted effort. Stream metadata verifies it later.
            return
        await effort_pill.click()

        deadline = asyncio.get_running_loop().time() + 10
        while asyncio.get_running_loop().time() < deadline:
            choices = await self.browser.page.query_selector_all(
                '[role="menuitem"],[role="option"]'
            )
            high = next(
                (
                    choice
                    for choice in choices
                    if (choice.text_all or "").strip() == "High"
                ),
                None,
            )
            if high is not None:
                await high.click()
                return
            await asyncio.sleep(0.1)
        raise BrowserError("High reasoning effort menu item not found.")

    async def send(
        self,
        callback: StreamCallback | None = None,
        *,
        timeout: float = 900.0,
    ) -> ChatGPTResponse:
        """
        Send composer contents and stream visible reasoning/output DOM deltas.

        ChatGPT exposes visible reasoning summaries, not private hidden
        chain-of-thought. Those visible summaries are returned as `reasoning`.
        """
        await self.ensure_high()
        if not (await self.composer_text()).strip():
            raise BrowserError("Composer is empty. Use add_text first.")

        initial_turns = int(
            await self.browser.evaluate(
                'document.querySelectorAll(\'section[data-turn="assistant"]\').length'
            )
            or 0
        )

        from zendriver import cdp

        decoder = ChatGPTStreamDecoder()
        websocket_frames: asyncio.Queue[str] = asyncio.Queue()
        last_websocket_frame_at = asyncio.get_running_loop().time()

        async def on_websocket_frame(
            event: Any, _connection: Any = None
        ) -> None:
            nonlocal last_websocket_frame_at
            if not isinstance(event, cdp.network.WebSocketFrameReceived):
                return
            payload = event.response.payload_data
            if isinstance(payload, str):
                last_websocket_frame_at = asyncio.get_running_loop().time()
                await websocket_frames.put(payload)

        await self.browser.page.send(cdp.network.enable())
        self.browser.page.add_handler(
            cdp.network.WebSocketFrameReceived,
            on_websocket_frame,
        )

        send_button = await self.browser.wait_for(SEND_SELECTOR, timeout=30)
        await send_button.click()
        await self._emit(callback, StreamEvent(kind="status", delta="sent"))

        deadline = asyncio.get_running_loop().time() + timeout
        started_at = asyncio.get_running_loop().time()
        reasoning = ""
        output = ""
        model: str | None = None
        stable_complete_polls = 0
        websocket_seen = False

        while asyncio.get_running_loop().time() < deadline:
            while not websocket_frames.empty():
                websocket_seen = True
                payload = websocket_frames.get_nowait()
                for event in decoder.feed_websocket(payload):
                    await self._emit(callback, event)
            model = decoder.model or model

            state = await self._latest_turn_state(initial_turns)
            if state is None:
                await asyncio.sleep(0.08)
                continue

            model = state.get("model") or model
            next_reasoning = state.get("reasoning", "")
            next_output = state.get("output", "")

            # WebSocket stream is authoritative and arrives before DOM paint.
            # DOM remains fallback when ChatGPT changes transport or payload.
            if decoder.reasoning:
                reasoning = decoder.reasoning
            elif (
                not websocket_seen
                and asyncio.get_running_loop().time() - started_at >= 2
            ):
                reasoning = await self._stream_change(
                    callback, "reasoning", reasoning, next_reasoning, model
                )
            if decoder.output:
                output = decoder.output
            elif (
                not websocket_seen
                and asyncio.get_running_loop().time() - started_at >= 2
            ):
                output = await self._stream_change(
                    callback, "output", output, next_output, model
                )

            if state.get("complete") or decoder.complete:
                stable_complete_polls += 1
                if stable_complete_polls >= 2:
                    # Event callbacks are tasks. Let all frames queued before and
                    # immediately after message_stream_complete settle.
                    settle_deadline = (
                        asyncio.get_running_loop().time() + 1.5
                    )
                    while (
                        asyncio.get_running_loop().time() < settle_deadline
                    ):
                        while not websocket_frames.empty():
                            websocket_seen = True
                            payload = websocket_frames.get_nowait()
                            for event in decoder.feed_websocket(payload):
                                await self._emit(callback, event)
                        quiet_for = (
                            asyncio.get_running_loop().time()
                            - last_websocket_frame_at
                        )
                        if websocket_frames.empty() and quiet_for >= 0.3:
                            break
                        await asyncio.sleep(0.05)

                    # Final rendered DOM is authoritative. Reconcile any tail
                    # that CDP callback scheduling delivered after completion.
                    final_state = await self._latest_turn_state(initial_turns)
                    if final_state is not None:
                        next_reasoning = final_state.get("reasoning", "")
                        next_output = final_state.get("output", "")
                    if decoder.reasoning:
                        reasoning = decoder.reasoning
                    reasoning = await self._reconcile_terminal(
                        callback,
                        "reasoning",
                        reasoning,
                        next_reasoning,
                        model,
                    )
                    if decoder.output:
                        output = decoder.output
                    output = await self._reconcile_terminal(
                        callback,
                        "output",
                        output,
                        next_output,
                        model,
                    )
                    model = decoder.model or model
                    break
            else:
                stable_complete_polls = 0
            await asyncio.sleep(0.08)
        else:
            self.browser.page.remove_handlers(
                cdp.network.WebSocketFrameReceived,
                on_websocket_frame,
            )
            raise BrowserError(
                f"ChatGPT generation timed out after {timeout:g}s. "
                f"Partial output: {output[-200:]}"
            )

        self.browser.page.remove_handlers(
            cdp.network.WebSocketFrameReceived,
            on_websocket_frame,
        )
        await self._emit(
            callback,
            StreamEvent(
                kind="status",
                delta="complete",
                full_text=output,
                model=model,
                source="websocket" if decoder.output else "dom",
            ),
        )
        return ChatGPTResponse(
            reasoning=reasoning,
            output=output,
            model=model,
            effort=decoder.effort,
            conversation_url=await self.browser.url(),
        )

    async def stop(self) -> bool:
        """Stop active generation. Returns False if nothing is generating."""
        await self.ensure_open()
        button = await self.browser.page.query_selector(STOP_SELECTOR)
        if button is None:
            return False
        await button.click()
        return True

    async def model_info(self) -> str | None:
        """Return model slug from latest rendered assistant message."""
        model = await self.browser.evaluate(
            """(() => {
                const nodes = [...document.querySelectorAll(
                    '[data-message-model-slug]'
                )];
                return nodes.length
                    ? nodes[nodes.length - 1].dataset.messageModelSlug
                    : null;
            })()"""
        )
        return str(model) if model else None

    async def _latest_turn_state(
        self, initial_turns: int
    ) -> dict[str, Any] | None:
        script = f"""(() => {{
            const turns = [...document.querySelectorAll(
                'section[data-turn="assistant"]'
            )];
            if (turns.length <= {initial_turns}) return null;
            const turn = turns[turns.length - 1];
            const messages = [...turn.querySelectorAll(
                '[data-message-author-role="assistant"]'
            )];
            const buttons = [...turn.querySelectorAll("button")];
            const marker = buttons.find(button =>
                /^(Thought|Thinking)(\\s|$)/i.test(
                    (button.innerText || "").trim()
                )
            );
            const before = [];
            const after = [];
            for (const message of messages) {{
                const text = (message.innerText || "").trim();
                if (!text) continue;
                if (marker && (marker.compareDocumentPosition(message) & 4)) {{
                    after.push(text);
                }} else {{
                    before.push(text);
                }}
            }}
            const active = Boolean(document.querySelector(
                'button[data-testid="stop-button"],' +
                'button[aria-label="Stop generating"]'
            ));
            let reasoning = "";
            let output = "";
            if (marker) {{
                reasoning = before.join("\\n\\n");
                output = after.join("\\n\\n");
            }} else if (active) {{
                reasoning = before.join("\\n\\n");
            }} else {{
                output = before.join("\\n\\n");
            }}
            const modelNode = messages.find(
                node => node.dataset.messageModelSlug
            );
            const hasActions = Boolean(turn.querySelector(
                '[data-testid="copy-turn-action-button"]'
            ));
            return {{
                reasoning,
                output,
                active,
                complete: !active && hasActions,
                model: modelNode
                    ? modelNode.dataset.messageModelSlug
                    : null
            }};
        }})()"""
        raw = await self.browser.evaluate(f"JSON.stringify({script})")
        if raw in (None, "null"):
            return None
        if isinstance(raw, str):
            return json.loads(raw)
        return raw

    @staticmethod
    def _suffix(old: str, new: str) -> tuple[str, bool]:
        if new.startswith(old):
            return new[len(old) :], False
        common = 0
        limit = min(len(old), len(new))
        while common < limit and old[common] == new[common]:
            common += 1
        return new[common:], True

    async def _stream_change(
        self,
        callback: StreamCallback | None,
        kind: Literal["reasoning", "output"],
        old: str,
        new: str,
        model: str | None,
    ) -> str:
        if new == old:
            return old
        delta, reset = self._suffix(old, new)
        if reset:
            # DOM rewrites can correct earlier streamed text. Emit full value.
            delta = new
        await self._emit(
            callback,
            StreamEvent(
                kind=kind,
                delta=delta,
                full_text=new,
                model=model,
                source="dom",
            ),
        )
        return new

    async def _reconcile_terminal(
        self,
        callback: StreamCallback | None,
        kind: Literal["reasoning", "output"],
        streamed: str,
        rendered: str,
        model: str | None,
    ) -> str:
        """Append a tail present in final DOM but missing from live stream."""
        if not rendered or rendered == streamed:
            return streamed
        if not streamed:
            delta = rendered
        elif rendered.startswith(streamed):
            delta = rendered[len(streamed) :]
        elif streamed.startswith(rendered):
            return streamed
        else:
            # DOM strips Markdown syntax. Find a substantial streamed suffix
            # anywhere in rendered text, then append everything after it.
            limit = min(len(streamed), len(rendered))
            minimum_overlap = min(16, limit)
            for size in range(limit, minimum_overlap - 1, -1):
                position = rendered.rfind(streamed[-size:])
                if position >= 0:
                    delta = rendered[position + size :]
                    if not delta:
                        return rendered
                    break
            else:
                delta = ""

            if not delta:
                # Handle a late chunk whose boundary overlaps existing output.
                overlap = 0
                for size in range(limit, 0, -1):
                    if streamed.endswith(rendered[:size]):
                        overlap = size
                        break
                if overlap == 0:
                    # Avoid printing whole answer twice after a formatting-only
                    # DOM rewrite. Returned response still uses authoritative DOM.
                    return rendered
                delta = rendered[overlap:]

        await self._emit(
            callback,
            StreamEvent(
                kind=kind,
                delta=delta,
                full_text=rendered,
                model=model,
                source="dom",
            ),
        )
        return rendered

    @staticmethod
    async def _emit(
        callback: StreamCallback | None, event: StreamEvent
    ) -> None:
        if callback is None:
            return
        result = callback(event)
        if inspect.isawaitable(result):
            await result
