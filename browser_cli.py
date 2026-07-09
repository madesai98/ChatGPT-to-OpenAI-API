"""Interactive shell for browser.py and chatgpt.py."""

from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path

from browser import BrowserController, BrowserError
from chatgpt import ChatGPT, StreamEvent


Command = Callable[[str], Awaitable[None]]


class BrowserShell:
    """Small command registry; add future commands in _register_commands."""

    def __init__(self, *, headless: bool = True) -> None:
        self.browser = BrowserController(headless=headless)
        self.chatgpt = ChatGPT(self.browser)
        self.running = True
        self._stream_kind: str | None = None
        self.commands: dict[str, Command] = {}
        self.help_text: dict[str, str] = {}
        self._register_commands()

    def register(self, name: str, help_text: str, handler: Command) -> None:
        self.commands[name] = handler
        self.help_text[name] = help_text

    def _register_commands(self) -> None:
        self.register("help", "help [command]", self.cmd_help)
        self.register("sc", "sc  (open chatgpt.com)", self.cmd_sc)
        self.register("auth", "auth [url]  (manual headful login)", self.cmd_auth)
        self.register("add_text", "add_text TEXT", self.cmd_add_text)
        self.register("send", "send [TEXT]", self.cmd_send)
        self.register("ask", "ask TEXT  (replace composer and send)", self.cmd_ask)
        self.register("new_chat", "new_chat  (start empty conversation)", self.cmd_new_chat)
        self.register(
            "open_chat",
            "open_chat URL_OR_ID  (resume conversation)",
            self.cmd_open_chat,
        )
        self.register("clear_text", "clear_text", self.cmd_clear_text)
        self.register("stop", "stop  (stop ChatGPT generation)", self.cmd_stop)
        self.register("model", "model  (latest response model slug)", self.cmd_model)
        self.register("go", "go URL", self.cmd_go)
        self.register("goto", "goto URL", self.cmd_go)
        self.register("back", "back", self.cmd_back)
        self.register("forward", "forward", self.cmd_forward)
        self.register("reload", "reload", self.cmd_reload)
        self.register("url", "url", self.cmd_url)
        self.register("title", "title", self.cmd_title)
        self.register("click", "click CSS_SELECTOR", self.cmd_click)
        self.register("type", "type CSS_SELECTOR TEXT", self.cmd_type)
        self.register("text", "text CSS_SELECTOR", self.cmd_text)
        self.register("html", "html [CSS_SELECTOR]", self.cmd_html)
        self.register("js", "js JAVASCRIPT_EXPRESSION", self.cmd_js)
        self.register("screenshot", "screenshot PATH", self.cmd_screenshot)
        self.register("quit", "quit", self.cmd_quit)
        self.register("exit", "exit", self.cmd_quit)

    async def run(self) -> None:
        print(
            f"Zendriver shell. Persistent profile: {self.browser.profile_dir}\n"
            "Type help for commands."
        )
        try:
            while self.running:
                try:
                    line = await asyncio.to_thread(input, "browser> ")
                except EOFError:
                    break
                except KeyboardInterrupt:
                    print()
                    continue
                await self.execute(line)
        finally:
            await self.browser.stop()

    async def execute(self, line: str) -> None:
        line = line.strip()
        if not line:
            return
        command, _, argument = line.partition(" ")
        handler = self.commands.get(command.lower())
        if handler is None:
            print(f"Unknown command: {command}. Type help.")
            return
        try:
            await handler(argument.strip())
        except BrowserError as exc:
            print(f"error: {exc}", file=sys.stderr)
        except Exception as exc:
            print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)

    @staticmethod
    def _require(argument: str, usage: str) -> str:
        if not argument:
            raise BrowserError(f"Usage: {usage}")
        return argument

    async def cmd_help(self, argument: str) -> None:
        if argument:
            text = self.help_text.get(argument)
            print(text or f"Unknown command: {argument}")
            return
        for name in sorted(self.help_text):
            print(self.help_text[name])

    async def cmd_sc(self, _: str) -> None:
        print(await self.chatgpt.open())

    async def cmd_auth(self, argument: str) -> None:
        await self.browser.auth(argument or "https://chatgpt.com/")
        print("Auth window closed. Persistent session saved.")

    async def cmd_add_text(self, argument: str) -> None:
        await self.chatgpt.add_text(
            self._require(argument, "add_text TEXT")
        )
        print("Text added.")

    async def cmd_send(self, argument: str) -> None:
        if argument:
            await self.chatgpt.add_text(argument)
        self._stream_kind = None
        response = await self.chatgpt.send(self._print_stream)
        if self._stream_kind is not None:
            print()
        if response.model and response.model != "gpt-5-5-thinking":
            print(
                f"warning: rendered model was {response.model}, "
                "not gpt-5-5-thinking",
                file=sys.stderr,
            )
        if response.effort and response.effort not in {"high", "extended"}:
            print(
                f"warning: reported thinking effort was {response.effort}",
                file=sys.stderr,
            )
        print(
            f"[done] model={response.model or 'unknown'} "
            f"effort={response.effort or 'unknown'} "
            f"{response.conversation_url}"
        )

    async def cmd_ask(self, argument: str) -> None:
        text = self._require(argument, "ask TEXT")
        self._stream_kind = None
        response = await self.chatgpt.ask(text, self._print_stream)
        if self._stream_kind is not None:
            print()
        print(
            f"[done] model={response.model or 'unknown'} "
            f"effort={response.effort or 'unknown'} "
            f"{response.conversation_url}"
        )

    async def cmd_new_chat(self, _: str) -> None:
        print(await self.chatgpt.new_chat())

    async def cmd_open_chat(self, argument: str) -> None:
        print(
            await self.chatgpt.open_chat(
                self._require(argument, "open_chat URL_OR_ID")
            )
        )

    def _print_stream(self, event: StreamEvent) -> None:
        if event.kind == "status":
            return
        if event.kind != self._stream_kind:
            if self._stream_kind is not None:
                print()
            print(f"[{event.kind}]")
            self._stream_kind = event.kind
        print(event.delta, end="", flush=True)

    async def cmd_clear_text(self, _: str) -> None:
        await self.chatgpt.clear_text()
        print("Composer cleared.")

    async def cmd_stop(self, _: str) -> None:
        print("Stopped." if await self.chatgpt.stop() else "Not generating.")

    async def cmd_model(self, _: str) -> None:
        print(await self.chatgpt.model_info() or "No rendered model yet.")

    async def cmd_go(self, argument: str) -> None:
        print(await self.browser.goto(self._require(argument, "go URL")))

    async def cmd_back(self, _: str) -> None:
        print(await self.browser.back())

    async def cmd_forward(self, _: str) -> None:
        print(await self.browser.forward())

    async def cmd_reload(self, _: str) -> None:
        print(await self.browser.reload())

    async def cmd_url(self, _: str) -> None:
        print(await self.browser.url())

    async def cmd_title(self, _: str) -> None:
        print(await self.browser.title())

    async def cmd_click(self, argument: str) -> None:
        await self.browser.click(self._require(argument, "click CSS_SELECTOR"))
        print("Clicked.")

    async def cmd_type(self, argument: str) -> None:
        try:
            parts = shlex.split(argument, posix=False)
        except ValueError as exc:
            raise BrowserError(str(exc)) from exc
        if len(parts) < 2:
            raise BrowserError("Usage: type CSS_SELECTOR TEXT")
        selector = parts[0].strip("\"'")
        text = " ".join(parts[1:]).strip("\"'")
        await self.browser.type_text(selector, text)
        print("Text typed.")

    async def cmd_text(self, argument: str) -> None:
        print(await self.browser.text(self._require(argument, "text CSS_SELECTOR")))

    async def cmd_html(self, argument: str) -> None:
        print(await self.browser.html(argument or "html"))

    async def cmd_js(self, argument: str) -> None:
        result = await self.browser.evaluate(
            self._require(argument, "js JAVASCRIPT_EXPRESSION")
        )
        if isinstance(result, (dict, list)):
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(result)

    async def cmd_screenshot(self, argument: str) -> None:
        target = await self.browser.screenshot(
            self._require(argument, "screenshot PATH").strip("\"'")
        )
        print(target)

    async def cmd_quit(self, _: str) -> None:
        self.running = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headful",
        action="store_true",
        help="show normal browser window for all commands",
    )
    parser.add_argument(
        "-c",
        "--command",
        help="run one shell command, then exit",
    )
    return parser.parse_args()


async def async_main() -> None:
    args = parse_args()
    shell = BrowserShell(headless=not args.headful)
    if args.command:
        try:
            await shell.execute(args.command)
        finally:
            await shell.browser.stop()
    else:
        await shell.run()


if __name__ == "__main__":
    asyncio.run(async_main())
