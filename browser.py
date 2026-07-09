"""Reusable single-window Zendriver browser controls."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


DEFAULT_PROFILE_DIR = Path.home() / ".webllm2api-zendriver-profile"


class BrowserError(RuntimeError):
    """Raised when a browser operation cannot be completed."""


class BrowserController:
    """Own one Zendriver browser process, window, profile, and active page."""

    def __init__(
        self,
        profile_dir: str | os.PathLike[str] | None = None,
        *,
        headless: bool = True,
    ) -> None:
        configured_profile = os.environ.get("ZENDRIVER_PROFILE_DIR")
        self.profile_dir = Path(
            profile_dir or configured_profile or DEFAULT_PROFILE_DIR
        ).expanduser().resolve()
        self.default_headless = headless
        self.browser: Any | None = None
        self.page: Any | None = None
        self._headless: bool | None = None
        self._lock = asyncio.Lock()

    @staticmethod
    def _zendriver() -> Any:
        try:
            import zendriver as zd
        except ImportError as exc:
            raise BrowserError(
                "Zendriver is not installed. Run: python -m pip install "
                "'zendriver>=0.15.4'"
            ) from exc
        return zd

    @property
    def running(self) -> bool:
        return bool(
            self.browser is not None
            and not getattr(self.browser, "stopped", True)
            and self.page is not None
        )

    async def start(self, *, headless: bool | None = None) -> Any:
        """Start or reuse the sole browser and return its sole page."""
        requested_headless = (
            self.default_headless if headless is None else headless
        )
        async with self._lock:
            if self.running:
                if self._headless == requested_headless:
                    return self.page
                await self._stop_unlocked()

            self.profile_dir.mkdir(parents=True, exist_ok=True)
            zd = self._zendriver()
            try:
                self.browser = await zd.start(
                    user_data_dir=str(self.profile_dir),
                    headless=requested_headless,
                    browser_args=[
                        "--no-first-run",
                        "--disable-session-crashed-bubble",
                        "--window-size=1440,1000",
                    ],
                )
            except Exception as exc:
                self.browser = None
                raise BrowserError(
                    "Could not start Zendriver. Another process may be using "
                    f"profile {self.profile_dir}."
                ) from exc

            self._headless = requested_headless
            tabs = list(self.browser.tabs)
            if not tabs:
                self.page = await self.browser.get("about:blank")
            else:
                self.page = tabs[0]
                for extra_page in tabs[1:]:
                    try:
                        await extra_page.close()
                    except Exception:
                        pass
            return self.page

    async def stop(self) -> None:
        """Stop browser process without deleting persistent profile."""
        async with self._lock:
            await self._stop_unlocked()

    async def _stop_unlocked(self) -> None:
        browser, self.browser = self.browser, None
        self.page = None
        self._headless = None
        if browser is not None and not getattr(browser, "stopped", True):
            try:
                await browser.stop()
            except Exception:
                pass

    async def auth(self, url: str = "https://chatgpt.com/") -> None:
        """
        Open same persistent profile headful and wait for manual window close.

        Next command restarts headless and reuses cookies/local storage.
        """
        await self.stop()
        await self.start(headless=False)
        await self.goto(url)
        print(
            "Auth window open. Log in manually, then close whole browser "
            "window to save session."
        )
        try:
            while self.browser is not None and not getattr(
                self.browser, "stopped", True
            ):
                await asyncio.sleep(0.25)
        finally:
            self.browser = None
            self.page = None
            self._headless = None

    async def ensure_page(self) -> Any:
        if not self.running:
            return await self.start()
        return self.page

    @staticmethod
    def normalize_url(url: str) -> str:
        url = url.strip()
        if not url:
            raise BrowserError("URL is required.")
        if not urlparse(url).scheme:
            url = "https://" + url
        return url

    async def goto(self, url: str) -> str:
        url = self.normalize_url(url)
        await self.ensure_page()
        try:
            self.page = await self.browser.get(url)
        except asyncio.TimeoutError:
            # Navigation may have succeeded despite a missed load event.
            current = await self.url()
            if current != url:
                raise BrowserError(f"Navigation timed out: {url}")
        return await self.url()

    async def back(self) -> str:
        await self.evaluate("history.back(); true")
        await asyncio.sleep(0.4)
        return await self.url()

    async def forward(self) -> str:
        await self.evaluate("history.forward(); true")
        await asyncio.sleep(0.4)
        return await self.url()

    async def reload(self, *, ignore_cache: bool = False) -> str:
        await self.ensure_page()
        from zendriver import cdp

        await self.page.send(cdp.page.reload(ignore_cache=ignore_cache))
        await asyncio.sleep(0.4)
        return await self.url()

    async def url(self) -> str:
        return str(await self.evaluate("location.href"))

    async def title(self) -> str:
        return str(await self.evaluate("document.title"))

    async def evaluate(self, expression: str) -> Any:
        """Evaluate JavaScript in active page and return JSON-safe value."""
        await self.ensure_page()
        try:
            return await self.page.evaluate(expression, return_by_value=True)
        except Exception as exc:
            raise BrowserError(f"JavaScript failed: {exc}") from exc

    async def evaluate_json(self, expression: str) -> Any:
        """Evaluate expression wrapped in JSON.stringify."""
        raw = await self.evaluate(f"JSON.stringify(({expression}))")
        return json.loads(raw) if isinstance(raw, str) else raw

    async def wait_for(
        self,
        selector: str,
        *,
        timeout: float = 15.0,
        poll_interval: float = 0.1,
    ) -> Any:
        await self.ensure_page()
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            element = await self.page.query_selector(selector)
            if element is not None:
                return element
            if asyncio.get_running_loop().time() >= deadline:
                raise BrowserError(
                    f"Timed out waiting for selector: {selector}"
                )
            await asyncio.sleep(poll_interval)

    async def click(self, selector: str, *, timeout: float = 15.0) -> None:
        element = await self.wait_for(selector, timeout=timeout)
        await element.click()

    async def type_text(
        self,
        selector: str,
        text: str,
        *,
        clear: bool = False,
        timeout: float = 15.0,
    ) -> None:
        element = await self.wait_for(selector, timeout=timeout)
        if clear:
            try:
                await element.clear_input()
            except Exception:
                await element.apply(
                    """(el) => {
                        el.focus();
                        if (el.isContentEditable) el.innerHTML = "";
                        else el.value = "";
                        el.dispatchEvent(new InputEvent(
                            "input", {bubbles: true, inputType: "deleteContent"}
                        ));
                    }"""
                )
        await element.send_keys(text)

    async def text(self, selector: str, *, timeout: float = 15.0) -> str:
        element = await self.wait_for(selector, timeout=timeout)
        await element.update()
        return element.text_all

    async def html(self, selector: str = "html", *, timeout: float = 15.0) -> str:
        element = await self.wait_for(selector, timeout=timeout)
        return await element.get_html()

    async def screenshot(self, path: str | os.PathLike[str]) -> Path:
        await self.ensure_page()
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        await self.page.save_screenshot(str(target))
        return target

    async def close(self) -> None:
        """Alias used by callers that treat controller as a resource."""
        await self.stop()

    async def __aenter__(self) -> "BrowserController":
        await self.start()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.stop()
