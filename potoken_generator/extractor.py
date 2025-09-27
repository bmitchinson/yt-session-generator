import asyncio
import dataclasses
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp
from typing import Optional

import nodriver

logger = logging.getLogger("extractor")


@dataclass
class TokenInfo:
    updated: int
    potoken: str
    visitor_data: str

    def to_json(self) -> str:
        as_dict = dataclasses.asdict(self)
        as_json = json.dumps(as_dict)
        return as_json


class PotokenExtractor:
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        update_interval: float = 3600,
        browser_path: Optional[Path] = None,
    ) -> None:
        self.update_interval: float = update_interval
        self.browser_path: Optional[Path] = browser_path
        self.profile_path = mkdtemp()  # cleaned up on exit by nodriver
        self._loop = loop
        self._token_info: Optional[TokenInfo] = None
        self._ongoing_update: asyncio.Lock = asyncio.Lock()
        self._extraction_done: asyncio.Event = asyncio.Event()
        self._update_requested: asyncio.Event = asyncio.Event()

    def get(self) -> Optional[TokenInfo]:
        return self._token_info

    async def run_once(self) -> Optional[TokenInfo]:
        await self._update()
        return self.get()

    async def run(self) -> None:
        await self._update()
        while True:
            try:
                await asyncio.wait_for(
                    self._update_requested.wait(), timeout=self.update_interval
                )
                logger.debug("initiating force update")
            except asyncio.TimeoutError:
                logger.debug("initiating scheduled update")
            await self._update()
            self._update_requested.clear()

    def request_update(self) -> bool:
        """Request immediate update, return False if update request is already set"""
        if self._ongoing_update.locked():
            logger.debug("update process is already running")
            return False
        if self._update_requested.is_set():
            logger.debug("force update has already been requested")
            return False
        self._loop.call_soon_threadsafe(self._update_requested.set)
        logger.debug("force update requested")
        return True

    @staticmethod
    def _extract_token(request: nodriver.cdp.network.Request) -> Optional[TokenInfo]:
        post_data = request.post_data
        logger.info("fork logC: post_data: %s", post_data)
        logger.info("fork logD: post_data: %r", post_data)
        if not post_data:
            logger.debug(
                "matched /youtubei/v1/player but request has no post_data; url=%s",
                getattr(request, "url", "<unknown>"),
            )
            return None
        try:
            post_data_json = json.loads(post_data)
            logger.info("fork logA: post_data_json: %s", post_data_json)
            logger.info("fork logB: post_data_json: %r", post_data_json)
            visitor_data = post_data_json["context"]["client"]["visitorData"]
            potoken = post_data_json["serviceIntegrityDimensions"]["poToken"]
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.warning(f"failed to extract token from request: {type(e)}, {e}")
            return None
        token_info = TokenInfo(
            updated=int(time.time()), potoken=potoken, visitor_data=visitor_data
        )
        return token_info

    async def _update(self) -> None:
        try:
            await asyncio.wait_for(self._perform_update(), timeout=600)
        except asyncio.TimeoutError:
            logger.error(
                "update failed: hard limit timeout exceeded. Browser might be failing to start properly"
            )

    async def _perform_update(self) -> None:
        if self._ongoing_update.locked():
            logger.debug("update is already in progress")
            return

        async with self._ongoing_update:
            logger.info("update started")
            self._extraction_done.clear()
            try:
                logger.debug(
                    "starting browser with profile_path=%s, browser_path=%s",
                    self.profile_path,
                    self.browser_path,
                )
                browser = await nodriver.start(
                    headless=False,
                    browser_executable_path=self.browser_path,
                    user_data_dir=self.profile_path,
                )
            except FileNotFoundError as e:
                msg = "could not find Chromium. Make sure it's installed or provide direct path to the executable"
                raise FileNotFoundError(msg) from e
            tab = browser.main_tab
            tab.add_handler(
                nodriver.cdp.network.RequestWillBeSent, self._general_request_logger
            )
            tab.add_handler(nodriver.cdp.network.RequestWillBeSent, self._send_handler)
            logger.debug(
                "navigating to %s", "https://www.youtube.com/embed/jNQXAC9IVRw"
            )
            await tab.get("https://www.youtube.com/embed/jNQXAC9IVRw")
            logger.debug("navigation finished")
            player_clicked = await self._click_on_player(tab)
            if player_clicked:
                await self._wait_for_handler()
            else:
                logger.debug(
                    "skipping wait: player was not clicked (element not found)"
                )
            await tab.close()
            browser.stop()

    @staticmethod
    async def _click_on_player(tab: nodriver.Tab) -> bool:
        try:
            player = await tab.select("#movie_player", 10)
        except asyncio.TimeoutError:
            logger.warning("update failed: unable to locate video player on the page")
            return False
        else:
            logger.debug(
                "video player element located; attempting to click to trigger network requests"
            )
            await player.click()
            return True

    async def _wait_for_handler(self) -> bool:
        try:
            timeout_s = 60
            logger.debug(
                "waiting up to %ss for outgoing POST /youtubei/v1/player request",
                timeout_s,
            )
            await asyncio.wait_for(self._extraction_done.wait(), timeout=timeout_s)
        except asyncio.TimeoutError:
            logger.warning(
                "update failed: timeout waiting for outgoing API request (did not observe POST /youtubei/v1/player)"
            )
            return False
        else:
            logger.info("update was succeessful")
            return True

    async def _send_handler(
        self, event: nodriver.cdp.network.RequestWillBeSent
    ) -> None:
        req = event.request
        if not req.method == "POST":
            logger.debug(
                "ignoring request (method=%s) url=%s",
                getattr(req, "method", None),
                getattr(req, "url", None),
            )
            return
        if "/youtubei/v1/player" not in req.url:
            logger.debug(
                "ignoring POST to different endpoint: %s", getattr(req, "url", None)
            )
            return
        token_info = self._extract_token(req)
        if token_info is None:
            logger.debug("matched endpoint but failed to extract token from request")
            return
        logger.info(f"new token: {token_info.to_json()}")
        self._token_info = token_info
        self._extraction_done.set()

    async def _general_request_logger(
        self, event: nodriver.cdp.network.RequestWillBeSent
    ) -> None:
        req = event.request
        url = getattr(req, "url", None)
        method = getattr(req, "method", None)
        has_body = bool(getattr(req, "post_data", None))
        body_len = (
            len(req.post_data)
            if hasattr(req, "post_data") and isinstance(req.post_data, str)
            else None
        )
        logger.debug(
            "network request: method=%s url=%s has_body=%s body_len=%s",
            method,
            url,
            has_body,
            body_len,
        )
