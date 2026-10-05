import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass

import aiohttp

from app.config import Settings

logger = logging.getLogger(__name__)
TEMPERATURES = (0.0, 0.3, 0.7, 1.0)


class LLMError(Exception):
    def __init__(self, kind: str):
        self.kind = kind
        super().__init__(kind)


@dataclass(frozen=True)
class Completion:
    text: str
    usage: dict[str, int]
    finish_reason: str


class LLMClient:
    def __init__(self, settings: Settings, session: aiohttp.ClientSession):
        self.settings = settings
        self.session = session

    async def complete(self, messages: list[dict[str, str]], temperature: float) -> Completion:
        if temperature not in TEMPERATURES:
            raise ValueError("Invalid temperature")
        request_id = uuid.uuid4().hex
        started = time.monotonic()
        status = "error"
        logger.info("llm_start request=%s model=%s", request_id, self.settings.llm_model)
        try:
            async with self.session.post(
                self.settings.llm_base_url + "/chat/completions",
                headers={"Authorization": "Bearer " + self.settings.llm_api_key},
                json={
                    "model": self.settings.llm_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_completion_tokens": self.settings.llm_max_output_tokens,
                    "stream": False,
                    **(
                        {"reasoning_effort": self.settings.llm_reasoning_effort}
                        if self.settings.llm_reasoning_effort
                        else {}
                    ),
                },
                timeout=aiohttp.ClientTimeout(total=self.settings.llm_timeout),
                allow_redirects=False,
            ) as response:
                if response.status != 200:
                    raise LLMError("quota" if response.status == 429 else "unavailable")
                # Never read or log an unbounded provider error body.
                raw = bytearray()
                async for chunk in response.content.iter_chunked(8192):
                    raw.extend(chunk)
                    if len(raw) > 262144:
                        raise LLMError("invalid")
                try:
                    data = json.loads(raw)
                    choice = data["choices"][0]
                    text = choice["message"]["content"]
                    reason = choice["finish_reason"]
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError
                    if reason not in {"stop", "length"}:
                        raise ValueError
                    # Telegram rejects NUL and lone surrogates.
                    text.encode("utf-8")
                    if "\x00" in text:
                        raise ValueError
                    usage = data.get("usage") or {}
                    if not isinstance(usage, dict):
                        raise ValueError
                    usage = {
                        key: count
                        for key, count in usage.items()
                        if key in {"prompt_tokens", "completion_tokens", "total_tokens"}
                        and type(count) is int
                        and count >= 0
                    }
                except (ValueError, KeyError, IndexError, TypeError):
                    raise LLMError("invalid") from None
                status = "success"
                logger.info(
                    "llm_usage request=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s",
                    request_id,
                    usage.get("prompt_tokens"),
                    usage.get("completion_tokens"),
                    usage.get("total_tokens"),
                )
                return Completion(text, usage, reason)
        except TimeoutError:
            status = "timeout"
            raise LLMError("timeout") from None
        except (aiohttp.ClientError, OSError):
            raise LLMError("unavailable") from None
        except LLMError as exc:
            status = exc.kind
            raise
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        finally:
            logger.info(
                "llm_end request=%s model=%s status=%s duration_ms=%d",
                request_id,
                self.settings.llm_model,
                status,
                (time.monotonic() - started) * 1000,
            )
