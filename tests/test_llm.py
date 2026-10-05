import asyncio
from dataclasses import replace

import pytest
from aiohttp import ClientSession, web

from app.config import Settings
from app.llm import LLMClient, LLMError


@pytest.fixture
async def endpoint(unused_tcp_port):
    state = {
        "status": 200,
        "body": {
            "choices": [{"message": {"content": "Ответ"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
        },
        "delay": 0,
        "requests": [],
    }

    async def handler(request):
        state["requests"].append((request.headers.get("Authorization"), await request.json()))
        if state["delay"]:
            await asyncio.sleep(state["delay"])
        return web.json_response(state["body"], status=state["status"])

    app = web.Application()
    app.router.add_post("/v1/chat/completions", handler)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", unused_tcp_port).start()
    settings = Settings(
        bot_token="fake",
        postgres_password="fake",
        llm_api_key="fake-key",
        llm_base_url=f"http://127.0.0.1:{unused_tcp_port}/v1",
    )
    async with ClientSession() as session:
        yield state, LLMClient(settings, session)
    await runner.cleanup()


async def test_exact_http_payload_and_usage(endpoint):
    state, client = endpoint
    messages = [{"role": "system", "content": "prompt"}, {"role": "user", "content": "text"}]
    answer = await client.complete(messages, 0.0)
    assert answer.text == "Ответ"
    assert answer.usage == {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25}
    assert state["requests"] == [
        (
            "Bearer fake-key",
            {
                "model": "openai/gpt-oss-20b",
                "messages": messages,
                "temperature": 0.0,
                "max_completion_tokens": 2048,
                "reasoning_effort": "low",
                "stream": False,
            },
        )
    ]


@pytest.mark.parametrize(
    "status,kind",
    [(429, "quota"), (401, "unavailable"), (500, "unavailable"), (302, "unavailable")],
)
async def test_provider_errors_do_not_expose_body(endpoint, status, kind):
    state, client = endpoint
    state.update(status=status, body={"error": "fake-key secret dialogue"})
    with pytest.raises(LLMError) as exc:
        await client.complete([], 0.3)
    assert str(exc.value) == kind


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"choices": []},
        {"choices": [{"message": {"content": ""}, "finish_reason": "stop"}]},
        {"choices": [{"message": {"content": None}, "finish_reason": "stop"}]},
        {"choices": [{"message": {"content": "a"}, "finish_reason": "tool_calls"}]},
        {"choices": [{"message": {"content": "\x00"}, "finish_reason": "stop"}]},
    ],
)
async def test_invalid_success_response(endpoint, body):
    state, client = endpoint
    state["body"] = body
    with pytest.raises(LLMError, match="invalid"):
        await client.complete([], 0.3)


async def test_timeout_and_invalid_temperature(endpoint):
    state, client = endpoint
    state["delay"] = 0.1
    client.settings = replace(client.settings, llm_timeout=0.01)
    with pytest.raises(LLMError, match="timeout"):
        await client.complete([], 0.3)
    with pytest.raises(ValueError):
        await client.complete([], float("nan"))


async def test_response_size_bounded(endpoint):
    state, client = endpoint
    state["body"] = {"choices": [{"message": {"content": "x" * 300000}, "finish_reason": "stop"}]}
    with pytest.raises(LLMError, match="invalid"):
        await client.complete([], 0.3)
