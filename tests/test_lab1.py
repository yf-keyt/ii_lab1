from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot, Dispatcher
from aiogram.methods import SendMessage
from aiogram.types import Chat, Message, Update, User

from app.config import ConfigError, Settings
from app.context import InputTooLong, Turn, build_context, estimate_tokens
from app.dialogue import Dialogue
from app.handlers.assistant import ERRORS, create_router, split_answer
from app.llm import Completion, LLMError
from app.prompts import PROMPTS
from app.storage import BusyError, Preferences

TOKEN = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk"


class MemorySession:
    def __init__(self):
        self.mode = "study"
        self.temperature = 0.3
        self.turns = []

    async def preferences(self):
        return Preferences(self.mode, self.temperature)

    async def history(self):
        return self.turns[:]

    async def set_mode(self, mode):
        self.mode = mode
        await self.reset()

    async def reset(self):
        self.turns.clear()

    async def set_temperature(self, temperature):
        self.temperature = temperature

    async def save_turn(self, text, answer):
        self.turns.append(Turn(text, answer))


class MemoryRepository:
    def __init__(self):
        self.users = {}
        self.busy = False

    @asynccontextmanager
    async def session(self, user_id):
        if self.busy:
            raise BusyError
        yield self.users.setdefault(user_id, MemorySession())


@pytest.fixture
async def harness():
    settings = Settings(bot_token=TOKEN, postgres_password="fake", llm_api_key="fake")
    client = AsyncMock()
    client.complete.return_value = Completion("Ответ", {}, "stop")
    repository = MemoryRepository()
    dialogue = Dialogue(client, settings)
    bot = Bot(TOKEN)
    bot.session = AsyncMock()
    dispatcher = Dispatcher()
    dispatcher.include_router(create_router(repository, dialogue))

    async def send(text, user_id=41, chat_type="private"):
        bot.session.reset_mock()
        await dispatcher.feed_update(
            bot,
            Update(
                update_id=1,
                message=Message(
                    message_id=1,
                    date=datetime.now(UTC),
                    chat=Chat(id=user_id, type=chat_type),
                    from_user=User(id=user_id, is_bot=False, first_name="Test"),
                    text=text,
                ),
            ),
        )
        return [
            call.args[1].text
            for call in bot.session.call_args_list
            if isinstance(call.args[1], SendMessage)
        ]

    yield send, repository, client, settings
    await bot.session.close()


async def test_messages_roles_parameters_and_user_isolation(harness):
    send, repository, client, _ = harness
    assert await send("Первый") == ["Ответ"]
    assert await send("Второй") == ["Ответ"]
    client.complete.assert_awaited_with(
        [
            {"role": "system", "content": PROMPTS["study"]},
            {"role": "user", "content": "Первый"},
            {"role": "assistant", "content": "Ответ"},
            {"role": "user", "content": "Второй"},
        ],
        0.3,
    )
    await send("Чужой", 42)
    assert client.complete.call_args.args[0] == [
        {"role": "system", "content": PROMPTS["study"]},
        {"role": "user", "content": "Чужой"},
    ]
    assert len(repository.users[41].turns) == 2


async def test_mode_reset_and_settings_preserve_other_user(harness):
    send, repository, client, settings = harness
    await send("Первая история")
    await send("Другая история", 42)
    assert await send("/settings 0.7") == [
        f"Режим: /study\nМодель: {settings.llm_model}\ntemperature: 0.7"
    ]
    calls = client.complete.await_count
    assert await send("/translate") == ["Режим: /translate. История очищена."]
    assert repository.users[41].turns == []
    assert repository.users[41].mode == "translate"
    assert client.complete.await_count == calls
    await send("На английский: привет")
    assert client.complete.call_args.args[0][0]["content"] == PROMPTS["translate"]
    assert client.complete.call_args.args[1] == 0.7
    assert await send("/reset") == ["История очищена. Режим и temperature сохранены."]
    assert repository.users[41].turns == []
    assert await repository.users[41].preferences() == Preferences("translate", 0.7)
    assert repository.users[42].turns == [Turn("Другая история", "Ответ")]


@pytest.mark.parametrize("argument", ["nan", "0.5", "inf", "0.7 extra", "1", "-1"])
async def test_settings_rejects_bad_values_before_api(harness, argument):
    send, repository, client, _ = harness
    assert await send("/settings " + argument) == [
        "Допустимые значения: /settings 0.0, 0.3, 0.7 или 1.0."
    ]
    client.complete.assert_not_awaited()
    assert not repository.users


@pytest.mark.parametrize("value", ["0.0", "0.3", "0.7", "1.0"])
async def test_settings_accepts_all_required_values(harness, value):
    send, repository, _, _ = harness
    await send("/settings " + value)
    assert repository.users[41].temperature == float(value)


@pytest.mark.parametrize("kind", ["timeout", "quota", "invalid", "unavailable"])
async def test_llm_error_is_safe_and_does_not_create_history(harness, kind):
    send, repository, client, _ = harness
    client.complete.side_effect = LLMError(kind)
    assert await send("Запрос") == [ERRORS[kind]]
    assert repository.users[41].turns == []


async def test_long_response_preserved_in_order_without_markup(harness):
    send, _, client, _ = harness
    answer = "<b>текст</b>" + "\U0001f600" * 5000
    client.complete.return_value = Completion(answer, {}, "stop")
    parts = await send("Длинный ответ")
    assert "".join(parts) == answer
    assert all(0 < len(part.encode("utf-16-le")) // 2 <= 4096 for part in parts)


async def test_private_only_commands_unknown_and_busy(harness):
    send, repository, client, _ = harness
    assert await send("Запрос", chat_type="group") == []
    assert await send("/unexpected") == ["Неизвестная команда. Список команд: /start."]
    assert "/review" in (await send("/start"))[0]
    repository.busy = True
    assert await send("/reset") == ["Предыдущий запрос ещё обрабатывается. Повторите позже."]
    client.complete.assert_not_awaited()


async def test_review_and_truncated_response(harness):
    send, _, client, _ = harness
    await send("/review")
    client.complete.return_value = Completion("Часть", {}, "length")
    assert await send("Код") == [
        "Часть",
        "Ответ достиг лимита генерации. Можно попросить продолжить.",
    ]
    assert client.complete.call_args.args[0][0]["content"] == PROMPTS["review"]


def test_context_trims_complete_oldest_turns_and_counts_prompt():
    history = [Turn("old", "old answer"), Turn("new", "new answer")]
    expected = [
        {"role": "system", "content": "prompt"},
        {"role": "user", "content": "new"},
        {"role": "assistant", "content": "new answer"},
        {"role": "user", "content": "current"},
    ]
    budget = sum(estimate_tokens(item["content"]) for item in expected)
    assert build_context("prompt", history, "current", 20, budget) == expected
    assert build_context("prompt", history, "current", 2, 10000) == expected
    assert build_context("prompt", history, "current", 0, 10000) == [expected[0], expected[-1]]
    with pytest.raises(InputTooLong):
        build_context("prompt", history, "current", 20, 1)


async def test_oversized_input_never_calls_model(harness):
    send, repository, client, _ = harness
    assert await send("я" * 4096) == ["Сообщение слишком длинное для контекста. Сократите его."]
    client.complete.assert_not_awaited()
    assert repository.users[41].turns == []


def test_split_boundary_exact_and_empty():
    assert split_answer("x" * 4096) == ["x" * 4096]
    assert split_answer("x" * 4097) == ["x" * 4096, "x"]
    assert split_answer("") == []


def test_llm_config_required_validated_and_hidden(tmp_path):
    values = {"BOT_TOKEN": TOKEN, "POSTGRES_PASSWORD": "fake"}
    with pytest.raises(ConfigError, match="LLM_API_KEY"):
        Settings.load(tmp_path / "absent", environ=values)
    values["LLM_API_KEY"] = "hidden-api-key"
    settings = Settings.load(tmp_path / "absent", environ=values)
    assert "hidden-api-key" not in repr(settings)
    for key, bad in [
        ("LLM_TIMEOUT", "0"),
        ("HISTORY_MAX_TOKENS", "0"),
        ("LLM_BASE_URL", "https://user:secret@host/v1"),
        ("LLM_BASE_URL", "http://host/v1"),
        ("LLM_MODEL", "bad\nmodel"),
    ]:
        with pytest.raises(ConfigError, match=key):
            Settings.load(tmp_path / "absent", environ={**values, key: bad})
    import logging

    from app.logging_setup import SecretFilter

    record = logging.LogRecord("test", logging.ERROR, "", 1, "hidden-api-key", (), None)
    SecretFilter([settings.llm_api_key]).filter(record)
    assert record.getMessage() == "[скрыто]"
