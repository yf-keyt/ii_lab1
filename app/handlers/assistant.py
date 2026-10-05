import logging

import asyncpg
from aiogram import F, Router
from aiogram.enums import ChatAction
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import Message
from aiogram.utils.chat_action import ChatActionSender

from app.context import InputTooLong
from app.dialogue import Dialogue
from app.llm import LLMError
from app.storage import BusyError, Repository

logger = logging.getLogger(__name__)
START_TEXT = (
    "AI-ассистент студента.\n"
    "/study — вопросы по программированию\n"
    "/translate — перевод на указанный язык\n"
    "/review — проверка кода\n"
    "/settings — настройки; /settings 0.3 — изменить temperature\n"
    "/reset — очистить историю\n"
    "Сообщения сохраняются в базе и передаются Groq для ответа. Не отправляйте секреты."
)
ERRORS = {
    "timeout": "Модель не ответила вовремя. Попробуйте позже.",
    "quota": "Лимит запросов к модели исчерпан. Попробуйте позже.",
    "invalid": "Модель вернула некорректный ответ. Попробуйте ещё раз.",
    "unavailable": "Сервис модели недоступен. Попробуйте позже.",
}


def split_answer(text: str, limit: int = 4096) -> list[str]:
    if limit < 2:
        raise ValueError("Message limit must be at least 2")
    parts = []
    start = units = 0
    # UTF-16 units also respect Telegram's handling of astral Unicode characters.
    for index, char in enumerate(text):
        width = 2 if ord(char) > 0xFFFF else 1
        if units + width > limit:
            parts.append(text[start:index])
            start, units = index, 0
        units += width
    if start < len(text):
        parts.append(text[start:])
    return parts


def create_router(repository: Repository, dialogue: Dialogue) -> Router:
    router = Router(name="assistant")
    router.message.filter(F.chat.type == "private", F.from_user, ~F.from_user.is_bot)

    async def handle(message: Message, action, *, typing: bool = False) -> None:
        try:
            async with repository.session(message.from_user.id) as session:
                if typing:
                    async with ChatActionSender(
                        bot=message.bot, chat_id=message.chat.id, action=ChatAction.TYPING
                    ):
                        await action(session)
                else:
                    await action(session)
        except BusyError:
            await message.answer("Предыдущий запрос ещё обрабатывается. Повторите позже.")
        except InputTooLong:
            await message.answer("Сообщение слишком длинное для контекста. Сократите его.")
        except LLMError as exc:
            await message.answer(ERRORS[exc.kind])
        except (asyncpg.PostgresError, asyncpg.InterfaceError, OSError, TimeoutError):
            logger.error("storage_operation_failed")
            await message.answer("Не удалось обработать запрос. Попробуйте позже.")

    @router.message(CommandStart())
    async def start(message: Message) -> None:
        await message.answer(START_TEXT, parse_mode=None)

    @router.message(Command("study", "translate", "review"))
    async def mode(message: Message, command: CommandObject) -> None:
        async def action(session):
            await session.set_mode(command.command)
            await message.answer(f"Режим: /{command.command}. История очищена.", parse_mode=None)

        await handle(message, action)

    @router.message(Command("reset"))
    async def reset(message: Message) -> None:
        async def action(session):
            await session.reset()
            await message.answer("История очищена. Режим и temperature сохранены.")

        await handle(message, action)

    @router.message(Command("settings"))
    async def settings(message: Message, command: CommandObject) -> None:
        choices = {"0.0": 0.0, "0.3": 0.3, "0.7": 0.7, "1.0": 1.0}
        argument = command.args.strip() if command.args is not None else None
        if argument is not None and argument not in choices:
            await message.answer("Допустимые значения: /settings 0.0, 0.3, 0.7 или 1.0.")
            return

        async def action(session):
            if argument is not None:
                await session.set_temperature(choices[argument])
            preferences = await session.preferences()
            await message.answer(
                f"Режим: /{preferences.mode}\nМодель: {dialogue.settings.llm_model}\n"
                f"temperature: {preferences.temperature:.1f}",
                parse_mode=None,
            )

        await handle(message, action)

    @router.message(F.text.startswith("/"))
    async def unknown(message: Message) -> None:
        await message.answer("Неизвестная команда. Список команд: /start.")

    @router.message(F.text)
    async def text(message: Message) -> None:
        if not message.text.strip():
            await message.answer("Отправьте непустое текстовое сообщение.")
            return

        async def action(session):
            answer = await dialogue.respond(session, message.text)
            for part in split_answer(answer.text):
                await message.answer(part, parse_mode=None)
            if answer.finish_reason == "length":
                await message.answer("Ответ достиг лимита генерации. Можно попросить продолжить.")

        await handle(message, action, typing=True)

    return router
