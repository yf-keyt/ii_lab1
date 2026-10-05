import argparse
import asyncio
import logging

from aiogram import Dispatcher
from aiohttp import ClientSession

from app.config import ConfigError, Settings
from app.db import create_pool
from app.dialogue import Dialogue
from app.handlers.assistant import create_router
from app.health import HealthState, start_health_server
from app.lifecycle import TaskTracker
from app.llm import LLMClient
from app.logging_setup import configure_logging
from app.storage import Repository
from app.telegram import create_bot

logger = logging.getLogger("app")


async def run(settings: Settings) -> None:
    state = HealthState()
    bot = create_bot(settings)
    runner = None
    llm_session = None
    tracker = TaskTracker()
    try:
        state.pool = await create_pool(settings)
        logger.info("PostgreSQL подключён: SELECT 1 выполнен.")
        # Начальная проверка токена и маршрута через прокси ограничена по времени.
        async with asyncio.timeout(30):
            me = await bot.get_me()
            webhook = await bot.get_webhook_info()
        if webhook.url:
            raise ConfigError(
                "У бота установлен webhook. Удалите его перед запуском polling "
                "или используйте отдельного учебного бота."
            )
        logger.info("Telegram доступен. Бот @%s запускает polling.", me.username)
        repository = Repository(state.pool, settings)
        await repository.initialize()
        llm_session = ClientSession()
        dialogue = Dialogue(LLMClient(settings, llm_session), settings)
        dispatcher = Dispatcher()
        dispatcher.update.outer_middleware(tracker)
        dispatcher.include_router(create_router(repository, dialogue))
        runner = await start_health_server(state, settings.health_port)
        state.polling_task = asyncio.create_task(
            dispatcher.start_polling(
                bot,
                db=state.pool,
                allowed_updates=dispatcher.resolve_used_update_types(),
                close_bot_session=False,
                tasks_concurrency_limit=32,
            )
        )
        state.initialized = True
        await state.polling_task
    finally:
        state.initialized = False
        if state.polling_task and not state.polling_task.done():
            state.polling_task.cancel()
            await asyncio.gather(state.polling_task, return_exceptions=True)
        await tracker.close()
        if runner:
            await runner.cleanup()
        await bot.session.close()
        if llm_session:
            await llm_session.close()
        if state.pool:
            try:
                async with asyncio.timeout(10):
                    await state.pool.close()
            except TimeoutError:
                state.pool.terminate()


def main() -> int:
    parser = argparse.ArgumentParser(description="AI-ассистент студента")
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()
    try:
        settings = Settings.load(args.env_file)
    except ConfigError as exc:
        print(str(exc))
        return 1
    configure_logging(settings)
    try:
        asyncio.run(run(settings))
    except KeyboardInterrupt:
        logger.info("Бот остановлен.")
    except Exception:
        logger.exception("Не удалось запустить бот. Проверьте БД, токен и прокси.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
