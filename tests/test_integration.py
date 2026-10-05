"""RUN_INTEGRATION=1 python -m pytest -m integration -v.

Тест создаёт собственные контейнер и volume; пользовательскую БД не использует.
"""

import asyncio
import json
import os
import secrets
import socket
import time
from dataclasses import replace

import asyncpg
import pytest

from app.config import Settings
from app.db import create_pool
from app.health import HealthState, health_result
from scripts.common import CommandError, run_command

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1",
        reason="Включите RUN_INTEGRATION=1 для Docker-тестов",
    ),
]


@pytest.fixture
def database():
    name = "itmo-test-" + secrets.token_hex(6)
    volume = name + "-data"
    password = secrets.token_urlsafe(24)

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    def docker(*args):
        return run_command(["docker", *args], label="Интеграционный Docker-тест", timeout=180)

    def wait_until_ready():
        deadline = time.monotonic() + 60
        while True:
            try:
                docker("exec", name, "pg_isready", "-h", "127.0.0.1", "-U", "bot", "-d", "bot")
                return
            except CommandError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(1)

    docker("info")
    docker("volume", "create", volume)
    try:
        docker(
            "run",
            "-d",
            "--name",
            name,
            "-p",
            f"127.0.0.1:{port}:5432",
            "-e",
            "POSTGRES_DB=bot",
            "-e",
            "POSTGRES_USER=bot",
            "-e",
            f"POSTGRES_PASSWORD={password}",
            "--mount",
            f"type=volume,source={volume},target=/var/lib/postgresql/data",
            "postgres:16-bookworm",
        )
        info = json.loads(docker("inspect", name))[0]
        assert int(info["NetworkSettings"]["Ports"]["5432/tcp"][0]["HostPort"]) == port
        wait_until_ready()
        yield (
            Settings(
                bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
                postgres_password=password,
                postgres_port=port,
            ),
            docker,
            wait_until_ready,
            name,
        )
    finally:
        try:
            docker("rm", "-f", name)
        finally:
            docker("volume", "rm", volume)


async def test_real_db_password_restart_health_and_persistence(database):
    # Arrange
    settings, docker, wait_until_ready, name = database
    pool = await create_pool(settings)
    task = asyncio.create_task(asyncio.Event().wait())
    state = HealthState(pool=pool, initialized=True, polling_task=task)
    try:
        await pool.execute("CREATE TABLE integration_probe (value text NOT NULL)")
        await pool.execute("INSERT INTO integration_probe VALUES ($1)", "Сохранилось 👋")
        # Act / Assert: реальное подключение и ошибка пароля.
        assert (await health_result(state))[0] == 200
        with pytest.raises(asyncpg.InvalidPasswordError):
            await create_pool(replace(settings, postgres_password="wrong-password"))
        # Act / Assert: потеря соединения и сохранность данных после запуска.
        await asyncio.to_thread(docker, "stop", name)
        assert (await health_result(state))[0] == 503
        await asyncio.to_thread(docker, "start", name)
        await asyncio.to_thread(wait_until_ready)
        for _ in range(60):
            if (await health_result(state))[0] == 200:
                break
            await asyncio.sleep(1)
        assert (await health_result(state))[0] == 200
        assert await pool.fetchval("SELECT value FROM integration_probe") == "Сохранилось 👋"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await pool.close()


async def test_lab1_storage_persistence_isolation_locks_and_retention(database):
    from app.context import Turn
    from app.storage import BusyError, Preferences, Repository

    settings, _, _, _ = database
    pool = await create_pool(settings)
    repository = Repository(pool, settings)
    try:
        await repository.initialize()
        await repository.initialize()
        async with repository.session(41) as user:
            await user.save_turn("Вопрос", "Ответ")
            await user.set_temperature(0.7)
            with pytest.raises(BusyError):
                async with repository.session(41):
                    pytest.fail("Concurrent user session must be rejected")
            async with repository.session(42) as other:
                assert await other.history() == []
                await other.save_turn("Другой", "Другой ответ")
        # Recreate the pool to prove state is stored in PostgreSQL, not process memory.
        await pool.close()
        pool = await create_pool(settings)
        repository = Repository(pool, settings)
        async with repository.session(41) as user:
            assert await user.preferences() == Preferences("study", 0.7)
            assert await user.history() == [Turn("Вопрос", "Ответ")]
            await user.set_mode("review")
            assert await user.history() == []
            assert await user.preferences() == Preferences("review", 0.7)
            await user.save_turn("Код", "Проверка")
            await user.reset()
            assert await user.preferences() == Preferences("review", 0.7)
        async with repository.session(42) as other:
            assert await other.history() == [Turn("Другой", "Другой ответ")]
        bounded = Repository(pool, replace(settings, history_max_messages=2))
        async with bounded.session(41) as user:
            await user.save_turn("Первый", "1")
            await user.save_turn("Второй", "2")
            assert await user.history() == [Turn("Второй", "2")]
        with pytest.raises(RuntimeError):
            async with repository.session(41):
                raise RuntimeError("Simulated failure")
        async with repository.session(41) as user:
            assert await user.history() == [Turn("Второй", "2")]
    finally:
        await pool.close()
