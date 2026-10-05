from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import asyncpg

from app.config import Settings
from app.context import Turn, estimate_tokens
from app.llm import TEMPERATURES
from app.prompts import PROMPTS


class BusyError(Exception):
    pass


@dataclass(frozen=True)
class Preferences:
    mode: str
    temperature: float


class UserSession:
    def __init__(self, connection: asyncpg.Connection, user_id: int, settings: Settings):
        self.connection = connection
        self.user_id = user_id
        self.settings = settings

    async def preferences(self) -> Preferences:
        row = await self.connection.fetchrow(
            "SELECT mode, temperature FROM assistant_users WHERE user_id=$1", self.user_id
        )
        return Preferences(row["mode"], float(row["temperature"]))

    async def history(self) -> list[Turn]:
        rows = await self.connection.fetch(
            "SELECT user_text, assistant_text FROM assistant_turns "
            "WHERE user_id=$1 ORDER BY id DESC LIMIT $2",
            self.user_id,
            self.settings.history_max_messages // 2,
        )
        return [Turn(row["user_text"], row["assistant_text"]) for row in reversed(rows)]

    async def reset(self) -> None:
        await self.connection.execute("DELETE FROM assistant_turns WHERE user_id=$1", self.user_id)

    async def set_mode(self, mode: str) -> None:
        if mode not in PROMPTS:
            raise ValueError("Invalid mode")
        async with self.connection.transaction():
            await self.connection.execute(
                "UPDATE assistant_users SET mode=$2 WHERE user_id=$1", self.user_id, mode
            )
            await self.reset()

    async def set_temperature(self, temperature: float) -> None:
        if temperature not in TEMPERATURES:
            raise ValueError("Invalid temperature")
        await self.connection.execute(
            "UPDATE assistant_users SET temperature=$2 WHERE user_id=$1",
            self.user_id,
            temperature,
        )

    async def save_turn(self, text: str, answer: str) -> None:
        async with self.connection.transaction():
            await self.connection.execute(
                "INSERT INTO assistant_turns (user_id, user_text, assistant_text) "
                "VALUES ($1, $2, $3)",
                self.user_id,
                text,
                answer,
            )
            rows = await self.connection.fetch(
                "SELECT id, user_text, assistant_text FROM assistant_turns "
                "WHERE user_id=$1 ORDER BY id DESC",
                self.user_id,
            )
            keep = []
            remaining = self.settings.history_max_tokens
            for row in rows:
                cost = estimate_tokens(row["user_text"]) + estimate_tokens(row["assistant_text"])
                if len(keep) * 2 + 2 > self.settings.history_max_messages or cost > remaining:
                    break
                keep.append(row["id"])
                remaining -= cost
            await self.connection.execute(
                "DELETE FROM assistant_turns WHERE user_id=$1 AND NOT (id=ANY($2::bigint[]))",
                self.user_id,
                keep,
            )


class Repository:
    def __init__(self, pool: asyncpg.Pool, settings: Settings):
        self.pool = pool
        self.settings = settings

    async def initialize(self) -> None:
        schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
        async with self.pool.acquire(timeout=5) as connection, connection.transaction():
            await connection.execute("SELECT pg_advisory_xact_lock(-901201)")
            await connection.execute(schema)
            versions = await connection.fetch("SELECT version FROM lab_schema_version")
            if {row["version"] for row in versions} != {1}:
                raise RuntimeError("Unsupported database schema version")

    @asynccontextmanager
    async def session(self, user_id: int):
        if not 0 < user_id < 2**63:
            raise ValueError("Invalid user identifier")
        try:
            connection = await self.pool.acquire(timeout=2)
        except TimeoutError:
            raise BusyError from None
        try:
            # Session lock spans the LLM call without holding a SQL transaction.
            locked = await connection.fetchval("SELECT pg_try_advisory_lock($1)", user_id)
            if not locked:
                raise BusyError
            await connection.execute(
                "INSERT INTO assistant_users (user_id) VALUES ($1) ON CONFLICT DO NOTHING",
                user_id,
            )
            yield UserSession(connection, user_id, self.settings)
        finally:
            # asyncpg resets session locks on release, including cancellation/error paths.
            await self.pool.release(connection)
