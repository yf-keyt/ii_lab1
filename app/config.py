"""Единственное место чтения и проверки настроек приложения."""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from aiogram.utils.token import TokenValidationError, validate_token
from dotenv import dotenv_values


class ConfigError(ValueError):
    """Ошибка настройки без секретных значений в сообщении."""


@dataclass(frozen=True)
class Settings:
    bot_token: str = field(repr=False)
    postgres_password: str = field(repr=False)
    telegram_proxy_url: str = field(default="", repr=False)
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432
    postgres_db: str = "bot"
    postgres_user: str = "bot"
    log_level: str = "INFO"
    health_port: int = 8080
    llm_api_key: str = field(default="", repr=False)
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_model: str = "openai/gpt-oss-20b"
    llm_reasoning_effort: str = "low"
    llm_timeout: int = 45
    llm_max_output_tokens: int = 2048
    history_max_messages: int = 20
    history_max_tokens: int = 5000

    @classmethod
    def load(
        cls,
        env_file: Path | str = ".env",
        *,
        environ: Mapping[str, str] | None = None,
        require_llm: bool = True,
    ) -> "Settings":
        values = {
            **dotenv_values(env_file, interpolate=False),
            **(os.environ if environ is None else environ),
        }

        def value(key: str, default: str = "") -> str:
            return values.get(key) or default

        def port(key: str, default: str) -> int:
            try:
                result = int(value(key, default))
                if not 1 <= result <= 65535:
                    raise ValueError
                return result
            except ValueError:
                raise ConfigError(f"{key}: нужен номер порта от 1 до 65535.") from None

        def integer(key: str, default: int, low: int, high: int) -> int:
            try:
                result = int(value(key, str(default)))
                if not low <= result <= high:
                    raise ValueError
                return result
            except ValueError:
                raise ConfigError(f"{key}: нужно целое число от {low} до {high}.") from None

        token = value("BOT_TOKEN")
        try:
            validate_token(token)
        except TokenValidationError:
            raise ConfigError("BOT_TOKEN: укажите токен, полученный у BotFather.") from None
        password = value("POSTGRES_PASSWORD")
        if not password:
            raise ConfigError("POSTGRES_PASSWORD: пароль базы данных не задан.")
        proxy = value("TELEGRAM_PROXY_URL")
        if proxy:
            try:
                parsed = urlsplit(proxy)
                if (
                    parsed.scheme not in {"http", "socks5"}
                    or not parsed.hostname
                    or not parsed.port
                    or parsed.path not in {"", "/"}
                    or parsed.query
                    or parsed.fragment
                ):
                    raise ValueError
            except ValueError:
                raise ConfigError(
                    "TELEGRAM_PROXY_URL: нужен http://host:port или socks5://host:port; "
                    "при необходимости добавьте user:password@."
                ) from None
        level = value("LOG_LEVEL", "INFO").upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ConfigError("LOG_LEVEL: используйте DEBUG, INFO, WARNING, ERROR или CRITICAL.")
        api_key = value("LLM_API_KEY")
        if require_llm and not api_key:
            raise ConfigError("LLM_API_KEY: укажите ключ Groq в локальном .env.")
        base_url = value("LLM_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
        try:
            parsed = urlsplit(base_url)
            if (
                parsed.scheme != "https"
                or not parsed.hostname
                or parsed.username
                or parsed.password
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError
            if parsed.port is not None and not 1 <= parsed.port <= 65535:
                raise ValueError
        except ValueError:
            raise ConfigError("LLM_BASE_URL: нужен HTTPS URL без реквизитов и query.") from None
        model = value("LLM_MODEL", "openai/gpt-oss-20b")
        effort = values.get("LLM_REASONING_EFFORT", "low") or ""
        if effort not in {"", "low", "medium", "high"}:
            raise ConfigError("LLM_REASONING_EFFORT: low, medium, high или пустое значение.")
        if len(model) > 128 or any(c.isspace() or ord(c) < 32 for c in model):
            raise ConfigError("LLM_MODEL: некорректное имя модели.")
        if any(ord(c) < 32 for c in api_key):
            raise ConfigError("LLM_API_KEY: ключ содержит недопустимые символы.")
        return cls(
            bot_token=token,
            postgres_password=password,
            telegram_proxy_url=proxy,
            postgres_host=value("POSTGRES_HOST", "127.0.0.1"),
            postgres_port=port("POSTGRES_PORT", "5432"),
            postgres_db=value("POSTGRES_DB", "bot"),
            postgres_user=value("POSTGRES_USER", "bot"),
            log_level=level,
            health_port=port("HEALTH_PORT", "8080"),
            llm_api_key=api_key,
            llm_base_url=base_url,
            llm_model=model,
            llm_reasoning_effort=effort,
            llm_timeout=integer("LLM_TIMEOUT", 45, 1, 120),
            llm_max_output_tokens=integer("LLM_MAX_OUTPUT_TOKENS", 2048, 64, 8192),
            history_max_messages=integer("HISTORY_MAX_MESSAGES", 20, 0, 100),
            history_max_tokens=integer("HISTORY_MAX_TOKENS", 5000, 2048, 65536),
        )
