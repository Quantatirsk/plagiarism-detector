"""SQLModel engine/session helpers for the plagiarism detector."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession

from backend.core.config import get_settings

_engine: Optional[AsyncEngine] = None
_sessionmaker: Optional[sessionmaker] = None


def get_engine() -> AsyncEngine:
    """Return a singleton async engine configured from settings."""
    global _engine, _sessionmaker
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(settings.database_url, echo=False, future=True)
        _sessionmaker = sessionmaker(
            bind=_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    assert _engine is not None
    return _engine


def _get_sessionmaker() -> sessionmaker:
    if _sessionmaker is None:
        get_engine()
    assert _sessionmaker is not None
    return _sessionmaker


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Provide an async SQLAlchemy session."""
    session_factory = _get_sessionmaker()
    session: AsyncSession = session_factory()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def init_db() -> None:
    """Create database tables based on SQLModel metadata."""
    # Import models to ensure they are registered with metadata
    from backend.db import models  # noqa: F401

    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
