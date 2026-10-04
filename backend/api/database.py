import os

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5433/imka")

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


class Base(DeclarativeBase):
    pass


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


async def create_tables():
    from api import db_models  # noqa: F401 — registers models with Base

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # create_all never alters existing tables — add columns introduced after first deploy.
        await conn.execute(text("ALTER TABLE conversations ADD COLUMN IF NOT EXISTS persona VARCHAR(30)"))
        await conn.execute(text("ALTER TABLE messages ADD COLUMN IF NOT EXISTS topic VARCHAR(120)"))
