"""
database.py — Prisma async client setup and lifecycle management.

Replaces the previous SQLAlchemy engine. The Prisma client reads DATABASE_URL
from the .env file and handles connection pooling automatically.
"""

import logging

from prisma import Prisma

logger = logging.getLogger(__name__)

# Singleton Prisma client
prisma = Prisma()


async def init_db() -> None:
    """Connect the Prisma client to the database."""
    await prisma.connect()
    logger.info("Prisma connected to database.")


async def close_db() -> None:
    """Disconnect the Prisma client gracefully."""
    await prisma.disconnect()
    logger.info("Prisma disconnected from database.")
