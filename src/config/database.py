"""
Database connection management for FPL ML System.
"""

import logging
from typing import Generator
from sqlalchemy import create_engine, MetaData, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
from .settings import settings

logger = logging.getLogger(__name__)

# Database engine (singleton)
_engine = None
_SessionLocal = None

# Base class for ORM models
Base = declarative_base()


def get_engine():
    """Get database engine singleton."""
    global _engine
    if _engine is None:
        _engine = create_engine(
            settings.database_url,
            echo=settings.debug,
            pool_pre_ping=True,
            pool_recycle=300,
        )
        logger.info(f"Created database engine for: {settings.database_url}")
    return _engine


def get_session_factory():
    """Get session factory singleton.""" 
    global _SessionLocal
    if _SessionLocal is None:
        engine = get_engine()
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        logger.info("Created database session factory")
    return _SessionLocal


def get_db() -> Generator[Session, None, None]:
    """
    Get database session with proper cleanup.
    
    Usage:
        async def some_function():
            db = next(get_db())
            try:
                # Use db session
                result = db.query(Model).all()
                db.commit()
            except Exception:
                db.rollback()
                raise
            finally:
                db.close()
    """
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def create_tables():
    """Create all database tables."""
    try:
        engine = get_engine()
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")
        raise


def drop_tables():
    """Drop all database tables (use with caution)."""
    try:
        engine = get_engine()
        Base.metadata.drop_all(bind=engine)
        logger.info("Database tables dropped successfully")
    except Exception as e:
        logger.error(f"Failed to drop database tables: {e}")
        raise


def test_connection() -> bool:
    """
    Test database connection.
    
    Returns:
        True if connection is successful
    """
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection test successful")
        return True
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False


def get_database_info() -> dict:
    """
    Get database information.
    
    Returns:
        Dictionary with database info
    """
    try:
        engine = get_engine()
        with engine.connect() as conn:
            # Get database version info
            if "sqlite" in settings.database_url:
                result = conn.execute(text("SELECT sqlite_version()"))
                version = result.scalar()
                db_type = "SQLite"
            elif "postgresql" in settings.database_url:
                result = conn.execute(text("SELECT version()"))
                version = result.scalar()
                db_type = "PostgreSQL"
            else:
                version = "Unknown"
                db_type = "Unknown"
            
            return {
                "database_type": db_type,
                "database_version": version,
                "database_url": settings.database_url,
                "connection_pool_size": engine.pool.size(),
                "connection_pool_checked_out": engine.pool.checkedout(),
            }
    except Exception as e:
        logger.error(f"Failed to get database info: {e}")
        return {
            "database_type": "Unknown",
            "database_version": "Unknown", 
            "database_url": settings.database_url,
            "error": str(e),
        }


def initialize_database():
    """Initialize database with tables and basic data."""
    try:
        logger.info("Initializing database...")
        
        # Test connection first
        if not test_connection():
            raise Exception("Database connection failed")
        
        # Create tables
        create_tables()
        
        logger.info("Database initialization completed successfully")
        return True
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False