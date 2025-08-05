"""
Caching utilities for FPL ML System.
Redis-based caching with fallback to in-memory cache.
"""

import json
import pickle
import hashlib
import time
from typing import Any, Optional, Dict, Union, Callable
from datetime import datetime, timedelta
from functools import wraps
import asyncio
import os

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

from .logging import get_logger

logger = get_logger(__name__)


class InMemoryCache:
    """
    Simple in-memory cache with TTL support.
    Fallback when Redis is not available.
    """
    
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        self.max_size = 1000
        self.default_ttl = 3600  # 1 hour
    
    def _cleanup(self):
        """Remove expired entries."""
        current_time = time.time()
        expired_keys = []
        
        for key, data in self._cache.items():
            if current_time > data['expires_at']:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._cache.pop(key, None)
            self._access_times.pop(key, None)
    
    def _evict_lru(self):
        """Evict least recently used items if cache is full."""
        if len(self._cache) >= self.max_size:
            # Remove 10% of least recently used items
            num_to_remove = max(1, self.max_size // 10)
            lru_keys = sorted(self._access_times.items(), key=lambda x: x[1])[:num_to_remove]
            
            for key, _ in lru_keys:
                self._cache.pop(key, None)
                self._access_times.pop(key, None)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        self._cleanup()
        
        if key not in self._cache:
            return None
        
        data = self._cache[key]
        current_time = time.time()
        
        if current_time > data['expires_at']:
            self._cache.pop(key, None)
            self._access_times.pop(key, None)
            return None
        
        self._access_times[key] = current_time
        return data['value']
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        self._cleanup()
        self._evict_lru()
        
        ttl = ttl or self.default_ttl
        expires_at = time.time() + ttl
        
        self._cache[key] = {
            'value': value,
            'expires_at': expires_at,
            'created_at': time.time()
        }
        self._access_times[key] = time.time()
    
    def delete(self, key: str) -> None:
        """Delete key from cache."""
        self._cache.pop(key, None)
        self._access_times.pop(key, None)
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._access_times.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        current_time = time.time()
        valid_entries = sum(
            1 for data in self._cache.values()
            if current_time <= data['expires_at']
        )
        
        return {
            'total_entries': len(self._cache),
            'valid_entries': valid_entries,
            'expired_entries': len(self._cache) - valid_entries,
            'max_size': self.max_size,
            'utilization': len(self._cache) / self.max_size
        }


class RedisCache:
    """
    Redis-based cache implementation.
    """
    
    def __init__(self, redis_url: str = None, default_ttl: int = 3600):
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.default_ttl = default_ttl
        self._client = None
        self.connected = False
        
        if REDIS_AVAILABLE:
            self._connect()
    
    def _connect(self):
        """Connect to Redis."""
        try:
            self._client = redis.from_url(
                self.redis_url,
                decode_responses=False,  # We handle encoding ourselves
                socket_connect_timeout=5,
                socket_timeout=5
            )
            # Test connection
            self._client.ping()
            self.connected = True
            logger.info("Connected to Redis cache", redis_url=self.redis_url)
        except Exception as e:
            logger.warning("Failed to connect to Redis", error=str(e))
            self.connected = False
            self._client = None
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        try:
            # Try JSON first for simple types
            json.dumps(value)
            return json.dumps(value).encode('utf-8')
        except (TypeError, ValueError):
            # Fall back to pickle for complex objects
            return pickle.dumps(value)
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        try:
            # Try JSON first
            return json.loads(data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Fall back to pickle
            return pickle.loads(data)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache."""
        if not self.connected or not self._client:
            return None
        
        try:
            data = self._client.get(key)
            if data is None:
                return None
            
            return self._deserialize(data)
        except Exception as e:
            logger.warning("Failed to get from Redis cache", key=key, error=str(e))
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in Redis cache."""
        if not self.connected or not self._client:
            return
        
        try:
            ttl = ttl or self.default_ttl
            serialized = self._serialize(value)
            self._client.setex(key, ttl, serialized)
        except Exception as e:
            logger.warning("Failed to set in Redis cache", key=key, error=str(e))
    
    def delete(self, key: str) -> None:
        """Delete key from Redis cache."""
        if not self.connected or not self._client:
            return
        
        try:
            self._client.delete(key)
        except Exception as e:
            logger.warning("Failed to delete from Redis cache", key=key, error=str(e))
    
    def clear(self) -> None:
        """Clear all cache entries."""
        if not self.connected or not self._client:
            return
        
        try:
            self._client.flushdb()
        except Exception as e:
            logger.warning("Failed to clear Redis cache", error=str(e))
    
    def stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics."""
        if not self.connected or not self._client:
            return {'connected': False}
        
        try:
            info = self._client.info()
            return {
                'connected': True,
                'used_memory': info.get('used_memory', 0),
                'used_memory_human': info.get('used_memory_human', '0B'),
                'connected_clients': info.get('connected_clients', 0),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'keyspace_hits': info.get('keyspace_hits', 0),
                'keyspace_misses': info.get('keyspace_misses', 0),
                'hit_rate': info.get('keyspace_hits', 0) / max(1, info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0))
            }
        except Exception as e:
            logger.warning("Failed to get Redis stats", error=str(e))
            return {'connected': False, 'error': str(e)}


class CacheManager:
    """
    Unified cache manager with Redis primary and in-memory fallback.
    """
    
    def __init__(self, redis_url: str = None, default_ttl: int = 3600):
        self.default_ttl = default_ttl
        
        # Initialize caches
        self.redis_cache = RedisCache(redis_url, default_ttl) if REDIS_AVAILABLE else None
        self.memory_cache = InMemoryCache()
        self.memory_cache.default_ttl = default_ttl
        
        # Determine primary cache
        self.use_redis = self.redis_cache and self.redis_cache.connected
        
        logger.info(
            "Cache manager initialized",
            primary_cache="redis" if self.use_redis else "memory",
            redis_available=REDIS_AVAILABLE,
            redis_connected=self.use_redis
        )
    
    def _get_cache_key(self, key: str, prefix: str = "fpl") -> str:
        """Generate cache key with prefix."""
        return f"{prefix}:{key}"
    
    def get(self, key: str, prefix: str = "fpl") -> Optional[Any]:
        """Get value from cache."""
        cache_key = self._get_cache_key(key, prefix)
        
        # Try Redis first if available
        if self.use_redis:
            value = self.redis_cache.get(cache_key)
            if value is not None:
                return value
        
        # Fall back to memory cache
        return self.memory_cache.get(cache_key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, prefix: str = "fpl") -> None:
        """Set value in cache."""
        cache_key = self._get_cache_key(key, prefix)
        ttl = ttl or self.default_ttl
        
        # Set in both caches for redundancy
        if self.use_redis:
            self.redis_cache.set(cache_key, value, ttl)
        
        self.memory_cache.set(cache_key, value, ttl)
    
    def delete(self, key: str, prefix: str = "fpl") -> None:
        """Delete key from cache."""
        cache_key = self._get_cache_key(key, prefix)
        
        if self.use_redis:
            self.redis_cache.delete(cache_key)
        
        self.memory_cache.delete(cache_key)
    
    def clear(self, prefix: str = "fpl") -> None:
        """Clear all cache entries with prefix."""
        if self.use_redis:
            # For Redis, we need to find and delete keys with prefix
            try:
                if self.redis_cache.connected and self.redis_cache._client:
                    keys = self.redis_cache._client.keys(f"{prefix}:*")
                    if keys:
                        self.redis_cache._client.delete(*keys)
            except Exception as e:
                logger.warning("Failed to clear Redis cache with prefix", prefix=prefix, error=str(e))
        
        # For memory cache, we need to filter keys
        keys_to_delete = [
            key for key in self.memory_cache._cache.keys()
            if key.startswith(f"{prefix}:")
        ]
        for key in keys_to_delete:
            self.memory_cache.delete(key)
    
    def stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        stats = {
            'primary_cache': 'redis' if self.use_redis else 'memory',
            'memory_cache': self.memory_cache.stats()
        }
        
        if self.redis_cache:
            stats['redis_cache'] = self.redis_cache.stats()
        
        return stats


# Global cache manager instance
_cache_manager = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def cache_key(*args, **kwargs) -> str:
    """Generate cache key from arguments."""
    # Create a string representation of arguments
    key_parts = [str(arg) for arg in args]
    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
    key_string = "|".join(key_parts)
    
    # Hash for consistent key length
    return hashlib.md5(key_string.encode()).hexdigest()


def cached(ttl: int = 3600, prefix: str = "fpl", key_func: Optional[Callable] = None):
    """
    Decorator for caching function results.
    
    Args:
        ttl: Time to live in seconds
        prefix: Cache key prefix
        key_func: Custom function to generate cache key
    """
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_manager = get_cache_manager()
            
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key = f"{func.__module__}.{func.__name__}:{cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = cache_manager.get(key, prefix)
            if cached_result is not None:
                logger.debug("Cache hit", function=func.__name__, key=key[:32])
                return cached_result
            
            # Execute function and cache result
            logger.debug("Cache miss", function=func.__name__, key=key[:32])
            result = func(*args, **kwargs)
            
            # Only cache non-None results
            if result is not None:
                cache_manager.set(key, result, ttl, prefix)
            
            return result
        
        return wrapper
    return decorator


def async_cached(ttl: int = 3600, prefix: str = "fpl", key_func: Optional[Callable] = None):
    """
    Decorator for caching async function results.
    """
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_manager = get_cache_manager()
            
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = f"{func.__module__}.{func.__name__}:{cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = cache_manager.get(key, prefix)
            if cached_result is not None:
                logger.debug("Cache hit (async)", function=func.__name__, key=key[:32])
                return cached_result
            
            # Execute function and cache result
            logger.debug("Cache miss (async)", function=func.__name__, key=key[:32])
            result = await func(*args, **kwargs)
            
            # Only cache non-None results
            if result is not None:
                cache_manager.set(key, result, ttl, prefix)
            
            return result
        
        return wrapper
    return decorator


class CacheWarmer:
    """
    Utility for warming up cache with commonly accessed data.
    """
    
    def __init__(self, cache_manager: CacheManager = None):
        self.cache_manager = cache_manager or get_cache_manager()
        self.logger = get_logger(f"{__name__}.CacheWarmer")
    
    async def warm_bootstrap_data(self):
        """Warm cache with FPL bootstrap data."""
        from ..data.fetchers import FPLAPIClient
        
        try:
            client = FPLAPIClient()
            bootstrap_data = client.get_bootstrap_data()
            
            if bootstrap_data:
                self.cache_manager.set("bootstrap_data", bootstrap_data, ttl=900)  # 15 minutes
                self.logger.info("Warmed bootstrap data cache")
        except Exception as e:
            self.logger.warning("Failed to warm bootstrap data cache", error=str(e))
    
    async def warm_player_data(self, player_ids: list):
        """Warm cache with player data."""
        from ..data.fetchers import FPLAPIClient
        
        client = FPLAPIClient()
        
        for player_id in player_ids:
            try:
                player_data = client.get_player_history(player_id)
                if player_data:
                    self.cache_manager.set(f"player_history:{player_id}", player_data, ttl=3600)
                
                await asyncio.sleep(0.1)  # Rate limiting
            except Exception as e:
                self.logger.warning("Failed to warm player data cache", player_id=player_id, error=str(e))
        
        self.logger.info("Warmed player data cache", player_count=len(player_ids))
    
    async def warm_predictions(self):
        """Warm cache with ML predictions."""
        try:
            from ..models.ml_models import PlayerPredictor
            
            predictor = PlayerPredictor()
            if predictor.is_trained:
                # This would generate predictions for all players
                # Implementation depends on your ML pipeline
                self.logger.info("Warmed prediction cache")
        except Exception as e:
            self.logger.warning("Failed to warm prediction cache", error=str(e))


# Cache monitoring
class CacheMonitor:
    """
    Monitor cache performance and health.
    """
    
    def __init__(self, cache_manager: CacheManager = None):
        self.cache_manager = cache_manager or get_cache_manager()
        self.logger = get_logger(f"{__name__}.CacheMonitor")
    
    def log_stats(self):
        """Log current cache statistics."""
        stats = self.cache_manager.stats()
        self.logger.info("Cache statistics", **stats)
    
    def health_check(self) -> Dict[str, Any]:
        """Perform cache health check."""
        try:
            stats = self.cache_manager.stats()
            
            health = {
                'healthy': True,
                'primary_cache': stats['primary_cache'],
                'memory_utilization': stats['memory_cache']['utilization'],
                'issues': []
            }
            
            # Check memory cache utilization
            if stats['memory_cache']['utilization'] > 0.9:
                health['issues'].append('Memory cache utilization high')
            
            # Check Redis connection if available
            if 'redis_cache' in stats:
                if not stats['redis_cache'].get('connected', False):
                    health['issues'].append('Redis cache disconnected')
                    health['healthy'] = False
                else:
                    hit_rate = stats['redis_cache'].get('hit_rate', 0)
                    if hit_rate < 0.5:
                        health['issues'].append(f'Redis hit rate low: {hit_rate:.2%}')
            
            return health
        
        except Exception as e:
            self.logger.error("Cache health check failed", error=str(e))
            return {'healthy': False, 'error': str(e)}