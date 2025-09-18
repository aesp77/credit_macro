"""
Caching utilities for Bloomberg data
Reduces API calls and improves performance
"""
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


class DataCache:
    """Simple time-based cache for market data"""
    
    def __init__(self, expiry_minutes: int = 5):
        """
        Initialize cache with expiry time
        
        Args:
            expiry_minutes: How long to cache data in minutes
        """
        self.cache_expiry = timedelta(minutes=expiry_minutes)
        self._cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._hit_count = 0
        self._miss_count = 0
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache if valid
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        if self.is_valid(key):
            self._hit_count += 1
            logger.debug(f"Cache hit for key: {key}")
            return self._cache.get(key)
        
        self._miss_count += 1
        logger.debug(f"Cache miss for key: {key}")
        return None
    
    def set(self, key: str, value: Any):
        """
        Store value in cache
        
        Args:
            key: Cache key
            value: Value to cache
        """
        self._cache[key] = value
        self._cache_timestamps[key] = datetime.now()
        logger.debug(f"Cached value for key: {key}")
    
    def is_valid(self, key: str) -> bool:
        """
        Check if cached value is still valid
        
        Args:
            key: Cache key
            
        Returns:
            True if cache entry exists and hasn't expired
        """
        if key not in self._cache:
            return False
        
        timestamp = self._cache_timestamps.get(key)
        if timestamp is None:
            return False
        
        age = datetime.now() - timestamp
        return age < self.cache_expiry
    
    def clear(self, pattern: str = None):
        """
        Clear cache entries
        
        Args:
            pattern: Optional pattern to match keys (None clears all)
        """
        if pattern is None:
            self._cache.clear()
            self._cache_timestamps.clear()
            logger.info("Cleared entire cache")
        else:
            keys_to_delete = [k for k in self._cache.keys() if pattern in k]
            for key in keys_to_delete:
                del self._cache[key]
                del self._cache_timestamps[key]
            logger.info(f"Cleared {len(keys_to_delete)} cache entries matching '{pattern}'")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self._hit_count + self._miss_count
        hit_rate = self._hit_count / total_requests if total_requests > 0 else 0
        
        return {
            'entries': len(self._cache),
            'hits': self._hit_count,
            'misses': self._miss_count,
            'hit_rate': hit_rate,
            'size_bytes': sum(len(str(v)) for v in self._cache.values())
        }
    
    def make_key(self, *args, **kwargs) -> str:
        """
        Generate cache key from arguments
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Unique cache key
        """
        key_data = {
            'args': args,
            'kwargs': kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.md5(key_str.encode()).hexdigest()


class CachedFunction:
    """Decorator for caching function results"""
    
    def __init__(self, cache: DataCache = None, expiry_minutes: int = 5):
        """
        Initialize cached function decorator
        
        Args:
            cache: DataCache instance (creates new if None)
            expiry_minutes: Cache expiry time
        """
        self.cache = cache or DataCache(expiry_minutes)
    
    def __call__(self, func: Callable) -> Callable:
        """
        Decorator implementation
        
        Args:
            func: Function to cache
            
        Returns:
            Wrapped function with caching
        """
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{func.__name__}_{self.cache.make_key(*args, **kwargs)}"
            
            # Check cache
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Call function and cache result
            result = func(*args, **kwargs)
            self.cache.set(cache_key, result)
            return result
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.cache = self.cache  # Expose cache for management
        return wrapper


# Specialized caches for different data types
class SpreadCache(DataCache):
    """Cache specifically for spread data with shorter expiry"""
    
    def __init__(self, expiry_minutes: int = 1):
        super().__init__(expiry_minutes)


class ReferenceDataCache(DataCache):
    """Cache for reference data with longer expiry"""
    
    def __init__(self, expiry_minutes: int = 1440):  # 24 hours
        super().__init__(expiry_minutes)


class CurveCache(DataCache):
    """Cache for curve data with medium expiry"""
    
    def __init__(self, expiry_minutes: int = 15):
        super().__init__(expiry_minutes)