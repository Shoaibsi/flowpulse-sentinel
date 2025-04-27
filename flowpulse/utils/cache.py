import os
import json
import time
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Any, Optional

logger = logging.getLogger(__name__)

class CacheManager:
    def __init__(self, cache_dir: str, default_ttl_seconds: int):
        self.cache_dir = cache_dir
        self.default_ttl_seconds = default_ttl_seconds
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"CacheManager initialized. Cache directory: {self.cache_dir}, Default TTL: {self.default_ttl_seconds}s")

    def _get_file_path(self, key: str) -> str:
        # Use a hash of the key for the filename to avoid issues with invalid chars
        hashed_key = hashlib.md5(key.encode('utf-8')).hexdigest()
        return os.path.join(self.cache_dir, f"{hashed_key}.json")

    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None):
        """Stores an item in the cache."""
        if ttl_seconds is None:
            ttl_seconds = self.default_ttl_seconds

        filepath = self._get_file_path(key)
        expiry_timestamp = time.time() + ttl_seconds
        data_to_store = {
            'expiry': expiry_timestamp,
            'value': value
        }
        try:
            with open(filepath, 'w') as f:
                json.dump(data_to_store, f)
            logger.debug(f"Cache SET: Key='{key}', TTL={ttl_seconds}s, Path='{filepath}'")
        except (IOError, TypeError, json.JSONDecodeError) as e:
            logger.error(f"Failed to set cache for key '{key}': {e}")

    def get(self, key: str) -> Optional[Any]:
        """Retrieves an item from the cache if it exists and hasn't expired."""
        filepath = self._get_file_path(key)
        if not os.path.exists(filepath):
            logger.debug(f"Cache MISS (Not Found): Key='{key}', Path='{filepath}'")
            return None

        try:
            with open(filepath, 'r') as f:
                cached_data = json.load(f)

            expiry_timestamp = cached_data.get('expiry')
            if expiry_timestamp is None:
                 logger.warning(f"Cache MISS (No Expiry): Key='{key}', Path='{filepath}'")
                 return None # Treat items without expiry as invalid

            if time.time() > expiry_timestamp:
                logger.info(f"Cache MISS (Expired): Key='{key}', Path='{filepath}'")
                self.delete(key) # Clean up expired file
                return None
            else:
                logger.debug(f"Cache HIT: Key='{key}', Path='{filepath}'")
                return cached_data.get('value')

        except (IOError, json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to get cache for key '{key}': {e}")
            # Attempt to delete corrupted cache file
            self.delete(key)
            return None

    def delete(self, key: str):
        """Deletes an item from the cache."""
        filepath = self._get_file_path(key)
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.debug(f"Cache DELETE: Key='{key}', Path='{filepath}'")
        except OSError as e:
            logger.error(f"Failed to delete cache file '{filepath}': {e}")

    def clear_all(self):
        """Clears all items from the cache directory."""
        logger.warning(f"Clearing all cache files from {self.cache_dir}...")
        for filename in os.listdir(self.cache_dir):
            filepath = os.path.join(self.cache_dir, filename)
            try:
                if os.path.isfile(filepath):
                    os.remove(filepath)
            except Exception as e:
                logger.error(f"Failed to delete cache file {filepath} during clear_all: {e}")
        logger.info("Cache cleared.")
