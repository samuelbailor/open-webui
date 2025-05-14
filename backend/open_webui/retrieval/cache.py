import time
import hashlib
import logging
from typing import Dict, List, Optional, Any, Tuple

from open_webui.env import SRC_LOG_LEVELS

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["RAG"])

class EmbeddingCache:
    """
    Cache for embeddings based on file metadata.
    """
    def __init__(self, ttl_seconds: int = 86400):
        """
        Initialize the embedding cache.
        
        Args:
            ttl_seconds: Time-to-live for cache entries in seconds (default: 24 hours)
        """
        self.cache: Dict[str, Tuple[float, Any]] = {}
        self.ttl = ttl_seconds
        log.info(f"Embedding cache initialized with TTL of {ttl_seconds} seconds")
    
    def get(self, file_id: str, updated_at: int) -> Optional[List[List[float]]]:
        """
        Get embeddings from cache if available and not expired.
        
        Args:
            file_id: The file ID
            updated_at: The file's last updated timestamp (not used directly)
            
        Returns:
            The cached embeddings if available, None otherwise
        """
        # Check if we have a hash in the metadata
        from open_webui.models.files import Files
        file = Files.get_file_by_id(file_id)
        
        if file and file.hash:
            # Use ONLY the hash as the cache key to work across different file uploads
            cache_key = f"content_{file.hash}"
            log.debug(f"Using content hash as cache key: {cache_key}")
        else:
            # Fallback to file_id if no hash is available
            cache_key = f"file_{file_id}"
            log.debug(f"No hash available, using file_id as cache key: {cache_key}")
        if cache_key in self.cache:
            timestamp, embeddings = self.cache[cache_key]
            if time.time() - timestamp < self.ttl:
                log.info(f"Cache hit for file {file_id} - using cached embeddings")
                return embeddings
            else:
                # Expired entry
                log.info(f"Cache expired for file {file_id} - recalculating embeddings")
                del self.cache[cache_key]
        else:
            log.info(f"Cache miss for file {file_id} - calculating embeddings for the first time")
        return None
    
    def set(self, file_id: str, updated_at: int, embeddings: List[List[float]]) -> None:
        """
        Store embeddings in cache.
        
        Args:
            file_id: The file ID
            updated_at: The file's last updated timestamp (not used directly)
            embeddings: The embeddings to cache
        """
        # Check if we have a hash in the metadata
        from open_webui.models.files import Files
        file = Files.get_file_by_id(file_id)
        
        if file and file.hash:
            # Use ONLY the hash as the cache key to work across different file uploads
            cache_key = f"content_{file.hash}"
            log.debug(f"Using content hash as cache key for storage: {cache_key}")
        else:
            # Fallback to file_id if no hash is available
            cache_key = f"file_{file_id}"
            log.debug(f"No hash available, using file_id as cache key for storage: {cache_key}")
        self.cache[cache_key] = (time.time(), embeddings)
        log.info(f"Cached embeddings for file {file_id} - {len(embeddings)} vectors stored")
    
    def clear(self) -> None:
        """Clear the entire cache."""
        self.cache.clear()
        log.debug("Embedding cache cleared")

# Global cache instance
embedding_cache = EmbeddingCache()