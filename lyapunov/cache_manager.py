"""
Module C: Cache Manager (Multi-level Caching System)
Implements intelligent caching with SHA-256 hashing for computed results.
"""

import hashlib
import json
import pickle
import os
from pathlib import Path
from typing import Dict, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import threading

import sympy as sp
from sympy import Expr, Symbol, sympify


@dataclass
class CacheEntry:
    """Represents a cached computation result."""
    key: str
    value: Any
    created_at: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def touch(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = datetime.now()


class CacheManager:
    """
    Multi-level caching system for Lyapunov computations.
    
    Levels:
    1. Memory cache (fastest, volatile)
    2. Disk cache (persistent between sessions)
    
    Features:
    - SHA-256 based unique identification
    - Automatic serialization of SymPy expressions
    - Thread-safe operations
    - Cache statistics and management
    
    Attributes:
        cache_dir: Directory for disk cache storage
        memory_cache: In-memory cache dictionary
        max_memory_entries: Maximum entries in memory cache
    """
    
    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        max_memory_entries: int = 1000,
        enable_disk_cache: bool = True
    ):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Directory for disk cache (default: ~/.lyapunov_cache)
            max_memory_entries: Maximum entries in memory cache
            enable_disk_cache: Whether to use disk caching
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".lyapunov_cache"
        
        self.cache_dir = Path(cache_dir)
        self.max_memory_entries = max_memory_entries
        self.enable_disk_cache = enable_disk_cache
        
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._binary_cache: Dict[str, Path] = {}
        self._lock = threading.RLock()
        
        self._stats = {
            'memory_hits': 0,
            'memory_misses': 0,
            'disk_hits': 0,
            'disk_misses': 0,
            'total_saves': 0
        }
        
        if self.enable_disk_cache:
            self._init_disk_cache()
    
    def _init_disk_cache(self) -> None:
        """Initialize disk cache directory structure."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        (self.cache_dir / "symbolic").mkdir(exist_ok=True)
        (self.cache_dir / "binary").mkdir(exist_ok=True)
        (self.cache_dir / "metadata").mkdir(exist_ok=True)
    
    def get_hash(self, system_key: str, order: int, extra: str = "") -> str:
        """
        Generate SHA-256 hash for unique identification.
        
        Args:
            system_key: String representation of the system
            order: Order of Lyapunov coefficient
            extra: Additional distinguishing information
            
        Returns:
            Hexadecimal hash string
        """
        content = f"{system_key}|order={order}|{extra}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def get_cached_symbolic(
        self,
        system_key: str,
        order: int
    ) -> Optional[Expr]:
        """
        Retrieve cached symbolic expression.
        
        Search order:
        1. Memory cache
        2. Disk cache
        
        Args:
            system_key: System identification string
            order: Order of Lyapunov coefficient
            
        Returns:
            Cached SymPy expression or None if not found
        """
        cache_key = self.get_hash(system_key, order)
        
        with self._lock:
            if cache_key in self._memory_cache:
                entry = self._memory_cache[cache_key]
                entry.touch()
                self._stats['memory_hits'] += 1
                return entry.value
            
            self._stats['memory_misses'] += 1
        
        if self.enable_disk_cache:
            disk_result = self._load_from_disk(cache_key, "symbolic")
            if disk_result is not None:
                self._stats['disk_hits'] += 1
                with self._lock:
                    self._add_to_memory_cache(cache_key, disk_result)
                return disk_result
            
            self._stats['disk_misses'] += 1
        
        return None
    
    def save_symbolic(
        self,
        system_key: str,
        order: int,
        expression: Expr,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Save symbolic expression to cache.
        
        Args:
            system_key: System identification string
            order: Order of Lyapunov coefficient
            expression: SymPy expression to cache
            metadata: Optional metadata to store
            
        Returns:
            Cache key (hash) for the entry
        """
        cache_key = self.get_hash(system_key, order)
        
        with self._lock:
            self._add_to_memory_cache(cache_key, expression, metadata)
        
        if self.enable_disk_cache:
            self._save_to_disk(cache_key, expression, "symbolic", metadata)
        
        self._stats['total_saves'] += 1
        return cache_key
    
    def _add_to_memory_cache(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add entry to memory cache with LRU eviction."""
        if len(self._memory_cache) >= self.max_memory_entries:
            self._evict_lru()
        
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(),
            metadata=metadata or {}
        )
        entry.touch()
        self._memory_cache[key] = entry
    
    def _evict_lru(self) -> None:
        """Evict least recently used entry from memory cache."""
        if not self._memory_cache:
            return
        
        lru_key = min(
            self._memory_cache.keys(),
            key=lambda k: (
                self._memory_cache[k].last_accessed or 
                self._memory_cache[k].created_at
            )
        )
        del self._memory_cache[lru_key]
    
    def _save_to_disk(
        self,
        key: str,
        value: Any,
        cache_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Save value to disk cache.
        
        Args:
            key: Cache key (hash)
            value: Value to save
            cache_type: "symbolic" or "binary"
            metadata: Optional metadata
            
        Returns:
            True if successful
        """
        try:
            cache_path = self.cache_dir / cache_type / f"{key}.pkl"
            
            if isinstance(value, Expr):
                serialized = {
                    'type': 'sympy_expr',
                    'srepr': sp.srepr(value),
                    'str': str(value)
                }
            else:
                serialized = {'type': 'raw', 'value': value}
            
            with open(cache_path, 'wb') as f:
                pickle.dump(serialized, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            if metadata:
                meta_path = self.cache_dir / "metadata" / f"{key}.json"
                with open(meta_path, 'w') as f:
                    json.dump({
                        'key': key,
                        'cache_type': cache_type,
                        'created_at': datetime.now().isoformat(),
                        **metadata
                    }, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Warning: Failed to save to disk cache: {e}")
            return False
    
    def _load_from_disk(self, key: str, cache_type: str) -> Optional[Any]:
        """
        Load value from disk cache.
        
        Args:
            key: Cache key (hash)
            cache_type: "symbolic" or "binary"
            
        Returns:
            Cached value or None if not found
        """
        try:
            cache_path = self.cache_dir / cache_type / f"{key}.pkl"
            
            if not cache_path.exists():
                return None
            
            with open(cache_path, 'rb') as f:
                serialized = pickle.load(f)
            
            if serialized.get('type') == 'sympy_expr':
                return sympify(serialized['srepr'])
            else:
                return serialized.get('value')
                
        except Exception as e:
            print(f"Warning: Failed to load from disk cache: {e}")
            return None
    
    def get_cached_binary(self, system_key: str, order: int) -> Optional[Path]:
        """
        Get path to cached compiled binary library.
        
        Args:
            system_key: System identification string
            order: Order of Lyapunov coefficient
            
        Returns:
            Path to binary library or None if not found
        """
        cache_key = self.get_hash(system_key, order, extra="binary")
        
        if cache_key in self._binary_cache:
            path = self._binary_cache[cache_key]
            if path.exists():
                return path
        
        if self.enable_disk_cache:
            binary_dir = self.cache_dir / "binary"
            
            for ext in ['.so', '.dll', '.dylib', '.pyd']:
                binary_path = binary_dir / f"{cache_key}{ext}"
                if binary_path.exists():
                    self._binary_cache[cache_key] = binary_path
                    return binary_path
        
        return None
    
    def save_binary(
        self,
        system_key: str,
        order: int,
        binary_path: Path
    ) -> str:
        """
        Register a compiled binary in the cache.
        
        Args:
            system_key: System identification string
            order: Order of Lyapunov coefficient
            binary_path: Path to the compiled binary
            
        Returns:
            Cache key for the binary
        """
        cache_key = self.get_hash(system_key, order, extra="binary")
        self._binary_cache[cache_key] = binary_path
        return cache_key
    
    def clear_memory_cache(self) -> int:
        """
        Clear all entries from memory cache.
        
        Returns:
            Number of entries cleared
        """
        with self._lock:
            count = len(self._memory_cache)
            self._memory_cache.clear()
            return count
    
    def clear_disk_cache(self) -> int:
        """
        Clear all entries from disk cache.
        
        Returns:
            Number of files deleted
        """
        if not self.enable_disk_cache:
            return 0
        
        count = 0
        for subdir in ["symbolic", "binary", "metadata"]:
            cache_subdir = self.cache_dir / subdir
            if cache_subdir.exists():
                for file in cache_subdir.iterdir():
                    try:
                        file.unlink()
                        count += 1
                    except:
                        pass
        
        return count
    
    def clear_all(self) -> Dict[str, int]:
        """
        Clear both memory and disk caches.
        
        Returns:
            Dictionary with counts of cleared entries
        """
        return {
            'memory_cleared': self.clear_memory_cache(),
            'disk_cleared': self.clear_disk_cache()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            memory_size = len(self._memory_cache)
        
        disk_size = 0
        if self.enable_disk_cache:
            symbolic_dir = self.cache_dir / "symbolic"
            if symbolic_dir.exists():
                disk_size = len(list(symbolic_dir.glob("*.pkl")))
        
        hit_rate_memory = 0
        total_memory = self._stats['memory_hits'] + self._stats['memory_misses']
        if total_memory > 0:
            hit_rate_memory = self._stats['memory_hits'] / total_memory
        
        hit_rate_disk = 0
        total_disk = self._stats['disk_hits'] + self._stats['disk_misses']
        if total_disk > 0:
            hit_rate_disk = self._stats['disk_hits'] / total_disk
        
        return {
            'memory_entries': memory_size,
            'disk_entries': disk_size,
            'memory_hit_rate': hit_rate_memory,
            'disk_hit_rate': hit_rate_disk,
            **self._stats
        }
    
    def list_cached_systems(self) -> list:
        """
        List all cached system computations.
        
        Returns:
            List of metadata dictionaries for cached entries
        """
        entries = []
        
        if self.enable_disk_cache:
            meta_dir = self.cache_dir / "metadata"
            if meta_dir.exists():
                for meta_file in meta_dir.glob("*.json"):
                    try:
                        with open(meta_file, 'r') as f:
                            entries.append(json.load(f))
                    except:
                        pass
        
        return entries
    
    def invalidate(self, system_key: str, order: Optional[int] = None) -> int:
        """
        Invalidate cache entries for a system.
        
        Args:
            system_key: System identification string
            order: Specific order to invalidate (None = all orders)
            
        Returns:
            Number of entries invalidated
        """
        count = 0
        
        if order is not None:
            cache_key = self.get_hash(system_key, order)
            
            with self._lock:
                if cache_key in self._memory_cache:
                    del self._memory_cache[cache_key]
                    count += 1
            
            if self.enable_disk_cache:
                for subdir in ["symbolic", "metadata"]:
                    path = self.cache_dir / subdir / f"{cache_key}.*"
                    for file in self.cache_dir.glob(str(path)):
                        try:
                            file.unlink()
                            count += 1
                        except:
                            pass
        
        return count
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (f"CacheManager(memory={stats['memory_entries']}, "
                f"disk={stats['disk_entries']}, "
                f"hit_rate={stats['memory_hit_rate']:.2%})")
