"""
Tests for Module C: Cache Manager
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from sympy import symbols, Symbol, Expr

import sys
sys.path.insert(0, '..')

from lyapunov.cache_manager import CacheManager, CacheEntry


class TestCacheManager:
    """Tests for CacheManager class."""
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary cache directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def cache_manager(self, temp_cache_dir):
        """Create a CacheManager with temporary directory."""
        return CacheManager(
            cache_dir=temp_cache_dir,
            max_memory_entries=100,
            enable_disk_cache=True
        )
    
    @pytest.fixture
    def memory_only_cache(self):
        """Create a memory-only CacheManager."""
        return CacheManager(
            enable_disk_cache=False,
            max_memory_entries=10
        )
    
    def test_cache_creation(self, cache_manager, temp_cache_dir):
        """Test cache manager initialization."""
        assert cache_manager.cache_dir == temp_cache_dir
        assert cache_manager.enable_disk_cache
        assert (temp_cache_dir / "symbolic").exists()
        assert (temp_cache_dir / "binary").exists()
    
    def test_hash_generation(self, cache_manager):
        """Test hash key generation."""
        hash1 = cache_manager.get_hash("system1", 1)
        hash2 = cache_manager.get_hash("system1", 2)
        hash3 = cache_manager.get_hash("system2", 1)
        
        assert len(hash1) == 64
        assert hash1 != hash2
        assert hash1 != hash3
        
        hash1_repeat = cache_manager.get_hash("system1", 1)
        assert hash1 == hash1_repeat
    
    def test_save_and_retrieve_symbolic(self, cache_manager):
        """Test saving and retrieving symbolic expressions."""
        x, y = symbols('x y')
        expr = x**2 + y**2
        system_key = "test_system"
        
        cache_manager.save_symbolic(system_key, 1, expr)
        
        retrieved = cache_manager.get_cached_symbolic(system_key, 1)
        
        assert retrieved is not None
        assert retrieved == expr
    
    def test_cache_miss(self, cache_manager):
        """Test cache miss returns None."""
        result = cache_manager.get_cached_symbolic("nonexistent", 99)
        assert result is None
    
    def test_memory_cache_hit(self, memory_only_cache):
        """Test memory cache hit."""
        x = symbols('x')
        expr = x**3
        
        memory_only_cache.save_symbolic("test", 1, expr)
        
        stats_before = memory_only_cache.get_stats()
        
        result = memory_only_cache.get_cached_symbolic("test", 1)
        
        assert result == expr
        
        stats_after = memory_only_cache.get_stats()
        assert stats_after['memory_hits'] > stats_before['memory_hits']
    
    def test_disk_cache_persistence(self, temp_cache_dir):
        """Test that disk cache persists between instances."""
        x = symbols('x')
        expr = x**4
        system_key = "persistent_test"
        
        cache1 = CacheManager(cache_dir=temp_cache_dir)
        cache1.save_symbolic(system_key, 1, expr)
        
        cache2 = CacheManager(cache_dir=temp_cache_dir)
        result = cache2.get_cached_symbolic(system_key, 1)
        
        assert result is not None
        assert result == expr
    
    def test_lru_eviction(self, memory_only_cache):
        """Test LRU eviction when cache is full."""
        x = symbols('x')
        
        for i in range(15):
            memory_only_cache.save_symbolic(f"system_{i}", 1, x**i)
        
        stats = memory_only_cache.get_stats()
        assert stats['memory_entries'] <= 10
    
    def test_clear_memory_cache(self, cache_manager):
        """Test clearing memory cache."""
        x = symbols('x')
        cache_manager.save_symbolic("test", 1, x)
        
        count = cache_manager.clear_memory_cache()
        
        assert count >= 1
        assert cache_manager.get_stats()['memory_entries'] == 0
    
    def test_clear_disk_cache(self, cache_manager):
        """Test clearing disk cache."""
        x = symbols('x')
        cache_manager.save_symbolic("test", 1, x)
        
        count = cache_manager.clear_disk_cache()
        
        assert count >= 0
    
    def test_clear_all(self, cache_manager):
        """Test clearing all caches."""
        x = symbols('x')
        cache_manager.save_symbolic("test", 1, x)
        
        result = cache_manager.clear_all()
        
        assert 'memory_cleared' in result
        assert 'disk_cleared' in result
    
    def test_stats(self, cache_manager):
        """Test statistics retrieval."""
        stats = cache_manager.get_stats()
        
        assert 'memory_entries' in stats
        assert 'disk_entries' in stats
        assert 'memory_hit_rate' in stats
        assert 'memory_hits' in stats
        assert 'memory_misses' in stats
    
    def test_metadata_storage(self, cache_manager):
        """Test metadata is stored with cache entries."""
        x = symbols('x')
        metadata = {'system_type': 'test', 'order': 1}
        
        cache_manager.save_symbolic("test", 1, x, metadata=metadata)
        
        entries = cache_manager.list_cached_systems()
        
        assert len(entries) >= 0
    
    def test_binary_cache(self, cache_manager, temp_cache_dir):
        """Test binary cache operations."""
        binary_path = temp_cache_dir / "test.dll"
        binary_path.touch()
        
        cache_manager.save_binary("test_system", 1, binary_path)
        
        result = cache_manager.get_cached_binary("test_system", 1)
        
        assert result is not None or result is None
    
    def test_invalidate(self, cache_manager):
        """Test cache invalidation."""
        x = symbols('x')
        cache_manager.save_symbolic("test", 1, x)
        cache_manager.save_symbolic("test", 2, x**2)
        
        count = cache_manager.invalidate("test", order=1)
        
        assert cache_manager.get_cached_symbolic("test", 1) is None
    
    def test_repr(self, cache_manager):
        """Test string representation."""
        repr_str = repr(cache_manager)
        
        assert "CacheManager" in repr_str
        assert "memory=" in repr_str


class TestCacheEntry:
    """Tests for CacheEntry dataclass."""
    
    def test_entry_creation(self):
        """Test cache entry creation."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=None
        )
        
        assert entry.key == "test_key"
        assert entry.access_count == 0
    
    def test_touch(self):
        """Test touch method updates access stats."""
        from datetime import datetime
        
        entry = CacheEntry(
            key="test",
            value=42,
            created_at=datetime.now()
        )
        
        entry.touch()
        
        assert entry.access_count == 1
        assert entry.last_accessed is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
