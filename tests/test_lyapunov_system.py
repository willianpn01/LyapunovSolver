"""
Tests for Layer 4: High-Level API (LyapunovSystem)
"""

import pytest
import tempfile
from pathlib import Path
from sympy import symbols, simplify, Rational

import sys
sys.path.insert(0, '..')

from lyapunov import LyapunovSystem
from lyapunov.lyapunov_system import create_system, from_full_system


class TestLyapunovSystem:
    """Tests for LyapunovSystem facade class."""
    
    @pytest.fixture
    def basic_symbols(self):
        """Create basic symbols for testing."""
        return symbols('x y mu')
    
    @pytest.fixture
    def simple_system(self, basic_symbols):
        """Create a simple test system."""
        x, y, mu = basic_symbols
        P = mu * x - x**3
        Q = -y**3
        return LyapunovSystem(P, Q, params=[mu], enable_cache=False)
    
    def test_system_creation(self, simple_system, basic_symbols):
        """Test system creation."""
        x, y, mu = basic_symbols
        
        assert simple_system.P == mu * x - x**3
        assert simple_system.Q == -y**3
        assert mu in simple_system.params
    
    def test_system_from_strings(self, basic_symbols):
        """Test system creation from string expressions."""
        x, y, mu = basic_symbols
        
        system = LyapunovSystem("mu*x - x**3", "-y**3", params=[mu])
        
        assert system.P is not None
        assert system.Q is not None
    
    def test_validation(self, simple_system):
        """Test system validation."""
        result = simple_system.validate()
        
        assert result.is_valid
    
    def test_compute_lyapunov(self, simple_system):
        """Test Lyapunov coefficient computation."""
        L1 = simple_system.compute_lyapunov(1)
        
        assert L1 is not None
    
    def test_compute_lyapunov_sequence(self, simple_system):
        """Test computing sequence of coefficients."""
        coeffs = simple_system.compute_lyapunov_sequence(2)
        
        assert 1 in coeffs
        assert 2 in coeffs
    
    def test_evaluate_lyapunov(self, simple_system, basic_symbols):
        """Test numerical evaluation."""
        x, y, mu = basic_symbols
        
        value = simple_system.evaluate_lyapunov(1, {mu: 0.5})
        
        assert isinstance(value, float)
    
    def test_evaluate_batch(self, simple_system, basic_symbols):
        """Test batch evaluation."""
        x, y, mu = basic_symbols
        
        values = simple_system.evaluate_batch(1, {mu: [0.0, 0.5, 1.0]})
        
        assert len(values) == 3
    
    def test_classify_bifurcation(self, simple_system, basic_symbols):
        """Test bifurcation classification."""
        x, y, mu = basic_symbols
        
        bif_type = simple_system.classify_bifurcation({mu: 0.5})
        
        assert bif_type in ['supercritical', 'subcritical', 'degenerate']
    
    def test_stability_info(self, simple_system):
        """Test stability information retrieval."""
        info = simple_system.get_stability_info(k=1)
        
        assert 'coefficient' in info
        assert 'order' in info
        assert 'latex' in info
    
    def test_to_latex(self, simple_system):
        """Test LaTeX generation."""
        latex = simple_system.to_latex(1)
        
        assert isinstance(latex, str)
        assert len(latex) > 0
    
    def test_export_latex(self, simple_system):
        """Test LaTeX file export."""
        with tempfile.NamedTemporaryFile(suffix='.tex', delete=False) as f:
            filepath = Path(f.name)
        
        try:
            simple_system.export_latex(filepath, k=1)
            
            assert filepath.exists()
            content = filepath.read_text()
            assert len(content) > 0
        finally:
            filepath.unlink(missing_ok=True)
    
    def test_get_c_code(self, simple_system):
        """Test C code generation."""
        c_code = simple_system.get_c_code(1)
        
        assert isinstance(c_code, str)
        assert 'double' in c_code
        assert 'return' in c_code
    
    def test_get_stats(self, simple_system):
        """Test statistics retrieval."""
        simple_system.compute_lyapunov(1)
        
        stats = simple_system.get_stats()
        
        assert 'system' in stats
        assert 'computation' in stats
    
    def test_repr_and_str(self, simple_system):
        """Test string representations."""
        repr_str = repr(simple_system)
        str_str = str(simple_system)
        
        assert "LyapunovSystem" in repr_str
        assert "ẋ" in str_str or "x" in str_str
    
    def test_properties(self, basic_symbols):
        """Test system properties detection."""
        x, y, _ = basic_symbols
        
        P = x**2 * y
        Q = -x * y**2
        
        system = LyapunovSystem(P, Q, params=[], enable_cache=False)
        
        assert len(system.properties) > 0


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_create_system(self):
        """Test create_system function."""
        x, y, mu = symbols('x y mu')
        
        system = create_system(x**2, y**2, params=[mu])
        
        assert isinstance(system, LyapunovSystem)
    
    def test_from_full_system(self):
        """Test from_full_system function."""
        x, y = symbols('x y')
        
        x_dot = -y + x**2
        y_dot = x + y**2
        
        system = from_full_system(x_dot, y_dot, x=x, y=y)
        
        assert isinstance(system, LyapunovSystem)
        assert simplify(system.P - x**2) == 0
        assert simplify(system.Q - y**2) == 0


class TestCacheIntegration:
    """Tests for cache integration."""
    
    def test_cache_enabled(self):
        """Test system with cache enabled."""
        x, y, mu = symbols('x y mu')
        
        with tempfile.TemporaryDirectory() as temp_dir:
            system = LyapunovSystem(
                x**2, y**2, 
                params=[mu],
                cache_dir=temp_dir,
                enable_cache=True
            )
            
            L1_first = system.compute_lyapunov(1)
            L1_second = system.compute_lyapunov(1)
            
            assert L1_first == L1_second
    
    def test_clear_cache(self):
        """Test cache clearing."""
        x, y, mu = symbols('x y mu')
        
        with tempfile.TemporaryDirectory() as temp_dir:
            system = LyapunovSystem(
                x**2, y**2,
                params=[mu],
                cache_dir=temp_dir,
                enable_cache=True
            )
            
            system.compute_lyapunov(1)
            result = system.clear_cache()
            
            assert 'memory_cleared' in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
