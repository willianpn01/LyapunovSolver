"""
Tests for Module B: Symbolic Engine
"""

import pytest
import sympy as sp
from sympy import symbols, simplify, Rational, I, pi, sqrt

import sys
sys.path.insert(0, '..')

from lyapunov.system_definition import SystemDefinition, SystemProperty
from lyapunov.symbolic_engine import SymbolicEngine, NormalFormComputer


class TestSymbolicEngine:
    """Tests for SymbolicEngine class."""
    
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
        return SystemDefinition(P, Q, params=[mu])
    
    @pytest.fixture
    def engine(self, simple_system):
        """Create a SymbolicEngine for testing."""
        return SymbolicEngine(simple_system)
    
    def test_engine_creation(self, engine, simple_system):
        """Test engine initialization."""
        assert engine.system == simple_system
        assert engine.max_computed_order == 0
        assert len(engine.lyapunov_coefficients) == 0
    
    def test_compute_L1(self, engine, basic_symbols):
        """Test computation of first Lyapunov coefficient."""
        x, y, mu = basic_symbols
        
        L1 = engine.compute_Lk_symbolic(1)
        
        assert L1 is not None
        assert engine.max_computed_order == 1
        assert 1 in engine.lyapunov_coefficients
    
    def test_L1_caching(self, engine):
        """Test that L1 is cached after first computation."""
        L1_first = engine.compute_Lk_symbolic(1)
        L1_second = engine.compute_Lk_symbolic(1)
        
        assert L1_first == L1_second
    
    def test_hamiltonian_L1_zero(self, basic_symbols):
        """Test that Hamiltonian systems have L1 = 0."""
        x, y, _ = basic_symbols
        
        P = x**2 * y
        Q = -x * y**2
        
        system = SystemDefinition(P, Q, params=[])
        engine = SymbolicEngine(system)
        
        L1 = engine.compute_Lk_symbolic(1)
        
        assert simplify(L1) == 0
    
    def test_numerical_evaluation(self, engine, basic_symbols):
        """Test numerical evaluation of L1."""
        x, y, mu = basic_symbols
        
        L1 = engine.compute_Lk_symbolic(1)
        
        value = engine.evaluate_numeric(1, {mu: 0.5})
        
        assert isinstance(value, complex)
        assert abs(value.imag) < 1e-10
    
    def test_lyapunov_sequence(self, engine):
        """Test computing sequence of coefficients."""
        sequence = engine.compute_lyapunov_sequence(2)
        
        assert len(sequence) == 2
        assert engine.max_computed_order >= 2
    
    def test_stability_condition(self, engine, basic_symbols):
        """Test stability condition extraction."""
        x, y, mu = basic_symbols
        
        conditions = engine.get_stability_condition(1)
        
        assert 'coefficient' in conditions
        assert 'stable_condition' in conditions
        assert 'unstable_condition' in conditions
    
    def test_to_latex(self, engine):
        """Test LaTeX generation."""
        latex = engine.to_latex(1)
        
        assert isinstance(latex, str)
        assert len(latex) > 0
    
    def test_computation_stats(self, engine):
        """Test computation statistics."""
        engine.compute_Lk_symbolic(1)
        
        stats = engine.get_computation_stats()
        
        assert 'max_computed_order' in stats
        assert stats['max_computed_order'] == 1
        assert 'num_cached_Lk' in stats
    
    def test_simplify_levels(self, basic_symbols):
        """Test different simplification levels."""
        x, y, mu = basic_symbols
        P = mu * x - x**3
        Q = -y**3
        system = SystemDefinition(P, Q, params=[mu])
        
        for level in [0, 1, 2, 3]:
            engine = SymbolicEngine(system, simplify_level=level)
            L1 = engine.compute_Lk_symbolic(1)
            assert L1 is not None


class TestSpecificSystems:
    """Tests for specific known systems."""
    
    def test_cubic_system_L1(self):
        """Test L1 for standard cubic system."""
        x, y = symbols('x y')
        
        P = -x**3 - x*y**2
        Q = -x**2*y - y**3
        
        system = SystemDefinition(P, Q, params=[])
        engine = SymbolicEngine(system)
        
        L1 = engine.compute_Lk_symbolic(1)
        L1_value = float(L1.evalf())
        
        assert L1_value < 0
    
    def test_symmetric_system(self):
        """Test system with Z2 symmetry."""
        x, y, a = symbols('x y a')
        
        P = a*x**3 + a*x*y**2
        Q = a*x**2*y + a*y**3
        
        system = SystemDefinition(P, Q, params=[a])
        engine = SymbolicEngine(system)
        
        L1 = engine.compute_Lk_symbolic(1)
        
        assert L1 is not None
    
    def test_parameter_dependence(self):
        """Test that L1 depends on parameters correctly."""
        x, y, mu = symbols('x y mu')
        
        P = mu * x**2 * y
        Q = mu * x * y**2
        
        system = SystemDefinition(P, Q, params=[mu])
        engine = SymbolicEngine(system)
        
        L1 = engine.compute_Lk_symbolic(1)
        
        assert L1.has(mu) or simplify(L1) == 0


class TestNormalFormComputer:
    """Tests for NormalFormComputer class."""
    
    @pytest.fixture
    def basic_engine(self):
        """Create a basic engine for testing."""
        x, y = symbols('x y')
        P = -x**3
        Q = -y**3
        system = SystemDefinition(P, Q, params=[])
        return SymbolicEngine(system)
    
    def test_normal_form_creation(self, basic_engine):
        """Test NormalFormComputer creation."""
        nf = NormalFormComputer(basic_engine)
        
        assert nf.engine == basic_engine
    
    def test_compute_normal_form(self, basic_engine):
        """Test normal form computation."""
        nf = NormalFormComputer(basic_engine)
        
        coeffs = nf.compute_normal_form(3)
        
        assert isinstance(coeffs, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
