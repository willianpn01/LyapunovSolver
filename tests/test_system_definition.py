"""
Tests for Module A: System Definition and Validation
"""

import pytest
import sympy as sp
from sympy import symbols, simplify, Rational

import sys
sys.path.insert(0, '..')

from lyapunov.system_definition import (
    SystemDefinition, 
    SystemProperty, 
    ValidationResult
)


class TestSystemDefinition:
    """Tests for SystemDefinition class."""
    
    @pytest.fixture
    def basic_symbols(self):
        """Create basic symbols for testing."""
        return symbols('x y mu')
    
    def test_basic_creation(self, basic_symbols):
        """Test basic system creation."""
        x, y, mu = basic_symbols
        
        P = mu * x - x**3
        Q = -y**3
        
        system = SystemDefinition(P, Q, params=[mu])
        
        assert system.P == P
        assert system.Q == Q
        assert system.x == x
        assert system.y == y
        assert mu in system.params
    
    def test_validation_valid_system(self, basic_symbols):
        """Test validation of a valid system."""
        x, y, mu = basic_symbols
        
        P = x**2 + y**2
        Q = x * y
        
        system = SystemDefinition(P, Q, params=[mu])
        result = system.validate()
        
        assert result.is_valid
        assert "valid" in result.message.lower()
    
    def test_validation_invalid_constant_term(self, basic_symbols):
        """Test validation fails with constant term."""
        x, y, mu = basic_symbols
        
        P = 1 + x**2
        Q = y**2
        
        system = SystemDefinition(P, Q, params=[mu], auto_validate=False)
        result = system.validate()
        
        assert not result.is_valid
        assert "constant" in result.message.lower()
    
    def test_validation_invalid_linear_term(self, basic_symbols):
        """Test validation fails with linear term."""
        x, y, mu = basic_symbols
        
        P = x + x**2
        Q = y**2
        
        system = SystemDefinition(P, Q, params=[mu], auto_validate=False)
        result = system.validate()
        
        assert not result.is_valid
        assert "linear" in result.message.lower()
    
    def test_hamiltonian_detection(self, basic_symbols):
        """Test detection of Hamiltonian systems."""
        x, y, _ = basic_symbols
        
        P = x**2 * y
        Q = -x * y**2
        
        system = SystemDefinition(P, Q, params=[])
        
        assert SystemProperty.HAMILTONIAN in system.properties
    
    def test_non_hamiltonian(self, basic_symbols):
        """Test non-Hamiltonian system is not marked as Hamiltonian."""
        x, y, mu = basic_symbols
        
        P = mu * x - x**3
        Q = -y**3
        
        system = SystemDefinition(P, Q, params=[mu])
        
        assert SystemProperty.HAMILTONIAN not in system.properties
    
    def test_z2_symmetry_detection(self, basic_symbols):
        """Test detection of Z2 symmetry."""
        x, y, _ = basic_symbols
        
        P = -x**3 - x * y**2
        Q = -y**3 - x**2 * y
        
        system = SystemDefinition(P, Q, params=[])
        
        assert SystemProperty.Z2_SYMMETRIC in system.properties
    
    def test_get_coefficient(self, basic_symbols):
        """Test coefficient extraction."""
        x, y, mu = basic_symbols
        
        P = 2*x**2 + 3*x*y + mu*y**2
        Q = x**3
        
        system = SystemDefinition(P, Q, params=[mu])
        
        assert system.get_coefficient('P', 2, 0) == 2
        assert system.get_coefficient('P', 1, 1) == 3
        assert system.get_coefficient('P', 0, 2) == mu
        assert system.get_coefficient('Q', 3, 0) == 1
        assert system.get_coefficient('Q', 0, 0) == 0
    
    def test_full_system(self, basic_symbols):
        """Test get_full_system method."""
        x, y, mu = basic_symbols
        
        P = x**2
        Q = y**2
        
        system = SystemDefinition(P, Q, params=[mu])
        x_dot, y_dot = system.get_full_system()
        
        assert simplify(x_dot - (-y + x**2)) == 0
        assert simplify(y_dot - (x + y**2)) == 0
    
    def test_complex_form(self, basic_symbols):
        """Test transformation to complex coordinates."""
        x, y, _ = basic_symbols
        
        P = x**2
        Q = y**2
        
        system = SystemDefinition(P, Q, params=[])
        F, z, z_bar = system.to_complex_form()
        
        assert F.has(z) or F.has(z_bar) or F.is_number
    
    def test_truncate_to_order(self, basic_symbols):
        """Test polynomial truncation."""
        x, y, _ = basic_symbols
        
        P = x**2 + x**3 + x**4 + x**5
        Q = y**2 + y**3 + y**4
        
        system = SystemDefinition(P, Q, params=[])
        truncated = system.truncate_to_order(3)
        
        P_trunc = truncated.P
        Q_trunc = truncated.Q
        
        assert P_trunc.has(x**2)
        assert P_trunc.has(x**3)
        assert not P_trunc.has(x**4)
        assert not P_trunc.has(x**5)
    
    def test_hash_key_uniqueness(self, basic_symbols):
        """Test that different systems have different hash keys."""
        x, y, mu = basic_symbols
        
        system1 = SystemDefinition(x**2, y**2, params=[mu])
        system2 = SystemDefinition(x**3, y**2, params=[mu])
        system3 = SystemDefinition(x**2, y**2, params=[])
        
        assert system1.get_hash_key() != system2.get_hash_key()
        assert system1.get_hash_key() != system3.get_hash_key()
    
    def test_repr_and_str(self, basic_symbols):
        """Test string representations."""
        x, y, mu = basic_symbols
        
        system = SystemDefinition(x**2, y**2, params=[mu])
        
        repr_str = repr(system)
        str_str = str(system)
        
        assert "SystemDefinition" in repr_str
        assert "ẋ" in str_str or "x" in str_str


class TestValidationResult:
    """Tests for ValidationResult dataclass."""
    
    def test_valid_result(self):
        """Test valid result creation."""
        result = ValidationResult(
            is_valid=True,
            message="System is valid"
        )
        
        assert result.is_valid
        assert result.warnings == []
    
    def test_invalid_result_with_warnings(self):
        """Test invalid result with warnings."""
        result = ValidationResult(
            is_valid=False,
            message="Validation failed",
            warnings=["Warning 1", "Warning 2"]
        )
        
        assert not result.is_valid
        assert len(result.warnings) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
