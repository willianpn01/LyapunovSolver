"""
Module A: System Definition and Validation
Handles input of dynamical systems and validation of canonical form.
"""

from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum, auto

import sympy as sp
from sympy import (
    Symbol, Expr, symbols, expand, degree, 
    diff, simplify, Poly, I, sqrt, Rational,
    collect, Add, Mul, Pow, S
)


class SystemProperty(Enum):
    """Properties that a dynamical system may possess."""
    HAMILTONIAN = auto()
    REVERSIBLE = auto()
    Z2_SYMMETRIC = auto()
    SO2_SYMMETRIC = auto()
    POLYNOMIAL = auto()
    ANALYTIC = auto()


@dataclass
class ValidationResult:
    """Result of system validation."""
    is_valid: bool
    message: str
    warnings: List[str] = field(default_factory=list)
    linearization_eigenvalues: Optional[Tuple[Expr, Expr]] = None


class SystemDefinition:
    """
    Defines and validates a planar dynamical system for Lyapunov analysis.
    
    The system must be in the canonical form:
        ẋ = -y + P(x, y, μ)
        ẏ = x + Q(x, y, μ)
    
    where P and Q contain only nonlinear terms (degree ≥ 2).
    
    Attributes:
        P: Nonlinear part of the x-equation
        Q: Nonlinear part of the y-equation
        x, y: State variables
        params: List of parameter symbols
        properties: Detected system properties
    """
    
    def __init__(
        self,
        P: Expr,
        Q: Expr,
        params: Optional[List[Symbol]] = None,
        x: Optional[Symbol] = None,
        y: Optional[Symbol] = None,
        auto_validate: bool = True
    ):
        """
        Initialize a system definition.
        
        Args:
            P: Nonlinear terms in ẋ = -y + P(x,y,μ)
            Q: Nonlinear terms in ẏ = x + Q(x,y,μ)
            params: List of parameter symbols (e.g., [mu, epsilon])
            x, y: State variable symbols (default: symbols('x y'))
            auto_validate: Whether to validate on construction
        """
        self.x = x if x is not None else symbols('x')
        self.y = y if y is not None else symbols('y')
        self.params = params if params is not None else []
        
        self.P = sp.sympify(P)
        self.Q = sp.sympify(Q)
        
        self._P_expanded = expand(self.P)
        self._Q_expanded = expand(self.Q)
        
        self.properties: Set[SystemProperty] = set()
        self._validation_result: Optional[ValidationResult] = None
        self._taylor_coeffs_P: Dict[Tuple[int, int], Expr] = {}
        self._taylor_coeffs_Q: Dict[Tuple[int, int], Expr] = {}
        
        self._extract_taylor_coefficients()
        
        if auto_validate:
            self.validate()
            self.detect_properties()
    
    def _extract_taylor_coefficients(self) -> None:
        """Extract Taylor coefficients from P and Q."""
        for expr, coeff_dict in [(self._P_expanded, self._taylor_coeffs_P),
                                  (self._Q_expanded, self._taylor_coeffs_Q)]:
            if expr == 0:
                continue
            
            expr_expanded = expand(expr)
            if isinstance(expr_expanded, Add):
                terms = expr_expanded.args
            else:
                terms = [expr_expanded]
            
            for term in terms:
                coeff, powers = self._extract_monomial_info(term)
                if powers is not None:
                    i, j = powers
                    if (i, j) in coeff_dict:
                        coeff_dict[(i, j)] = simplify(coeff_dict[(i, j)] + coeff)
                    else:
                        coeff_dict[(i, j)] = coeff
    
    def _extract_monomial_info(self, term: Expr) -> Tuple[Expr, Optional[Tuple[int, int]]]:
        """
        Extract coefficient and powers from a monomial term.
        
        Returns:
            Tuple of (coefficient, (x_power, y_power)) or (term, None) if not a monomial
        """
        term = expand(term)
        
        try:
            poly = Poly(term, self.x, self.y)
            monoms = poly.monoms()
            if len(monoms) == 1:
                i, j = monoms[0]
                coeff = poly.coeffs()[0]
                return (coeff, (i, j))
        except:
            pass
        
        return (term, None)
    
    def get_coefficient(self, expr_type: str, i: int, j: int) -> Expr:
        """
        Get the coefficient of x^i * y^j in P or Q.
        
        Args:
            expr_type: 'P' or 'Q'
            i: Power of x
            j: Power of y
            
        Returns:
            The coefficient (0 if not present)
        """
        coeff_dict = self._taylor_coeffs_P if expr_type == 'P' else self._taylor_coeffs_Q
        return coeff_dict.get((i, j), S.Zero)
    
    def validate(self) -> ValidationResult:
        """
        Validate that the system is in proper canonical form.
        
        Checks:
        1. P and Q contain no constant or linear terms
        2. The linearization at origin has eigenvalues ±i
        
        Returns:
            ValidationResult with validation status and details
        """
        warnings = []
        
        for name, expr, coeff_dict in [('P', self.P, self._taylor_coeffs_P),
                                        ('Q', self.Q, self._taylor_coeffs_Q)]:
            const_term = coeff_dict.get((0, 0), S.Zero)
            if const_term != 0:
                self._validation_result = ValidationResult(
                    is_valid=False,
                    message=f"{name} contains constant term: {const_term}. "
                           f"System must have equilibrium at origin."
                )
                return self._validation_result
            
            linear_x = coeff_dict.get((1, 0), S.Zero)
            linear_y = coeff_dict.get((0, 1), S.Zero)
            if linear_x != 0 or linear_y != 0:
                self._validation_result = ValidationResult(
                    is_valid=False,
                    message=f"{name} contains linear terms: {linear_x}*x + {linear_y}*y. "
                           f"Linear part must be in canonical form ẋ=-y, ẏ=x."
                )
                return self._validation_result
        
        eigenvalues = (I, -I)
        
        self._validation_result = ValidationResult(
            is_valid=True,
            message="System is in valid canonical form.",
            warnings=warnings,
            linearization_eigenvalues=eigenvalues
        )
        return self._validation_result
    
    def detect_properties(self) -> Set[SystemProperty]:
        """
        Detect special properties of the system.
        
        Detects:
        - Hamiltonian: ∂P/∂x + ∂Q/∂y = 0
        - Reversible: P(-x,y) = P(x,y), Q(-x,y) = -Q(x,y)
        - Z2 symmetric: P(-x,-y) = -P(x,y), Q(-x,-y) = -Q(x,y)
        - SO2 symmetric: System is equivariant under rotations
        
        Returns:
            Set of detected properties
        """
        self.properties.clear()
        
        self.properties.add(SystemProperty.POLYNOMIAL)
        
        if self._is_hamiltonian():
            self.properties.add(SystemProperty.HAMILTONIAN)
        
        if self._is_reversible():
            self.properties.add(SystemProperty.REVERSIBLE)
        
        if self._is_z2_symmetric():
            self.properties.add(SystemProperty.Z2_SYMMETRIC)
        
        if self._is_so2_symmetric():
            self.properties.add(SystemProperty.SO2_SYMMETRIC)
        
        return self.properties
    
    def _is_hamiltonian(self) -> bool:
        """Check if system is Hamiltonian (divergence-free)."""
        div = diff(self.P, self.x) + diff(self.Q, self.y)
        return simplify(div) == 0
    
    def _is_reversible(self) -> bool:
        """Check if system is reversible under (x,y,t) -> (-x,y,-t)."""
        P_reversed = self.P.subs(self.x, -self.x)
        Q_reversed = self.Q.subs(self.x, -self.x)
        
        cond1 = simplify(P_reversed - self.P) == 0
        cond2 = simplify(Q_reversed + self.Q) == 0
        
        return cond1 and cond2
    
    def _is_z2_symmetric(self) -> bool:
        """Check for Z2 symmetry: (x,y) -> (-x,-y)."""
        P_sym = self.P.subs([(self.x, -self.x), (self.y, -self.y)])
        Q_sym = self.Q.subs([(self.x, -self.x), (self.y, -self.y)])
        
        cond1 = simplify(P_sym + self.P) == 0
        cond2 = simplify(Q_sym + self.Q) == 0
        
        return cond1 and cond2
    
    def _is_so2_symmetric(self) -> bool:
        """
        Check for SO(2) symmetry (rotational invariance).
        This requires P and Q to depend only on r² = x² + y².
        """
        r_sq = self.x**2 + self.y**2
        r = symbols('r', positive=True)
        
        try:
            P_polar = self.P.subs([(self.x**2 + self.y**2, r**2)])
            Q_polar = self.Q.subs([(self.x**2 + self.y**2, r**2)])
            
            has_mixed = False
            for term in Add.make_args(expand(self.P)) + Add.make_args(expand(self.Q)):
                poly = Poly(term, self.x, self.y)
                for monom in poly.monoms():
                    if monom[0] != monom[1] and monom[0] > 0 and monom[1] > 0:
                        has_mixed = True
                        break
            
            return not has_mixed
        except:
            return False
    
    def get_full_system(self) -> Tuple[Expr, Expr]:
        """
        Get the full system equations including linear part.
        
        Returns:
            Tuple (ẋ, ẏ) = (-y + P, x + Q)
        """
        x_dot = -self.y + self.P
        y_dot = self.x + self.Q
        return (x_dot, y_dot)
    
    def to_complex_form(self) -> Tuple[Expr, Symbol, Symbol]:
        """
        Transform system to complex coordinates z = x + iy.
        
        Returns:
            Tuple (F, z, z_bar) where ż = iz + F(z, z̄)
        """
        z = symbols('z')
        z_bar = symbols('z_bar')
        
        x_expr = (z + z_bar) / 2
        y_expr = (z - z_bar) / (2 * I)
        
        P_complex = self.P.subs([(self.x, x_expr), (self.y, y_expr)])
        Q_complex = self.Q.subs([(self.x, x_expr), (self.y, y_expr)])
        
        F = (P_complex + I * Q_complex) / 2
        F = expand(F)
        
        return (F, z, z_bar)
    
    def truncate_to_order(self, max_order: int) -> 'SystemDefinition':
        """
        Create a new system truncated to given polynomial order.
        
        Args:
            max_order: Maximum total degree to keep
            
        Returns:
            New SystemDefinition with truncated P and Q
        """
        def truncate_expr(expr: Expr) -> Expr:
            if expr == 0:
                return S.Zero
            
            result = S.Zero
            for term in Add.make_args(expand(expr)):
                try:
                    poly = Poly(term, self.x, self.y)
                    for monom, coeff in zip(poly.monoms(), poly.coeffs()):
                        if sum(monom) <= max_order:
                            result += coeff * self.x**monom[0] * self.y**monom[1]
                except:
                    result += term
            return result
        
        P_trunc = truncate_expr(self.P)
        Q_trunc = truncate_expr(self.Q)
        
        return SystemDefinition(
            P=P_trunc,
            Q=Q_trunc,
            params=self.params.copy(),
            x=self.x,
            y=self.y,
            auto_validate=False
        )
    
    def get_hash_key(self) -> str:
        """
        Generate a unique hash key for this system definition.
        Used for caching purposes.
        
        Returns:
            String representation suitable for hashing
        """
        components = [
            str(self.P),
            str(self.Q),
            str(sorted([str(p) for p in self.params])),
            str(self.x),
            str(self.y)
        ]
        return "|".join(components)
    
    def __repr__(self) -> str:
        return (f"SystemDefinition(\n"
                f"  ẋ = -y + {self.P}\n"
                f"  ẏ = x + {self.Q}\n"
                f"  params = {self.params}\n"
                f"  properties = {[p.name for p in self.properties]}\n"
                f")")
    
    def __str__(self) -> str:
        x_dot, y_dot = self.get_full_system()
        return f"ẋ = {x_dot}\nẏ = {y_dot}"
