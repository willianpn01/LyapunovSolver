"""
Canonical Transformer Module
Transforms a general 2D system to canonical Hopf normal form.

The canonical form for Hopf bifurcation analysis is:
    ẋ = -y + P(x, y)
    ẏ = x + Q(x, y)

where P and Q contain only nonlinear terms (degree ≥ 2).
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

import sympy as sp
from sympy import (
    Symbol, Expr, symbols, Matrix, simplify, expand,
    sqrt, I, S, Poly, degree, collect, diff
)

logger = logging.getLogger(__name__)


@dataclass
class CanonicalForm:
    """Represents a system in canonical Hopf form."""
    
    P: Expr  # Nonlinear part of ẋ = -ωy + P(x,y)
    Q: Expr  # Nonlinear part of ẏ = ωx + Q(x,y)
    omega: Expr  # Natural frequency ω
    x: Symbol
    y: Symbol
    params: List[Symbol]
    
    # Transformation info
    x0: Expr  # Original equilibrium x-coordinate
    y0: Expr  # Original equilibrium y-coordinate
    transformation_matrix: Optional[Matrix] = None
    
    def get_full_system(self) -> Tuple[Expr, Expr]:
        """Return the full canonical system equations."""
        return (
            -self.omega * self.y + self.P,
            self.omega * self.x + self.Q
        )
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'P': str(self.P),
            'Q': str(self.Q),
            'omega': str(self.omega),
            'equilibrium': (str(self.x0), str(self.y0))
        }


class CanonicalTransformer:
    """
    Transforms a general 2D dynamical system to canonical Hopf form.
    
    Given a system:
        ẋ = f(x, y, μ)
        ẏ = g(x, y, μ)
    
    with a Hopf candidate equilibrium at (x₀, y₀), this class:
    1. Translates to origin: (x, y) → (x - x₀, y - y₀)
    2. Linearizes and extracts the frequency ω
    3. Applies linear transformation to get canonical form
    4. Extracts nonlinear parts P and Q
    
    The result is:
        ẋ = -ωy + P(x, y)
        ẏ = ωx + Q(x, y)
    """
    
    def __init__(
        self,
        f: Expr,
        g: Expr,
        x: Symbol = None,
        y: Symbol = None,
        params: List[Symbol] = None
    ):
        """
        Initialize the transformer.
        
        Args:
            f: Right-hand side of ẋ = f(x, y, μ)
            g: Right-hand side of ẏ = g(x, y, μ)
            x, y: State variable symbols
            params: List of parameter symbols
        """
        self.f = f
        self.g = g
        self.x = x if x is not None else symbols('x')
        self.y = y if y is not None else symbols('y')
        self.params = params if params is not None else []
        
        # Compute Jacobian
        self._jacobian = Matrix([
            [diff(f, self.x), diff(f, self.y)],
            [diff(g, self.x), diff(g, self.y)]
        ])
        
        logger.debug("CanonicalTransformer initialized")
    
    def transform(
        self,
        equilibrium: Dict[Symbol, Expr],
        param_values: Dict[Symbol, float] = None
    ) -> CanonicalForm:
        """
        Transform the system to canonical form at given equilibrium.
        
        Args:
            equilibrium: Dict with {x: x0, y: y0} coordinates
            param_values: Numerical values for parameters (optional)
        
        Returns:
            CanonicalForm object with P, Q, and ω
        """
        x0 = equilibrium.get(self.x, S.Zero)
        y0 = equilibrium.get(self.y, S.Zero)
        
        logger.info(f"Transforming system at equilibrium ({x0}, {y0})")
        
        # Step 1: Translate to origin
        f_translated, g_translated = self._translate_to_origin(x0, y0)
        
        # Step 2: Compute Jacobian at origin (of translated system)
        J = self._compute_jacobian_at_origin(f_translated, g_translated, param_values)
        
        # Step 3: Extract frequency ω from eigenvalues
        omega = self._extract_frequency(J, param_values)
        
        if omega is None or omega == 0:
            logger.warning("Could not extract valid frequency from Jacobian")
            # Return with omega = 1 as fallback
            omega = S.One
        
        # Step 4: Transform to canonical form
        P, Q, T = self._transform_to_canonical(
            f_translated, g_translated, J, omega, param_values
        )
        
        return CanonicalForm(
            P=P,
            Q=Q,
            omega=omega,
            x=self.x,
            y=self.y,
            params=self.params,
            x0=x0,
            y0=y0,
            transformation_matrix=T
        )
    
    def _translate_to_origin(
        self,
        x0: Expr,
        y0: Expr
    ) -> Tuple[Expr, Expr]:
        """Translate the system so equilibrium is at origin."""
        # New coordinates: x_new = x - x0, y_new = y - y0
        # So x = x_new + x0, y = y_new + y0
        
        f_trans = self.f.subs({self.x: self.x + x0, self.y: self.y + y0})
        g_trans = self.g.subs({self.x: self.x + x0, self.y: self.y + y0})
        
        f_trans = expand(f_trans)
        g_trans = expand(g_trans)
        
        logger.debug(f"Translated system: f={f_trans}, g={g_trans}")
        
        return f_trans, g_trans
    
    def _compute_jacobian_at_origin(
        self,
        f: Expr,
        g: Expr,
        param_values: Dict[Symbol, float] = None
    ) -> Matrix:
        """Compute Jacobian at origin."""
        df_dx = diff(f, self.x).subs({self.x: 0, self.y: 0})
        df_dy = diff(f, self.y).subs({self.x: 0, self.y: 0})
        dg_dx = diff(g, self.x).subs({self.x: 0, self.y: 0})
        dg_dy = diff(g, self.y).subs({self.x: 0, self.y: 0})
        
        J = Matrix([
            [df_dx, df_dy],
            [dg_dx, dg_dy]
        ])
        
        if param_values:
            J = J.subs(param_values)
        
        J = simplify(J)
        logger.debug(f"Jacobian at origin: {J}")
        
        return J
    
    def _extract_frequency(
        self,
        J: Matrix,
        param_values: Dict[Symbol, float] = None
    ) -> Expr:
        """
        Extract the natural frequency ω from Jacobian eigenvalues.
        
        For a Hopf point, eigenvalues are ±iω.
        """
        try:
            eigenvals = J.eigenvals()
            
            for eig, _ in eigenvals.items():
                eig = simplify(eig)
                
                # Try to evaluate numerically
                try:
                    eig_val = complex(eig.evalf())
                    if abs(eig_val.real) < 1e-8 and abs(eig_val.imag) > 1e-8:
                        return simplify(abs(sp.im(eig)))
                except:
                    pass
                
                # Symbolic: check if purely imaginary
                if eig.has(I):
                    # Extract coefficient of I
                    real_part = sp.re(eig)
                    imag_part = sp.im(eig)
                    
                    if simplify(real_part) == 0:
                        return simplify(abs(imag_part))
            
            # Fallback: use sqrt(det(J)) if trace = 0
            trace = J.trace()
            det = J.det()
            
            if simplify(trace) == 0 and simplify(det) > 0:
                return simplify(sqrt(det))
            
        except Exception as e:
            logger.warning(f"Could not extract frequency: {e}")
        
        return S.One  # Default
    
    def _transform_to_canonical(
        self,
        f: Expr,
        g: Expr,
        J: Matrix,
        omega: Expr,
        param_values: Dict[Symbol, float] = None
    ) -> Tuple[Expr, Expr, Matrix]:
        """
        Transform to canonical form ẋ = -ωy + P, ẏ = ωx + Q.
        
        Returns P, Q, and transformation matrix T.
        """
        # The canonical Jacobian is:
        # J_canonical = [[0, -ω], [ω, 0]]
        #
        # We need T such that T⁻¹ J T = J_canonical
        # Or equivalently: J = T J_canonical T⁻¹
        
        # For simplicity, we'll check if J is already in canonical form
        # or close to it, and extract P, Q accordingly
        
        a, b = J[0, 0], J[0, 1]
        c, d = J[1, 0], J[1, 1]
        
        # Check if already canonical (a=d=0, b=-ω, c=ω)
        is_canonical = (
            simplify(a) == 0 and
            simplify(d) == 0 and
            simplify(b + omega) == 0 and
            simplify(c - omega) == 0
        )
        
        if is_canonical:
            logger.debug("System is already in canonical form")
            T = sp.eye(2)
            P = self._extract_nonlinear(f)
            Q = self._extract_nonlinear(g)
        else:
            # Need to transform
            T = self._compute_transformation_matrix(J, omega)
            
            if T is not None:
                # Apply transformation
                # New variables: [u, v]^T = T^(-1) [x, y]^T
                # So [x, y]^T = T [u, v]^T
                
                T_inv = T.inv()
                
                # Substitute x = T[0,0]*u + T[0,1]*v, y = T[1,0]*u + T[1,1]*v
                u, v = symbols('_u _v')
                
                x_expr = T[0, 0] * u + T[0, 1] * v
                y_expr = T[1, 0] * u + T[1, 1] * v
                
                f_new = f.subs({self.x: x_expr, self.y: y_expr})
                g_new = g.subs({self.x: x_expr, self.y: y_expr})
                
                # Transform the vector field
                # [ẋ, ẏ]^T = [f, g]^T
                # [u̇, v̇]^T = T^(-1) [f, g]^T
                
                vec = Matrix([f_new, g_new])
                vec_new = T_inv * vec
                
                f_transformed = expand(vec_new[0])
                g_transformed = expand(vec_new[1])
                
                # Replace back to x, y
                f_transformed = f_transformed.subs({u: self.x, v: self.y})
                g_transformed = g_transformed.subs({u: self.x, v: self.y})
                
                # Extract nonlinear parts
                P = self._extract_nonlinear(f_transformed)
                Q = self._extract_nonlinear(g_transformed)
            else:
                logger.warning("Could not compute transformation matrix, using original")
                T = sp.eye(2)
                P = self._extract_nonlinear(f)
                Q = self._extract_nonlinear(g)
        
        P = simplify(expand(P))
        Q = simplify(expand(Q))
        
        logger.info(f"Canonical form: P={P}, Q={Q}, ω={omega}")
        
        return P, Q, T
    
    def _compute_transformation_matrix(
        self,
        J: Matrix,
        omega: Expr
    ) -> Optional[Matrix]:
        """
        Compute transformation matrix T to canonical form.
        
        For J with eigenvalues ±iω, find T such that
        T⁻¹ J T = [[0, -ω], [ω, 0]]
        """
        try:
            # Get eigenvectors
            eigenvecs = J.eigenvects()
            
            # Find eigenvector for iω
            for eig, mult, vecs in eigenvecs:
                if sp.im(eig) != 0:
                    v = vecs[0]  # Eigenvector (complex)
                    
                    # T = [Re(v), Im(v)]
                    v_real = Matrix([sp.re(v[0]), sp.re(v[1])])
                    v_imag = Matrix([sp.im(v[0]), sp.im(v[1])])
                    
                    T = Matrix([[v_real[0], v_imag[0]],
                               [v_real[1], v_imag[1]]])
                    
                    # Normalize
                    det_T = T.det()
                    if det_T != 0:
                        return simplify(T)
            
        except Exception as e:
            logger.warning(f"Could not compute transformation: {e}")
        
        return None
    
    def _extract_nonlinear(self, expr: Expr) -> Expr:
        """Extract nonlinear part (degree >= 2) from expression."""
        expr = expand(expr)
        
        # Remove constant and linear terms
        nonlinear = expr
        
        # Remove constant
        const = expr.subs({self.x: 0, self.y: 0})
        nonlinear = nonlinear - const
        
        # Remove linear terms
        linear_x = diff(nonlinear, self.x).subs({self.x: 0, self.y: 0}) * self.x
        linear_y = diff(nonlinear, self.y).subs({self.x: 0, self.y: 0}) * self.y
        
        nonlinear = nonlinear - linear_x - linear_y
        
        return simplify(expand(nonlinear))
    
    def transform_from_equilibrium_point(
        self,
        eq_point,  # EquilibriumPoint
        param_values: Dict[Symbol, float] = None
    ) -> CanonicalForm:
        """
        Transform using an EquilibriumPoint object.
        
        Args:
            eq_point: EquilibriumPoint from EquilibriumScanner
            param_values: Parameter values (uses eq_point's if not provided)
        
        Returns:
            CanonicalForm object
        """
        equilibrium = {self.x: eq_point.x, self.y: eq_point.y}
        
        if param_values is None:
            param_values = eq_point.parameter_values
        
        return self.transform(equilibrium, param_values)
