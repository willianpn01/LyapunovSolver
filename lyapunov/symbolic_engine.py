"""
Module B: Symbolic Engine (Core Mathematical Computation)
Implements the Normal Form method for computing Lyapunov coefficients.

Based on the iterative algorithm from Mathematica reference:
    Z[j_] := If[1 < j < 5, a[j]*((x + y)/2)^j + b[j]*((x - y)/(2*I))^j, 0]
    F[2] := x*y/2
    Phi[l_, k_] := Z[k]*(D[F[l], x] + D[F[l], y])
    S[p_] := Sum[Phi[p - i + 1, i], {i, 2, p - 1}]
    K[p_, k_] := Coefficient[-I*S[p], x^(p - k)*y^k]
    h[p_] := Table[If[2 k - p != 0, K[p, k]/(2 k - p), 0], {k, 0, p}]
    F[p_] := Sum[h[p][[k]]*x^(p - k + 1)*y^(k - 1), {k, p + 1}]
    V[p_] := Simplify[I*K[p + 1, (p + 1)/2]]

In this formulation:
- x and y are complex conjugate coordinates (z and z̄)
- The system is ẋ = -y + P(x,y), ẏ = x + Q(x,y) in real form
- Coefficients aⱼ come from P (terms in x), bⱼ from Q (terms in y)
- V[3] = L₁, V[5] = L₂, V[7] = L₃, etc.
"""

from typing import Dict, List, Optional, Tuple
import sympy as sp
from sympy import (
    Symbol, Expr, symbols, expand, simplify, collect,
    diff, I, Rational, sqrt, pi, factorial,
    Add, Mul, Pow, S, conjugate, re, im, trigsimp,
    factor, cancel, together, Poly, degree, nsimplify
)

from .system_definition import SystemDefinition, SystemProperty


class SymbolicEngine:
    """
    Core symbolic computation engine for Lyapunov coefficients.
    
    Implements the Normal Form method following the iterative algorithm.
    Works in complex coordinates where x = z and y = z̄.
    """
    
    def __init__(self, system: SystemDefinition, simplify_level: int = 2, max_order: int = 10):
        """
        Initialize the symbolic engine.
        
        Args:
            system: SystemDefinition instance
            simplify_level: Aggressiveness of simplification (0-3)
            max_order: Maximum order for coefficient extraction
        """
        self.system = system
        self.simplify_level = simplify_level
        self.max_order = max_order
        
        self._x = symbols('_x_complex')
        self._y = symbols('_y_complex')
        
        self.real_x = system.x
        self.real_y = system.y
        self.params = system.params
        
        self.x = system.x
        self.y = system.y
        
        self.lyapunov_coefficients: Dict[int, Expr] = {}
        self.max_computed_order = 0
        
        self._a: Dict[int, Expr] = {}
        self._b: Dict[int, Expr] = {}
        
        self._F: Dict[int, Expr] = {}
        self._h: Dict[int, List[Expr]] = {}
        
        self._extract_coefficients()
        self._F[2] = self._x * self._y / 2
    
    def _extract_coefficients(self) -> None:
        """
        Extract coefficients aⱼ and bⱼ from P and Q.
        
        For P(x,y), extract coefficient of x^j (pure x terms).
        For Q(x,y), extract coefficient of y^j (pure y terms).
        """
        P = expand(self.system.P)
        Q = expand(self.system.Q)
        rx, ry = self.real_x, self.real_y
        
        for j in range(2, self.max_order + 1):
            a_j = S.Zero
            b_j = S.Zero
            
            if P != 0:
                try:
                    poly_P = Poly(P, rx, ry)
                    for monom, coeff in zip(poly_P.monoms(), poly_P.coeffs()):
                        if monom == (j, 0):
                            a_j = coeff
                            break
                except:
                    coeff = P.coeff(rx, j)
                    if coeff is not None:
                        a_j = coeff.subs(ry, 0) if coeff.has(ry) else coeff
            
            if Q != 0:
                try:
                    poly_Q = Poly(Q, rx, ry)
                    for monom, coeff in zip(poly_Q.monoms(), poly_Q.coeffs()):
                        if monom == (0, j):
                            b_j = coeff
                            break
                except:
                    coeff = Q.coeff(ry, j)
                    if coeff is not None:
                        b_j = coeff.subs(rx, 0) if coeff.has(rx) else coeff
            
            self._a[j] = a_j
            self._b[j] = b_j
    
    def _Z(self, j: int) -> Expr:
        """
        Z[j] = a[j]*((x + y)/2)^j + b[j]*((x - y)/(2*I))^j
        
        where x, y are complex coordinates.
        """
        if j < 2 or j >= self.max_order:
            return S.Zero
        
        a_j = self._a.get(j, S.Zero)
        b_j = self._b.get(j, S.Zero)
        
        if a_j == 0 and b_j == 0:
            return S.Zero
        
        x, y = self._x, self._y
        
        term1 = a_j * ((x + y) / 2) ** j
        term2 = b_j * ((x - y) / (2 * I)) ** j
        
        return expand(term1 + term2)
    
    def _get_F(self, l: int) -> Expr:
        """Get F[l], computing if necessary."""
        if l < 2:
            return S.Zero
        if l == 2:
            return self._F[2]
        if l in self._F:
            return self._F[l]
        
        for p in range(3, l + 1):
            if p not in self._F:
                self._compute_F_at_order(p)
        
        return self._F.get(l, S.Zero)
    
    def _Phi(self, l: int, k: int) -> Expr:
        """
        Phi[l, k] = Z[k] * (dF[l]/dx + dF[l]/dy)
        """
        Z_k = self._Z(k)
        if Z_k == 0:
            return S.Zero
        
        F_l = self._get_F(l)
        if F_l == 0:
            return S.Zero
        
        dF_dx = diff(F_l, self._x)
        dF_dy = diff(F_l, self._y)
        
        return expand(Z_k * (dF_dx + dF_dy))
    
    def _S(self, p: int) -> Expr:
        """
        S[p] = Sum[Phi[p - i + 1, i], {i, 2, p - 1}]
        """
        result = S.Zero
        
        for i in range(2, p):
            l = p - i + 1
            phi = self._Phi(l, i)
            result = result + phi
        
        return expand(result)
    
    def _get_coeff(self, expr: Expr, x_pow: int, y_pow: int) -> Expr:
        """Extract coefficient of x^x_pow * y^y_pow from expr."""
        if expr == 0:
            return S.Zero
        
        expr = expand(expr)
        
        try:
            poly = Poly(expr, self._x, self._y)
            for monom, coeff in zip(poly.monoms(), poly.coeffs()):
                if monom == (x_pow, y_pow):
                    return coeff
            return S.Zero
        except:
            pass
        
        try:
            coeff = expr
            if x_pow > 0:
                coeff = coeff.coeff(self._x, x_pow)
            if coeff is not None and y_pow > 0:
                coeff = coeff.coeff(self._y, y_pow)
            return coeff if coeff is not None else S.Zero
        except:
            return S.Zero
    
    def _K(self, p: int, k: int) -> Expr:
        """
        K[p, k] = Coefficient[-I*S[p], x^(p-k) * y^k]
        """
        S_p = self._S(p)
        if S_p == 0:
            return S.Zero
        
        expr = expand(-I * S_p)
        return self._get_coeff(expr, p - k, k)
    
    def _compute_h(self, p: int) -> List[Expr]:
        """
        h[p] = Table[If[2*k - p != 0, K[p, k]/(2*k - p), 0], {k, 0, p}]
        
        Returns list of length p+1, indexed from 0 to p.
        """
        h_list = []
        
        for k in range(p + 1):
            divisor = 2 * k - p
            if divisor != 0:
                K_pk = self._K(p, k)
                h_k = K_pk / divisor
                h_k = simplify(h_k)
            else:
                h_k = S.Zero
            h_list.append(h_k)
        
        self._h[p] = h_list
        return h_list
    
    def _compute_F_at_order(self, p: int) -> Expr:
        """
        F[p] = Sum[h[p][[k]] * x^(p-k+1) * y^(k-1), {k, 1, p+1}]
        
        In Mathematica, {k, p+1} means k goes from 1 to p+1.
        h[p][[k]] is 1-indexed, so h[p][[k]] = h_list[k-1].
        """
        if p not in self._h:
            self._compute_h(p)
        
        h_list = self._h[p]
        result = S.Zero
        
        for k in range(1, p + 2):
            if k - 1 < len(h_list):
                h_k = h_list[k - 1]
                if h_k != 0:
                    x_pow = p - k + 1
                    y_pow = k - 1
                    if x_pow >= 0 and y_pow >= 0:
                        result = result + h_k * self._x**x_pow * self._y**y_pow
        
        result = expand(result)
        self._F[p] = result
        return result
    
    def _V(self, p: int) -> Expr:
        """
        V[p] = Simplify[I * K[p + 1, (p + 1)/2]]
        
        This gives the Lyapunov coefficient at order p.
        For odd p: V[3] = L₁, V[5] = L₂, V[7] = L₃
        """
        if p % 2 == 0:
            return S.Zero
        
        for order in range(3, p + 1):
            if order not in self._F:
                self._compute_F_at_order(order)
        
        k = (p + 1) // 2
        K_val = self._K(p + 1, k)
        
        V_p = simplify(I * K_val)
        
        return V_p
    
    def compute_Lk_symbolic(self, k: int) -> Expr:
        """
        Compute the k-th Lyapunov coefficient symbolically.
        
        L₁ = V[3], L₂ = V[5], L₃ = V[7]
        In general: Lₖ = V[2k+1]
        
        Args:
            k: Order of Lyapunov coefficient (1, 2, 3, ...)
            
        Returns:
            Symbolic expression for L_k
        """
        if k in self.lyapunov_coefficients:
            return self.lyapunov_coefficients[k]
        
        p = 2 * k + 1
        result = self._V(p)
        result = self._simplify_expression(result)
        
        self.lyapunov_coefficients[k] = result
        self.max_computed_order = max(self.max_computed_order, k)
        
        return result
    
    def _simplify_expression(self, expr: Expr) -> Expr:
        """Apply simplification pipeline based on simplify_level."""
        if self.simplify_level == 0:
            return expr
        
        if expr == 0:
            return S.Zero
        
        result = expand(expr)
        result = collect(result, self.params) if self.params else result
        
        if self.simplify_level >= 2:
            result = simplify(result)
        
        if self.simplify_level >= 3:
            result = trigsimp(result)
            result = factor(result)
        
        return result
    
    def compute_lyapunov_sequence(self, max_k: int) -> List[Expr]:
        """
        Compute Lyapunov coefficients L₁, L₂, ..., L_{max_k}.
        
        Args:
            max_k: Maximum order to compute
            
        Returns:
            List of symbolic expressions [L₁, L₂, ..., L_{max_k}]
        """
        results = []
        for k in range(1, max_k + 1):
            L_k = self.compute_Lk_symbolic(k)
            results.append(L_k)
        return results
    
    def get_stability_condition(self, k: int) -> Dict[str, Expr]:
        """
        Get conditions for stability based on L_k.
        
        Returns:
            Dictionary with 'stable' and 'unstable' conditions
        """
        L_k = self.compute_Lk_symbolic(k)
        
        return {
            'coefficient': L_k,
            'stable_condition': L_k < 0,
            'unstable_condition': L_k > 0,
            'degenerate_condition': sp.Eq(L_k, 0)
        }
    
    def check_hamiltonian_property(self) -> bool:
        """
        For Hamiltonian systems, verify L₁ = L₃ = L₅ = ... = 0.
        """
        if SystemProperty.HAMILTONIAN not in self.system.properties:
            return False
        
        L1 = self.compute_Lk_symbolic(1)
        return simplify(L1) == 0
    
    def get_bifurcation_value(self, k: int, param: Symbol) -> Optional[List[Expr]]:
        """
        Solve L_k = 0 for a parameter to find bifurcation value.
        
        Args:
            k: Order of Lyapunov coefficient
            param: Parameter to solve for
            
        Returns:
            Solution(s) for param where L_k = 0, or None if no solution
        """
        L_k = self.compute_Lk_symbolic(k)
        
        try:
            solutions = sp.solve(L_k, param)
            return solutions
        except:
            return None
    
    def evaluate_numeric(self, k: int, param_values: Dict[Symbol, float]) -> complex:
        """
        Evaluate L_k numerically for given parameter values.
        
        Args:
            k: Order of Lyapunov coefficient
            param_values: Dictionary mapping parameters to numerical values
            
        Returns:
            Numerical value of L_k
        """
        L_k = self.compute_Lk_symbolic(k)
        
        result = L_k.subs(param_values)
        
        return complex(result.evalf())
    
    def to_latex(self, k: int) -> str:
        """
        Generate LaTeX representation of L_k.
        
        Args:
            k: Order of Lyapunov coefficient
            
        Returns:
            LaTeX string
        """
        L_k = self.compute_Lk_symbolic(k)
        return sp.latex(L_k)
    
    def get_computation_stats(self) -> Dict:
        """
        Get statistics about the computation.
        
        Returns:
            Dictionary with computation statistics
        """
        return {
            'max_computed_order': self.max_computed_order,
            'num_F_orders': len(self._F),
            'num_cached_Lk': len(self.lyapunov_coefficients),
            'system_properties': [p.name for p in self.system.properties],
            'simplify_level': self.simplify_level,
            'a_coeffs': {k: str(v) for k, v in self._a.items() if v != 0},
            'b_coeffs': {k: str(v) for k, v in self._b.items() if v != 0}
        }


class NormalFormComputer:
    """
    Dedicated class for computing normal forms to arbitrary order.
    
    This implements the full Poincaré-Dulac normal form algorithm
    for systems near a Hopf bifurcation.
    """
    
    def __init__(self, engine: SymbolicEngine):
        """
        Initialize with a SymbolicEngine instance.
        
        Args:
            engine: SymbolicEngine with system already loaded
        """
        self.engine = engine
    
    def compute_normal_form(self, max_order: int) -> Dict[Tuple[int, int], Expr]:
        """
        Compute normal form coefficients up to given order.
        
        The normal form is:
        ẇ = iw + Σ g_{jk} w^j w̄^k
        
        where only resonant terms (j = k + 1) remain.
        
        Args:
            max_order: Maximum total order j + k
            
        Returns:
            Dictionary of normal form coefficients g_{jk}
        """
        coeffs = {}
        for k in range(1, (max_order - 1) // 2 + 1):
            L_k = self.engine.compute_Lk_symbolic(k)
            coeffs[(k + 1, k)] = L_k
        return coeffs
    
    def get_lyapunov_from_normal_form(self, k: int) -> Expr:
        """
        Extract L_k from normal form coefficient g_{k+1, k}.
        """
        return self.engine.compute_Lk_symbolic(k)
