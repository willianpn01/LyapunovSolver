"""
Symbolic Solver for Equilibrium Points
Finds equilibrium points of 2D dynamical systems symbolically.
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import logging
import time

import sympy as sp
from sympy import (
    Symbol, Expr, symbols, solve, simplify, groebner,
    Poly, sqrt, I, S, Eq, And, Or, Ne
)

logger = logging.getLogger(__name__)


@dataclass
class SymbolicSolution:
    """Represents a symbolic equilibrium solution."""
    
    x: Expr
    y: Expr
    is_parametric: bool
    valid_domain: Optional[str] = None
    method: str = "direct"
    
    def to_dict(self) -> Dict[Symbol, Expr]:
        """Convert to dictionary format."""
        x_sym, y_sym = symbols('x y')
        return {x_sym: self.x, y_sym: self.y}
    
    def __repr__(self):
        domain_str = f" (valid: {self.valid_domain})" if self.valid_domain else ""
        return f"SymbolicSolution(x={self.x}, y={self.y}{domain_str})"


class SymbolicSolver:
    """
    Symbolic solver for finding equilibrium points.
    
    Solves the system:
        f(x, y, μ) = 0
        g(x, y, μ) = 0
    
    Uses multiple strategies with fallback:
    1. Direct solve
    2. Groebner basis (for polynomials)
    3. Elimination method
    """
    
    def __init__(
        self,
        f: Expr,
        g: Expr,
        x: Symbol = None,
        y: Symbol = None,
        params: List[Symbol] = None,
        timeout: float = 30.0
    ):
        """
        Initialize the solver.
        
        Args:
            f: Right-hand side of ẋ = f(x, y, μ)
            g: Right-hand side of ẏ = g(x, y, μ)
            x, y: State variable symbols
            params: List of parameter symbols
            timeout: Maximum time for symbolic solve (seconds)
        """
        self.f = f
        self.g = g
        self.x = x if x is not None else symbols('x')
        self.y = y if y is not None else symbols('y')
        self.params = params if params is not None else []
        self.timeout = timeout
        
        # Check if system is polynomial
        self._is_polynomial = self._check_polynomial()
        
        logger.debug(f"SymbolicSolver initialized, polynomial: {self._is_polynomial}")
    
    def _check_polynomial(self) -> bool:
        """Check if f and g are polynomials in x and y."""
        try:
            Poly(self.f, self.x, self.y)
            Poly(self.g, self.x, self.y)
            return True
        except:
            return False
    
    def solve(self) -> List[SymbolicSolution]:
        """
        Find all equilibrium points symbolically.
        
        Returns:
            List of SymbolicSolution objects
        """
        start_time = time.time()
        
        # Try direct solve first
        logger.debug("Attempting direct symbolic solve...")
        solutions = self._solve_with_timeout(self._solve_direct)
        
        if solutions is not None:
            elapsed = time.time() - start_time
            logger.info(f"Direct solve succeeded in {elapsed:.2f}s, found {len(solutions)} solutions")
            return solutions
        
        # Try Groebner if polynomial
        if self._is_polynomial:
            logger.debug("Direct solve failed, attempting Groebner basis...")
            solutions = self._solve_with_timeout(self._solve_groebner)
            
            if solutions is not None:
                elapsed = time.time() - start_time
                logger.info(f"Groebner solve succeeded in {elapsed:.2f}s, found {len(solutions)} solutions")
                return solutions
        
        # Try elimination method
        logger.debug("Attempting elimination method...")
        solutions = self._solve_with_timeout(self._solve_elimination)
        
        if solutions is not None:
            elapsed = time.time() - start_time
            logger.info(f"Elimination solve succeeded in {elapsed:.2f}s, found {len(solutions)} solutions")
            return solutions
        
        logger.warning("All symbolic solve methods failed")
        return []
    
    def _solve_with_timeout(self, solve_func) -> Optional[List[SymbolicSolution]]:
        """Execute solver function with timeout."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(solve_func)
            try:
                return future.result(timeout=self.timeout)
            except FuturesTimeoutError:
                logger.warning(f"Solver timed out after {self.timeout}s")
                return None
            except Exception as e:
                logger.warning(f"Solver failed with error: {e}")
                return None
    
    def _solve_direct(self) -> List[SymbolicSolution]:
        """Direct symbolic solve using SymPy."""
        try:
            solutions = solve([self.f, self.g], [self.x, self.y], dict=True)
            return self._process_solutions(solutions, method="direct")
        except Exception as e:
            logger.debug(f"Direct solve exception: {e}")
            raise
    
    def _solve_groebner(self) -> List[SymbolicSolution]:
        """Solve using Groebner basis for polynomial systems."""
        try:
            # Compute Groebner basis
            G = groebner([self.f, self.g], self.x, self.y, order='lex')
            
            # Solve the reduced system
            solutions = solve(list(G), [self.x, self.y], dict=True)
            return self._process_solutions(solutions, method="groebner")
        except Exception as e:
            logger.debug(f"Groebner solve exception: {e}")
            raise
    
    def _solve_elimination(self) -> List[SymbolicSolution]:
        """Solve by variable elimination."""
        solutions = []
        
        # Try eliminating y first
        try:
            y_solutions = solve(self.g, self.y)
            for y_expr in y_solutions:
                f_sub = self.f.subs(self.y, y_expr)
                x_solutions = solve(f_sub, self.x)
                for x_sol in x_solutions:
                    y_sol = y_expr.subs(self.x, x_sol)
                    solutions.append({self.x: x_sol, self.y: simplify(y_sol)})
        except:
            pass
        
        # Try eliminating x if no solutions found
        if not solutions:
            try:
                x_solutions = solve(self.f, self.x)
                for x_expr in x_solutions:
                    g_sub = self.g.subs(self.x, x_expr)
                    y_solutions = solve(g_sub, self.y)
                    for y_sol in y_solutions:
                        x_sol = x_expr.subs(self.y, y_sol)
                        solutions.append({self.x: simplify(x_sol), self.y: y_sol})
            except:
                pass
        
        if solutions:
            return self._process_solutions(solutions, method="elimination")
        raise ValueError("Elimination method found no solutions")
    
    def _process_solutions(
        self,
        solutions: List[Dict],
        method: str
    ) -> List[SymbolicSolution]:
        """Process raw solutions into SymbolicSolution objects."""
        processed = []
        seen = set()  # Avoid duplicates
        
        for sol in solutions:
            x_val = sol.get(self.x, S.Zero)
            y_val = sol.get(self.y, S.Zero)
            
            # Simplify
            x_val = simplify(x_val)
            y_val = simplify(y_val)
            
            # Check for duplicates
            key = (str(x_val), str(y_val))
            if key in seen:
                continue
            seen.add(key)
            
            # Check if solution depends on parameters
            is_parametric = any(
                x_val.has(p) or y_val.has(p) 
                for p in self.params
            )
            
            # Compute valid domain
            valid_domain = self._compute_domain(x_val, y_val) if is_parametric else None
            
            processed.append(SymbolicSolution(
                x=x_val,
                y=y_val,
                is_parametric=is_parametric,
                valid_domain=valid_domain,
                method=method
            ))
        
        return processed
    
    def _compute_domain(self, x_val: Expr, y_val: Expr) -> Optional[str]:
        """
        Compute the valid parameter domain for a solution.
        
        Examples:
            x = sqrt(mu) → "μ ≥ 0"
            x = 1/mu → "μ ≠ 0"
        """
        constraints = []
        
        for param in self.params:
            # Check for square roots
            if x_val.has(sqrt) or y_val.has(sqrt):
                # Find arguments of sqrt
                for atom in (x_val.atoms() | y_val.atoms()):
                    if hasattr(atom, 'args') and len(atom.args) > 0:
                        arg = atom.args[0]
                        if arg.has(param):
                            constraints.append(f"{param} ≥ 0")
            
            # Check for divisions
            try:
                # Get denominators
                x_numer, x_denom = x_val.as_numer_denom()
                y_numer, y_denom = y_val.as_numer_denom()
                
                if x_denom.has(param):
                    constraints.append(f"{param} ≠ 0")
                if y_denom.has(param):
                    constraints.append(f"{param} ≠ 0")
            except:
                pass
        
        # Remove duplicates
        constraints = list(set(constraints))
        
        return ", ".join(constraints) if constraints else None
    
    def solve_for_parameter(
        self,
        param_values: Dict[Symbol, float]
    ) -> List[SymbolicSolution]:
        """
        Solve for specific parameter values.
        
        Args:
            param_values: Dict mapping parameters to numerical values
        
        Returns:
            List of solutions with parameters substituted
        """
        f_sub = self.f.subs(param_values)
        g_sub = self.g.subs(param_values)
        
        # Create new solver with substituted expressions
        solver = SymbolicSolver(
            f_sub, g_sub,
            self.x, self.y,
            params=[],  # No parameters left
            timeout=self.timeout
        )
        
        return solver.solve()
    
    def get_trivial_solutions(self) -> List[SymbolicSolution]:
        """
        Quick check for common trivial solutions like (0, 0).
        
        This is fast and can be used before full solve.
        """
        trivial = []
        
        # Check origin
        f_origin = self.f.subs({self.x: 0, self.y: 0})
        g_origin = self.g.subs({self.x: 0, self.y: 0})
        
        if simplify(f_origin) == 0 and simplify(g_origin) == 0:
            trivial.append(SymbolicSolution(
                x=S.Zero,
                y=S.Zero,
                is_parametric=False,
                method="trivial"
            ))
        
        return trivial
