"""
Layer 4: High-Level API (Facade Pattern)
Provides a unified interface to the LyapunovSolver system.
"""

from typing import Dict, List, Optional, Union, Tuple, Any, Callable
from pathlib import Path

import sympy as sp
from sympy import Symbol, Expr, symbols, simplify, N

from .system_definition import SystemDefinition, SystemProperty, ValidationResult
from .symbolic_engine import SymbolicEngine, NormalFormComputer
from .cache_manager import CacheManager
from .code_generator import CCodeGenerator, NumbaGenerator, CompilationResult


class LyapunovSystem:
    """
    High-level facade for Lyapunov stability analysis.
    
    This class provides a unified interface to:
    - Define and validate dynamical systems
    - Compute Lyapunov coefficients symbolically
    - Evaluate coefficients numerically (Python, C, or Numba)
    - Visualize results and export to LaTeX
    
    Example:
        >>> from lyapunov import LyapunovSystem
        >>> from sympy import symbols
        >>> 
        >>> x, y, mu = symbols('x y mu')
        >>> P = mu * x - x**3
        >>> Q = -y**3
        >>> 
        >>> system = LyapunovSystem(P, Q, params=[mu])
        >>> L1 = system.compute_lyapunov(1)
        >>> print(f"L₁ = {L1}")
        >>> 
        >>> value = system.evaluate_lyapunov(1, {mu: 0.5})
        >>> print(f"L₁(μ=0.5) = {value}")
    
    Attributes:
        system_def: The underlying SystemDefinition
        engine: The SymbolicEngine for computations
        cache: The CacheManager for caching results
    """
    
    def __init__(
        self,
        P: Union[Expr, str],
        Q: Union[Expr, str],
        params: Optional[List[Symbol]] = None,
        x: Optional[Symbol] = None,
        y: Optional[Symbol] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        enable_cache: bool = True,
        simplify_level: int = 2,
        auto_compile: bool = False
    ):
        """
        Initialize a Lyapunov analysis system.
        
        Args:
            P: Nonlinear terms in ẋ = -y + P(x,y,μ)
            Q: Nonlinear terms in ẏ = x + Q(x,y,μ)
            params: List of parameter symbols
            x, y: State variable symbols (default: symbols('x y'))
            cache_dir: Directory for disk cache
            enable_cache: Whether to enable caching
            simplify_level: Simplification aggressiveness (0-3)
            auto_compile: Whether to auto-compile C code for evaluation
        """
        self._x = x if x is not None else symbols('x')
        self._y = y if y is not None else symbols('y')
        self._params = params if params is not None else []
        
        if isinstance(P, str):
            P = sp.sympify(P)
        if isinstance(Q, str):
            Q = sp.sympify(Q)
        
        self.system_def = SystemDefinition(
            P=P, Q=Q,
            params=self._params,
            x=self._x, y=self._y,
            auto_validate=True
        )
        
        self.cache = CacheManager(
            cache_dir=cache_dir,
            enable_disk_cache=enable_cache
        ) if enable_cache else None
        
        self.engine = SymbolicEngine(
            system=self.system_def,
            simplify_level=simplify_level
        )
        
        self._code_gen: Optional[CCodeGenerator] = None
        self._numba_gen: Optional[NumbaGenerator] = None
        self._compiled_evaluators: Dict[int, Callable] = {}
        
        self._auto_compile = auto_compile
        self._simplify_level = simplify_level
    
    @property
    def P(self) -> Expr:
        """Nonlinear part of x-equation."""
        return self.system_def.P
    
    @property
    def Q(self) -> Expr:
        """Nonlinear part of y-equation."""
        return self.system_def.Q
    
    @property
    def params(self) -> List[Symbol]:
        """System parameters."""
        return self._params
    
    @property
    def properties(self) -> set:
        """Detected system properties."""
        return self.system_def.properties
    
    def validate(self) -> ValidationResult:
        """
        Validate the system definition.
        
        Returns:
            ValidationResult with status and details
        """
        return self.system_def.validate()
    
    def compute_lyapunov(
        self,
        k: int,
        use_cache: bool = True,
        simplify_level: Optional[int] = None
    ) -> Expr:
        """
        Compute the k-th Lyapunov coefficient symbolically.
        
        Args:
            k: Order of coefficient (1, 2, 3, ...)
            use_cache: Whether to use cached results
            
        Returns:
            Symbolic expression for L_k
        """
        if use_cache and self.cache is not None:
            system_key = self.system_def.get_hash_key()
            cached = self.cache.get_cached_symbolic(system_key, k)
            if cached is not None:
                return cached
        
        result = self.engine.compute_Lk_symbolic(k)
        
        if use_cache and self.cache is not None:
            self.cache.save_symbolic(
                system_key=self.system_def.get_hash_key(),
                order=k,
                expression=result,
                metadata={
                    'system_P': str(self.P),
                    'system_Q': str(self.Q),
                    'params': [str(p) for p in self._params]
                }
            )
        
        return result
    
    def compute_lyapunov_sequence(
        self,
        max_k: int,
        stop_on_nonzero: bool = False
    ) -> Dict[int, Expr]:
        """
        Compute Lyapunov coefficients L₁ through L_{max_k}.
        
        Args:
            max_k: Maximum order to compute
            stop_on_nonzero: Stop when first nonzero coefficient found
            
        Returns:
            Dictionary mapping k to L_k
        """
        results = {}
        
        for k in range(1, max_k + 1):
            L_k = self.compute_lyapunov(k)
            results[k] = L_k
            
            if stop_on_nonzero and simplify(L_k) != 0:
                break
        
        return results
    
    def evaluate_lyapunov(
        self,
        k: int,
        param_values: Dict[Symbol, float],
        method: str = 'auto'
    ) -> float:
        """
        Evaluate L_k numerically for given parameter values.
        
        Args:
            k: Order of Lyapunov coefficient
            param_values: Dictionary mapping parameters to values
            method: Evaluation method ('auto', 'sympy', 'numpy', 'c', 'numba')
            
        Returns:
            Numerical value of L_k
        """
        if method == 'auto':
            if k in self._compiled_evaluators:
                method = 'compiled'
            elif self._auto_compile and self._code_gen is not None:
                method = 'c'
            else:
                method = 'sympy'
        
        if method == 'compiled' and k in self._compiled_evaluators:
            evaluator = self._compiled_evaluators[k]
            args = [param_values.get(p, 0.0) for p in self._params]
            return evaluator(*args)
        
        if method == 'c':
            return self._evaluate_c(k, param_values)
        
        if method == 'numba':
            return self._evaluate_numba(k, param_values)
        
        L_k = self.compute_lyapunov(k)
        result = L_k.subs(param_values)
        return float(N(result))
    
    def _evaluate_c(self, k: int, param_values: Dict[Symbol, float]) -> float:
        """Evaluate using compiled C code."""
        if self._code_gen is None:
            self._code_gen = CCodeGenerator(self.cache)
        
        if k not in self._compiled_evaluators:
            L_k = self.compute_lyapunov(k)
            param_names = [str(p) for p in self._params]
            
            c_code = self._code_gen.generate_c_function(
                L_k, param_names, f"eval_L{k}"
            )
            
            result = self._code_gen.compile_to_library(c_code, f"lyapunov_L{k}")
            
            if result.success:
                lib = self._code_gen.load_library(result.library_path)
                if lib is not None:
                    evaluator = self._code_gen.create_evaluator(
                        lib, f"eval_L{k}", len(self._params)
                    )
                    self._compiled_evaluators[k] = evaluator
        
        if k in self._compiled_evaluators:
            args = [param_values.get(p, 0.0) for p in self._params]
            return self._compiled_evaluators[k](*args)
        
        return self.evaluate_lyapunov(k, param_values, method='sympy')
    
    def _evaluate_numba(self, k: int, param_values: Dict[Symbol, float]) -> float:
        """Evaluate using Numba JIT."""
        if self._numba_gen is None:
            self._numba_gen = NumbaGenerator()
        
        L_k = self.compute_lyapunov(k)
        
        func = self._numba_gen.generate_numba_function(L_k, self._params)
        if func is not None:
            args = [param_values.get(p, 0.0) for p in self._params]
            return func(*args)
        
        return self.evaluate_lyapunov(k, param_values, method='sympy')
    
    def evaluate_batch(
        self,
        k: int,
        param_arrays: Dict[Symbol, list],
        method: str = 'numpy'
    ) -> list:
        """
        Evaluate L_k for multiple parameter combinations.
        
        Args:
            k: Order of Lyapunov coefficient
            param_arrays: Dictionary mapping parameters to arrays of values
            method: Evaluation method
            
        Returns:
            Array of L_k values
        """
        import numpy as np
        
        L_k = self.compute_lyapunov(k)
        
        from sympy.utilities.lambdify import lambdify
        numpy_func = lambdify(self._params, L_k, modules=['numpy'])
        
        arrays = [np.array(param_arrays.get(p, [0.0])) for p in self._params]
        
        return numpy_func(*arrays)
    
    def get_stability_info(self, k: int = 1) -> Dict[str, Any]:
        """
        Get stability analysis information.
        
        Args:
            k: Order of Lyapunov coefficient to analyze
            
        Returns:
            Dictionary with stability information
        """
        L_k = self.compute_lyapunov(k)
        
        info = {
            'coefficient': L_k,
            'order': k,
            'latex': sp.latex(L_k),
            'is_zero': simplify(L_k) == 0,
            'system_properties': [p.name for p in self.properties]
        }
        
        if SystemProperty.HAMILTONIAN in self.properties:
            info['note'] = "Hamiltonian system: odd Lyapunov coefficients are zero"
        
        if self._params:
            try:
                critical_values = sp.solve(L_k, self._params[0])
                info['critical_values'] = {str(self._params[0]): critical_values}
            except:
                info['critical_values'] = None
        
        return info
    
    def find_hopf_bifurcation(
        self,
        param: Symbol,
        k: int = 1
    ) -> Optional[List[Expr]]:
        """
        Find parameter values where Hopf bifurcation occurs.
        
        Args:
            param: Parameter to solve for
            k: Order of Lyapunov coefficient
            
        Returns:
            List of parameter values where L_k = 0
        """
        L_k = self.compute_lyapunov(k)
        
        try:
            solutions = sp.solve(L_k, param)
            return solutions
        except:
            return None
    
    def classify_bifurcation(
        self,
        param_value: Dict[Symbol, float],
        k: int = 1
    ) -> str:
        """
        Classify the type of Hopf bifurcation.
        
        Args:
            param_value: Parameter values to evaluate at
            k: Order of Lyapunov coefficient
            
        Returns:
            'supercritical', 'subcritical', or 'degenerate'
        """
        L_k_value = self.evaluate_lyapunov(k, param_value)
        
        if abs(L_k_value) < 1e-12:
            return 'degenerate'
        elif L_k_value < 0:
            return 'supercritical'
        else:
            return 'subcritical'
    
    def to_latex(self, k: Optional[int] = None) -> str:
        """
        Generate LaTeX representation.
        
        Args:
            k: Specific order to export (None = all computed)
            
        Returns:
            LaTeX string
        """
        lines = [
            r"\begin{align}",
            r"\dot{x} &= -y + " + sp.latex(self.P) + r" \\",
            r"\dot{y} &= x + " + sp.latex(self.Q),
            r"\end{align}",
            ""
        ]
        
        if k is not None:
            L_k = self.compute_lyapunov(k)
            lines.append(f"$L_{k} = {sp.latex(L_k)}$")
        else:
            for order, coeff in self.engine.lyapunov_coefficients.items():
                lines.append(f"$L_{order} = {sp.latex(coeff)}$")
                lines.append("")
        
        return "\n".join(lines)
    
    def export_latex(self, filepath: Union[str, Path], k: Optional[int] = None) -> None:
        """
        Export results to a LaTeX file.
        
        Args:
            filepath: Output file path
            k: Specific order to export (None = all computed)
        """
        latex_content = self.to_latex(k)
        
        with open(filepath, 'w') as f:
            f.write(latex_content)
    
    def get_c_code(self, k: int) -> str:
        """
        Get generated C code for L_k evaluation.
        
        Args:
            k: Order of Lyapunov coefficient
            
        Returns:
            C source code string
        """
        if self._code_gen is None:
            self._code_gen = CCodeGenerator(self.cache)
        
        L_k = self.compute_lyapunov(k)
        param_names = [str(p) for p in self._params]
        
        return self._code_gen.generate_c_function(L_k, param_names, f"eval_L{k}")
    
    def compile_evaluator(self, k: int) -> CompilationResult:
        """
        Compile C evaluator for L_k.
        
        Args:
            k: Order of Lyapunov coefficient
            
        Returns:
            CompilationResult with status
        """
        if self._code_gen is None:
            self._code_gen = CCodeGenerator(self.cache)
        
        c_code = self.get_c_code(k)
        result = self._code_gen.compile_to_library(c_code, f"lyapunov_L{k}")
        
        if result.success:
            lib = self._code_gen.load_library(result.library_path)
            if lib is not None:
                evaluator = self._code_gen.create_evaluator(
                    lib, f"eval_L{k}", len(self._params)
                )
                self._compiled_evaluators[k] = evaluator
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get computation and cache statistics.
        
        Returns:
            Dictionary with statistics
        """
        stats = {
            'system': {
                'P': str(self.P),
                'Q': str(self.Q),
                'params': [str(p) for p in self._params],
                'properties': [p.name for p in self.properties]
            },
            'computation': self.engine.get_computation_stats(),
            'compiled_evaluators': list(self._compiled_evaluators.keys())
        }
        
        if self.cache is not None:
            stats['cache'] = self.cache.get_stats()
        
        return stats
    
    def clear_cache(self) -> Dict[str, int]:
        """
        Clear all cached results.
        
        Returns:
            Dictionary with counts of cleared entries
        """
        if self.cache is not None:
            return self.cache.clear_all()
        return {'memory_cleared': 0, 'disk_cleared': 0}
    
    def __repr__(self) -> str:
        return (f"LyapunovSystem(\n"
                f"  ẋ = -y + {self.P}\n"
                f"  ẏ = x + {self.Q}\n"
                f"  params = {self._params}\n"
                f"  computed_orders = {list(self.engine.lyapunov_coefficients.keys())}\n"
                f")")
    
    def __str__(self) -> str:
        return str(self.system_def)


def create_system(
    P: Union[Expr, str],
    Q: Union[Expr, str],
    params: Optional[List[Symbol]] = None,
    **kwargs
) -> LyapunovSystem:
    """
    Convenience function to create a LyapunovSystem.
    
    Args:
        P: Nonlinear terms in ẋ = -y + P(x,y,μ)
        Q: Nonlinear terms in ẏ = x + Q(x,y,μ)
        params: List of parameter symbols
        **kwargs: Additional arguments for LyapunovSystem
        
    Returns:
        Configured LyapunovSystem instance
    """
    return LyapunovSystem(P, Q, params=params, **kwargs)


def from_full_system(
    x_dot: Union[Expr, str],
    y_dot: Union[Expr, str],
    x: Optional[Symbol] = None,
    y: Optional[Symbol] = None,
    params: Optional[List[Symbol]] = None,
    **kwargs
) -> LyapunovSystem:
    """
    Create LyapunovSystem from full system equations.
    
    Automatically extracts P and Q from:
        ẋ = -y + P(x,y)
        ẏ = x + Q(x,y)
    
    Args:
        x_dot: Full expression for ẋ
        y_dot: Full expression for ẏ
        x, y: State variable symbols
        params: Parameter symbols
        **kwargs: Additional arguments for LyapunovSystem
        
    Returns:
        Configured LyapunovSystem instance
    """
    if x is None:
        x = symbols('x')
    if y is None:
        y = symbols('y')
    
    if isinstance(x_dot, str):
        x_dot = sp.sympify(x_dot)
    if isinstance(y_dot, str):
        y_dot = sp.sympify(y_dot)
    
    P = sp.expand(x_dot + y)
    Q = sp.expand(y_dot - x)
    
    return LyapunovSystem(P, Q, params=params, x=x, y=y, **kwargs)
