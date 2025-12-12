"""
Numerical Solver for Equilibrium Points
Multi-start optimization approach for finding equilibria.
"""

from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import logging
import time

import numpy as np
from sympy import Symbol, Expr, lambdify

logger = logging.getLogger(__name__)


@dataclass
class NumericalSolution:
    """Represents a numerically found equilibrium."""
    
    x: float
    y: float
    residual: float
    seed_index: int
    converged: bool
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {'x': self.x, 'y': self.y}


class NumericalSolver:
    """
    Numerical solver using multi-start optimization.
    
    Uses scipy.optimize.root with multiple random starting points
    to find equilibrium points of the system:
        f(x, y) = 0
        g(x, y) = 0
    """
    
    def __init__(
        self,
        f: Expr,
        g: Expr,
        x: Symbol,
        y: Symbol,
        param_values: Dict[Symbol, float] = None,
        domain: List[Tuple[float, float]] = None,
        n_starts: int = 100,
        tolerance: float = 1e-8,
        cluster_tolerance: float = 1e-6
    ):
        """
        Initialize the numerical solver.
        
        Args:
            f, g: Symbolic expressions for the system
            x, y: State variable symbols
            param_values: Numerical values for parameters
            domain: Search domain [(x_min, x_max), (y_min, y_max)]
            n_starts: Number of random starting points
            tolerance: Convergence tolerance for root finding
            cluster_tolerance: Tolerance for clustering similar solutions
        """
        self.x_sym = x
        self.y_sym = y
        self.param_values = param_values or {}
        self.domain = domain or [(-10.0, 10.0), (-10.0, 10.0)]
        self.n_starts = n_starts
        self.tolerance = tolerance
        self.cluster_tolerance = cluster_tolerance
        
        # Substitute parameters and create numerical functions
        f_sub = f.subs(param_values) if param_values else f
        g_sub = g.subs(param_values) if param_values else g
        
        self.f_func = lambdify((x, y), f_sub, modules=['numpy'])
        self.g_func = lambdify((x, y), g_sub, modules=['numpy'])
        
        logger.debug(f"NumericalSolver initialized with domain {self.domain}, {n_starts} starts")
    
    def solve(self) -> List[NumericalSolution]:
        """
        Find equilibrium points using multi-start optimization.
        
        Returns:
            List of NumericalSolution objects
        """
        try:
            from scipy.optimize import root
        except ImportError:
            logger.error("scipy is required for numerical solving")
            return []
        
        start_time = time.time()
        logger.info(f"Starting numerical solve with {self.n_starts} seeds...")
        
        # Generate starting points
        seeds = self._generate_seeds()
        
        # Find roots from each seed
        candidates = []
        converged_count = 0
        
        for idx, seed in enumerate(seeds):
            result = self._find_root(seed, root)
            if result is not None:
                result.seed_index = idx
                candidates.append(result)
                if result.converged:
                    converged_count += 1
        
        logger.debug(f"Found {len(candidates)} candidates, {converged_count} converged")
        
        # Cluster similar solutions
        unique_solutions = self._cluster_solutions(candidates)
        
        # Validate solutions
        validated = [sol for sol in unique_solutions if self._validate_solution(sol)]
        
        elapsed = time.time() - start_time
        logger.info(f"Numerical solve completed in {elapsed:.2f}s, found {len(validated)} unique points")
        
        return validated
    
    def _generate_seeds(self) -> np.ndarray:
        """
        Generate starting points using mixed strategies.
        
        - 25% uniform grid
        - 50% random
        - 25% Sobol quasi-random (better coverage)
        """
        n_grid = max(1, self.n_starts // 4)
        n_random = self.n_starts // 2
        n_sobol = self.n_starts - n_grid - n_random
        
        seeds = []
        
        # Uniform grid
        grid_seeds = self._uniform_grid(n_grid)
        seeds.extend(grid_seeds)
        
        # Random uniform
        x_min, x_max = self.domain[0]
        y_min, y_max = self.domain[1]
        
        random_seeds = np.column_stack([
            np.random.uniform(x_min, x_max, n_random),
            np.random.uniform(y_min, y_max, n_random)
        ])
        seeds.extend(random_seeds.tolist())
        
        # Sobol quasi-random (if scipy.stats.qmc available)
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=2, scramble=True)
            sobol_points = sampler.random(n_sobol)
            sobol_scaled = qmc.scale(
                sobol_points,
                [x_min, y_min],
                [x_max, y_max]
            )
            seeds.extend(sobol_scaled.tolist())
        except ImportError:
            # Fall back to more random points
            extra_random = np.column_stack([
                np.random.uniform(x_min, x_max, n_sobol),
                np.random.uniform(y_min, y_max, n_sobol)
            ])
            seeds.extend(extra_random.tolist())
        
        return np.array(seeds)
    
    def _uniform_grid(self, n: int) -> List[List[float]]:
        """Generate uniform grid of points."""
        # Determine grid dimensions
        n_per_dim = max(2, int(np.sqrt(n)))
        
        x_min, x_max = self.domain[0]
        y_min, y_max = self.domain[1]
        
        x_vals = np.linspace(x_min, x_max, n_per_dim)
        y_vals = np.linspace(y_min, y_max, n_per_dim)
        
        grid = []
        for x in x_vals:
            for y in y_vals:
                grid.append([x, y])
                if len(grid) >= n:
                    return grid
        
        return grid
    
    def _find_root(self, seed: np.ndarray, root_func) -> Optional[NumericalSolution]:
        """Find a root starting from a seed point."""
        def system(vars):
            x, y = vars
            return [self.f_func(x, y), self.g_func(x, y)]
        
        try:
            result = root_func(
                system,
                seed,
                method='hybr',
                tol=self.tolerance
            )
            
            x, y = result.x
            residual = np.sqrt(self.f_func(x, y)**2 + self.g_func(x, y)**2)
            
            return NumericalSolution(
                x=float(x),
                y=float(y),
                residual=float(residual),
                seed_index=0,
                converged=result.success and residual < self.tolerance * 100
            )
        except Exception as e:
            logger.debug(f"Root finding failed from seed {seed}: {e}")
            return None
    
    def _cluster_solutions(
        self,
        candidates: List[NumericalSolution]
    ) -> List[NumericalSolution]:
        """
        Cluster similar solutions and keep the best from each cluster.
        """
        if not candidates:
            return []
        
        # Sort by residual (best first)
        sorted_candidates = sorted(candidates, key=lambda s: s.residual)
        
        unique = []
        used_indices = set()
        
        for i, sol in enumerate(sorted_candidates):
            if i in used_indices:
                continue
            
            # Find all solutions close to this one
            cluster_indices = [i]
            for j, other in enumerate(sorted_candidates):
                if j <= i or j in used_indices:
                    continue
                
                dist = np.sqrt((sol.x - other.x)**2 + (sol.y - other.y)**2)
                if dist < self.cluster_tolerance:
                    cluster_indices.append(j)
            
            # Mark all in cluster as used
            used_indices.update(cluster_indices)
            
            # Keep the best (first, since sorted by residual)
            unique.append(sol)
        
        return unique
    
    def _validate_solution(self, sol: NumericalSolution) -> bool:
        """Validate that the solution is actually a root."""
        # Check if in domain
        x_min, x_max = self.domain[0]
        y_min, y_max = self.domain[1]
        
        if not (x_min <= sol.x <= x_max and y_min <= sol.y <= y_max):
            return False
        
        # Check residual
        if sol.residual > self.tolerance * 1000:
            return False
        
        # Check for NaN/Inf
        if np.isnan(sol.x) or np.isnan(sol.y):
            return False
        if np.isinf(sol.x) or np.isinf(sol.y):
            return False
        
        return True
    
    def set_domain(self, x_range: Tuple[float, float], y_range: Tuple[float, float]):
        """Update the search domain."""
        self.domain = [x_range, y_range]
        logger.debug(f"Domain updated to {self.domain}")
