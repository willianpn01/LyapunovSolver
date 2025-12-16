"""
Equilibrium Scanner - Main API
Unified interface for equilibrium point analysis.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict, field
import logging
import time

import sympy as sp
from sympy import Symbol, Expr, symbols, simplify, N, latex

from .symbolic_solver import SymbolicSolver, SymbolicSolution
from .numerical_solver import NumericalSolver, NumericalSolution
from .classification import EquilibriumClassifier, EquilibriumType, JacobianAnalysis
from .canonical_transformer import CanonicalTransformer, CanonicalForm

logger = logging.getLogger(__name__)


@dataclass
class BifurcationCandidate:
    name: str
    condition: str
    importance: str
    details: Optional[str] = None


@dataclass
class EquilibriumPoint:
    """Represents a classified equilibrium point."""
    
    # Identification
    index: int = 0
    
    # Coordinates
    x: Union[float, Expr] = 0
    y: Union[float, Expr] = 0
    is_symbolic: bool = True
    
    # Parameter info
    parameter_values: Optional[Dict] = None
    valid_domain: Optional[str] = None
    
    # Jacobian analysis
    eigenvalues: List = field(default_factory=list)
    trace: Optional[Expr] = None
    determinant: Optional[Expr] = None
    
    # Classification
    eq_type: EquilibriumType = EquilibriumType.UNKNOWN
    stability: str = "unknown"
    hypotheses: Optional[List] = None  # ClassificationHypothesis list
    classification_reason: Optional[str] = None  # Explanation for classification
    
    # Hopf-specific
    hopf_frequency: Optional[float] = None
    is_hopf: bool = False

    # Bifurcation candidates (heuristics)
    bifurcations: List[BifurcationCandidate] = field(default_factory=list)
    
    # Metadata
    solver_method: str = "symbolic"
    numerical_residual: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        result = {
            'index': self.index,
            'x': str(self.x),
            'y': str(self.y),
            'is_symbolic': self.is_symbolic,
            'type': self.eq_type.value if isinstance(self.eq_type, EquilibriumType) else str(self.eq_type),
            'stability': self.stability,
            'eigenvalues': [str(e) for e in self.eigenvalues],
            'solver_method': self.solver_method
        }
        if self.hopf_frequency is not None:
            result['hopf_frequency'] = self.hopf_frequency
        if self.valid_domain:
            result['valid_domain'] = self.valid_domain
        return result
    
    def to_latex(self) -> str:
        """Generate LaTeX representation."""
        x_latex = latex(self.x) if hasattr(self.x, '__class__') and self.x.__class__.__module__ == 'sympy.core.numbers' else latex(sp.sympify(str(self.x)))
        y_latex = latex(self.y) if hasattr(self.y, '__class__') else latex(sp.sympify(str(self.y)))
        return f"({x_latex}, {y_latex})"
    
    @property
    def type_display(self) -> str:
        """Human-readable type name."""
        type_names = {
            EquilibriumType.NODE_STABLE: "Nó Estável",
            EquilibriumType.NODE_UNSTABLE: "Nó Instável",
            EquilibriumType.SADDLE: "Sela",
            EquilibriumType.FOCUS_STABLE: "Foco Estável",
            EquilibriumType.FOCUS_UNSTABLE: "Foco Instável",
            EquilibriumType.CENTER: "Centro",
            EquilibriumType.HOPF_CANDIDATE: "Candidato Hopf",
            EquilibriumType.DEGENERATE: "Degenerado",
            EquilibriumType.UNKNOWN: "Desconhecido"
        }
        return type_names.get(self.eq_type, str(self.eq_type))


class EquilibriumScanner:
    """
    Unified interface for equilibrium point analysis.
    
    Combines symbolic solving, numerical solving, and classification
    to find and analyze equilibrium points of 2D dynamical systems.
    
    Example:
        >>> from sympy import symbols
        >>> x, y, mu = symbols('x y mu')
        >>> f = mu*x - x**3
        >>> g = -y
        >>> 
        >>> scanner = EquilibriumScanner(f, g, params=[mu])
        >>> points = scanner.scan()
        >>> 
        >>> for pt in points:
        ...     print(f"({pt.x}, {pt.y}) - {pt.type_display}")
    """
    
    def __init__(
        self,
        f: Union[Expr, str],
        g: Union[Expr, str],
        x: Symbol = None,
        y: Symbol = None,
        params: List[Symbol] = None,
        timeout: float = 30.0
    ):
        """
        Initialize the scanner.
        
        Args:
            f: Right-hand side of ẋ = f(x, y, μ)
            g: Right-hand side of ẏ = g(x, y, μ)
            x, y: State variable symbols
            params: List of parameter symbols
            timeout: Timeout for symbolic solver (seconds)
        """
        # Parse string expressions if needed
        if isinstance(f, str):
            f = sp.sympify(f)
        if isinstance(g, str):
            g = sp.sympify(g)
        
        self.f = f
        self.g = g
        self.x = x if x is not None else symbols('x')
        self.y = y if y is not None else symbols('y')
        self.params = params if params is not None else []
        self.timeout = timeout
        
        # Initialize components
        self.symbolic_solver = SymbolicSolver(
            f, g, self.x, self.y, self.params, timeout
        )
        self.classifier = EquilibriumClassifier(
            f, g, self.x, self.y, self.params
        )
        
        # Cache
        self._cached_points: Optional[List[EquilibriumPoint]] = None
        self._last_param_values: Optional[Dict] = None
        
        logger.info(f"EquilibriumScanner initialized for system ẋ={f}, ẏ={g}")
    
    def scan(
        self,
        param_values: Dict[Symbol, float] = None,
        method: str = 'auto',
        domain: List[Tuple[float, float]] = None,
        n_starts: int = 100
    ) -> List[EquilibriumPoint]:
        """
        Scan for equilibrium points and classify them.
        
        Args:
            param_values: Numerical values for parameters (optional)
            method: 'auto', 'symbolic', or 'numeric'
            domain: Search domain for numerical solver [(x_min, x_max), (y_min, y_max)]
            n_starts: Number of starting points for numerical solver
        
        Returns:
            List of classified EquilibriumPoint objects
        """
        start_time = time.time()
        logger.info(f"Starting equilibrium scan (method={method})")
        
        points = []
        
        # Stage A: Symbolic solve
        if method in ['auto', 'symbolic']:
            symbolic_solutions = self.symbolic_solver.solve()
            
            if symbolic_solutions:
                logger.info(f"Found {len(symbolic_solutions)} symbolic solutions")
                points = self._classify_solutions(symbolic_solutions, param_values)
        
        # Stage B: Numerical solve (fallback or explicit)
        if method == 'numeric' or (method == 'auto' and not points):
            if param_values is None and self.params:
                logger.warning("Numerical solver requires param_values when parameters exist")
            else:
                logger.info("Using numerical solver...")
                numerical_solutions = self._solve_numerical(param_values, domain, n_starts)
                
                if numerical_solutions:
                    logger.info(f"Found {len(numerical_solutions)} numerical solutions")
                    numerical_points = self._classify_numerical_solutions(
                        numerical_solutions, param_values
                    )
                    
                    # Merge with symbolic (avoid duplicates)
                    points = self._merge_solutions(points, numerical_points)

        self._annotate_bifurcations(points)
        
        # Cache results
        self._cached_points = points
        self._last_param_values = param_values
        
        elapsed = time.time() - start_time
        logger.info(f"Scan completed in {elapsed:.2f}s, found {len(points)} points")
        
        # Log summary
        self._log_summary(points)
        
        return points

    def _annotate_bifurcations(self, points: List[EquilibriumPoint]) -> None:
        for pt in points:
            pt.bifurcations = self._detect_local_bifurcations(pt)

        self._detect_transcritical_and_pitchfork(points)

    def _to_float(self, expr: Any) -> Optional[float]:
        try:
            if expr is None:
                return None
            if hasattr(expr, 'evalf'):
                return float(expr.evalf())
            return float(expr)
        except Exception:
            return None

    def _is_close_to_zero(self, expr: Any, tol: float = 1e-8) -> Optional[bool]:
        val = self._to_float(expr)
        if val is not None:
            return abs(val) < tol
        try:
            return simplify(expr) == 0
        except Exception:
            return None

    def _detect_local_bifurcations(self, pt: EquilibriumPoint) -> List[BifurcationCandidate]:
        candidates: List[BifurcationCandidate] = []

        tr = pt.trace
        det = pt.determinant

        if pt.eq_type == EquilibriumType.HOPF_CANDIDATE:
            candidates.append(BifurcationCandidate(
                name="Hopf",
                condition="Tr(J) = 0, Det(J) > 0 (ω ≠ 0)",
                importance="Geração/desaparecimento de ciclo limite (via foco fraco)"
            ))

        det_zero = self._is_close_to_zero(det)
        tr_zero = self._is_close_to_zero(tr)

        if det_zero is True and tr_zero is False:
            candidates.append(BifurcationCandidate(
                name="Sela-Nó",
                condition="Det(J) = 0, Tr(J) ≠ 0",
                importance="Criação/aniquilação de equilíbrios"
            ))

        if det_zero is True and tr_zero is True:
            candidates.append(BifurcationCandidate(
                name="Bogdanov–Takens",
                condition="Tr(J) = 0 e Det(J) = 0",
                importance="Bifurcação de codimensão 2 (organiza Hopf/Sela-Nó/órbitas)"
            ))

        return candidates

    def _detect_transcritical_and_pitchfork(self, points: List[EquilibriumPoint]) -> None:
        if len(self.params) != 1:
            return

        mu = self.params[0]
        symbolic_points = [pt for pt in points if pt.is_symbolic]
        if len(symbolic_points) < 2:
            return

        for i in range(len(symbolic_points)):
            for j in range(i + 1, len(symbolic_points)):
                p1 = symbolic_points[i]
                p2 = symbolic_points[j]

                try:
                    dx = simplify(sp.sympify(p1.x) - sp.sympify(p2.x))
                    dy = simplify(sp.sympify(p1.y) - sp.sympify(p2.y))
                except Exception:
                    continue

                if dx == 0 and dy == 0:
                    continue

                try:
                    sols = sp.solve([dx, dy], [mu], dict=True)
                except Exception:
                    continue

                if not sols:
                    continue

                mu_val = sols[0].get(mu)
                if mu_val is None:
                    continue

                cond = f"Dois equilíbrios colidem quando {mu} = {mu_val}"
                for pt in (p1, p2):
                    pt.bifurcations.append(BifurcationCandidate(
                        name="Transcrítica",
                        condition=cond,
                        importance="Dois equilíbrios colidem; possível troca de estabilidade"
                    ))

        origin_pts = [pt for pt in symbolic_points if simplify(sp.sympify(pt.x)) == 0 and simplify(sp.sympify(pt.y)) == 0]
        if not origin_pts:
            return

        for origin in origin_pts:
            for a in symbolic_points:
                if a is origin:
                    continue
                for b in symbolic_points:
                    if b is origin or b is a:
                        continue

                    try:
                        ax = simplify(sp.sympify(a.x))
                        ay = simplify(sp.sympify(a.y))
                        bx = simplify(sp.sympify(b.x))
                        by = simplify(sp.sympify(b.y))
                    except Exception:
                        continue

                    if simplify(ax + bx) != 0:
                        continue
                    if simplify(ay - by) != 0:
                        continue

                    try:
                        sols_a = sp.solve([ax, ay], [mu], dict=True)
                        sols_b = sp.solve([bx, by], [mu], dict=True)
                    except Exception:
                        continue

                    if not sols_a or not sols_b:
                        continue

                    mu_a = sols_a[0].get(mu)
                    mu_b = sols_b[0].get(mu)
                    if mu_a is None or mu_b is None:
                        continue
                    if simplify(mu_a - mu_b) != 0:
                        continue

                    cond = f"x₀ = 0 e simetria (ramificações ±x) com colisão em {mu} = {mu_a}"
                    for pt in (origin, a, b):
                        pt.bifurcations.append(BifurcationCandidate(
                            name="Pitchfork",
                            condition=cond,
                            importance="Quebra espontânea de simetria"
                        ))

                    return
    
    def _solve_numerical(
        self,
        param_values: Dict[Symbol, float],
        domain: List[Tuple[float, float]] = None,
        n_starts: int = 100
    ) -> List[NumericalSolution]:
        """Run numerical solver."""
        solver = NumericalSolver(
            self.f, self.g,
            self.x, self.y,
            param_values=param_values,
            domain=domain,
            n_starts=n_starts
        )
        return solver.solve()
    
    def _classify_numerical_solutions(
        self,
        solutions: List[NumericalSolution],
        param_values: Dict[Symbol, float]
    ) -> List[EquilibriumPoint]:
        """Classify numerical solutions."""
        points = []
        
        for idx, sol in enumerate(solutions):
            point_dict = {self.x: sol.x, self.y: sol.y}
            
            try:
                eq_type, analysis = self.classifier.classify(point_dict, param_values)
            except Exception as e:
                logger.warning(f"Classification failed for numerical point {idx}: {e}")
                eq_type = EquilibriumType.UNKNOWN
                analysis = None
            
            stability = self._get_stability(eq_type)
            
            # Check Hopf
            hopf_freq = None
            is_hopf = False
            
            if eq_type == EquilibriumType.HOPF_CANDIDATE and self.params and param_values:
                hopf_check = self.classifier.check_hopf_conditions(
                    point_dict,
                    self.params[0],
                    param_values.get(self.params[0], 0)
                )
                is_hopf = hopf_check.get('is_hopf', False)
                hopf_freq = hopf_check.get('hopf_frequency')
            
            eq_point = EquilibriumPoint(
                index=idx,
                x=sol.x,
                y=sol.y,
                is_symbolic=False,
                parameter_values=param_values,
                eigenvalues=analysis.eigenvalues if analysis else [],
                trace=analysis.trace if analysis else None,
                determinant=analysis.determinant if analysis else None,
                eq_type=eq_type,
                stability=stability,
                hopf_frequency=hopf_freq,
                is_hopf=is_hopf,
                solver_method="numeric",
                numerical_residual=sol.residual
            )
            
            points.append(eq_point)
        
        return points
    
    def _merge_solutions(
        self,
        symbolic_points: List[EquilibriumPoint],
        numerical_points: List[EquilibriumPoint],
        tolerance: float = 1e-6
    ) -> List[EquilibriumPoint]:
        """Merge symbolic and numerical solutions, avoiding duplicates."""
        merged = list(symbolic_points)
        
        for num_pt in numerical_points:
            is_duplicate = False
            
            for sym_pt in symbolic_points:
                try:
                    sym_x = float(sym_pt.x.evalf()) if hasattr(sym_pt.x, 'evalf') else float(sym_pt.x)
                    sym_y = float(sym_pt.y.evalf()) if hasattr(sym_pt.y, 'evalf') else float(sym_pt.y)
                    
                    dist = ((num_pt.x - sym_x)**2 + (num_pt.y - sym_y)**2)**0.5
                    if dist < tolerance:
                        is_duplicate = True
                        break
                except:
                    pass
            
            if not is_duplicate:
                num_pt.index = len(merged)
                merged.append(num_pt)
        
        return merged
    
    def _classify_solutions(
        self,
        solutions: List[SymbolicSolution],
        param_values: Dict[Symbol, float] = None
    ) -> List[EquilibriumPoint]:
        """Classify a list of symbolic solutions."""
        points = []
        
        for idx, sol in enumerate(solutions):
            point_dict = {self.x: sol.x, self.y: sol.y}
            
            # Classify with hypotheses for symbolic cases
            hypotheses = None
            reason = None
            try:
                eq_type, analysis, hypotheses, reason = self.classifier.classify_with_hypotheses(
                    point_dict, param_values
                )
            except Exception as e:
                logger.warning(f"Classification failed for point {idx}: {e}")
                eq_type = EquilibriumType.UNKNOWN
                analysis = None
                hypotheses = None
                reason = f"Erro na classificação: {e}"
            
            # Determine stability
            stability = self._get_stability(eq_type)
            
            # Check Hopf conditions if candidate
            hopf_freq = None
            is_hopf = False
            
            if eq_type == EquilibriumType.HOPF_CANDIDATE and self.params and param_values:
                hopf_check = self.classifier.check_hopf_conditions(
                    point_dict,
                    self.params[0],
                    param_values.get(self.params[0], 0)
                )
                is_hopf = hopf_check.get('is_hopf', False)
                hopf_freq = hopf_check.get('hopf_frequency')
            
            # Create EquilibriumPoint
            eq_point = EquilibriumPoint(
                index=idx,
                x=sol.x,
                y=sol.y,
                is_symbolic=sol.is_parametric or not self._is_numeric(sol.x, sol.y),
                parameter_values=param_values,
                valid_domain=sol.valid_domain,
                eigenvalues=analysis.eigenvalues if analysis else [],
                trace=analysis.trace if analysis else None,
                determinant=analysis.determinant if analysis else None,
                eq_type=eq_type,
                stability=stability,
                hypotheses=hypotheses,
                classification_reason=reason,
                hopf_frequency=hopf_freq,
                is_hopf=is_hopf,
                solver_method=sol.method
            )
            
            points.append(eq_point)
        
        return points
    
    def _is_numeric(self, x: Expr, y: Expr) -> bool:
        """Check if coordinates are purely numeric."""
        try:
            float(x)
            float(y)
            return True
        except (TypeError, ValueError):
            return False
    
    def _get_stability(self, eq_type: EquilibriumType) -> str:
        """Get stability string from equilibrium type."""
        stable_types = {
            EquilibriumType.NODE_STABLE,
            EquilibriumType.FOCUS_STABLE
        }
        unstable_types = {
            EquilibriumType.NODE_UNSTABLE,
            EquilibriumType.FOCUS_UNSTABLE,
            EquilibriumType.SADDLE
        }
        neutral_types = {
            EquilibriumType.CENTER,
            EquilibriumType.HOPF_CANDIDATE
        }
        
        if eq_type in stable_types:
            return "stable"
        elif eq_type in unstable_types:
            return "unstable"
        elif eq_type in neutral_types:
            return "neutral"
        else:
            return "unknown"
    
    def _log_summary(self, points: List[EquilibriumPoint]) -> None:
        """Log a summary of found points."""
        if not points:
            logger.warning("No equilibrium points found")
            return
        
        type_counts = {}
        for pt in points:
            t = pt.eq_type.value
            type_counts[t] = type_counts.get(t, 0) + 1
        
        hopf_count = sum(1 for pt in points if pt.is_hopf or pt.eq_type == EquilibriumType.HOPF_CANDIDATE)
        
        logger.info(f"Summary: {len(points)} points, {hopf_count} Hopf candidates")
        for t, count in type_counts.items():
            logger.debug(f"  {t}: {count}")
    
    def find_hopf_points(
        self,
        param_values: Dict[Symbol, float] = None
    ) -> List[EquilibriumPoint]:
        """
        Find only Hopf bifurcation candidates.
        
        Returns:
            List of points that are Hopf candidates
        """
        # Use cached results if available with same parameters
        if self._cached_points is not None and self._last_param_values == param_values:
            points = self._cached_points
        else:
            points = self.scan(param_values)
        
        hopf_points = [
            pt for pt in points 
            if pt.is_hopf or pt.eq_type == EquilibriumType.HOPF_CANDIDATE
        ]
        
        logger.info(f"Found {len(hopf_points)} Hopf candidates")
        return hopf_points
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the last scan.
        
        Returns:
            Dict with summary information
        """
        if not self._cached_points:
            return {'total_points': 0, 'message': 'No scan performed yet'}
        
        points = self._cached_points
        
        type_dist = {}
        for pt in points:
            t = pt.eq_type.value
            type_dist[t] = type_dist.get(t, 0) + 1
        
        return {
            'total_points': len(points),
            'hopf_candidates': sum(1 for pt in points if pt.eq_type == EquilibriumType.HOPF_CANDIDATE),
            'stable_points': sum(1 for pt in points if pt.stability == 'stable'),
            'unstable_points': sum(1 for pt in points if pt.stability == 'unstable'),
            'type_distribution': type_dist,
            'has_hopf': any(pt.is_hopf for pt in points)
        }
    
    def to_dataframe(self):
        """
        Convert results to pandas DataFrame.
        
        Returns:
            pandas DataFrame with equilibrium points
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for DataFrame export")
        
        if not self._cached_points:
            return pd.DataFrame()
        
        data = []
        for pt in self._cached_points:
            row = {
                '#': pt.index,
                'x': str(pt.x),
                'y': str(pt.y),
                'Tipo': pt.type_display,
                'λ₁': str(pt.eigenvalues[0]) if pt.eigenvalues else '-',
                'λ₂': str(pt.eigenvalues[1]) if len(pt.eigenvalues) > 1 else '-',
                'ω': f"{pt.hopf_frequency:.4f}" if pt.hopf_frequency else '-',
                'Estabilidade': pt.stability,
                'Método': pt.solver_method
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def print_results(self) -> None:
        """Print results in a formatted way."""
        if not self._cached_points:
            print("Nenhum ponto de equilíbrio encontrado. Execute scan() primeiro.")
            return
        
        print("\n" + "="*60)
        print("PONTOS DE EQUILÍBRIO ENCONTRADOS")
        print("="*60)
        
        for pt in self._cached_points:
            print(f"\n[{pt.index}] ({pt.x}, {pt.y})")
            print(f"    Tipo: {pt.type_display}")
            print(f"    Estabilidade: {pt.stability}")
            if pt.eigenvalues:
                print(f"    Autovalores: {pt.eigenvalues}")
            if pt.hopf_frequency:
                print(f"    Frequência Hopf: ω = {pt.hopf_frequency:.4f} rad/s")
            if pt.valid_domain:
                print(f"    Domínio válido: {pt.valid_domain}")
        
        print("\n" + "="*60)
        summary = self.get_summary()
        print(f"Total: {summary['total_points']} pontos")
        print(f"Candidatos Hopf: {summary['hopf_candidates']}")
        print("="*60 + "\n")
    
    def to_canonical_form(
        self,
        point: EquilibriumPoint,
        param_values: Dict[Symbol, float] = None
    ) -> CanonicalForm:
        """
        Transform the system to canonical Hopf form at a given equilibrium.
        
        Args:
            point: EquilibriumPoint (should be a Hopf candidate)
            param_values: Parameter values (uses point's if not provided)
        
        Returns:
            CanonicalForm with P, Q, ω ready for Lyapunov analysis
        """
        transformer = CanonicalTransformer(
            self.f, self.g, self.x, self.y, self.params
        )
        
        if param_values is None:
            param_values = point.parameter_values
        
        return transformer.transform_from_equilibrium_point(point, param_values)
    
    def create_lyapunov_system(
        self,
        point: EquilibriumPoint,
        param_values: Dict[Symbol, float] = None
    ):
        """
        Create a LyapunovSystem from a Hopf candidate point.
        
        This transforms the system to canonical form and creates a
        LyapunovSystem ready for computing Lyapunov coefficients.
        
        Args:
            point: EquilibriumPoint (should be a Hopf candidate)
            param_values: Parameter values
        
        Returns:
            LyapunovSystem ready for coefficient computation
        
        Example:
            >>> scanner = EquilibriumScanner(f, g, params=[mu])
            >>> hopf_points = scanner.find_hopf_points(param_values={mu: 0})
            >>> lyap_sys = scanner.create_lyapunov_system(hopf_points[0])
            >>> L1 = lyap_sys.compute_lyapunov(1)
        """
        from ..lyapunov_system import LyapunovSystem
        
        # Get canonical form
        canonical = self.to_canonical_form(point, param_values)
        
        # Create LyapunovSystem with the canonical P and Q
        # Note: LyapunovSystem expects ẋ = -y + P, ẏ = x + Q (ω=1)
        # If ω ≠ 1, we need to rescale or use P and Q as-is
        
        logger.info(f"Creating LyapunovSystem with P={canonical.P}, Q={canonical.Q}")
        
        return LyapunovSystem(
            P=canonical.P,
            Q=canonical.Q,
            params=self.params,
            x=self.x,
            y=self.y
        )
    
    def analyze_hopf_point(
        self,
        point: EquilibriumPoint,
        max_k: int = 3,
        param_values: Dict[Symbol, float] = None
    ) -> Dict[str, Any]:
        """
        Complete Hopf analysis: transform and compute Lyapunov coefficients.
        
        Args:
            point: Hopf candidate equilibrium point
            max_k: Maximum order of Lyapunov coefficients to compute
            param_values: Parameter values
        
        Returns:
            Dict with canonical form and Lyapunov coefficients
        """
        if point.eq_type != EquilibriumType.HOPF_CANDIDATE:
            logger.warning(f"Point is not a Hopf candidate (type: {point.eq_type})")
        
        # Transform to canonical form
        canonical = self.to_canonical_form(point, param_values)
        
        # Create LyapunovSystem and compute coefficients
        lyap_sys = self.create_lyapunov_system(point, param_values)
        
        coefficients = {}
        for k in range(1, max_k + 1):
            try:
                L_k = lyap_sys.compute_lyapunov(k)
                coefficients[k] = L_k
                logger.info(f"L{k} = {L_k}")
            except Exception as e:
                logger.error(f"Failed to compute L{k}: {e}")
                break
        
        return {
            'equilibrium': (point.x, point.y),
            'canonical_form': canonical,
            'omega': canonical.omega,
            'lyapunov_coefficients': coefficients,
            'lyapunov_system': lyap_sys
        }
