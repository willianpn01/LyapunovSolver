"""
Equilibrium Classification Module
Classifies equilibrium points based on Jacobian eigenvalue analysis.
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging

import sympy as sp
from sympy import (
    Symbol, Expr, Matrix, symbols, diff, simplify,
    re as sym_re, im as sym_im, sqrt, I, S
)

logger = logging.getLogger(__name__)


class EquilibriumType(Enum):
    """Classification of equilibrium points."""
    
    # Hyperbolic points (no zero eigenvalues)
    NODE_STABLE = "node_stable"
    NODE_UNSTABLE = "node_unstable"
    SADDLE = "saddle"
    FOCUS_STABLE = "focus_stable"
    FOCUS_UNSTABLE = "focus_unstable"
    
    # Non-hyperbolic points
    CENTER = "center"
    HOPF_CANDIDATE = "hopf_candidate"
    
    # Degenerate cases
    DEGENERATE = "degenerate"
    UNKNOWN = "unknown"


@dataclass
class ClassificationHypothesis:
    """A hypothesis about parameter conditions and resulting classification."""
    condition: str  # Human-readable condition (e.g., "a*d > 0")
    condition_expr: Expr  # Symbolic expression
    eq_type: 'EquilibriumType'  # Classification if condition holds
    description: str  # Explanation


@dataclass
class JacobianAnalysis:
    """Results of Jacobian analysis at an equilibrium point."""
    
    jacobian: Matrix
    trace: Expr
    determinant: Expr
    eigenvalues: List[Expr]
    eigenvectors: Optional[List] = None
    hypotheses: Optional[List[ClassificationHypothesis]] = None
    
    def is_hyperbolic(self, tolerance: float = 1e-8) -> bool:
        """Check if the equilibrium is hyperbolic (no zero real parts)."""
        for eig in self.eigenvalues:
            try:
                real_part = complex(eig).real
                if abs(real_part) < tolerance:
                    return False
            except (TypeError, ValueError):
                # Symbolic eigenvalue, can't determine numerically
                return None
        return True


class EquilibriumClassifier:
    """
    Classifies equilibrium points based on Jacobian eigenvalue analysis.
    
    Given a 2D dynamical system:
        ẋ = f(x, y, μ)
        ẏ = g(x, y, μ)
    
    Computes and analyzes the Jacobian at equilibrium points.
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
        Initialize the classifier.
        
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
        
        # Compute Jacobian symbolically (once)
        self._jacobian = self._compute_jacobian()
        
        logger.debug(f"Jacobian computed: {self._jacobian}")
    
    def _compute_jacobian(self) -> Matrix:
        """Compute the symbolic Jacobian matrix."""
        df_dx = diff(self.f, self.x)
        df_dy = diff(self.f, self.y)
        dg_dx = diff(self.g, self.x)
        dg_dy = diff(self.g, self.y)
        
        return Matrix([
            [df_dx, df_dy],
            [dg_dx, dg_dy]
        ])
    
    def evaluate_jacobian(
        self,
        point: Dict[Symbol, Expr],
        param_values: Dict[Symbol, float] = None
    ) -> JacobianAnalysis:
        """
        Evaluate the Jacobian at a specific point.
        
        Args:
            point: Dict with {x: x0, y: y0} coordinates
            param_values: Optional numerical values for parameters
        
        Returns:
            JacobianAnalysis with eigenvalues and invariants
        """
        # Get coordinates, substituting parameter values if provided
        x_coord = point.get(self.x, 0)
        y_coord = point.get(self.y, 0)
        
        if param_values:
            x_coord = x_coord.subs(param_values) if hasattr(x_coord, 'subs') else x_coord
            y_coord = y_coord.subs(param_values) if hasattr(y_coord, 'subs') else y_coord
        
        subs_dict = {self.x: x_coord, self.y: y_coord}
        
        if param_values:
            subs_dict.update(param_values)
        
        J = self._jacobian.subs(subs_dict)
        
        # Compute invariants
        trace = simplify(J.trace())
        det = simplify(J.det())
        
        # Compute eigenvalues
        try:
            eigenvals = J.eigenvals()
            eigs = []
            for eig, mult in eigenvals.items():
                eigs.extend([simplify(eig)] * mult)
        except Exception as e:
            logger.warning(f"Could not compute eigenvalues: {e}")
            # Fallback: use characteristic equation
            eigs = self._eigenvalues_from_invariants(trace, det)
        
        return JacobianAnalysis(
            jacobian=J,
            trace=trace,
            determinant=det,
            eigenvalues=eigs
        )
    
    def _eigenvalues_from_invariants(self, trace: Expr, det: Expr) -> List[Expr]:
        """Compute eigenvalues from trace and determinant."""
        # λ = (tr ± sqrt(tr² - 4*det)) / 2
        discriminant = trace**2 - 4 * det
        
        lambda1 = (trace + sqrt(discriminant)) / 2
        lambda2 = (trace - sqrt(discriminant)) / 2
        
        return [simplify(lambda1), simplify(lambda2)]
    
    def classify(
        self,
        point: Dict[Symbol, Expr],
        param_values: Dict[Symbol, float] = None,
        tolerance: float = 1e-8
    ) -> Tuple[EquilibriumType, JacobianAnalysis]:
        """
        Classify an equilibrium point.
        
        Args:
            point: Coordinates of the equilibrium
            param_values: Numerical values for parameters
            tolerance: Numerical tolerance for comparisons
        
        Returns:
            Tuple of (EquilibriumType, JacobianAnalysis)
        """
        analysis = self.evaluate_jacobian(point, param_values)
        
        # Try numerical classification first
        eq_type = self._classify_numerical(analysis, tolerance)
        
        if eq_type == EquilibriumType.UNKNOWN:
            # Try symbolic classification
            eq_type = self._classify_symbolic(analysis)
        
        return eq_type, analysis
    
    def _classify_numerical(
        self,
        analysis: JacobianAnalysis,
        tolerance: float
    ) -> EquilibriumType:
        """Classify using numerical eigenvalue evaluation."""
        try:
            eigs = [complex(simplify(e).evalf()) for e in analysis.eigenvalues]
        except (TypeError, ValueError):
            return EquilibriumType.UNKNOWN
        
        λ1, λ2 = eigs[0], eigs[1]
        
        # Check if eigenvalues are real or complex
        is_real_1 = abs(λ1.imag) < tolerance
        is_real_2 = abs(λ2.imag) < tolerance
        
        if is_real_1 and is_real_2:
            # Real eigenvalues
            r1, r2 = λ1.real, λ2.real
            
            if r1 < -tolerance and r2 < -tolerance:
                return EquilibriumType.NODE_STABLE
            elif r1 > tolerance and r2 > tolerance:
                return EquilibriumType.NODE_UNSTABLE
            elif r1 * r2 < -tolerance**2:
                return EquilibriumType.SADDLE
            else:
                # At least one eigenvalue near zero
                return EquilibriumType.DEGENERATE
        else:
            # Complex conjugate eigenvalues
            real_part = λ1.real
            imag_part = abs(λ1.imag)
            
            if imag_part < tolerance:
                return EquilibriumType.DEGENERATE
            
            if abs(real_part) < tolerance:
                # Pure imaginary → Center or Hopf candidate
                return EquilibriumType.HOPF_CANDIDATE
            elif real_part < -tolerance:
                return EquilibriumType.FOCUS_STABLE
            elif real_part > tolerance:
                return EquilibriumType.FOCUS_UNSTABLE
            else:
                return EquilibriumType.UNKNOWN
        
        return EquilibriumType.UNKNOWN
    
    def _classify_symbolic(self, analysis: JacobianAnalysis) -> EquilibriumType:
        """Classify using symbolic analysis of trace and determinant."""
        tr = analysis.trace
        det = analysis.determinant
        
        # Try to simplify
        tr = simplify(tr)
        det = simplify(det)
        
        # Check trace
        if tr == 0:
            if det > 0:
                return EquilibriumType.CENTER
            elif det < 0:
                return EquilibriumType.SADDLE
            else:
                return EquilibriumType.DEGENERATE
        
        # Check determinant
        if det < 0:
            return EquilibriumType.SADDLE
        
        # det > 0: node or focus
        discriminant = tr**2 - 4 * det
        
        if tr < 0:
            # Stable
            try:
                disc_val = float(discriminant.evalf())
                if disc_val >= 0:
                    return EquilibriumType.NODE_STABLE
                else:
                    return EquilibriumType.FOCUS_STABLE
            except:
                return EquilibriumType.UNKNOWN
        elif tr > 0:
            # Unstable
            try:
                disc_val = float(discriminant.evalf())
                if disc_val >= 0:
                    return EquilibriumType.NODE_UNSTABLE
                else:
                    return EquilibriumType.FOCUS_UNSTABLE
            except:
                return EquilibriumType.UNKNOWN
        
        return EquilibriumType.UNKNOWN
    
    def classify_with_hypotheses(
        self,
        point: Dict[Symbol, Expr],
        param_values: Dict[Symbol, float] = None
    ) -> Tuple[EquilibriumType, JacobianAnalysis, List[ClassificationHypothesis], str]:
        """
        Classify with hypothesis generation for symbolic cases.
        
        When classification depends on unknown parameter signs, generates
        hypotheses about what the classification would be under different
        assumptions.
        
        Returns:
            Tuple of (primary_type, analysis, hypotheses, reason)
        """
        analysis = self.evaluate_jacobian(point, param_values)
        hypotheses = []
        reason = ""
        
        tr = simplify(analysis.trace)
        det = simplify(analysis.determinant)
        discriminant = simplify(tr**2 - 4 * det)
        
        # Try standard classification first
        try:
            eq_type = self._classify_numerical(analysis, 1e-8)
            if eq_type != EquilibriumType.UNKNOWN:
                reason = self._generate_reason(eq_type, tr, det, analysis.eigenvalues, is_numerical=True)
                return eq_type, analysis, hypotheses, reason
        except:
            pass
        
        # Generate hypotheses based on trace and determinant
        hypotheses = self._generate_hypotheses(tr, det, discriminant)
        analysis.hypotheses = hypotheses
        
        # Try symbolic classification
        try:
            eq_type = self._classify_symbolic(analysis)
        except:
            eq_type = EquilibriumType.UNKNOWN
        
        # If still unknown but we have hypotheses, mark as conditional
        if eq_type == EquilibriumType.UNKNOWN and hypotheses:
            # Check for special cases we can determine
            # Case: tr = 0 symbolically
            if simplify(tr) == 0:
                eq_type = EquilibriumType.CENTER
                reason = (f"Traço = 0 (simbolicamente). Autovalores puramente imaginários ±i√(det). "
                         f"⚠️ ATENÇÃO: Esta é uma classificação LINEAR. Para sistemas não-lineares, "
                         f"traço zero indica um CENTRO LINEAR, mas o comportamento real pode ser "
                         f"um foco fraco (estável ou instável). É necessário calcular os coeficientes "
                         f"de Lyapunov para determinar a estabilidade não-linear.")
            # Case: det is always negative (e.g., -x^2 - 1)
            elif self._is_always_negative(det):
                eq_type = EquilibriumType.SADDLE
                reason = f"Determinante sempre negativo (det = {det}). Autovalores reais de sinais opostos."
            # Case: det is always positive and tr = 0
            elif self._is_always_positive(det) and simplify(tr) == 0:
                eq_type = EquilibriumType.HOPF_CANDIDATE
                reason = (f"Traço = 0 e det > 0. Candidato a bifurcação de Hopf. "
                         f"⚠️ Para sistemas não-lineares, calcule L₁ para determinar estabilidade.")
        elif eq_type != EquilibriumType.UNKNOWN:
            reason = self._generate_reason(eq_type, tr, det, analysis.eigenvalues, is_numerical=False)
        
        return eq_type, analysis, hypotheses, reason
    
    def _generate_reason(
        self,
        eq_type: EquilibriumType,
        trace: Expr,
        det: Expr,
        eigenvalues: List[Expr],
        is_numerical: bool = False
    ) -> str:
        """Generate human-readable explanation for classification."""
        method = "numérica" if is_numerical else "simbólica"
        eig_str = ", ".join([str(e) for e in eigenvalues[:2]]) if eigenvalues else "N/A"
        
        reasons = {
            EquilibriumType.NODE_STABLE: (
                f"Nó Estável (classificação {method}). "
                f"Autovalores reais negativos: λ = {eig_str}. "
                f"Traço = {trace} < 0, Det = {det} > 0, Δ = tr² - 4det ≥ 0."
            ),
            EquilibriumType.NODE_UNSTABLE: (
                f"Nó Instável (classificação {method}). "
                f"Autovalores reais positivos: λ = {eig_str}. "
                f"Traço = {trace} > 0, Det = {det} > 0, Δ = tr² - 4det ≥ 0."
            ),
            EquilibriumType.SADDLE: (
                f"Sela (classificação {method}). "
                f"Autovalores reais de sinais opostos: λ = {eig_str}. "
                f"Det = {det} < 0. Ponto hiperbólico instável."
            ),
            EquilibriumType.FOCUS_STABLE: (
                f"Foco Estável (classificação {method}). "
                f"Autovalores complexos com parte real negativa: λ = {eig_str}. "
                f"Traço = {trace} < 0, Det = {det} > 0, Δ = tr² - 4det < 0."
            ),
            EquilibriumType.FOCUS_UNSTABLE: (
                f"Foco Instável (classificação {method}). "
                f"Autovalores complexos com parte real positiva: λ = {eig_str}. "
                f"Traço = {trace} > 0, Det = {det} > 0, Δ = tr² - 4det < 0."
            ),
            EquilibriumType.CENTER: (
                f"Centro Linear (classificação {method}). "
                f"Autovalores puramente imaginários: λ = {eig_str}. "
                f"Traço = {trace} = 0, Det = {det} > 0. "
                f"⚠️ ATENÇÃO: Para sistemas NÃO-LINEARES, esta classificação é baseada apenas "
                f"na linearização. O comportamento real pode ser um FOCO FRACO (estável ou instável). "
                f"Calcule os coeficientes de Lyapunov (L₁, L₂, ...) para determinar a estabilidade."
            ),
            EquilibriumType.HOPF_CANDIDATE: (
                f"Candidato a Hopf (classificação {method}). "
                f"Autovalores: λ = {eig_str}. "
                f"Traço ≈ 0, Det > 0. Ponto não-hiperbólico. "
                f"⚠️ Calcule L₁ para determinar se há ciclo limite e sua estabilidade."
            ),
            EquilibriumType.DEGENERATE: (
                f"Ponto Degenerado (classificação {method}). "
                f"Pelo menos um autovalor zero: λ = {eig_str}. "
                f"Det = {det} = 0. Análise de ordem superior necessária."
            ),
            EquilibriumType.UNKNOWN: (
                f"Classificação indeterminada. "
                f"Traço = {trace}, Det = {det}. "
                f"Não foi possível determinar o sinal das expressões simbolicamente."
            )
        }
        
        return reasons.get(eq_type, f"Tipo: {eq_type.value}")
    
    def _generate_hypotheses(
        self,
        trace: Expr,
        det: Expr,
        discriminant: Expr
    ) -> List[ClassificationHypothesis]:
        """Generate classification hypotheses based on parameter conditions."""
        hypotheses = []
        
        # Hypothesis 1: det < 0 → Saddle
        hypotheses.append(ClassificationHypothesis(
            condition=f"{det} < 0",
            condition_expr=det < 0,
            eq_type=EquilibriumType.SADDLE,
            description="Se o determinante for negativo, o ponto é uma Sela (hiperbólico)"
        ))
        
        # Hypothesis 2: det > 0 and tr < 0 → Stable (node or focus)
        hypotheses.append(ClassificationHypothesis(
            condition=f"{det} > 0 e {trace} < 0",
            condition_expr=sp.And(det > 0, trace < 0),
            eq_type=EquilibriumType.FOCUS_STABLE,
            description="Se det > 0 e traço < 0, o ponto é estável (Nó ou Foco estável)"
        ))
        
        # Hypothesis 3: det > 0 and tr > 0 → Unstable (node or focus)
        hypotheses.append(ClassificationHypothesis(
            condition=f"{det} > 0 e {trace} > 0",
            condition_expr=sp.And(det > 0, trace > 0),
            eq_type=EquilibriumType.FOCUS_UNSTABLE,
            description="Se det > 0 e traço > 0, o ponto é instável (Nó ou Foco instável)"
        ))
        
        # Hypothesis 4: det > 0 and tr = 0 → Center/Hopf
        hypotheses.append(ClassificationHypothesis(
            condition=f"{det} > 0 e {trace} = 0",
            condition_expr=sp.And(det > 0, sp.Eq(trace, 0)),
            eq_type=EquilibriumType.HOPF_CANDIDATE,
            description="Se det > 0 e traço = 0, o ponto é um Centro/Candidato a Hopf (não-hiperbólico)"
        ))
        
        # Hypothesis 5: det = 0 → Degenerate
        hypotheses.append(ClassificationHypothesis(
            condition=f"{det} = 0",
            condition_expr=sp.Eq(det, 0),
            eq_type=EquilibriumType.DEGENERATE,
            description="Se det = 0, o ponto é degenerado (autovalor zero)"
        ))
        
        return hypotheses
    
    def _is_always_negative(self, expr: Expr) -> bool:
        """Check if expression is always negative (e.g., -x^2 - 1)."""
        try:
            # Check if it's a sum of negative terms
            simplified = simplify(expr)
            # Try to evaluate if it's a constant
            val = float(simplified.evalf())
            return val < 0
        except:
            pass
        
        # Check for patterns like -a^2 - b^2 - c (always negative if c > 0)
        # or -(something)^2 - positive_constant
        try:
            # If expr + positive_number is still always negative
            if simplify(expr + 1).is_negative:
                return True
        except:
            pass
        
        return False
    
    def _is_always_positive(self, expr: Expr) -> bool:
        """Check if expression is always positive (e.g., x^2 + 1)."""
        try:
            simplified = simplify(expr)
            val = float(simplified.evalf())
            return val > 0
        except:
            pass
        
        try:
            if simplify(expr).is_positive:
                return True
        except:
            pass
        
        return False
    
    def check_hopf_conditions(
        self,
        point: Dict[Symbol, Expr],
        param: Symbol,
        param_value: float = 0.0,
        tolerance: float = 1e-6
    ) -> Dict:
        """
        Check Hopf bifurcation conditions at a point.
        
        Conditions:
        1. Re(λ) = 0 at μ = μ₀
        2. Im(λ) ≠ 0 (ω ≠ 0)
        3. d/dμ Re(λ) ≠ 0 at μ = μ₀ (transversality)
        
        Returns:
            Dict with condition checks and Hopf frequency
        """
        param_values = {param: param_value}
        eq_type, analysis = self.classify(point, param_values, tolerance)
        
        result = {
            'is_hopf_candidate': eq_type == EquilibriumType.HOPF_CANDIDATE,
            'eigenvalues': analysis.eigenvalues,
            'trace': analysis.trace,
            'determinant': analysis.determinant,
            'conditions': {}
        }
        
        if not result['is_hopf_candidate']:
            return result
        
        try:
            eig = complex(analysis.eigenvalues[0].evalf())
            
            # Condition 1: Re(λ) ≈ 0
            result['conditions']['real_part_zero'] = abs(eig.real) < tolerance
            
            # Condition 2: Im(λ) ≠ 0
            omega = abs(eig.imag)
            result['conditions']['nonzero_frequency'] = omega > tolerance
            result['hopf_frequency'] = omega
            
            # Condition 3: Transversality (d/dμ Re(λ) ≠ 0)
            transversality = self._check_transversality(point, param, param_value)
            result['conditions']['transversality'] = transversality
            
            # All conditions met?
            result['is_hopf'] = all(result['conditions'].values())
            
        except Exception as e:
            logger.warning(f"Error checking Hopf conditions: {e}")
            result['error'] = str(e)
        
        return result
    
    def _check_transversality(
        self,
        point: Dict[Symbol, Expr],
        param: Symbol,
        param_value: float
    ) -> bool:
        """
        Check transversality condition: d/dμ Re(λ(μ)) ≠ 0.
        
        Uses the fact that for Jacobian J(μ):
        d(Re λ)/dμ = (1/2) * d(trace)/dμ
        """
        subs_dict = {self.x: point.get(self.x, 0), self.y: point.get(self.y, 0)}
        
        # Get trace as function of parameter
        J = self._jacobian.subs(subs_dict)
        trace = J.trace()
        
        # Derivative of trace w.r.t. parameter
        d_trace = diff(trace, param)
        
        # Evaluate at parameter value
        d_trace_val = d_trace.subs(param, param_value)
        
        try:
            val = float(d_trace_val.evalf())
            return abs(val) > 1e-8
        except:
            # Symbolic result, assume transversal if non-zero
            return simplify(d_trace_val) != 0
