"""
Basic Usage Examples for LyapunovSolver-Hybrid v2.0

This file demonstrates the core functionality of the library.
"""

import sys
sys.path.insert(0, '..')

from sympy import symbols, sqrt, Rational, pprint
from lyapunov import LyapunovSystem, LyapunovVisualizer


def example_1_simple_system():
    """
    Example 1: Simple cubic nonlinearity
    
    System:
        ẋ = -y + μx - x³
        ẏ = x - y³
    """
    print("=" * 60)
    print("Example 1: Simple Cubic System")
    print("=" * 60)
    
    x, y, mu = symbols('x y mu')
    
    P = mu * x - x**3
    Q = -y**3
    
    system = LyapunovSystem(P, Q, params=[mu])
    
    print("\nSystem Definition:")
    print(system)
    
    print("\nSystem Properties:")
    for prop in system.properties:
        print(f"  - {prop.name}")
    
    L1 = system.compute_lyapunov(1)
    print(f"\nFirst Lyapunov Coefficient:")
    print(f"  L₁ = {L1}")
    
    print("\nNumerical Evaluation:")
    for mu_val in [0.0, 0.5, 1.0, -0.5]:
        L1_val = system.evaluate_lyapunov(1, {mu: mu_val})
        bif_type = system.classify_bifurcation({mu: mu_val})
        print(f"  μ = {mu_val:5.1f}: L₁ = {L1_val:10.6f} ({bif_type})")
    
    print("\nLaTeX representation:")
    print(f"  {system.to_latex(1)}")
    
    return system


def example_2_van_der_pol():
    """
    Example 2: Van der Pol oscillator (transformed to canonical form)
    
    Original: ẍ + μ(x² - 1)ẋ + x = 0
    Canonical form near origin with small amplitude.
    """
    print("\n" + "=" * 60)
    print("Example 2: Van der Pol Type System")
    print("=" * 60)
    
    x, y, mu = symbols('x y mu')
    
    P = -mu * x * y
    Q = mu * (x**2 - 1) * x / 2
    
    system = LyapunovSystem(P, Q, params=[mu])
    
    print("\nSystem Definition:")
    print(system)
    
    L1 = system.compute_lyapunov(1)
    print(f"\nFirst Lyapunov Coefficient:")
    print(f"  L₁ = {L1}")
    
    return system


def example_3_hamiltonian():
    """
    Example 3: Hamiltonian system (should have L₁ = 0)
    
    For Hamiltonian systems, the divergence is zero:
    ∂P/∂x + ∂Q/∂y = 0
    """
    print("\n" + "=" * 60)
    print("Example 3: Hamiltonian System")
    print("=" * 60)
    
    x, y = symbols('x y')
    
    P = x**2 * y
    Q = -x * y**2
    
    system = LyapunovSystem(P, Q, params=[])
    
    print("\nSystem Definition:")
    print(system)
    
    print("\nSystem Properties:")
    for prop in system.properties:
        print(f"  - {prop.name}")
    
    L1 = system.compute_lyapunov(1)
    print(f"\nFirst Lyapunov Coefficient:")
    print(f"  L₁ = {L1}")
    
    from sympy import simplify
    if simplify(L1) == 0:
        print("  ✓ As expected for Hamiltonian system, L₁ = 0")
    
    return system


def example_4_two_parameters():
    """
    Example 4: System with two parameters
    
    Demonstrates multi-parameter analysis.
    """
    print("\n" + "=" * 60)
    print("Example 4: Two-Parameter System")
    print("=" * 60)
    
    x, y, alpha, beta = symbols('x y alpha beta')
    
    P = alpha * x - beta * x**2 * y - x**3
    Q = alpha * y + beta * x * y**2 - y**3
    
    system = LyapunovSystem(P, Q, params=[alpha, beta])
    
    print("\nSystem Definition:")
    print(system)
    
    L1 = system.compute_lyapunov(1)
    print(f"\nFirst Lyapunov Coefficient:")
    print(f"  L₁ = {L1}")
    
    print("\nParameter sweep (α = 0.1):")
    for beta_val in [-1.0, 0.0, 1.0, 2.0]:
        L1_val = system.evaluate_lyapunov(1, {alpha: 0.1, beta: beta_val})
        print(f"  β = {beta_val:5.1f}: L₁ = {L1_val:10.6f}")
    
    return system


def example_5_batch_evaluation():
    """
    Example 5: Batch numerical evaluation
    
    Demonstrates efficient evaluation over parameter arrays.
    """
    print("\n" + "=" * 60)
    print("Example 5: Batch Evaluation")
    print("=" * 60)
    
    import numpy as np
    
    x, y, mu = symbols('x y mu')
    
    P = mu * x - x**3 - x * y**2
    Q = mu * y - y**3 - x**2 * y
    
    system = LyapunovSystem(P, Q, params=[mu])
    
    L1 = system.compute_lyapunov(1)
    print(f"\nL₁ = {L1}")
    
    mu_values = np.linspace(-1, 1, 11)
    L1_values = system.evaluate_batch(1, {mu: mu_values.tolist()})
    
    print("\nBatch evaluation results:")
    print("  μ        L₁")
    print("  " + "-" * 20)
    for m, L in zip(mu_values, L1_values):
        print(f"  {m:6.2f}   {L:10.6f}")
    
    return system


def example_6_c_code_generation():
    """
    Example 6: C code generation
    
    Demonstrates generating optimized C code for numerical evaluation.
    """
    print("\n" + "=" * 60)
    print("Example 6: C Code Generation")
    print("=" * 60)
    
    x, y, mu, epsilon = symbols('x y mu epsilon')
    
    P = mu * x + epsilon * x**2 - x**3
    Q = -y**3
    
    system = LyapunovSystem(P, Q, params=[mu, epsilon])
    
    L1 = system.compute_lyapunov(1)
    print(f"\nL₁ = {L1}")
    
    c_code = system.get_c_code(1)
    print("\nGenerated C code:")
    print("-" * 40)
    print(c_code[:500] + "..." if len(c_code) > 500 else c_code)
    print("-" * 40)
    
    return system


def example_7_stability_analysis():
    """
    Example 7: Complete stability analysis
    
    Demonstrates the full analysis workflow.
    """
    print("\n" + "=" * 60)
    print("Example 7: Complete Stability Analysis")
    print("=" * 60)
    
    x, y, mu = symbols('x y mu')
    
    P = mu * x - x**3 + x * y**2
    Q = mu * y - y**3 + x**2 * y
    
    system = LyapunovSystem(P, Q, params=[mu])
    
    print("\nSystem:")
    print(system)
    
    info = system.get_stability_info(k=1)
    
    print("\nStability Analysis:")
    print(f"  L₁ = {info['coefficient']}")
    print(f"  Is zero: {info['is_zero']}")
    print(f"  Properties: {info['system_properties']}")
    
    if info.get('critical_values'):
        print(f"  Critical values: {info['critical_values']}")
    
    print("\nBifurcation classification at different μ values:")
    for mu_val in [-0.5, 0.0, 0.5]:
        bif_type = system.classify_bifurcation({mu: mu_val})
        L1_val = system.evaluate_lyapunov(1, {mu: mu_val})
        print(f"  μ = {mu_val:5.1f}: {bif_type:15s} (L₁ = {L1_val:.6f})")
    
    return system


def example_8_visualization():
    """
    Example 8: Visualization (requires matplotlib)
    
    Demonstrates plotting capabilities.
    """
    print("\n" + "=" * 60)
    print("Example 8: Visualization")
    print("=" * 60)
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        HAS_MATPLOTLIB = True
    except ImportError:
        HAS_MATPLOTLIB = False
        print("  Matplotlib not available, skipping visualization example")
        return None
    
    x, y, mu = symbols('x y mu')
    
    P = mu * x - x**3
    Q = -y**3
    
    system = LyapunovSystem(P, Q, params=[mu])
    viz = LyapunovVisualizer(system)
    
    print("\nGenerating phase portrait...")
    fig, ax = viz.plot_phase_portrait(
        param_values={mu: 0.1},
        x_range=(-1.5, 1.5),
        y_range=(-1.5, 1.5),
        save_path='phase_portrait.png'
    )
    print("  Saved to: phase_portrait.png")
    plt.close(fig)
    
    print("\nGenerating bifurcation diagram...")
    fig, ax = viz.plot_bifurcation_diagram(
        param=mu,
        param_range=(-1, 1),
        k=1,
        save_path='bifurcation_diagram.png'
    )
    print("  Saved to: bifurcation_diagram.png")
    plt.close(fig)
    
    return system


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("LyapunovSolver-Hybrid v2.0 - Examples")
    print("=" * 60)
    
    example_1_simple_system()
    example_2_van_der_pol()
    example_3_hamiltonian()
    example_4_two_parameters()
    example_5_batch_evaluation()
    example_6_c_code_generation()
    example_7_stability_analysis()
    example_8_visualization()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
