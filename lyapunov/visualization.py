"""
Layer 5: Visualization and LaTeX Export
Provides plotting capabilities and document generation.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path
import numpy as np

import sympy as sp
from sympy import Symbol, Expr, lambdify

from .lyapunov_system import LyapunovSystem


class LyapunovVisualizer:
    """
    Visualization tools for Lyapunov stability analysis.
    
    Features:
    - Phase portraits with vector fields
    - Bifurcation diagrams
    - Parameter sensitivity plots
    - LaTeX document generation
    
    Attributes:
        system: The LyapunovSystem to visualize
        fig_size: Default figure size
        dpi: Default DPI for saved figures
    """
    
    def __init__(
        self,
        system: LyapunovSystem,
        fig_size: Tuple[float, float] = (10, 8),
        dpi: int = 150,
        style: str = 'default'
    ):
        """
        Initialize the visualizer.
        
        Args:
            system: LyapunovSystem instance
            fig_size: Default figure size (width, height) in inches
            dpi: Resolution for saved figures
            style: Matplotlib style to use
        """
        self.system = system
        self.fig_size = fig_size
        self.dpi = dpi
        self.style = style
        
        self._check_matplotlib()
    
    def _check_matplotlib(self) -> bool:
        """Check if matplotlib is available."""
        try:
            import matplotlib
            return True
        except ImportError:
            return False
    
    def plot_phase_portrait(
        self,
        param_values: Dict[Symbol, float],
        x_range: Tuple[float, float] = (-2, 2),
        y_range: Tuple[float, float] = (-2, 2),
        grid_points: int = 20,
        show_nullclines: bool = True,
        show_trajectories: bool = True,
        n_trajectories: int = 10,
        t_max: float = 20.0,
        save_path: Optional[Union[str, Path]] = None,
        title: Optional[str] = None
    ):
        """
        Plot the phase portrait of the system.
        
        Args:
            param_values: Dictionary of parameter values
            x_range: Range for x-axis
            y_range: Range for y-axis
            grid_points: Number of grid points for vector field
            show_nullclines: Whether to show nullclines
            show_trajectories: Whether to show sample trajectories
            n_trajectories: Number of trajectories to plot
            t_max: Maximum integration time for trajectories
            save_path: Path to save figure (None = display)
            title: Plot title
            
        Returns:
            Matplotlib figure and axes
        """
        import matplotlib.pyplot as plt
        from scipy.integrate import odeint
        
        x_dot_full, y_dot_full = self.system.system_def.get_full_system()
        
        x_sym, y_sym = self.system._x, self.system._y
        
        x_dot_func = lambdify(
            [x_sym, y_sym] + self.system.params,
            x_dot_full,
            modules=['numpy']
        )
        y_dot_func = lambdify(
            [x_sym, y_sym] + self.system.params,
            y_dot_full,
            modules=['numpy']
        )
        
        param_vals = [param_values.get(p, 0.0) for p in self.system.params]
        
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        x_grid = np.linspace(x_range[0], x_range[1], grid_points)
        y_grid = np.linspace(y_range[0], y_range[1], grid_points)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        U = x_dot_func(X, Y, *param_vals)
        V = y_dot_func(X, Y, *param_vals)
        
        speed = np.sqrt(U**2 + V**2)
        speed = np.where(speed == 0, 1, speed)
        U_norm = U / speed
        V_norm = V / speed
        
        ax.quiver(X, Y, U_norm, V_norm, speed, cmap='viridis', alpha=0.7)
        
        if show_nullclines:
            x_fine = np.linspace(x_range[0], x_range[1], 200)
            y_fine = np.linspace(y_range[0], y_range[1], 200)
            X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
            
            U_fine = x_dot_func(X_fine, Y_fine, *param_vals)
            V_fine = y_dot_func(X_fine, Y_fine, *param_vals)
            
            ax.contour(X_fine, Y_fine, U_fine, levels=[0], colors='red', 
                      linestyles='--', linewidths=1.5, alpha=0.8)
            ax.contour(X_fine, Y_fine, V_fine, levels=[0], colors='blue',
                      linestyles='--', linewidths=1.5, alpha=0.8)
        
        if show_trajectories:
            import warnings
            
            def system_ode(state, t):
                x, y = state
                return [x_dot_func(x, y, *param_vals),
                        y_dot_func(x, y, *param_vals)]
            
            t_span = np.linspace(0, t_max, 1000)
            
            np.random.seed(42)
            for _ in range(n_trajectories):
                x0 = np.random.uniform(x_range[0], x_range[1])
                y0 = np.random.uniform(y_range[0], y_range[1])
                
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        sol = odeint(system_ode, [x0, y0], t_span)
                    
                    mask = (np.abs(sol[:, 0]) < 10 * max(abs(x_range[0]), abs(x_range[1]))) & \
                           (np.abs(sol[:, 1]) < 10 * max(abs(y_range[0]), abs(y_range[1])))
                    sol_filtered = sol[mask]
                    
                    if len(sol_filtered) > 1:
                        ax.plot(sol_filtered[:, 0], sol_filtered[:, 1], 'k-', linewidth=0.5, alpha=0.6)
                        ax.plot(x0, y0, 'go', markersize=3)
                except:
                    pass
        
        ax.plot(0, 0, 'ro', markersize=8, label='Origin')
        
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        if title is None:
            param_str = ", ".join([f"{p}={v:.3g}" for p, v in param_values.items()])
            title = f"Phase Portrait ({param_str})"
        ax.set_title(title, fontsize=14)
        
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig, ax
    
    def plot_bifurcation_diagram(
        self,
        param: Symbol,
        param_range: Tuple[float, float],
        k: int = 1,
        n_points: int = 200,
        other_params: Optional[Dict[Symbol, float]] = None,
        save_path: Optional[Union[str, Path]] = None,
        title: Optional[str] = None
    ):
        """
        Plot bifurcation diagram showing L_k vs parameter.
        
        Args:
            param: Parameter to vary
            param_range: Range of parameter values
            k: Order of Lyapunov coefficient
            n_points: Number of points to evaluate
            other_params: Values for other parameters
            save_path: Path to save figure
            title: Plot title
            
        Returns:
            Matplotlib figure and axes
        """
        import matplotlib.pyplot as plt
        
        other_params = other_params or {}
        
        L_k = self.system.compute_lyapunov(k)
        
        all_params = self.system.params
        eval_func = lambdify(all_params, L_k, modules=['numpy'])
        
        param_values = np.linspace(param_range[0], param_range[1], n_points)
        L_k_values = []
        
        param_idx = all_params.index(param) if param in all_params else 0
        
        for val in param_values:
            args = []
            for p in all_params:
                if p == param:
                    args.append(val)
                else:
                    args.append(other_params.get(p, 0.0))
            
            try:
                L_k_val = float(eval_func(*args))
                L_k_values.append(L_k_val)
            except:
                L_k_values.append(np.nan)
        
        L_k_values = np.array(L_k_values)
        
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        ax.plot(param_values, L_k_values, 'b-', linewidth=2, label=f'$L_{k}$')
        ax.axhline(y=0, color='r', linestyle='--', linewidth=1, alpha=0.7)
        
        zero_crossings = np.where(np.diff(np.sign(L_k_values)))[0]
        for idx in zero_crossings:
            if idx < len(param_values) - 1:
                p1, p2 = param_values[idx], param_values[idx + 1]
                L1, L2 = L_k_values[idx], L_k_values[idx + 1]
                if L1 != L2:
                    p_zero = p1 - L1 * (p2 - p1) / (L2 - L1)
                    ax.plot(p_zero, 0, 'ro', markersize=10, 
                           label=f'Bifurcation at {param}≈{p_zero:.4g}')
        
        ax.fill_between(param_values, L_k_values, 0,
                       where=(L_k_values < 0),
                       color='green', alpha=0.2, label='Supercritical')
        ax.fill_between(param_values, L_k_values, 0,
                       where=(L_k_values > 0),
                       color='red', alpha=0.2, label='Subcritical')
        
        ax.set_xlabel(f'${sp.latex(param)}$', fontsize=12)
        ax.set_ylabel(f'$L_{k}$', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best')
        
        if title is None:
            title = f'Bifurcation Diagram: $L_{k}$ vs ${sp.latex(param)}$'
        ax.set_title(title, fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig, ax
    
    def plot_parameter_sensitivity(
        self,
        param1: Symbol,
        param2: Symbol,
        param1_range: Tuple[float, float],
        param2_range: Tuple[float, float],
        k: int = 1,
        n_points: int = 50,
        other_params: Optional[Dict[Symbol, float]] = None,
        save_path: Optional[Union[str, Path]] = None,
        title: Optional[str] = None
    ):
        """
        Plot 2D parameter sensitivity showing sign of L_k.
        
        Args:
            param1, param2: Parameters to vary
            param1_range, param2_range: Ranges for parameters
            k: Order of Lyapunov coefficient
            n_points: Grid resolution
            other_params: Values for other parameters
            save_path: Path to save figure
            title: Plot title
            
        Returns:
            Matplotlib figure and axes
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
        
        other_params = other_params or {}
        
        L_k = self.system.compute_lyapunov(k)
        all_params = self.system.params
        eval_func = lambdify(all_params, L_k, modules=['numpy'])
        
        p1_vals = np.linspace(param1_range[0], param1_range[1], n_points)
        p2_vals = np.linspace(param2_range[0], param2_range[1], n_points)
        P1, P2 = np.meshgrid(p1_vals, p2_vals)
        
        L_k_grid = np.zeros_like(P1)
        
        for i in range(n_points):
            for j in range(n_points):
                args = []
                for p in all_params:
                    if p == param1:
                        args.append(P1[i, j])
                    elif p == param2:
                        args.append(P2[i, j])
                    else:
                        args.append(other_params.get(p, 0.0))
                
                try:
                    L_k_grid[i, j] = float(eval_func(*args))
                except:
                    L_k_grid[i, j] = np.nan
        
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        vmax = np.nanmax(np.abs(L_k_grid))
        if vmax == 0:
            vmax = 1
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        
        im = ax.pcolormesh(P1, P2, L_k_grid, cmap='RdBu_r', norm=norm, shading='auto')
        plt.colorbar(im, ax=ax, label=f'$L_{k}$')
        
        ax.contour(P1, P2, L_k_grid, levels=[0], colors='black', linewidths=2)
        
        ax.set_xlabel(f'${sp.latex(param1)}$', fontsize=12)
        ax.set_ylabel(f'${sp.latex(param2)}$', fontsize=12)
        
        if title is None:
            title = f'Parameter Sensitivity: $L_{k}$'
        ax.set_title(title, fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig, ax
    
    def plot_lyapunov_coefficients(
        self,
        max_k: int,
        param_values: Optional[Dict[Symbol, float]] = None,
        save_path: Optional[Union[str, Path]] = None,
        title: Optional[str] = None
    ):
        """
        Plot bar chart of Lyapunov coefficients.
        
        Args:
            max_k: Maximum order to compute
            param_values: Parameter values for numerical evaluation
            save_path: Path to save figure
            title: Plot title
            
        Returns:
            Matplotlib figure and axes
        """
        import matplotlib.pyplot as plt
        
        coefficients = self.system.compute_lyapunov_sequence(max_k)
        
        if param_values:
            values = []
            for k in range(1, max_k + 1):
                val = self.system.evaluate_lyapunov(k, param_values)
                values.append(val)
        else:
            values = [float(sp.N(coefficients[k])) if coefficients[k].is_number 
                     else 0.0 for k in range(1, max_k + 1)]
        
        fig, ax = plt.subplots(figsize=self.fig_size)
        
        k_values = list(range(1, max_k + 1))
        colors = ['green' if v < 0 else 'red' if v > 0 else 'gray' for v in values]
        
        bars = ax.bar(k_values, values, color=colors, edgecolor='black', alpha=0.7)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        ax.set_xlabel('Order k', fontsize=12)
        ax.set_ylabel('$L_k$', fontsize=12)
        ax.set_xticks(k_values)
        ax.grid(True, axis='y', alpha=0.3)
        
        if title is None:
            if param_values:
                param_str = ", ".join([f"{p}={v:.3g}" for p, v in param_values.items()])
                title = f'Lyapunov Coefficients ({param_str})'
            else:
                title = 'Lyapunov Coefficients'
        ax.set_title(title, fontsize=14)
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Supercritical ($L_k < 0$)'),
            Patch(facecolor='red', alpha=0.7, label='Subcritical ($L_k > 0$)')
        ]
        ax.legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        
        return fig, ax
    
    def generate_latex_report(
        self,
        output_path: Union[str, Path],
        max_k: int = 3,
        param_values: Optional[Dict[Symbol, float]] = None,
        include_plots: bool = True,
        plot_dir: Optional[Union[str, Path]] = None
    ) -> None:
        """
        Generate a complete LaTeX report.
        
        Args:
            output_path: Path for the .tex file
            max_k: Maximum order of coefficients to include
            param_values: Parameter values for numerical evaluation
            include_plots: Whether to include plot references
            plot_dir: Directory to save plots (relative to output)
        """
        output_path = Path(output_path)
        
        if include_plots and plot_dir:
            plot_dir = Path(plot_dir)
            plot_dir.mkdir(parents=True, exist_ok=True)
        
        lines = [
            r"\documentclass{article}",
            r"\usepackage{amsmath, amssymb}",
            r"\usepackage{graphicx}",
            r"\usepackage{booktabs}",
            r"\usepackage[margin=1in]{geometry}",
            r"",
            r"\title{Lyapunov Stability Analysis Report}",
            r"\author{Generated by LyapunovSolver-Hybrid v2.0}",
            r"\date{\today}",
            r"",
            r"\begin{document}",
            r"\maketitle",
            r"",
            r"\section{System Definition}",
            r"The analyzed system is:",
            r"\begin{align}",
            r"\dot{x} &= -y + " + sp.latex(self.system.P) + r" \\",
            r"\dot{y} &= x + " + sp.latex(self.system.Q),
            r"\end{align}",
            r""
        ]
        
        if self.system.params:
            param_str = ", ".join([f"${sp.latex(p)}$" for p in self.system.params])
            lines.append(f"with parameters: {param_str}.")
            lines.append("")
        
        if self.system.properties:
            lines.append(r"\subsection{System Properties}")
            lines.append("The system has the following properties:")
            lines.append(r"\begin{itemize}")
            for prop in self.system.properties:
                lines.append(f"  \\item {prop.name}")
            lines.append(r"\end{itemize}")
            lines.append("")
        
        lines.append(r"\section{Lyapunov Coefficients}")
        lines.append("")
        
        coefficients = self.system.compute_lyapunov_sequence(max_k)
        
        for k, L_k in coefficients.items():
            lines.append(f"\\subsection{{First Lyapunov Coefficient $L_{k}$}}" if k == 1 
                        else f"\\subsection{{Lyapunov Coefficient $L_{k}$}}")
            lines.append(r"\[")
            lines.append(f"L_{k} = {sp.latex(L_k)}")
            lines.append(r"\]")
            
            if param_values:
                val = self.system.evaluate_lyapunov(k, param_values)
                lines.append(f"Numerical value: $L_{k} = {val:.6g}$")
                
                if val < 0:
                    lines.append(r"\textbf{Supercritical} (stable limit cycle)")
                elif val > 0:
                    lines.append(r"\textbf{Subcritical} (unstable limit cycle)")
                else:
                    lines.append(r"\textbf{Degenerate} (higher order analysis needed)")
            
            lines.append("")
        
        if include_plots and plot_dir:
            lines.append(r"\section{Visualizations}")
            lines.append(r"\begin{figure}[h]")
            lines.append(r"\centering")
            lines.append(r"\includegraphics[width=0.8\textwidth]{" + str(plot_dir) + r"/phase_portrait.pdf}")
            lines.append(r"\caption{Phase portrait of the system}")
            lines.append(r"\end{figure}")
            lines.append("")
        
        lines.append(r"\section{Stability Analysis}")
        lines.append(r"\begin{itemize}")
        lines.append(r"\item $L_1 < 0$: Supercritical Hopf bifurcation (stable limit cycle emerges)")
        lines.append(r"\item $L_1 > 0$: Subcritical Hopf bifurcation (unstable limit cycle)")
        lines.append(r"\item $L_1 = 0$: Degenerate case, examine $L_2$")
        lines.append(r"\end{itemize}")
        lines.append("")
        
        lines.append(r"\end{document}")
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
    
    def create_interactive_plot(
        self,
        param_values: Dict[Symbol, float],
        x_range: Tuple[float, float] = (-2, 2),
        y_range: Tuple[float, float] = (-2, 2)
    ):
        """
        Create an interactive Plotly phase portrait.
        
        Args:
            param_values: Parameter values
            x_range, y_range: Plot ranges
            
        Returns:
            Plotly figure object
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
        except ImportError:
            raise ImportError("Plotly is required for interactive plots")
        
        x_dot_full, y_dot_full = self.system.system_def.get_full_system()
        x_sym, y_sym = self.system._x, self.system._y
        
        x_dot_func = lambdify(
            [x_sym, y_sym] + self.system.params,
            x_dot_full,
            modules=['numpy']
        )
        y_dot_func = lambdify(
            [x_sym, y_sym] + self.system.params,
            y_dot_full,
            modules=['numpy']
        )
        
        param_vals = [param_values.get(p, 0.0) for p in self.system.params]
        
        x_grid = np.linspace(x_range[0], x_range[1], 20)
        y_grid = np.linspace(y_range[0], y_range[1], 20)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        U = x_dot_func(X, Y, *param_vals)
        V = y_dot_func(X, Y, *param_vals)
        
        fig = go.Figure()
        
        fig.add_trace(go.Cone(
            x=X.flatten(),
            y=Y.flatten(),
            z=np.zeros_like(X.flatten()),
            u=U.flatten(),
            v=V.flatten(),
            w=np.zeros_like(U.flatten()),
            colorscale='Viridis',
            sizemode='absolute',
            sizeref=0.3
        ))
        
        fig.update_layout(
            title='Interactive Phase Portrait',
            scene=dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='',
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.1)
            )
        )
        
        return fig
