"""
Streamlit GUI for LyapunovSolver-Hybrid v2.0
A simple and intuitive web interface for Lyapunov stability analysis.

Run with: streamlit run lyapunov/gui.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import sympy as sp
from sympy import symbols, simplify, latex
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import warnings

from lyapunov.phase_portrait_utils import build_stream_seeds

# Configure matplotlib for complex paths
plt.rcParams['agg.path.chunksize'] = 10000
plt.rcParams['path.simplify_threshold'] = 0.5

st.set_page_config(
    page_title="LyapunovSolver v2.0",
    page_icon="📊",
    layout="wide"
)

from lyapunov.analysis import EquilibriumScanner, EquilibriumType


def init_session_state():
    """Initialize session state variables."""
    # Tab 1: Análise Completa
    if 'scanner' not in st.session_state:
        st.session_state.scanner = None
    if 'eq_points' not in st.session_state:
        st.session_state.eq_points = []
    if 'selected_hopf' not in st.session_state:
        st.session_state.selected_hopf = None
    if 'canonical_result' not in st.session_state:
        st.session_state.canonical_result = None
    if 'analysis_coeffs' not in st.session_state:
        st.session_state.analysis_coeffs = {}
    
    # Tab 2: Cálculo Direto
    if 'direct_system' not in st.session_state:
        st.session_state.direct_system = None
    if 'direct_params' not in st.session_state:
        st.session_state.direct_params = {}
    if 'direct_coeffs' not in st.session_state:
        st.session_state.direct_coeffs = {}


def clear_analysis_state():
    """Clear state for Tab 1 - called as on_click callback."""
    st.session_state.scanner = None
    st.session_state.eq_points = []
    st.session_state.selected_hopf = None
    st.session_state.canonical_result = None
    st.session_state.analysis_coeffs = {}
    # Reset selectbox by setting its key value BEFORE widget renders
    st.session_state.analysis_example = "Personalizado"


def clear_direct_state():
    """Clear state for Tab 2 - called as on_click callback."""
    st.session_state.direct_system = None
    st.session_state.direct_params = {}
    st.session_state.direct_coeffs = {}
    # Reset selectbox by setting its key value BEFORE widget renders
    st.session_state.direct_example = "Personalizado"


def clear_phase_state():
    """Clear state for Tab 3 - called as on_click callback."""
    # Reset selectbox and tracking variable
    st.session_state.phase_example = "Personalizado"
    st.session_state.phase_last_example = "Personalizado"
    # Clear input fields
    st.session_state.phase_f = ""
    st.session_state.phase_g = ""
    st.session_state.phase_params = ""
    st.session_state.phase_xmin = -3.0
    st.session_state.phase_xmax = 3.0
    st.session_state.phase_ymin = -3.0
    st.session_state.phase_ymax = 3.0


def create_lyapunov_system(P_str: str, Q_str: str, params_str: str):
    """Create a LyapunovSystem from string inputs."""
    from lyapunov.lyapunov_system import LyapunovSystem
    
    x, y = symbols('x y')
    
    params = {}
    param_list = []
    if params_str.strip():
        for p in params_str.split(','):
            p = p.strip()
            if p:
                sym = symbols(p)
                params[p] = sym
                param_list.append(sym)
    
    local_dict = {'x': x, 'y': y, **params}
    P = sp.sympify(P_str if P_str.strip() else "0", locals=local_dict)
    Q = sp.sympify(Q_str if Q_str.strip() else "0", locals=local_dict)
    
    system = LyapunovSystem(
        P, Q,
        params=param_list,
        x=x, y=y,
        enable_cache=True
    )
    
    return system, params


def sidebar_config():
    """Sidebar with global configurations only."""
    st.sidebar.header("⚙️ Configurações Globais")
    
    # Debug mode
    debug_mode = st.sidebar.checkbox("🐛 Modo Debug", value=False)
    if debug_mode:
        import lyapunov as lyap_module
        lyap_module.enable_debug_logging(level="DEBUG")
        st.sidebar.info("Logs no terminal")
    
    st.sidebar.divider()
    
    # Cache management
    if st.sidebar.button("🗑️ Limpar Cache Global"):
        try:
            import shutil
            from pathlib import Path
            
            cache_dir = Path.home() / ".lyapunov_cache"
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            
            clear_analysis_state()
            clear_direct_state()
            
            st.sidebar.success("✅ Cache limpo!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Erro: {e}")
    
    st.sidebar.divider()
    
    # Help
    st.sidebar.markdown("""
    ### 📚 Ajuda Rápida
    
    **Aba 1 - Análise Completa:**
    - Insira sistema geral
    - Encontra equilíbrios
    - Classifica automaticamente
    - Transforma para forma canônica
    
    **Aba 2 - Cálculo Direto:**
    - Sistema já na forma canônica
    - Calcula L₁, L₂, ... diretamente
    """)
    
    st.sidebar.divider()
    st.sidebar.markdown("**LyapunovSolver v2.0**")
    st.sidebar.markdown("[GitHub](https://github.com/willianpn01/LyapunovSolver)")


def tab_analise_completa():
    """Tab 1: Análise Completa & Normalização."""
    
    st.markdown("""
    Insira um sistema dinâmico geral. O software encontrará os pontos de equilíbrio, 
    classificará como hiperbólicos ou não, e permitirá transformar para forma canônica.
    """)
    
    # ============ INPUT SECTION ============
    st.subheader("📝 Definir Sistema")
    
    col_ex, col_clear = st.columns([3, 1])
    
    with col_ex:
        example = st.selectbox(
            "Exemplos pré-definidos:",
            [
                "Personalizado",
                "Hopf Normal Form",
                "Van der Pol",
                "Pitchfork",
                "Brusselator",
                "Lotka-Volterra"
            ],
            key="analysis_example"
        )
    
    with col_clear:
        st.write("")  # Spacing
        st.write("")
        st.button("🧹 Limpar Campos", key="clear_analysis", on_click=clear_analysis_state)
    
    examples = {
        "Personalizado": ("", "", ""),
        "Hopf Normal Form": ("mu*x - y - x*(x**2 + y**2)", "x + mu*y - y*(x**2 + y**2)", "mu"),
        "Van der Pol": ("y", "mu*(1 - x**2)*y - x", "mu"),
        "Pitchfork": ("mu*x - x**3", "-y", "mu"),
        "Brusselator": ("1 - (b+1)*x + a*x**2*y", "b*x - a*x**2*y", "a, b"),
        "Lotka-Volterra": ("x*(alpha - beta*y)", "y*(-gamma + delta*x)", "alpha, beta, gamma, delta")
    }
    
    default_f, default_g, default_p = examples[example]
    
    # Ensure defaults are strings
    default_f = str(default_f) if default_f else ""
    default_g = str(default_g) if default_g else ""
    default_p = str(default_p) if default_p else ""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Sistema:** ẋ = f(x,y,μ), ẏ = g(x,y,μ)")
        f_str = st.text_input("f(x, y, μ):", value=default_f, placeholder="ex: mu*x - y - x*(x**2 + y**2)")
        g_str = st.text_input("g(x, y, μ):", value=default_g, placeholder="ex: x + mu*y - y*(x**2 + y**2)")
        params_str = st.text_input("Parâmetros (separados por vírgula):", value=default_p, placeholder="ex: mu, alpha")
    
    with col2:
        st.markdown("**Configurações do Solver:**")
        
        solver_method = st.selectbox("Método:", ["auto", "symbolic", "numeric"], key="solver_method")
        
        col_cfg1, col_cfg2 = st.columns(2)
        with col_cfg1:
            n_starts = st.number_input("Seeds numéricos:", value=100, min_value=10, max_value=500, key="n_starts")
        with col_cfg2:
            timeout = st.number_input("Timeout (s):", value=30, min_value=5, max_value=120, key="timeout")
        
        # Parameter values for classification (optional)
        param_vals = {}
        if params_str.strip():
            use_numeric = st.checkbox(
                "Usar valores numéricos para classificação",
                value=False,
                help="Se desativado, a classificação será simbólica (pode indicar 'Indeterminado' se depender dos parâmetros)"
            )
            if use_numeric:
                st.markdown("**Valores dos parâmetros:**")
                param_names = [p.strip() for p in params_str.split(',') if p.strip()]
                cols_p = st.columns(min(len(param_names), 4))
                for i, p_name in enumerate(param_names):
                    with cols_p[i % 4]:
                        param_vals[symbols(p_name)] = st.number_input(
                            f"{p_name}:", value=0.0, step=0.1, key=f"param_{p_name}"
                        )
    
    # Analyze button
    if st.button("🔍 Analisar Equilíbrios", type="primary", use_container_width=True):
        if not f_str.strip() or not g_str.strip():
            st.error("Por favor, insira as expressões f e g.")
        else:
            try:
                x, y = symbols('x y')
                
                # Parse parameters
                param_list = []
                local_dict = {'x': x, 'y': y}
                if params_str.strip():
                    for p in params_str.split(','):
                        p = p.strip()
                        if p:
                            sym = symbols(p)
                            param_list.append(sym)
                            local_dict[p] = sym
                
                # Rebuild param_vals with correct symbols from param_list
                final_param_vals = None
                if param_vals:
                    final_param_vals = {}
                    for sym in param_list:
                        # Find matching value by symbol name
                        for k, v in param_vals.items():
                            if str(k) == str(sym):
                                final_param_vals[sym] = v
                                break
                
                # Parse expressions
                f_expr = sp.sympify(f_str, locals=local_dict)
                g_expr = sp.sympify(g_str, locals=local_dict)
                
                with st.spinner("Analisando sistema..."):
                    # Create scanner
                    scanner = EquilibriumScanner(
                        f_expr, g_expr, x, y, param_list,
                        timeout=timeout
                    )
                    
                    # Scan for equilibria
                    points = scanner.scan(
                        param_values=final_param_vals,
                        method=solver_method,
                        n_starts=n_starts
                    )
                    
                    st.session_state.scanner = scanner
                    st.session_state.eq_points = points
                    st.session_state.eq_param_vals = param_vals
                    st.session_state.selected_hopf = None
                    st.session_state.canonical_result = None
                    st.session_state.analysis_coeffs = {}
                
                st.success(f"✅ Encontrados {len(points)} ponto(s) de equilíbrio!")
                
            except Exception as e:
                st.error(f"❌ Erro: {e}")
    
    # ============ RESULTS SECTION ============
    if st.session_state.eq_points:
        st.divider()
        st.subheader("📊 Pontos de Equilíbrio Encontrados")
        
        points = st.session_state.eq_points
        
        # Summary metrics
        hopf_count = sum(1 for p in points if p.eq_type == EquilibriumType.HOPF_CANDIDATE)
        stable_count = sum(1 for p in points if p.stability == 'stable')
        unstable_count = sum(1 for p in points if p.stability == 'unstable')
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Total", len(points))
        col_m2.metric("Candidatos Hopf", hopf_count, delta="não-hiperbólico" if hopf_count > 0 else None)
        col_m3.metric("Estáveis", stable_count)
        col_m4.metric("Instáveis", unstable_count)
        
        # Points table
        st.markdown("**Selecione um ponto para análise detalhada:**")
        
        for pt in points:
            is_hopf = pt.eq_type == EquilibriumType.HOPF_CANDIDATE
            icon = "🎯" if is_hopf else "📍"
            
            with st.expander(f"{icon} [{pt.index}] ({pt.x}, {pt.y}) — {pt.type_display}", expanded=is_hopf):
                col_info1, col_info2, col_info3 = st.columns([1, 1, 1])
                
                with col_info1:
                    st.markdown(f"**Tipo:** {pt.type_display}")
                    st.markdown(f"**Estabilidade:** {pt.stability}")
                
                with col_info2:
                    if pt.eigenvalues:
                        eig_str = ", ".join([str(e) for e in pt.eigenvalues])
                        st.markdown(f"**Autovalores:** {eig_str}")
                    st.markdown(f"**Método:** {pt.solver_method}")
                
                with col_info3:
                    if pt.hopf_frequency:
                        st.markdown(f"**Frequência ω:** {pt.hopf_frequency:.4f} rad/s")
                    if pt.trace is not None:
                        st.markdown(f"**Traço:** {pt.trace}")
                    if pt.determinant is not None:
                        st.markdown(f"**Det:** {pt.determinant}")
                
                # Show classification reasoning
                if pt.classification_reason:
                    st.divider()
                    st.info(f"📋 **Justificativa:** {pt.classification_reason}")
                
                # Show hypotheses if classification is unknown/conditional
                if pt.hypotheses and pt.eq_type == EquilibriumType.UNKNOWN:
                    st.divider()
                    st.markdown("⚠️ **Classificação depende dos parâmetros:**")
                    for hyp in pt.hypotheses:
                        st.markdown(f"- Se `{hyp.condition}` → **{hyp.eq_type.value}**")
                        st.caption(f"  _{hyp.description}_")
                
                # Action buttons - normalization available for all points
                st.divider()
                
                # Determine if point is suitable for Lyapunov analysis
                can_normalize = pt.eq_type in [
                    EquilibriumType.HOPF_CANDIDATE, 
                    EquilibriumType.CENTER,
                    EquilibriumType.FOCUS_STABLE,
                    EquilibriumType.FOCUS_UNSTABLE
                ]
                
                if can_normalize:
                    col_btn1, col_btn2 = st.columns(2)
                    
                    with col_btn1:
                        if st.button("📐 Transformar para Forma Canônica", key=f"transform_{pt.index}"):
                            st.session_state.selected_hopf = pt
                            try:
                                canonical = st.session_state.scanner.to_canonical_form(pt)
                                st.session_state.canonical_result = canonical
                                st.session_state.analysis_coeffs = {}
                            except Exception as e:
                                st.error(f"Erro na transformação: {e}")
                    
                    with col_btn2:
                        if st.button("🚀 Análise Completa (Auto)", key=f"auto_{pt.index}"):
                            st.session_state.selected_hopf = pt
                            with st.spinner("Transformando e calculando coeficientes..."):
                                try:
                                    result = st.session_state.scanner.analyze_hopf_point(pt, max_k=3)
                                    st.session_state.canonical_result = result['canonical_form']
                                    st.session_state.analysis_coeffs = result['lyapunov_coefficients']
                                except Exception as e:
                                    st.error(f"Erro: {e}")
                    
                    # Explain why normalization is useful
                    if pt.eq_type == EquilibriumType.CENTER:
                        st.caption("💡 Centro linear detectado. A normalização permite calcular L₁ para determinar se é um centro verdadeiro ou foco fraco.")
                    elif pt.eq_type in [EquilibriumType.FOCUS_STABLE, EquilibriumType.FOCUS_UNSTABLE]:
                        st.caption("💡 Foco detectado. A normalização permite analisar a estrutura não-linear do sistema.")
                else:
                    # For saddles, nodes, etc. - explain why normalization doesn't apply
                    if pt.eq_type == EquilibriumType.SADDLE:
                        st.caption("ℹ️ Selas são pontos hiperbólicos. A análise de Lyapunov não se aplica (não há ciclos limite próximos).")
                    elif pt.eq_type in [EquilibriumType.NODE_STABLE, EquilibriumType.NODE_UNSTABLE]:
                        st.caption("ℹ️ Nós são pontos hiperbólicos. A análise de Lyapunov não se aplica.")
                    elif pt.eq_type == EquilibriumType.UNKNOWN:
                        st.caption("⚠️ Classificação indeterminada. Forneça valores numéricos para os parâmetros para habilitar a normalização.")
        
        # ============ CANONICAL FORM SECTION ============
        if st.session_state.canonical_result is not None:
            st.divider()
            st.subheader("🔬 Forma Canônica")
            
            canonical = st.session_state.canonical_result
            pt = st.session_state.selected_hopf
            
            st.markdown(f"**Ponto de equilíbrio:** ({pt.x}, {pt.y})")
            
            col_can1, col_can2 = st.columns(2)
            
            with col_can1:
                st.markdown("**Sistema transformado:**")
                st.latex(r"\dot{x} = -\omega y + P(x,y)")
                st.latex(r"\dot{y} = \omega x + Q(x,y)")
                st.markdown(f"**ω =** {canonical.omega}")
            
            with col_can2:
                st.markdown("**Termos não-lineares:**")
                st.latex(f"P(x,y) = {latex(canonical.P)}")
                st.latex(f"Q(x,y) = {latex(canonical.Q)}")
            
            # Lyapunov coefficients
            if not st.session_state.analysis_coeffs:
                st.divider()
                col_calc1, col_calc2 = st.columns([2, 1])
                
                with col_calc1:
                    max_k = st.slider("Ordem máxima dos coeficientes:", 1, 5, 2, key="max_k_analysis")
                
                with col_calc2:
                    st.write("")
                    if st.button("🧮 Calcular Coeficientes de Lyapunov", key="calc_lyap"):
                        with st.spinner("Calculando..."):
                            try:
                                result = st.session_state.scanner.analyze_hopf_point(pt, max_k=max_k)
                                st.session_state.analysis_coeffs = result['lyapunov_coefficients']
                            except Exception as e:
                                st.error(f"Erro: {e}")
            
            # Display coefficients
            if st.session_state.analysis_coeffs:
                st.divider()
                st.subheader("📊 Coeficientes de Lyapunov")
                
                for k, Lk in st.session_state.analysis_coeffs.items():
                    col_lk1, col_lk2 = st.columns([3, 1])
                    
                    with col_lk1:
                        st.latex(f"L_{k} = {latex(Lk)}")
                    
                    with col_lk2:
                        try:
                            val = float(Lk.evalf())
                            if val < -1e-12:
                                st.success(f"≈ {val:.6g} (Supercrítico)")
                            elif val > 1e-12:
                                st.warning(f"≈ {val:.6g} (Subcrítico)")
                            else:
                                st.info("≈ 0 (Degenerado)")
                        except:
                            st.info("Simbólico")
                
                # Export
                if st.button("📄 Exportar LaTeX", key="export_analysis"):
                    lines = [
                        r"\textbf{Sistema Original:}",
                        f"$\\dot{{x}} = {latex(st.session_state.scanner.f)}$",
                        f"$\\dot{{y}} = {latex(st.session_state.scanner.g)}$",
                        "",
                        r"\textbf{Ponto de Equilíbrio:}",
                        f"$({latex(pt.x)}, {latex(pt.y)})$",
                        "",
                        r"\textbf{Forma Canônica:}",
                        f"$P(x,y) = {latex(canonical.P)}$",
                        f"$Q(x,y) = {latex(canonical.Q)}$",
                        f"$\\omega = {latex(canonical.omega)}$",
                        "",
                        r"\textbf{Coeficientes de Lyapunov:}",
                    ]
                    for k, Lk in st.session_state.analysis_coeffs.items():
                        lines.append(f"$L_{{{k}}} = {latex(Lk)}$")
                    
                    latex_code = "\n".join(lines)
                    st.code(latex_code, language="latex")
                    st.download_button("📥 Baixar .tex", latex_code, "analise_hopf.tex", "text/plain")


def tab_calculo_direto():
    """Tab 2: Cálculo Direto (Sistema Canônico)."""
    
    st.markdown("""
    Insira um sistema que **já está na forma canônica** (ẋ = -y + P, ẏ = x + Q).
    O cálculo dos coeficientes de Lyapunov será feito diretamente, sem busca de equilíbrios.
    """)
    
    # ============ INPUT SECTION ============
    st.subheader("📝 Sistema na Forma Canônica")
    
    col_ex, col_clear = st.columns([3, 1])
    
    with col_ex:
        example = st.selectbox(
            "Exemplos pré-definidos:",
            [
                "Personalizado",
                "Cúbico Simples",
                "Sistema Simétrico",
                "Hamiltoniano (L₁=0)",
                "Dois Parâmetros",
                "Teste L2 (Mathematica)"
            ],
            key="direct_example"
        )
    
    with col_clear:
        st.write("")
        st.write("")
        st.button("🧹 Limpar Campos", key="clear_direct", on_click=clear_direct_state)
    
    examples = {
        "Personalizado": ("", "", ""),
        "Cúbico Simples": ("mu*x - x**3", "-y**3", "mu"),
        "Sistema Simétrico": ("-x**3 - x*y**2", "-x**2*y - y**3", ""),
        "Hamiltoniano (L₁=0)": ("x**2*y", "-x*y**2", ""),
        "Dois Parâmetros": ("alpha*x - beta*x**2*y - x**3", "alpha*y + beta*x*y**2 - y**3", "alpha, beta"),
        "Teste L2 (Mathematica)": ("a2*x**2 + a3*x**3", "b2*y**2 + b3*y**3", "a2, a3, b2, b3")
    }
    
    default_P, default_Q, default_params = examples[example]
    
    # Ensure defaults are strings
    default_P = str(default_P) if default_P else ""
    default_Q = str(default_Q) if default_Q else ""
    default_params = str(default_params) if default_params else ""
    
    st.markdown("**Sistema:** ẋ = -y + P(x,y), ẏ = x + Q(x,y)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        P_str = st.text_input("P(x, y, μ):", value=default_P, placeholder="ex: mu*x - x**3")
        Q_str = st.text_input("Q(x, y, μ):", value=default_Q, placeholder="ex: -y**3")
    
    with col2:
        params_str = st.text_input("Parâmetros:", value=default_params, placeholder="ex: mu, alpha")
        max_k = st.slider("Ordem máxima k:", 1, 10, 3, key="direct_max_k")
        
        if max_k > 5:
            st.warning("⚠️ Ordens > 5 podem demorar vários minutos.")
    
    # Create system button
    if st.button("🚀 Criar Sistema e Calcular", type="primary", use_container_width=True):
        if not P_str.strip() and not Q_str.strip():
            st.error("Por favor, insira pelo menos P ou Q.")
        else:
            try:
                system, params = create_lyapunov_system(P_str, Q_str, params_str)
                st.session_state.direct_system = system
                st.session_state.direct_params = params
                st.session_state.direct_coeffs = {}
                
                # Calculate coefficients
                with st.spinner("Calculando coeficientes..."):
                    progress = st.progress(0)
                    for k in range(1, max_k + 1):
                        L_k = system.compute_lyapunov(k)
                        st.session_state.direct_coeffs[k] = L_k
                        progress.progress(k / max_k)
                    progress.empty()
                
                st.success("✅ Cálculo concluído!")
                
            except Exception as e:
                st.error(f"❌ Erro: {e}")
    
    # ============ RESULTS SECTION ============
    if st.session_state.direct_system is not None:
        st.divider()
        
        system = st.session_state.direct_system
        params = st.session_state.direct_params
        
        col_sys1, col_sys2 = st.columns(2)
        
        with col_sys1:
            st.subheader("📐 Sistema Definido")
            st.latex(r"\dot{x} = -y + " + latex(system.P))
            st.latex(r"\dot{y} = x + " + latex(system.Q))
            
            if params:
                st.markdown(f"**Parâmetros:** {', '.join(params.keys())}")
            
            props = [p.name for p in system.properties]
            if props:
                st.markdown(f"**Propriedades detectadas:** {', '.join(props)}")
        
        with col_sys2:
            if st.session_state.direct_coeffs:
                st.subheader("📊 Coeficientes de Lyapunov")
                
                for k, L_k in st.session_state.direct_coeffs.items():
                    col_a, col_b = st.columns([2, 1])
                    
                    with col_a:
                        st.latex(f"L_{k} = " + latex(L_k))
                    
                    with col_b:
                        if simplify(L_k) == 0:
                            st.info("= 0")
                        elif not L_k.free_symbols - {symbols('x'), symbols('y')}:
                            try:
                                val = float(L_k.evalf())
                                if val < 0:
                                    st.success(f"≈ {val:.6g}")
                                else:
                                    st.warning(f"≈ {val:.6g}")
                            except:
                                pass
        
        # Numerical evaluation
        if params and st.session_state.direct_coeffs:
            st.divider()
            st.subheader("🔢 Avaliação Numérica")
            
            param_values = {}
            cols = st.columns(min(len(params), 4))
            for i, (name, sym) in enumerate(params.items()):
                with cols[i % 4]:
                    val = st.number_input(f"{name}:", value=0.0, step=0.1, key=f"eval_{name}")
                    param_values[sym] = val
            
            if st.button("Avaliar", key="eval_direct"):
                st.markdown("**Resultados:**")
                for k in st.session_state.direct_coeffs.keys():
                    val = system.evaluate_lyapunov(k, param_values)
                    
                    if val < -1e-12:
                        st.success(f"L_{k} = {val:.10g} → **Supercrítico** (ciclo limite estável)")
                    elif val > 1e-12:
                        st.warning(f"L_{k} = {val:.10g} → **Subcrítico** (ciclo limite instável)")
                    else:
                        st.info(f"L_{k} = {val:.10g} → **Degenerado**")
        
        # Export
        st.divider()
        if st.button("📄 Exportar LaTeX", key="export_direct"):
            lines = [
                r"\begin{align}",
                r"\dot{x} &= -y + " + latex(system.P) + r" \\",
                r"\dot{y} &= x + " + latex(system.Q),
                r"\end{align}",
                "",
                r"\textbf{Coeficientes de Lyapunov:}",
                ""
            ]
            
            for k, L_k in st.session_state.direct_coeffs.items():
                lines.append(f"$L_{{{k}}} = {latex(L_k)}$")
                lines.append("")
            
            latex_code = "\n".join(lines)
            st.code(latex_code, language="latex")
            st.download_button("📥 Baixar .tex", latex_code, "lyapunov_direto.tex", "text/plain")


def tab_retrato_fase():
    """Tab 3: Phase Portrait Plotter."""
    
    st.markdown("""
    Visualize o retrato de fase de sistemas dinâmicos 2D. 
    Varie os parâmetros para observar bifurcações (Hopf, Sela-Nó, etc.).
    """)
    
    # ============ INPUT SECTION ============
    st.subheader("📝 Definir Sistema")
    
    col_ex, col_clear = st.columns([3, 1])
    
    with col_ex:
        example = st.selectbox(
            "Exemplos pré-definidos:",
            [
                "Personalizado",
                "Hopf Normal Form",
                "Van der Pol",
                "Lotka-Volterra",
                "Duffing",
                "Sela-Nó"
            ],
            key="phase_example"
        )
    
    with col_clear:
        st.write("")
        st.write("")
        st.button("🧹 Limpar", key="clear_phase", on_click=clear_phase_state)
    
    examples = {
        "Personalizado": ("", "", "", (-3, 3), (-3, 3)),
        "Hopf Normal Form": ("mu*x - y - x*(x**2 + y**2)", "x + mu*y - y*(x**2 + y**2)", "mu", (-2, 2), (-2, 2)),
        "Van der Pol": ("y", "mu*(1 - x**2)*y - x", "mu", (-4, 4), (-4, 4)),
        "Lotka-Volterra": ("x*(a - b*y)", "y*(-c + d*x)", "a, b, c, d", (0, 5), (0, 5)),
        "Duffing": ("y", "-delta*y - alpha*x - beta*x**3", "alpha, beta, delta", (-2, 2), (-2, 2)),
        "Sela-Nó": ("mu - x**2", "y", "mu", (-2, 2), (-2, 2))
    }
    
    default_f, default_g, default_p, default_xlim, default_ylim = examples[example]
    
    # Ensure defaults are strings
    default_f = str(default_f) if default_f else ""
    default_g = str(default_g) if default_g else ""
    default_p = str(default_p) if default_p else ""
    
    # Initialize session state for phase inputs if not present
    if 'phase_f' not in st.session_state:
        st.session_state.phase_f = default_f
    if 'phase_g' not in st.session_state:
        st.session_state.phase_g = default_g
    if 'phase_params' not in st.session_state:
        st.session_state.phase_params = default_p
    if 'phase_xmin' not in st.session_state:
        st.session_state.phase_xmin = float(default_xlim[0])
    if 'phase_xmax' not in st.session_state:
        st.session_state.phase_xmax = float(default_xlim[1])
    if 'phase_ymin' not in st.session_state:
        st.session_state.phase_ymin = float(default_ylim[0])
    if 'phase_ymax' not in st.session_state:
        st.session_state.phase_ymax = float(default_ylim[1])
    if 'phase_last_example' not in st.session_state:
        st.session_state.phase_last_example = "Personalizado"
    
    # Update session state when example changes
    if example != st.session_state.phase_last_example:
        st.session_state.phase_f = default_f
        st.session_state.phase_g = default_g
        st.session_state.phase_params = default_p
        st.session_state.phase_xmin = float(default_xlim[0])
        st.session_state.phase_xmax = float(default_xlim[1])
        st.session_state.phase_ymin = float(default_ylim[0])
        st.session_state.phase_ymax = float(default_ylim[1])
        st.session_state.phase_last_example = example
        st.rerun()  # Rerun to apply changes immediately
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Sistema:** ẋ = f(x,y,μ), ẏ = g(x,y,μ)")
        f_str = st.text_input("f(x, y, μ):", placeholder="ex: mu*x - y", key="phase_f")
        g_str = st.text_input("g(x, y, μ):", placeholder="ex: x + mu*y", key="phase_g")
        params_str = st.text_input("Parâmetros:", placeholder="ex: mu, alpha", key="phase_params")
    
    with col2:
        st.markdown("**Configurações do Plot:**")
        
        col_lim1, col_lim2 = st.columns(2)
        with col_lim1:
            x_min = st.number_input("x min:", key="phase_xmin")
            y_min = st.number_input("y min:", key="phase_ymin")
        with col_lim2:
            x_max = st.number_input("x max:", key="phase_xmax")
            y_max = st.number_input("y max:", key="phase_ymax")
        
        n_trajectories = st.slider("Número de trajetórias:", 5, 50, 20, key="phase_n_traj")
        t_max = st.slider("Tempo de integração:", 1.0, 50.0, 10.0, key="phase_tmax")
        st.divider()
        use_streamplot = st.checkbox("Usar streamlines (recomendado)", value=True, key="phase_use_streamplot")
        stream_density = st.slider("Densidade das streamlines:", 0.5, 3.0, 1.5, 0.1, key="phase_stream_density")
        seed_mode = st.selectbox(
            "Modo de seeds (streamlines):",
            ["Automático", "Grade (controlado)", "Borda (controlado)"],
            index=0,
            key="phase_seed_mode"
        )
        seed_grid_n = st.slider("Seeds da grade (N x N):", 3, 30, 12, 1, key="phase_seed_grid_n")
        seed_border_n = st.slider("Seeds na borda (por lado):", 5, 80, 25, 1, key="phase_seed_border_n")

        if 'phase_trajectories' not in st.session_state:
            st.session_state.phase_trajectories = []

        draw_trajectories = st.checkbox("Desenhar trajetória (solve_ivp) a partir de (x0,y0)", value=False, key="phase_draw_traj")
        traj_col1, traj_col2 = st.columns(2)
        with traj_col1:
            traj_x0 = st.number_input("x0:", value=0.0, key="phase_traj_x0")
        with traj_col2:
            traj_y0 = st.number_input("y0:", value=0.0, key="phase_traj_y0")

        traj_btn1, traj_btn2 = st.columns(2)
        with traj_btn1:
            if st.button("➕ Adicionar trajetória", key="phase_add_traj", use_container_width=True):
                st.session_state.phase_trajectories.append((float(traj_x0), float(traj_y0)))
        with traj_btn2:
            if st.button("🗑️ Limpar trajetórias", key="phase_clear_traj", use_container_width=True):
                st.session_state.phase_trajectories = []

        st.info(
            "**Como interpretar estas opções:**\n"
            "- **Streamlines (recomendado):** desenha linhas do campo vetorial preenchendo o domínio. É a melhor opção para ver **topologia** (separatrizes, regiões atratoras/repulsoras) e mudanças em **bifurcações**.\n"
            "- **Densidade:** controla quantas streamlines são desenhadas. Aumente para revelar detalhes perto de pontos críticos; reduza se ficar muito carregado.\n"
            "- **Seeds controlados:** força o `streamplot` a iniciar linhas em pontos escolhidos. Use **Grade** para preencher uniformemente o domínio e **Borda** para evidenciar separatrizes/fluxo entrando/saindo da janela.\n"
            "- **Trajetória (solve_ivp):** integra uma órbita a partir de uma condição inicial **(x0,y0)** que você fornece. Use **Adicionar trajetória** para desenhar uma a uma (fica salvo na sessão)."
        )
    
    # Parameter values
    param_vals = {}
    if params_str.strip():
        st.subheader("🎚️ Valores dos Parâmetros")
        param_names = [p.strip() for p in params_str.split(',') if p.strip()]
        
        cols_p = st.columns(min(len(param_names), 4))
        for i, p_name in enumerate(param_names):
            with cols_p[i % 4]:
                param_vals[p_name] = st.slider(
                    f"{p_name}:", -5.0, 5.0, 0.0, 0.1, key=f"phase_param_{p_name}"
                )
    
    # Plot button
    if st.button("🎨 Gerar Retrato de Fase", type="primary", use_container_width=True):
        if not f_str.strip() or not g_str.strip():
            st.error("Por favor, insira as expressões f e g.")
        else:
            try:
                x, y = symbols('x y')
                
                # Parse parameters
                local_dict = {'x': x, 'y': y}
                for p_name in param_vals.keys():
                    local_dict[p_name] = symbols(p_name)
                
                # Parse expressions
                f_expr = sp.sympify(f_str, locals=local_dict)
                g_expr = sp.sympify(g_str, locals=local_dict)
                
                # Substitute parameter values
                for p_name, p_val in param_vals.items():
                    f_expr = f_expr.subs(symbols(p_name), p_val)
                    g_expr = g_expr.subs(symbols(p_name), p_val)
                
                # Create numerical functions
                f_func = sp.lambdify((x, y), f_expr, 'numpy')
                g_func = sp.lambdify((x, y), g_expr, 'numpy')
                
                def system_ivp(t, state):
                    """System for solve_ivp (note: t, state order)."""
                    x_val, y_val = state
                    return [f_func(x_val, y_val), g_func(x_val, y_val)]
                
                # Event to stop integration when leaving bounds
                def out_of_bounds(t, state):
                    margin = 2.0
                    x_val, y_val = state
                    # Returns 0 when we hit the boundary
                    dx = min(x_val - (x_min - margin), (x_max + margin) - x_val)
                    dy = min(y_val - (y_min - margin), (y_max + margin) - y_val)
                    return min(dx, dy)
                out_of_bounds.terminal = True
                out_of_bounds.direction = -1
                
                # Create figure
                fig, ax = plt.subplots(figsize=(10, 8))
                
                # Plot vector field
                x_range = np.linspace(x_min, x_max, 20)
                y_range = np.linspace(y_min, y_max, 20)
                X, Y = np.meshgrid(x_range, y_range)
                
                U = np.zeros_like(X)
                V = np.zeros_like(Y)
                
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        try:
                            U[i, j] = f_func(X[i, j], Y[i, j])
                            V[i, j] = g_func(X[i, j], Y[i, j])
                        except:
                            U[i, j] = 0
                            V[i, j] = 0
                
                # Normalize arrows
                magnitude = np.sqrt(U**2 + V**2)
                magnitude[magnitude == 0] = 1
                U_norm = U / magnitude
                V_norm = V / magnitude

                # Primary renderer: streamlines (fills the domain and preserves topology)
                if use_streamplot:
                    try:
                        # Use the original vector field (U,V) for streamlines
                        start_points = build_stream_seeds(
                            seed_mode,
                            x_min=x_min,
                            x_max=x_max,
                            y_min=y_min,
                            y_max=y_max,
                            grid_n=int(seed_grid_n),
                            border_n=int(seed_border_n),
                        )
                        ax.streamplot(
                            X, Y, U, V,
                            density=stream_density,
                            color=np.log1p(magnitude),
                            cmap='coolwarm',
                            linewidth=1.0,
                            arrowsize=1.0,
                            start_points=start_points
                        )
                    except Exception:
                        # Fallback to quiver if streamplot fails
                        ax.quiver(X, Y, U_norm, V_norm, magnitude, cmap='coolwarm', alpha=0.6)
                else:
                    ax.quiver(X, Y, U_norm, V_norm, magnitude, cmap='coolwarm', alpha=0.6)

                # Optional: user-controlled trajectories
                if draw_trajectories and st.session_state.phase_trajectories:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        for x0, y0 in st.session_state.phase_trajectories:
                            try:
                                sol = solve_ivp(
                                    system_ivp, [0, t_max], [x0, y0],
                                    events=out_of_bounds,
                                    max_step=max(0.02, t_max / 400)
                                )
                                if sol.success and len(sol.t) > 1:
                                    ax.plot(sol.y[0], sol.y[1], 'k-', linewidth=1.2, alpha=0.9)

                                sol_back = solve_ivp(
                                    system_ivp, [0, -t_max], [x0, y0],
                                    events=out_of_bounds,
                                    max_step=max(0.02, t_max / 400)
                                )
                                if sol_back.success and len(sol_back.t) > 1:
                                    ax.plot(sol_back.y[0], sol_back.y[1], 'k-', linewidth=1.2, alpha=0.9)
                            except:
                                pass
                
                # Find and plot equilibrium points
                try:
                    from scipy.optimize import fsolve
                    
                    eq_points = []
                    for _ in range(20):
                        x0_eq = np.random.uniform(x_min, x_max)
                        y0_eq = np.random.uniform(y_min, y_max)
                        
                        def eq_system(state):
                            return [f_func(state[0], state[1]), g_func(state[0], state[1])]
                        
                        try:
                            sol = fsolve(eq_system, [x0_eq, y0_eq], full_output=True)
                            if sol[2] == 1:  # Solution found
                                pt = sol[0]
                                # Check if it's a new point
                                is_new = True
                                for existing in eq_points:
                                    if np.linalg.norm(np.array(pt) - np.array(existing)) < 0.1:
                                        is_new = False
                                        break
                                if is_new and x_min <= pt[0] <= x_max and y_min <= pt[1] <= y_max:
                                    eq_points.append(pt)
                        except:
                            pass
                    
                    for pt in eq_points:
                        ax.plot(pt[0], pt[1], 'ro', markersize=10, markeredgecolor='black', markeredgewidth=2)
                except:
                    pass
                
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_xlabel('x', fontsize=12)
                ax.set_ylabel('y', fontsize=12)
                ax.set_title(f'Retrato de Fase: ẋ = {f_str}, ẏ = {g_str}', fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal', adjustable='box')
                
                # Add parameter info
                if param_vals:
                    param_text = ", ".join([f"{k}={v:.2f}" for k, v in param_vals.items()])
                    ax.text(0.02, 0.98, f"Parâmetros: {param_text}", transform=ax.transAxes, 
                           fontsize=10, verticalalignment='top', 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Legend
                st.markdown("""
                **Legenda:**
                - 🔴 **Pontos vermelhos**: Pontos de equilíbrio
                - 🔵 **Linhas azuis**: Trajetórias do sistema
                - **Setas coloridas**: Campo vetorial (cor indica magnitude)
                """)
                
            except Exception as e:
                st.error(f"❌ Erro: {e}")
    
    # Bifurcation animation section
    st.divider()
    st.subheader("🎬 Animação de Bifurcação")
    
    st.markdown("""
    Varie um parâmetro para visualizar a bifurcação. 
    Use os sliders acima para ajustar os parâmetros e observe como o retrato de fase muda.
    """)
    
    if params_str.strip():
        param_names = [p.strip() for p in params_str.split(',') if p.strip()]
        
        if len(param_names) > 0:
            selected_param = st.selectbox("Parâmetro para variar:", param_names, key="bifurc_param")
            
            col_bif1, col_bif2, col_bif3 = st.columns(3)
            with col_bif1:
                bif_start = st.number_input("Valor inicial:", value=-1.0, key="bif_start")
            with col_bif2:
                bif_end = st.number_input("Valor final:", value=1.0, key="bif_end")
            with col_bif3:
                bif_steps = st.number_input("Número de frames:", value=5, min_value=2, max_value=20, key="bif_steps")
            
            if st.button("🎥 Gerar Sequência de Bifurcação", use_container_width=True):
                if not f_str.strip() or not g_str.strip():
                    st.error("Por favor, insira as expressões f e g.")
                else:
                    try:
                        x, y = symbols('x y')
                        
                        # Parse parameters
                        local_dict = {'x': x, 'y': y}
                        for p_name in param_vals.keys():
                            local_dict[p_name] = symbols(p_name)
                        
                        # Parse expressions
                        f_expr_base = sp.sympify(f_str, locals=local_dict)
                        g_expr_base = sp.sympify(g_str, locals=local_dict)
                        
                        param_values_range = np.linspace(bif_start, bif_end, int(bif_steps))

                        bif_start_points = build_stream_seeds(
                            seed_mode,
                            x_min=x_min,
                            x_max=x_max,
                            y_min=y_min,
                            y_max=y_max,
                            grid_n=int(seed_grid_n),
                            border_n=int(seed_border_n),
                        )
                        
                        # Create grid of plots
                        n_cols = min(3, int(bif_steps))
                        n_rows = int(np.ceil(bif_steps / n_cols))
                        
                        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
                        if n_rows == 1 and n_cols == 1:
                            axes = np.array([[axes]])
                        elif n_rows == 1:
                            axes = axes.reshape(1, -1)
                        elif n_cols == 1:
                            axes = axes.reshape(-1, 1)
                        
                        for idx, param_val in enumerate(param_values_range):
                            row = idx // n_cols
                            col = idx % n_cols
                            ax = axes[row, col]
                            
                            # Substitute all parameters
                            f_expr = f_expr_base
                            g_expr = g_expr_base
                            
                            for p_name, p_val in param_vals.items():
                                if p_name == selected_param:
                                    f_expr = f_expr.subs(symbols(p_name), param_val)
                                    g_expr = g_expr.subs(symbols(p_name), param_val)
                                else:
                                    f_expr = f_expr.subs(symbols(p_name), p_val)
                                    g_expr = g_expr.subs(symbols(p_name), p_val)
                            
                            # Create numerical functions
                            f_func = sp.lambdify((x, y), f_expr, 'numpy')
                            g_func = sp.lambdify((x, y), g_expr, 'numpy')
                            
                            def system_ivp_bif(t, state):
                                return [f_func(state[0], state[1]), g_func(state[0], state[1])]
                            
                            # Event to stop at boundary
                            def out_of_bounds_bif(t, state):
                                margin = 2.0
                                dx = min(state[0] - (x_min - margin), (x_max + margin) - state[0])
                                dy = min(state[1] - (y_min - margin), (y_max + margin) - state[1])
                                return min(dx, dy)
                            out_of_bounds_bif.terminal = True
                            out_of_bounds_bif.direction = -1
                            
                            # Plot vector field
                            x_range = np.linspace(x_min, x_max, 15)
                            y_range = np.linspace(y_min, y_max, 15)
                            X, Y = np.meshgrid(x_range, y_range)
                            
                            U = np.zeros_like(X)
                            V = np.zeros_like(Y)
                            
                            for i in range(X.shape[0]):
                                for j in range(X.shape[1]):
                                    try:
                                        U[i, j] = f_func(X[i, j], Y[i, j])
                                        V[i, j] = g_func(X[i, j], Y[i, j])
                                    except:
                                        U[i, j] = 0
                                        V[i, j] = 0
                            
                            magnitude = np.sqrt(U**2 + V**2)
                            magnitude[magnitude == 0] = 1
                            U_norm = U / magnitude
                            V_norm = V / magnitude

                            # Prefer streamlines for topology in the bifurcation grid
                            try:
                                ax.streamplot(
                                    X, Y, U, V,
                                    density=max(0.8, stream_density * 0.9),
                                    color=np.log1p(magnitude),
                                    cmap='coolwarm',
                                    linewidth=0.9,
                                    arrowsize=0.9,
                                    start_points=bif_start_points
                                )
                            except Exception:
                                ax.quiver(X, Y, U_norm, V_norm, magnitude, cmap='coolwarm', alpha=0.6)

                            # Trajectories are intentionally omitted in the bifurcation grid to keep frames stable
                            
                            ax.set_xlim(x_min, x_max)
                            ax.set_ylim(y_min, y_max)
                            ax.set_title(f'{selected_param} = {param_val:.3f}', fontsize=11)
                            ax.set_aspect('equal', adjustable='box')
                        
                        # Hide empty subplots
                        for idx in range(int(bif_steps), n_rows * n_cols):
                            row = idx // n_cols
                            col = idx % n_cols
                            axes[row, col].axis('off')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                        
                    except Exception as e:
                        st.error(f"❌ Erro: {e}")
    else:
        st.info("ℹ️ Defina parâmetros no sistema para habilitar a animação de bifurcação.")


def main():
    """Main Streamlit application."""
    init_session_state()
    
    # Title
    st.title("🔬 LyapunovSolver-Hybrid v2.0")
    st.markdown("**Análise de Estabilidade de Lyapunov para Sistemas Dinâmicos Planares**")
    
    # Sidebar (global config only)
    sidebar_config()
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs([
        "🎯 Análise Completa & Normalização",
        "📐 Cálculo Direto (Sistema Canônico)",
        "🎨 Retrato de Fase"
    ])
    
    with tab1:
        tab_analise_completa()
    
    with tab2:
        tab_calculo_direto()
    
    with tab3:
        tab_retrato_fase()


if __name__ == "__main__":
    main()
