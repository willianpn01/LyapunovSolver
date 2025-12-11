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

st.set_page_config(
    page_title="LyapunovSolver v2.0",
    page_icon="📊",
    layout="wide"
)


def init_session_state():
    """Initialize session state variables."""
    if 'system' not in st.session_state:
        st.session_state.system = None
    if 'params' not in st.session_state:
        st.session_state.params = {}
    if 'computed_coeffs' not in st.session_state:
        st.session_state.computed_coeffs = {}


def create_system(P_str: str, Q_str: str, params_str: str):
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


def main():
    """Main Streamlit application."""
    init_session_state()
    
    st.title("🔬 LyapunovSolver-Hybrid v2.0")
    st.markdown("**Análise de Estabilidade de Lyapunov para Sistemas Dinâmicos Planares**")
    
    st.sidebar.header("📝 Definir Sistema")
    st.sidebar.markdown("""
    Sistema na forma canônica:
    - ẋ = -y + P(x,y,μ)
    - ẏ = x + Q(x,y,μ)
    """)
    
    example = st.sidebar.selectbox(
        "Exemplos pré-definidos:",
        [
            "Personalizado",
            "Cúbico Simples",
            "Sistema Simétrico",
            "Hamiltoniano (L₁=0)",
            "Dois Parâmetros",
            "Teste L2 (Mathematica)"
        ]
    )
    
    examples_data = {
        "Personalizado": ("", "", ""),
        "Cúbico Simples": ("mu*x - x**3", "-y**3", "mu"),
        "Sistema Simétrico": ("-x**3 - x*y**2", "-x**2*y - y**3", ""),
        "Hamiltoniano (L₁=0)": ("x**2*y", "-x*y**2", ""),
        "Dois Parâmetros": ("alpha*x - beta*x**2*y - x**3", "alpha*y + beta*x*y**2 - y**3", "alpha, beta"),
        "Teste L2 (Mathematica)": ("a2*x**2 + a3*x**3", "b2*y**2 + b3*y**3", "a2, a3, b2, b3")
    }
    
    default_P, default_Q, default_params = examples_data[example]
    
    P_str = st.sidebar.text_input("P(x,y,μ):", value=default_P, placeholder="ex: mu*x - x**3")
    Q_str = st.sidebar.text_input("Q(x,y,μ):", value=default_Q, placeholder="ex: -y**3")
    params_str = st.sidebar.text_input("Parâmetros (vírgula):", value=default_params, placeholder="ex: mu, alpha")
    
    if st.sidebar.button("🚀 Criar Sistema", type="primary"):
        try:
            system, params = create_system(P_str, Q_str, params_str)
            st.session_state.system = system
            st.session_state.params = params
            st.session_state.computed_coeffs = {}
            st.sidebar.success("✅ Sistema criado!")
        except Exception as e:
            st.sidebar.error(f"❌ Erro: {e}")
    
    st.sidebar.divider()
    st.sidebar.subheader("⚙️ Configurações")
    
    debug_mode = st.sidebar.checkbox("🐛 Modo Debug (logs no console)", value=False)
    
    if debug_mode:
        import lyapunov as lyap_module
        lyap_module.enable_debug_logging(level="DEBUG")
        st.sidebar.info("Logs aparecerão no terminal")
    
    if st.sidebar.button("🗑️ Limpar Cache"):
        try:
            import shutil
            from pathlib import Path
            
            cache_dir = Path.home() / ".lyapunov_cache"
            if cache_dir.exists():
                shutil.rmtree(cache_dir)
            
            st.session_state.system = None
            st.session_state.params = {}
            st.session_state.computed_coeffs = {}
            
            st.sidebar.success("✅ Cache limpo com sucesso!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"❌ Erro ao limpar cache: {e}")
    
    if st.session_state.system is not None:
        system = st.session_state.system
        params = st.session_state.params
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("📐 Sistema Definido")
            
            st.latex(r"\dot{x} = -y + " + latex(system.P))
            st.latex(r"\dot{y} = x + " + latex(system.Q))
            
            if params:
                st.markdown(f"**Parâmetros:** {', '.join(params.keys())}")
            
            props = [p.name for p in system.properties]
            if props:
                st.markdown(f"**Propriedades:** {', '.join(props)}")
        
        with col2:
            st.header("🧮 Calcular Coeficientes")
            
            max_k = st.slider("Ordem máxima k:", 1, 10, 1)
            
            if max_k > 5:
                st.warning("⚠️ Ordens altas (k > 5) podem demorar alguns minutos ou até horas dependendo do sistema.")
            
            if st.button("Calcular L₁ ... Lₖ"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for k in range(1, max_k + 1):
                    status_text.text(f"Calculando L_{k}...")
                    L_k = system.compute_lyapunov(k)
                    st.session_state.computed_coeffs[k] = L_k
                    progress_bar.progress(k / max_k)
                
                status_text.empty()
                progress_bar.empty()
                st.success("✅ Cálculo concluído!")
        
        if st.session_state.computed_coeffs:
            st.header("📊 Resultados")
            
            max_display = 3
            displayed = 0
            
            for k, L_k in st.session_state.computed_coeffs.items():
                if displayed >= max_display:
                    break
                    
                col_a, col_b = st.columns([2, 1])
                
                with col_a:
                    st.latex(f"L_{k} = " + latex(L_k))
                
                with col_b:
                    if simplify(L_k) == 0:
                        st.info("= 0 (degenerado)")
                    elif not L_k.free_symbols - {symbols('x'), symbols('y')}:
                        try:
                            val = float(L_k.evalf())
                            if val < 0:
                                st.success(f"= {val:.6g} (supercrítico)")
                            else:
                                st.warning(f"= {val:.6g} (subcrítico)")
                        except:
                            pass
                
                displayed += 1
            
            if len(st.session_state.computed_coeffs) > max_display:
                hidden_count = len(st.session_state.computed_coeffs) - max_display
                st.info(f"📄 +{hidden_count} coeficiente(s) calculado(s) mas não exibido(s). Use a opção **Exportar LaTeX** abaixo para visualizar todos.")
        
        if params and st.session_state.computed_coeffs:
            st.header("🔢 Avaliação Numérica")
            
            param_values = {}
            cols = st.columns(len(params))
            for i, (name, sym) in enumerate(params.items()):
                with cols[i]:
                    val = st.number_input(f"{name}:", value=0.0, step=0.1, key=f"param_{name}")
                    param_values[sym] = val
            
            if st.button("Avaliar"):
                st.subheader("Valores Numéricos:")
                for k in st.session_state.computed_coeffs.keys():
                    val = system.evaluate_lyapunov(k, param_values)
                    
                    if val < -1e-12:
                        st.success(f"L_{k} = {val:.10g}  →  **Supercrítico** (ciclo limite estável)")
                    elif val > 1e-12:
                        st.warning(f"L_{k} = {val:.10g}  →  **Subcrítico** (ciclo limite instável)")
                    else:
                        st.info(f"L_{k} = {val:.10g}  →  **Degenerado**")
        
        st.header("📈 Visualização")
        
        viz_type = st.selectbox(
            "Tipo de gráfico:",
            ["Retrato de Fase", "Diagrama de Bifurcação", "Sensibilidade 2D"]
        )
        
        try:
            import matplotlib.pyplot as plt
            from lyapunov.visualization import LyapunovVisualizer
            
            viz = LyapunovVisualizer(system)
            
            if viz_type == "Retrato de Fase":
                col_p1, col_p2 = st.columns(2)
                
                with col_p1:
                    x_min = st.number_input("x mín:", value=-2.0)
                    x_max = st.number_input("x máx:", value=2.0)
                
                with col_p2:
                    y_min = st.number_input("y mín:", value=-2.0)
                    y_max = st.number_input("y máx:", value=2.0)
                
                param_vals = {}
                if params:
                    st.markdown("**Valores dos parâmetros:**")
                    cols = st.columns(len(params))
                    for i, (name, sym) in enumerate(params.items()):
                        with cols[i]:
                            val = st.number_input(f"{name}:", value=0.1, step=0.1, key=f"phase_{name}")
                            param_vals[sym] = val
                
                if st.button("Gerar Retrato de Fase"):
                    with st.spinner("Gerando..."):
                        fig, ax = viz.plot_phase_portrait(
                            param_vals,
                            x_range=(x_min, x_max),
                            y_range=(y_min, y_max)
                        )
                        st.pyplot(fig)
                        plt.close(fig)
            
            elif viz_type == "Diagrama de Bifurcação":
                if not params:
                    st.warning("Sistema não tem parâmetros.")
                else:
                    param_name = st.selectbox("Parâmetro:", list(params.keys()))
                    
                    col_b1, col_b2 = st.columns(2)
                    with col_b1:
                        p_min = st.number_input("Valor mínimo:", value=-1.0)
                    with col_b2:
                        p_max = st.number_input("Valor máximo:", value=1.0)
                    
                    k_bif = st.selectbox("Coeficiente:", [1, 2, 3])
                    
                    if st.button("Gerar Diagrama"):
                        with st.spinner("Gerando..."):
                            fig, ax = viz.plot_bifurcation_diagram(
                                params[param_name],
                                (p_min, p_max),
                                k=k_bif
                            )
                            st.pyplot(fig)
                            plt.close(fig)
            
            elif viz_type == "Sensibilidade 2D":
                if len(params) < 2:
                    st.warning("Necessário pelo menos 2 parâmetros.")
                else:
                    param_names = list(params.keys())
                    col_s1, col_s2 = st.columns(2)
                    
                    with col_s1:
                        p1_name = st.selectbox("Parâmetro 1:", param_names, index=0)
                        p1_min = st.number_input("Mín 1:", value=-1.0)
                        p1_max = st.number_input("Máx 1:", value=1.0)
                    
                    with col_s2:
                        p2_name = st.selectbox("Parâmetro 2:", param_names, index=min(1, len(param_names)-1))
                        p2_min = st.number_input("Mín 2:", value=-1.0)
                        p2_max = st.number_input("Máx 2:", value=1.0)
                    
                    if st.button("Gerar Mapa"):
                        with st.spinner("Gerando..."):
                            fig, ax = viz.plot_parameter_sensitivity(
                                params[p1_name],
                                params[p2_name],
                                (p1_min, p1_max),
                                (p2_min, p2_max)
                            )
                            st.pyplot(fig)
                            plt.close(fig)
                            
        except ImportError:
            st.warning("Matplotlib não disponível. Instale com: pip install matplotlib")
        
        st.header("📄 Exportar LaTeX")
        
        if st.button("Gerar LaTeX"):
            lines = [
                r"\begin{align}",
                r"\dot{x} &= -y + " + latex(system.P) + r" \\",
                r"\dot{y} &= x + " + latex(system.Q),
                r"\end{align}",
                "",
                r"\textbf{Coeficientes de Lyapunov:}",
                ""
            ]
            
            for k, L_k in st.session_state.computed_coeffs.items():
                lines.append(f"$L_{{{k}}} = {latex(L_k)}$")
                lines.append("")
            
            latex_code = "\n".join(lines)
            
            st.code(latex_code, language="latex")
            
            st.download_button(
                "📥 Baixar .tex",
                latex_code,
                file_name="lyapunov_analysis.tex",
                mime="text/plain"
            )
    
    else:
        st.info("👈 Defina um sistema na barra lateral para começar.")
        
        st.header("📚 Como usar")
        st.markdown("""
        1. **Defina o sistema** na barra lateral (ou escolha um exemplo)
        2. **Clique em "Criar Sistema"** para validar
        3. **Calcule os coeficientes** de Lyapunov L₁, L₂, ...
        4. **Avalie numericamente** para valores específicos dos parâmetros
        5. **Visualize** retratos de fase e diagramas de bifurcação
        6. **Exporte** os resultados em LaTeX
        
        ### Interpretação dos Resultados
        
        | L₁ | Tipo de Bifurcação | Significado |
        |----|--------------------|-------------|
        | < 0 | Supercrítica | Ciclo limite **estável** emerge |
        | > 0 | Subcrítica | Ciclo limite **instável** |
        | = 0 | Degenerada | Analisar L₂, L₃, ... |
        """)


if __name__ == "__main__":
    main()
