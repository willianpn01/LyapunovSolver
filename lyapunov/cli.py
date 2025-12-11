"""
Command Line Interface for LyapunovSolver-Hybrid v2.0
Interactive mode for analyzing dynamical systems.
"""

import sys
from typing import Optional, Dict, List
import sympy as sp
from sympy import symbols, Symbol, simplify

from .lyapunov_system import LyapunovSystem


class LyapunovCLI:
    """Interactive command-line interface for Lyapunov analysis."""
    
    def __init__(self):
        self.system: Optional[LyapunovSystem] = None
        self.x = symbols('x')
        self.y = symbols('y')
        self.params: Dict[str, Symbol] = {}
        self.running = True
        
    def print_header(self):
        """Print welcome header."""
        print("\n" + "=" * 60)
        print("  LyapunovSolver-Hybrid v2.0")
        print("  Análise de Estabilidade de Lyapunov")
        print("=" * 60)
        print("\nDigite 'help' para ver os comandos disponíveis.\n")
    
    def print_help(self):
        """Print available commands."""
        help_text = """
╔══════════════════════════════════════════════════════════════╗
║                    COMANDOS DISPONÍVEIS                      ║
╠══════════════════════════════════════════════════════════════╣
║  define    - Definir um novo sistema dinâmico                ║
║  show      - Mostrar o sistema atual                         ║
║  compute   - Calcular coeficiente de Lyapunov L_k            ║
║  evaluate  - Avaliar L_k numericamente                       ║
║  classify  - Classificar tipo de bifurcação                  ║
║  latex     - Gerar representação LaTeX                       ║
║  plot      - Gerar gráficos (requer matplotlib)              ║
║  examples  - Ver sistemas de exemplo                         ║
║  clear     - Limpar sistema atual                            ║
║  help      - Mostrar esta ajuda                              ║
║  quit      - Sair do programa                                ║
╚══════════════════════════════════════════════════════════════╝
"""
        print(help_text)
    
    def print_examples(self):
        """Print example systems."""
        examples = """
╔══════════════════════════════════════════════════════════════╗
║                   SISTEMAS DE EXEMPLO                        ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. Sistema Cúbico Simples:                                  ║
║     P = mu*x - x**3                                          ║
║     Q = -y**3                                                ║
║     params = mu                                              ║
║                                                              ║
║  2. Sistema Simétrico:                                       ║
║     P = -x**3 - x*y**2                                       ║
║     Q = -x**2*y - y**3                                       ║
║     params = (nenhum)                                        ║
║                                                              ║
║  3. Sistema Hamiltoniano (L1 = 0):                           ║
║     P = x**2*y                                               ║
║     Q = -x*y**2                                              ║
║     params = (nenhum)                                        ║
║                                                              ║
║  4. Sistema com dois parâmetros:                             ║
║     P = alpha*x - beta*x**2*y - x**3                         ║
║     Q = alpha*y + beta*x*y**2 - y**3                         ║
║     params = alpha, beta                                     ║
║                                                              ║
║  5. Teste L2 (Referência Mathematica):                       ║
║     P = a2*x**2 + a3*x**3                                    ║
║     Q = b2*y**2 + b3*y**3                                    ║
║     params = a2, a3, b2, b3                                  ║
║     L1 = 3*a3/8                                              ║
║     L2 = -a3*(53*a2² + 48*a2*b2 + 15*b2² + 3*b3)/32         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

Para usar um exemplo, digite 'define' e siga as instruções.
"""
        print(examples)
    
    def cmd_define(self):
        """Define a new dynamical system."""
        print("\n--- Definir Sistema Dinâmico ---")
        print("Sistema na forma: ẋ = -y + P(x,y,μ), ẏ = x + Q(x,y,μ)")
        print("Use x, y como variáveis de estado.")
        print("Exemplo: P = mu*x - x**3\n")
        
        try:
            P_str = input("Digite P(x,y,μ): ").strip()
            if not P_str:
                P_str = "0"
            
            Q_str = input("Digite Q(x,y,μ): ").strip()
            if not Q_str:
                Q_str = "0"
            
            params_str = input("Parâmetros (separados por vírgula, ou Enter para nenhum): ").strip()
            
            self.params = {}
            param_list = []
            if params_str:
                for p in params_str.split(','):
                    p = p.strip()
                    if p:
                        sym = symbols(p)
                        self.params[p] = sym
                        param_list.append(sym)
            
            local_dict = {'x': self.x, 'y': self.y, **self.params}
            P = sp.sympify(P_str, locals=local_dict)
            Q = sp.sympify(Q_str, locals=local_dict)
            
            self.system = LyapunovSystem(
                P, Q,
                params=param_list,
                x=self.x,
                y=self.y,
                enable_cache=True
            )
            
            print("\n✓ Sistema definido com sucesso!")
            self._show_system()
            
        except Exception as e:
            print(f"\n✗ Erro ao definir sistema: {e}")
    
    def _show_system(self):
        """Display current system."""
        if self.system is None:
            print("Nenhum sistema definido. Use 'define' primeiro.")
            return
        
        print("\n┌─────────────────────────────────────┐")
        print("│         Sistema Atual               │")
        print("├─────────────────────────────────────┤")
        print(f"│  ẋ = -y + {self.system.P}")
        print(f"│  ẏ = x + {self.system.Q}")
        if self.params:
            print(f"│  Parâmetros: {list(self.params.keys())}")
        
        props = [p.name for p in self.system.properties]
        if props:
            print(f"│  Propriedades: {props}")
        print("└─────────────────────────────────────┘")
    
    def cmd_show(self):
        """Show current system."""
        self._show_system()
    
    def cmd_compute(self):
        """Compute Lyapunov coefficient."""
        if self.system is None:
            print("Nenhum sistema definido. Use 'define' primeiro.")
            return
        
        try:
            k_str = input("Ordem k do coeficiente L_k (padrão: 1): ").strip()
            k = int(k_str) if k_str else 1
            
            if k < 1:
                print("Ordem deve ser >= 1")
                return
            
            print(f"\nCalculando L_{k}...")
            L_k = self.system.compute_lyapunov(k)
            
            print(f"\n╔═══════════════════════════════════════╗")
            print(f"║  L_{k} = {L_k}")
            print(f"╚═══════════════════════════════════════╝")
            
            if simplify(L_k) == 0:
                print("  → Coeficiente é ZERO (caso degenerado)")
            
        except ValueError:
            print("Ordem inválida. Digite um número inteiro.")
        except Exception as e:
            print(f"Erro no cálculo: {e}")
    
    def cmd_evaluate(self):
        """Evaluate Lyapunov coefficient numerically."""
        if self.system is None:
            print("Nenhum sistema definido. Use 'define' primeiro.")
            return
        
        if not self.params:
            print("Sistema não tem parâmetros. Use 'compute' para valor simbólico.")
            L1 = self.system.compute_lyapunov(1)
            try:
                val = float(L1.evalf())
                print(f"L_1 = {val}")
            except:
                print(f"L_1 = {L1}")
            return
        
        try:
            k_str = input("Ordem k (padrão: 1): ").strip()
            k = int(k_str) if k_str else 1
            
            param_values = {}
            print("\nDigite os valores dos parâmetros:")
            for name, sym in self.params.items():
                val_str = input(f"  {name} = ").strip()
                param_values[sym] = float(val_str)
            
            value = self.system.evaluate_lyapunov(k, param_values)
            
            print(f"\n╔═══════════════════════════════════════╗")
            print(f"║  L_{k} = {value:.10g}")
            print(f"╚═══════════════════════════════════════╝")
            
            if value < 0:
                print("  → Bifurcação SUPERCRÍTICA (ciclo limite estável)")
            elif value > 0:
                print("  → Bifurcação SUBCRÍTICA (ciclo limite instável)")
            else:
                print("  → Caso DEGENERADO (analisar ordem superior)")
                
        except ValueError as e:
            print(f"Valor inválido: {e}")
        except Exception as e:
            print(f"Erro na avaliação: {e}")
    
    def cmd_classify(self):
        """Classify bifurcation type."""
        if self.system is None:
            print("Nenhum sistema definido. Use 'define' primeiro.")
            return
        
        if not self.params:
            L1 = self.system.compute_lyapunov(1)
            try:
                val = float(L1.evalf())
                if val < 0:
                    print("Bifurcação SUPERCRÍTICA")
                elif val > 0:
                    print("Bifurcação SUBCRÍTICA")
                else:
                    print("Caso DEGENERADO")
            except:
                print(f"L_1 = {L1} (não pode classificar simbolicamente)")
            return
        
        try:
            param_values = {}
            print("Digite os valores dos parâmetros:")
            for name, sym in self.params.items():
                val_str = input(f"  {name} = ").strip()
                param_values[sym] = float(val_str)
            
            bif_type = self.system.classify_bifurcation(param_values)
            L1_val = self.system.evaluate_lyapunov(1, param_values)
            
            print(f"\n  L_1 = {L1_val:.10g}")
            print(f"  Tipo: {bif_type.upper()}")
            
        except Exception as e:
            print(f"Erro: {e}")
    
    def cmd_latex(self):
        """Generate LaTeX representation."""
        if self.system is None:
            print("Nenhum sistema definido. Use 'define' primeiro.")
            return
        
        try:
            k_str = input("Ordem k (padrão: 1, 'all' para todos calculados): ").strip()
            
            if k_str.lower() == 'all' or not k_str:
                latex = self.system.to_latex()
            else:
                k = int(k_str)
                latex = self.system.to_latex(k)
            
            print("\n--- LaTeX ---")
            print(latex)
            print("-------------")
            
            save = input("\nSalvar em arquivo? (s/n): ").strip().lower()
            if save == 's':
                filename = input("Nome do arquivo (padrão: lyapunov.tex): ").strip()
                if not filename:
                    filename = "lyapunov.tex"
                self.system.export_latex(filename)
                print(f"✓ Salvo em {filename}")
                
        except Exception as e:
            print(f"Erro: {e}")
    
    def cmd_plot(self):
        """Generate plots."""
        if self.system is None:
            print("Nenhum sistema definido. Use 'define' primeiro.")
            return
        
        try:
            from .visualization import LyapunovVisualizer
            import matplotlib
            matplotlib.use('TkAgg')
            import matplotlib.pyplot as plt
        except ImportError:
            print("Matplotlib não disponível. Instale com: pip install matplotlib")
            return
        
        print("\nTipos de gráfico:")
        print("  1. Retrato de fase")
        print("  2. Diagrama de bifurcação")
        print("  3. Coeficientes de Lyapunov")
        
        choice = input("\nEscolha (1-3): ").strip()
        
        try:
            viz = LyapunovVisualizer(self.system)
            
            if choice == '1':
                param_values = {}
                if self.params:
                    print("Digite os valores dos parâmetros:")
                    for name, sym in self.params.items():
                        val_str = input(f"  {name} = ").strip()
                        param_values[sym] = float(val_str)
                
                fig, ax = viz.plot_phase_portrait(param_values)
                plt.show()
                
            elif choice == '2':
                if not self.params:
                    print("Sistema não tem parâmetros para diagrama de bifurcação.")
                    return
                
                param_name = input(f"Parâmetro a variar ({list(self.params.keys())}): ").strip()
                if param_name not in self.params:
                    print("Parâmetro não encontrado.")
                    return
                
                p_min = float(input("Valor mínimo: ").strip())
                p_max = float(input("Valor máximo: ").strip())
                
                fig, ax = viz.plot_bifurcation_diagram(
                    self.params[param_name],
                    (p_min, p_max)
                )
                plt.show()
                
            elif choice == '3':
                max_k = int(input("Ordem máxima k: ").strip() or "3")
                param_values = {}
                if self.params:
                    print("Digite os valores dos parâmetros:")
                    for name, sym in self.params.items():
                        val_str = input(f"  {name} = ").strip()
                        param_values[sym] = float(val_str)
                
                fig, ax = viz.plot_lyapunov_coefficients(max_k, param_values or None)
                plt.show()
            else:
                print("Opção inválida.")
                
        except Exception as e:
            print(f"Erro ao gerar gráfico: {e}")
    
    def cmd_clear(self):
        """Clear current system."""
        self.system = None
        self.params = {}
        print("✓ Sistema limpo.")
    
    def run(self):
        """Main loop."""
        self.print_header()
        
        commands = {
            'help': self.print_help,
            'h': self.print_help,
            '?': self.print_help,
            'define': self.cmd_define,
            'd': self.cmd_define,
            'show': self.cmd_show,
            's': self.cmd_show,
            'compute': self.cmd_compute,
            'c': self.cmd_compute,
            'evaluate': self.cmd_evaluate,
            'eval': self.cmd_evaluate,
            'e': self.cmd_evaluate,
            'classify': self.cmd_classify,
            'latex': self.cmd_latex,
            'l': self.cmd_latex,
            'plot': self.cmd_plot,
            'p': self.cmd_plot,
            'examples': self.print_examples,
            'ex': self.print_examples,
            'clear': self.cmd_clear,
            'quit': lambda: setattr(self, 'running', False),
            'q': lambda: setattr(self, 'running', False),
            'exit': lambda: setattr(self, 'running', False),
        }
        
        while self.running:
            try:
                cmd = input("\nlyapunov> ").strip().lower()
                
                if not cmd:
                    continue
                
                if cmd in commands:
                    commands[cmd]()
                else:
                    print(f"Comando desconhecido: '{cmd}'. Digite 'help' para ajuda.")
                    
            except KeyboardInterrupt:
                print("\n\nUse 'quit' para sair.")
            except EOFError:
                break
        
        print("\nAté logo!")


def main():
    """Entry point for CLI."""
    cli = LyapunovCLI()
    cli.run()


if __name__ == "__main__":
    main()
