# LyapunovSolver-Hybrid v2.0

## Análise de Estabilidade de Lyapunov para Sistemas Dinâmicos Planares

Biblioteca Python de alto desempenho para cálculo de coeficientes de Lyapunov usando computação simbólica e o método da Forma Normal. O algoritmo implementado é baseado na referência do Mathematica e foi validado para produzir resultados corretos para L₁, L₂, L₃ e ordens superiores.

## ✨ Funcionalidades

- **Cálculo Simbólico**: Coeficientes de Lyapunov L₁, L₂, L₃, ... em forma simbólica
- **Algoritmo Validado**: Baseado no método iterativo do Mathematica (Forma Normal)
- **Interface Gráfica (GUI)**: Interface web intuitiva com Streamlit
- **Interface CLI**: Linha de comando interativa
- **Cache Inteligente**: Sistema de cache em memória e disco com SHA-256
- **Visualização**: Retratos de fase, diagramas de bifurcação
- **Exportação LaTeX**: Todos os coeficientes calculados em formato LaTeX
- **Detecção de Propriedades**: Sistemas Hamiltonianos, reversíveis e simétricos

## 📦 Instalação

```bash
# Clone o repositório
git clone <repo-url>
cd Lyapunov

# Crie um ambiente virtual (recomendado)
python -m venv .venv
.venv\Scripts\activate  # Windows
# ou: source .venv/bin/activate  # Linux/Mac

# Instale as dependências
pip install -r requirements.txt
```

## 🚀 Guia Rápido

### Interface Gráfica (Recomendado)

```bash
streamlit run lyapunov/gui.py
```

**Passo a passo:**
1. Selecione um exemplo pré-definido ou digite seu próprio sistema
2. Clique em **"🚀 Criar Sistema"**
3. Ajuste a ordem máxima k (1-5) e clique em **"Calcular L₁ ... Lₖ"**
4. Visualize os resultados (L₁, L₂, L₃ são exibidos na tela)
5. Para coeficientes adicionais (L₄, L₅...), use **"Gerar LaTeX"**
6. Use **"Limpar Cache"** se precisar recalcular do zero

### Interface de Linha de Comando

```bash
python run_cli.py
```

**Comandos disponíveis:**
- `define` - Definir um novo sistema
- `compute` - Calcular coeficiente L_k
- `evaluate` - Avaliar numericamente
- `classify` - Classificar tipo de bifurcação
- `latex` - Exportar para LaTeX
- `examples` - Ver sistemas de exemplo
- `help` - Ajuda

### Uso Programático (Python)

```python
from lyapunov import LyapunovSystem
from sympy import symbols

# Definir variáveis e parâmetros
x, y, mu = symbols('x y mu')

# Sistema: ẋ = -y + P(x,y,μ), ẏ = x + Q(x,y,μ)
P = mu * x - x**3
Q = -y**3

# Criar sistema e calcular
system = LyapunovSystem(P, Q, params=[mu])
L1 = system.compute_lyapunov(1)
print(f"L₁ = {L1}")

# Avaliação numérica
L1_value = system.evaluate_lyapunov(1, {mu: 0.5})
print(f"L₁(μ=0.5) = {L1_value}")
```

## 📐 Exemplo de Validação (Referência Mathematica)

O algoritmo foi validado contra a referência do Mathematica. Para o sistema:

```
P = a₂x² + a₃x³
Q = b₂y² + b₃y³
```

**Resultados:**
- **L₁ = 3·a₃/8**
- **L₂ = -a₃·(53·a₂² + 48·a₂·b₂ + 15·b₂² + 3·b₃)/32**

Estes valores correspondem exatamente à fórmula de referência do Mathematica.

## 📊 Interpretação dos Resultados

| Coeficiente | Valor | Tipo de Bifurcação | Significado |
|-------------|-------|-------------------|-------------|
| L₁ | < 0 | Supercrítica | Ciclo limite **estável** emerge |
| L₁ | > 0 | Subcrítica | Ciclo limite **instável** |
| L₁ | = 0 | Degenerada | Analisar L₂, L₃, ... |

Quando L₁ = 0, o próximo coeficiente não-nulo determina a estabilidade.

## 🔧 Funcionalidades da Interface

### Exibição de Coeficientes
- **Na tela**: L₁, L₂, L₃ (para não poluir a visualização)
- **No LaTeX**: Todos os coeficientes calculados (L₁ até Lₖ)

### Gerenciamento de Cache
- O sistema usa cache em disco (`~/.lyapunov_cache`) para acelerar recálculos
- Use o botão **"Limpar Cache"** na sidebar para forçar recálculo

### Exemplos Pré-definidos
- **Cúbico Simples**: Sistema clássico com bifurcação de Hopf
- **Sistema Simétrico**: Sem parâmetros
- **Hamiltoniano**: L₁ = 0 por construção
- **Dois Parâmetros**: Para análise de sensibilidade
- **Teste L2 (Mathematica)**: Para validação do algoritmo

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────────────┐
│  Layer 5: Visualization & LaTeX Export                  │
│  (visualization.py, gui.py, cli.py)                     │
└─────────────────────────────────────────────────────────┘
                         ↕
┌─────────────────────────────────────────────────────────┐
│  Layer 4: High-Level API (LyapunovSystem)               │
│  (lyapunov_system.py)                                   │
└─────────────────────────────────────────────────────────┘
                         ↕
┌─────────────────────────────────────────────────────────┐
│  Layer 3: Cache & Optimization                          │
│  (cache_manager.py)                                     │
└─────────────────────────────────────────────────────────┘
                         ↕
┌─────────────────────────────────────────────────────────┐
│  Layer 2: Symbolic Engine (Forma Normal Iterativa)      │
│  (symbolic_engine.py) - Algoritmo Mathematica           │
└─────────────────────────────────────────────────────────┘
                         ↕
┌─────────────────────────────────────────────────────────┐
│  Layer 1: System Definition & Validation                │
│  (system_definition.py)                                 │
└─────────────────────────────────────────────────────────┘
```

## 📚 Base Matemática

O sistema calcula coeficientes de Lyapunov para sistemas planares próximos a uma bifurcação de Hopf:

```
ẋ = -y + P(x, y, μ)
ẏ = x + Q(x, y, μ)
```

### Algoritmo (Forma Normal Iterativa)

Baseado na referência do Mathematica https://prp.unicamp.br/inscricao-congresso/resumos/2021P18120A35838O2645.pdf:

```
Z[j] = aⱼ·((x+y)/2)ʲ + bⱼ·((x-y)/(2i))ʲ
F[2] = x·y/2
Φ[l,k] = Z[k]·(∂F[l]/∂x + ∂F[l]/∂y)
S[p] = Σ Φ[p-i+1, i], para i de 2 a p-1
K[p,k] = Coeficiente de x^(p-k)·y^k em -i·S[p]
h[p] = K[p,k]/(2k-p) se 2k-p ≠ 0, senão 0
F[p] = Σ h[p][k]·x^(p-k+1)·y^(k-1)
V[p] = i·K[p+1, (p+1)/2]
```

Onde:
- **V[3] = L₁** (primeiro coeficiente de Lyapunov)
- **V[5] = L₂** (segundo coeficiente de Lyapunov)
- **V[7] = L₃** (terceiro coeficiente de Lyapunov)

## 📄 Licença

MIT License
