#!/usr/bin/env python
"""
Entry point for the LyapunovSolver GUI interface.
Run with: python run_gui.py
    or: streamlit run lyapunov/gui.py
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit GUI."""
    gui_path = os.path.join(os.path.dirname(__file__), "lyapunov", "gui.py")
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", gui_path], check=True)
    except FileNotFoundError:
        print("Erro: Streamlit não encontrado.")
        print("Instale com: pip install streamlit")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nGUI encerrada.")

if __name__ == "__main__":
    main()
