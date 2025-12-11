"""
Module D: C Code Generator and Compiler Bridge
Generates optimized C code from symbolic expressions and handles compilation.
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple, Union
from dataclasses import dataclass
from enum import Enum, auto
import ctypes

import sympy as sp
from sympy import (
    Expr, Symbol, symbols, cse, numbered_symbols,
    sin, cos, tan, exp, log, sqrt, Abs, Pow,
    Float, Integer, Rational, pi, E
)
from sympy.printing.c import C99CodePrinter

from .cache_manager import CacheManager


class CompilerType(Enum):
    """Supported C compilers."""
    MSVC = auto()
    GCC = auto()
    CLANG = auto()
    MINGW = auto()
    UNKNOWN = auto()


@dataclass
class CompilationResult:
    """Result of a compilation attempt."""
    success: bool
    library_path: Optional[Path]
    error_message: Optional[str] = None
    compiler_output: str = ""
    compilation_time: float = 0.0


class OptimizedCPrinter(C99CodePrinter):
    """
    Custom C code printer with optimizations for numerical evaluation.
    """
    
    def __init__(self, settings=None):
        super().__init__(settings or {})
        self._float_precision = 'double'
    
    def _print_Pow(self, expr):
        """Optimize power expressions."""
        base, exp = expr.as_base_exp()
        
        if exp == 2:
            base_str = self._print(base)
            return f"({base_str} * {base_str})"
        elif exp == 3:
            base_str = self._print(base)
            return f"({base_str} * {base_str} * {base_str})"
        elif exp == -1:
            base_str = self._print(base)
            return f"(1.0 / {base_str})"
        elif exp == Rational(1, 2):
            return f"sqrt({self._print(base)})"
        elif exp == Rational(-1, 2):
            return f"(1.0 / sqrt({self._print(base)}))"
        
        return super()._print_Pow(expr)
    
    def _print_Float(self, expr):
        """Print floats with sufficient precision."""
        return f"{float(expr):.17g}"
    
    def _print_Rational(self, expr):
        """Convert rationals to floating point division."""
        return f"({float(expr.p):.17g} / {float(expr.q):.17g})"


class CCodeGenerator:
    """
    Generates optimized C code from SymPy expressions.
    
    Features:
    - Common subexpression elimination (CSE)
    - Compiler detection and configuration
    - Automatic compilation to shared library
    - SIMD hints for modern compilers
    
    Attributes:
        cache: CacheManager for storing compiled binaries
        compiler: Detected compiler type
        compiler_path: Path to compiler executable
    """
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        """
        Initialize the code generator.
        
        Args:
            cache_manager: Optional CacheManager for binary caching
        """
        self.cache = cache_manager
        self.compiler, self.compiler_path = self._detect_compiler()
        self._printer = OptimizedCPrinter()
        
        self._temp_dir: Optional[Path] = None
    
    def _detect_compiler(self) -> Tuple[CompilerType, Optional[str]]:
        """
        Detect available C compiler on the system.
        
        Returns:
            Tuple of (CompilerType, path_to_compiler)
        """
        if sys.platform == 'win32':
            vswhere = r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
            if os.path.exists(vswhere):
                try:
                    result = subprocess.run(
                        [vswhere, '-latest', '-property', 'installationPath'],
                        capture_output=True, text=True
                    )
                    if result.returncode == 0:
                        vs_path = result.stdout.strip()
                        cl_path = os.path.join(
                            vs_path, 
                            'VC', 'Tools', 'MSVC'
                        )
                        if os.path.exists(cl_path):
                            return (CompilerType.MSVC, 'cl')
                except:
                    pass
            
            for gcc_name in ['gcc', 'x86_64-w64-mingw32-gcc']:
                if shutil.which(gcc_name):
                    return (CompilerType.MINGW, gcc_name)
        
        else:
            if shutil.which('gcc'):
                return (CompilerType.GCC, 'gcc')
            
            if shutil.which('clang'):
                return (CompilerType.CLANG, 'clang')
        
        return (CompilerType.UNKNOWN, None)
    
    def generate_c_function(
        self,
        expr: Expr,
        params: List[str],
        func_name: str = "eval_Lk",
        use_cse: bool = True,
        add_derivatives: bool = False
    ) -> str:
        """
        Generate C code for evaluating a symbolic expression.
        
        Args:
            expr: SymPy expression to convert
            params: List of parameter names (in order)
            func_name: Name for the generated function
            use_cse: Whether to apply common subexpression elimination
            add_derivatives: Whether to generate derivative functions
            
        Returns:
            Complete C source code as string
        """
        code_lines = [
            "/* Auto-generated by LyapunovSolver-Hybrid v2.0 */",
            "#include <math.h>",
            "#include <stddef.h>",
            "",
            "#ifdef _WIN32",
            "    #define EXPORT __declspec(dllexport)",
            "#else",
            "    #define EXPORT __attribute__((visibility(\"default\")))",
            "#endif",
            "",
        ]
        
        param_list = ", ".join([f"double {p}" for p in params])
        if not param_list:
            param_list = "void"
        
        code_lines.append(f"EXPORT double {func_name}({param_list}) {{")
        
        if use_cse and expr.count_ops() > 10:
            replacements, reduced = cse(expr, numbered_symbols('t'))
            
            for sym, sub_expr in replacements:
                c_expr = self._printer.doprint(sub_expr)
                code_lines.append(f"    double {sym} = {c_expr};")
            
            if reduced:
                result_expr = self._printer.doprint(reduced[0])
            else:
                result_expr = self._printer.doprint(expr)
        else:
            result_expr = self._printer.doprint(expr)
        
        code_lines.append(f"    return {result_expr};")
        code_lines.append("}")
        code_lines.append("")
        
        if add_derivatives and params:
            for i, param in enumerate(params):
                deriv = sp.diff(expr, symbols(param))
                deriv_func = self._generate_derivative_function(
                    deriv, params, f"{func_name}_d{param}", use_cse
                )
                code_lines.extend(deriv_func.split('\n'))
        
        code_lines.extend([
            f"/* Batch evaluation for arrays */",
            f"EXPORT void {func_name}_batch(",
            f"    const double* params,  /* {len(params)} params per point */",
            f"    double* results,",
            f"    size_t n_points",
            ") {",
            f"    for (size_t i = 0; i < n_points; i++) {{",
        ])
        
        if params:
            param_access = ", ".join([f"params[i * {len(params)} + {j}]" 
                                      for j in range(len(params))])
            code_lines.append(f"        results[i] = {func_name}({param_access});")
        else:
            code_lines.append(f"        results[i] = {func_name}();")
        
        code_lines.extend([
            "    }",
            "}",
            ""
        ])
        
        return "\n".join(code_lines)
    
    def _generate_derivative_function(
        self,
        expr: Expr,
        params: List[str],
        func_name: str,
        use_cse: bool
    ) -> str:
        """Generate C function for a derivative expression."""
        lines = []
        
        param_list = ", ".join([f"double {p}" for p in params])
        if not param_list:
            param_list = "void"
        
        lines.append(f"EXPORT double {func_name}({param_list}) {{")
        
        if use_cse and expr.count_ops() > 10:
            replacements, reduced = cse(expr, numbered_symbols('t'))
            for sym, sub_expr in replacements:
                c_expr = self._printer.doprint(sub_expr)
                lines.append(f"    double {sym} = {c_expr};")
            result_expr = self._printer.doprint(reduced[0]) if reduced else "0.0"
        else:
            result_expr = self._printer.doprint(expr)
        
        lines.append(f"    return {result_expr};")
        lines.append("}")
        lines.append("")
        
        return "\n".join(lines)
    
    def compile_to_library(
        self,
        c_code: str,
        output_name: str = "lyapunov_eval",
        optimization_level: int = 3,
        keep_source: bool = False
    ) -> CompilationResult:
        """
        Compile C code to a shared library.
        
        Args:
            c_code: C source code string
            output_name: Base name for output library
            optimization_level: Optimization level (0-3)
            keep_source: Whether to keep the .c source file
            
        Returns:
            CompilationResult with compilation status and library path
        """
        if self.compiler == CompilerType.UNKNOWN:
            return CompilationResult(
                success=False,
                library_path=None,
                error_message="No C compiler detected on system"
            )
        
        import time
        start_time = time.time()
        
        if self._temp_dir is None:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="lyapunov_"))
        
        source_path = self._temp_dir / f"{output_name}.c"
        with open(source_path, 'w') as f:
            f.write(c_code)
        
        if sys.platform == 'win32':
            lib_ext = '.dll'
        elif sys.platform == 'darwin':
            lib_ext = '.dylib'
        else:
            lib_ext = '.so'
        
        output_path = self._temp_dir / f"{output_name}{lib_ext}"
        
        try:
            if self.compiler == CompilerType.MSVC:
                result = self._compile_msvc(source_path, output_path, optimization_level)
            elif self.compiler in (CompilerType.GCC, CompilerType.MINGW):
                result = self._compile_gcc(source_path, output_path, optimization_level)
            elif self.compiler == CompilerType.CLANG:
                result = self._compile_clang(source_path, output_path, optimization_level)
            else:
                return CompilationResult(
                    success=False,
                    library_path=None,
                    error_message=f"Unsupported compiler: {self.compiler}"
                )
            
            compilation_time = time.time() - start_time
            
            if not keep_source:
                try:
                    source_path.unlink()
                except:
                    pass
            
            if result.success and self.cache is not None:
                pass
            
            result.compilation_time = compilation_time
            return result
            
        except Exception as e:
            return CompilationResult(
                success=False,
                library_path=None,
                error_message=str(e)
            )
    
    def _compile_gcc(
        self,
        source_path: Path,
        output_path: Path,
        opt_level: int
    ) -> CompilationResult:
        """Compile using GCC or MinGW."""
        cmd = [
            self.compiler_path,
            '-shared',
            '-fPIC',
            f'-O{opt_level}',
            '-ffast-math',
            '-march=native',
            '-o', str(output_path),
            str(source_path),
            '-lm'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and output_path.exists():
            return CompilationResult(
                success=True,
                library_path=output_path,
                compiler_output=result.stdout + result.stderr
            )
        else:
            return CompilationResult(
                success=False,
                library_path=None,
                error_message=result.stderr or "Compilation failed",
                compiler_output=result.stdout + result.stderr
            )
    
    def _compile_clang(
        self,
        source_path: Path,
        output_path: Path,
        opt_level: int
    ) -> CompilationResult:
        """Compile using Clang."""
        cmd = [
            self.compiler_path,
            '-shared',
            '-fPIC',
            f'-O{opt_level}',
            '-ffast-math',
            '-o', str(output_path),
            str(source_path),
            '-lm'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and output_path.exists():
            return CompilationResult(
                success=True,
                library_path=output_path,
                compiler_output=result.stdout + result.stderr
            )
        else:
            return CompilationResult(
                success=False,
                library_path=None,
                error_message=result.stderr or "Compilation failed",
                compiler_output=result.stdout + result.stderr
            )
    
    def _compile_msvc(
        self,
        source_path: Path,
        output_path: Path,
        opt_level: int
    ) -> CompilationResult:
        """Compile using MSVC."""
        opt_flags = {0: '/Od', 1: '/O1', 2: '/O2', 3: '/Ox'}
        
        cmd = [
            'cl',
            '/LD',
            opt_flags.get(opt_level, '/O2'),
            '/fp:fast',
            f'/Fe:{output_path}',
            str(source_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        
        if result.returncode == 0 and output_path.exists():
            return CompilationResult(
                success=True,
                library_path=output_path,
                compiler_output=result.stdout + result.stderr
            )
        else:
            return CompilationResult(
                success=False,
                library_path=None,
                error_message=result.stderr or "Compilation failed",
                compiler_output=result.stdout + result.stderr
            )
    
    def load_library(self, library_path: Path) -> Optional[ctypes.CDLL]:
        """
        Load a compiled shared library.
        
        Args:
            library_path: Path to the shared library
            
        Returns:
            Loaded ctypes.CDLL object or None on failure
        """
        try:
            lib = ctypes.CDLL(str(library_path))
            return lib
        except Exception as e:
            print(f"Failed to load library: {e}")
            return None
    
    def create_evaluator(
        self,
        library: ctypes.CDLL,
        func_name: str,
        n_params: int
    ) -> Callable[..., float]:
        """
        Create a Python callable from a compiled C function.
        
        Args:
            library: Loaded ctypes.CDLL
            func_name: Name of the function in the library
            n_params: Number of parameters the function takes
            
        Returns:
            Callable that evaluates the function
        """
        func = getattr(library, func_name)
        func.restype = ctypes.c_double
        func.argtypes = [ctypes.c_double] * n_params
        
        def evaluator(*args):
            if len(args) != n_params:
                raise ValueError(f"Expected {n_params} arguments, got {len(args)}")
            return func(*[float(a) for a in args])
        
        return evaluator
    
    def cleanup(self) -> None:
        """Clean up temporary files and directories."""
        if self._temp_dir is not None and self._temp_dir.exists():
            try:
                shutil.rmtree(self._temp_dir)
            except:
                pass
            self._temp_dir = None
    
    def __del__(self):
        """Destructor to clean up temporary files."""
        self.cleanup()
    
    def get_compiler_info(self) -> Dict[str, str]:
        """
        Get information about the detected compiler.
        
        Returns:
            Dictionary with compiler information
        """
        info = {
            'type': self.compiler.name,
            'path': self.compiler_path or 'Not found',
            'available': self.compiler != CompilerType.UNKNOWN
        }
        
        if self.compiler_path and self.compiler != CompilerType.MSVC:
            try:
                result = subprocess.run(
                    [self.compiler_path, '--version'],
                    capture_output=True, text=True
                )
                info['version'] = result.stdout.split('\n')[0]
            except:
                info['version'] = 'Unknown'
        
        return info


class NumbaGenerator:
    """
    Alternative code generator using Numba JIT compilation.
    
    Useful when C compiler is not available or for rapid prototyping.
    """
    
    def __init__(self):
        """Initialize Numba generator."""
        self._numba_available = self._check_numba()
    
    def _check_numba(self) -> bool:
        """Check if Numba is available."""
        try:
            import numba
            return True
        except ImportError:
            return False
    
    def generate_numba_function(
        self,
        expr: Expr,
        params: List[Symbol],
        func_name: str = "eval_Lk"
    ) -> Optional[Callable]:
        """
        Generate a Numba-JIT compiled function.
        
        Args:
            expr: SymPy expression
            params: List of parameter symbols
            func_name: Name for the function
            
        Returns:
            JIT-compiled callable or None if Numba unavailable
        """
        if not self._numba_available:
            return None
        
        import numba
        from sympy.utilities.lambdify import lambdify
        import numpy as np
        
        numpy_func = lambdify(params, expr, modules=['numpy'])
        
        @numba.jit(nopython=True, cache=True, fastmath=True)
        def jit_wrapper(*args):
            return numpy_func(*args)
        
        return numpy_func
    
    def generate_vectorized_function(
        self,
        expr: Expr,
        params: List[Symbol]
    ) -> Optional[Callable]:
        """
        Generate a vectorized NumPy function for batch evaluation.
        
        Args:
            expr: SymPy expression
            params: List of parameter symbols
            
        Returns:
            Vectorized callable
        """
        from sympy.utilities.lambdify import lambdify
        import numpy as np
        
        numpy_func = lambdify(params, expr, modules=['numpy'])
        
        def vectorized(*arrays):
            return numpy_func(*arrays)
        
        return vectorized
