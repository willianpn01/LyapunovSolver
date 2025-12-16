@echo off
setlocal

pushd "%~dp0"

echo ==========================================
echo LyapunovSolver v2.0 - GUI (Windows)
echo ==========================================
echo.

where python >nul 2>nul
if errorlevel 1 (
  echo ERRO: Python nao encontrado no PATH.
  echo Instale o Python 3.10+ e marque a opcao "Add Python to PATH".
  popd
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo Criando ambiente virtual em .venv...
  python -m venv .venv
  if errorlevel 1 (
    echo ERRO: Falha ao criar o ambiente virtual.
    popd
    pause
    exit /b 1
  )
)

call ".venv\Scripts\activate.bat"

echo Atualizando pip...
python -m pip install --upgrade pip

echo Instalando dependencias...
pip install -r requirements.txt
if errorlevel 1 (
  echo ERRO: Falha ao instalar dependencias.
  popd
  pause
  exit /b 1
)

echo.
echo Iniciando GUI...
echo (Feche a janela para encerrar)
echo.
python run_gui.py

popd
pause
