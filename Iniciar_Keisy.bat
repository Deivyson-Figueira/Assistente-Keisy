@echo off
:: Altera o diretório de execução para a pasta onde o arquivo .bat está localizado
cd /d "%~dp0"

echo 1. VERIFICANDO AMBIENTE DO WINDOWS...

:: 1. Tenta verificar se o Python já está instalado no sistema hospedeiro
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Python detectado no sistema.
    goto :inicializar_ambiente
)

echo ❌ Python nao encontrado no computador hospedeiro!
echo 🛠️ Iniciando instalacao automatica do Python...

:: Configura a versão e o link de download do instalador silencioso do Python (64-bit)
set "PYTHON_EXE=python_installer.exe"
set "PYTHON_URL=https://www.python.org/ftp/python/3.12.3/python-3.12.3-amd64.exe"

echo 📥 Baixando instalador oficial do Python...
powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object System.Net.WebClient).DownloadFile('%PYTHON_URL%', '%PYTHON_EXE%')"

if not exist "%PYTHON_EXE%" (
    echo ❌ Falha ao baixar o instalador do Python. Verifique a conexao com a Internet.
    pause
    exit /b
)

echo ⚙️ Instalando Python... Por favor, aguarde...
start /wait "" "%PYTHON_EXE%" /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_test=0 Shortcuts=0

:: Remove o arquivo instalador do pen drive para não acumular lixo
del "%PYTHON_EXE%"

set "PATH=%USERPROFILE%\AppData\Local\Programs\Python\Python312\;%USERPROFILE%\AppData\Local\Programs\Python\Python312\Scripts\;%PATH%"

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Falha crítica: O Python foi instalado, mas o terminal não conseguiu mapear o executavel.
    echo 💡 Dica: Tente fechar esta janela e abrir o 'Iniciar_Keisy.bat' novamente.
    pause
    exit /b
)
echo ✅ Python instalado e configurado com sucesso!

:inicializar_ambiente
echo.
echo 2. CONFIGURANDO AMBIENTE PORTATIL DA KEISY...

:: Verifica se o ambiente virtual existe no pen drive
if not exist ".Keisy\Scripts\activate.bat" (
    echo 📦 Criando ambiente isolado (.Keisy) no seu pen drive...
    python -m venv .Keisy
)

:: Ativa o ambiente virtual do pen drive
echo 🛠️ Ativando ambiente virtual...
call .Keisy\Scripts\activate.bat

:: Executa a Keisy diretamente
echo 🧠 Carregando a Keisy...
python Keisy_IA.py

:: Desativa o ambiente ao fechar a interface gráfica
call deactivate
echo 👋 Sistema Keisy encerrado.
pause
