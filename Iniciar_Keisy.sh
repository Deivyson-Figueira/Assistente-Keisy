#!/bin/bash
# Descobre a pasta atual do pen drive onde o script está a ser executado
DIR_ATUAL="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR_ATUAL"

echo "🚀 Ativando ambiente virtual móvel..."
source .Keisy/bin/activate

echo "🧠 Carregando a Keisy..."
python3 Keisy_IA.py

deactivate
