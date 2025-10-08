# 🤖 Assistente Virtual KEISY

[![Made with Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**KEISY** é um assistente virtual em português que combina **Processamento de Linguagem Natural (NLP)** para conversação, **Reconhecimento e Síntese de Voz (TTS)**, e um **Pipeline de Machine Learning (ML)** para tarefas de classificação.

A personalidade da Keisy é ajustada via *Fine-Tuning* no modelo DialoGPT, buscando um tom feminino, natural e envolvente.

---

## ✨ Funcionalidades Principais

* **Conversação Avançada:** Utiliza o modelo **DialoGPT-medium** (ajustado para personalidade KEISY) para diálogos com histórico e contexto.
* **Interação por Voz:** Reconhece comandos de voz (pt-BR) e responde com áudio.
* **Pipeline de ML Integrado:** Pode executar um pipeline de Machine Learning (classificação) com o dataset Iris sob comando de voz/texto.
* **Modularidade:** Usa bibliotecas padrão da indústria (Hugging Face, scikit-learn, PyAudio).

---

## ⚙️ Instalação

### 1. Pré-requisitos

O projeto requer o **FFmpeg** para manipulação de áudio (`pydub`).

```bash
# Para sistemas baseados em Debian/Ubuntu (Google Colab/Linux)
sudo apt-get install ffmpeg -y
