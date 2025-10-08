from transformers import pipeline, set_seed
import speech_recognition as sr
from plyer import notification
import logging
import requests
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from imblearn.over_sampling import SMOTE
import plotly.express as px
import joblib
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import _play_with_pyaudio
import pyaudio
import random
import os # Importação para lidar com arquivos

# --- Configurações Iniciais ---

# Configurar logging
logging.basicConfig(level=logging.INFO)

# Configurar reconhecimento de voz
recognizer = sr.Recognizer()

# Configurar modelo de NLP
set_seed(42)
# Tenta carregar o modelo sem prefixo "KEISY"
nlp = pipeline("text-generation", model="microsoft/DialoGPT-medium")

# Variáveis globais para o histórico de conversa
historico_conversa = []

# --- Funções de Áudio e Comunicação ---

def listar_dispositivos_audio():
    """Lista todos os dispositivos de áudio disponíveis no sistema."""
    p = pyaudio.PyAudio()
    print("\n--- Dispositivos de Áudio ---")
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        # Exibe apenas os dispositivos de saída
        if dev['maxOutputChannels'] > 0:
            print(f"OUTPUT {i}: {dev['name']}")
    print("---------------------------\n")
    p.terminate()

def reconhecer_voz():
    """Captura áudio do microfone e usa o Google Speech Recognition."""
    with sr.Microphone() as source:
        # Ajuste para ruído de fundo
        recognizer.adjust_for_ambient_noise(source)
        print("Diga algo (ou digite no console):")
        
        try:
            # Tenta capturar o áudio por 5 segundos
            audio = recognizer.listen(source, timeout=5)
            # Tenta reconhecer a voz em português
            texto = recognizer.recognize_google(audio, language='pt-BR')
            return texto
        except sr.WaitTimeoutError:
            print("Tempo esgotado. Nenhuma fala detectada.")
            return None
        except sr.UnknownValueError:
            print("Não consegui entender o que você disse.")
            return None
        except sr.RequestError as e:
            print(f"Erro ao solicitar resultados do serviço de reconhecimento de voz; {e}")
            return None
        except Exception as e:
            print(f"Erro inesperado no reconhecimento de voz: {e}")
            return None

def responder_mensagem(mensagem):
    """Gera uma resposta com o modelo DialoGPT, mantendo o histórico."""
    global historico_conversa
    
    # 1. Adiciona a entrada do usuário e limita o histórico
    historico_conversa.append("Você: " + mensagem)
    max_historico = 5
    historico_conversa = historico_conversa[-max_historico:]

    # 2. Prepara o prompt, garantindo que a próxima linha espere a resposta de KEISY
    prompt = "\n".join(historico_conversa) + "\nKEISY: " 
    
    # 3. Gera a conversa
    conversa = nlp(prompt, max_new_tokens=50, temperature=0.8, top_p=0.95, repetition_penalty=1.2, do_sample=True)
    
    # 4. Extrai a resposta limpa
    resposta_completa = conversa[0]['generated_text']
    
    # Busca a parte da resposta que vem depois do prefixo "KEISY: "
    if "KEISY: " in resposta_completa:
        resposta = resposta_completa.split("KEISY: ")[-1].strip()
    else:
        # Caso o modelo ignore o prefixo, usa a técnica original
        resposta = resposta_completa.split(prompt)[-1].strip()

    # 5. Adiciona a resposta de KEISY ao histórico
    historico_conversa.append("KEISY: " + resposta)
    return resposta

def play_with_specific_device(seg, device_index):
    """Reproduz um segmento de áudio em um dispositivo de saída específico."""
    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=p.get_format_from_width(seg.sample_width),
                        channels=seg.channels,
                        rate=seg.frame_rate,
                        output=True,
                        output_device_index=device_index)
        _play_with_pyaudio(seg, stream=stream)
        stream.stop_stream()
        stream.close()
    except ValueError as e:
        print(f"Erro: Dispositivo de áudio com índice {device_index} inválido ou inativo. Tentando dispositivo padrão (0). Erro original: {e}")
        # Tenta o dispositivo padrão (0) como fallback
        stream = p.open(format=p.get_format_from_width(seg.sample_width),
                        channels=seg.channels,
                        rate=seg.frame_rate,
                        output=True,
                        output_device_index=0)
        _play_with_pyaudio(seg, stream=stream)
        stream.stop_stream()
        stream.close()
    finally:
        p.terminate()


def responder_mensagem_com_audio(mensagem, device_index):
    """Gera a resposta do NLP, converte para TTS e reproduz no dispositivo."""
    
    # Tratar entradas de controle para evitar chamadas desnecessárias ao modelo
    if mensagem is None:
        return

    resposta = responder_mensagem(mensagem)
    print(f"KEISY: {resposta}")

    if not resposta:
        resposta = "Desculpe, não consegui gerar uma resposta."

    # Converter texto para áudio
    try:
        tts = gTTS(resposta, lang='pt')
        tts.save("resposta.mp3")

        # Reproduzir áudio usando dispositivo específico
        audio = AudioSegment.from_mp3("resposta.mp3")
        play_with_specific_device(audio, device_index)
        
        # Opcional: remover o arquivo temporário
        os.remove("resposta.mp3")
        
    except Exception as e:
        print(f"Erro ao gerar ou reproduzir áudio: {e}")


def enviar_notificacao(titulo, mensagem):
    """Envia uma notificação nativa (depende do SO)."""
    notification.notify(
        title=titulo,
        message=mensagem,
        timeout=10
    )

def buscar_informacoes(query):
    """Busca informações na web usando a API do Bing (requer chave válida)."""
    # **ATENÇÃO: Substitua 'sua_chave_de_api_aqui' por sua chave real**
    url = f"https://api.bing.microsoft.com/v7.0/search?q={query}"
    headers = {"Ocp-Apim-Subscription-Key": "sua_chave_de_api_aqui"}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status() # Lança exceção para status de erro (4xx ou 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Erro ao buscar informações na web: {e}")
        return {"error": str(e)}

# --- Funções de Machine Learning (ML) ---

def carregar_dados():
    """Carrega o dataset Iris."""
    try:
        iris = load_iris()
        X = iris.data
        y = iris.target
        # Retorna o conjunto completo para o pipeline cuidar da seleção e escalonamento
        return X, y
    except Exception as e:
        logging.error(f"Erro ao carregar os dados: {e}")
        return None, None

def preprocessar_dados(X, y):
    """Aplica SMOTE para rebalanceamento (embora Iris seja balanceado) e retorna."""
    try:
        # SMOTE é mantido conforme sua implementação original, embora para Iris seja redundante.
        smote = SMOTE(random_state=42)
        # O SelectKBest será aplicado dentro do Pipeline, então SMOTE é aplicado nos dados brutos.
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled
    except Exception as e:
        logging.error(f"Erro ao pré-processar os dados: {e}")
        return None, None

def treinar_modelo_incremental(X, y):
    """Define e treina o Pipeline com StandardScaler, SelectKBest e SGDClassifier."""
    try:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            # SelectKBest reduz de 4 para 2 features
            ('selector', SelectKBest(score_func=f_classif, k=2)),
            ('classifier', SGDClassifier(random_state=42, class_weight='balanced'))
        ])
        # Treina o modelo
        pipeline.fit(X, y)
        return pipeline
    except Exception as e:
        logging.error(f"Erro ao treinar o modelo: {e}")
        return None

def salvar_modelo(modelo, caminho):
    """Salva o modelo treinado em disco."""
    joblib.dump(modelo, caminho)
    print(f"Modelo salvo em {caminho}")

def carregar_modelo(caminho):
    """Carrega o modelo de disco."""
    try:
        modelo = joblib.load(caminho)
        print(f"Modelo carregado de {caminho}")
        return modelo
    except FileNotFoundError:
        print(f"Arquivo do modelo não encontrado em {caminho}. Retornando None.")
        return None

def avaliar_modelo(modelo, X_test, y_test):
    """Avalia o modelo e exibe a matriz de confusão."""
    try:
        y_pred = modelo.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        print("\n--- Avaliação do Modelo ---")
        print(f"Acurácia: {accuracy:.4f}")
        
        # Exibe a matriz de confusão com matplotlib
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                      display_labels=load_iris().target_names)
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Matriz de Confusão")
        plt.show()
        
    except Exception as e:
        logging.error(f"Erro ao avaliar o modelo: {e}")

def visualizar_dados(X, y):
    """Visualiza os dados usando Plotly (features selecionadas pelo pipeline)."""
    try:
        # Para visualizar, precisamos aplicar apenas o SelectKBest (o pipeline completo faria isso internamente)
        selector = SelectKBest(score_func=f_classif, k=2)
        X_2d = selector.fit_transform(X, y)
        
        # Cria o DataFrame para Plotly
        data = {
            'Feature 1': X_2d[:, 0], 
            'Feature 2': X_2d[:, 1], 
            'Species': [load_iris().target_names[i] for i in y]
        }

        fig = px.scatter(data, 
                         x='Feature 1', 
                         y='Feature 2', 
                         color='Species', 
                         title='Visualização Interativa dos Dados (2 Melhores Features)',
                         labels={'Feature 1': 'Melhor Feature (Eixo X)', 'Feature 2': 'Segunda Melhor Feature (Eixo Y)'})
        fig.show() # Abre o gráfico no navegador (ou exibe no ambiente notebook)
    except Exception as e:
        logging.error(f"Erro ao visualizar os dados com Plotly: {e}")

def realizar_previsao(modelo, novos_dados):
    """Realiza previsões com novos dados."""
    try:
        previsoes = modelo.predict(novos_dados)
        nomes_classes = [load_iris().target_names[p] for p in previsoes]
        print(f"Previsões (Classes): {previsoes}")
        print(f"Previsões (Nomes): {nomes_classes}")
        return previsoes
    except Exception as e:
        logging.error(f"Erro ao realizar previsão: {e}")
        return None

# --- Loop Principal de Interação ---

def main_ml():
    """Função que executa o pipeline de Machine Learning (ML)."""
    print("\n[KEISY - ML] Iniciando o Pipeline de Machine Learning com dataset Iris.")
    X, y = carregar_dados()
    
    if X is not None and y is not None:
        visualizar_dados(X, y) # Visualiza antes do SMOTE/Pipeline
        X_resampled, y_resampled = preprocessar_dados(X, y)
        
        if X_resampled is not None and y_resampled is not None:
            # Treinamento e avaliação
            modelo = treinar_modelo_incremental(X_resampled, y_resampled)
            if modelo is not None:
                print("[KEISY - ML] Modelo treinado com sucesso!")
                salvar_modelo(modelo, "modelo_incremental.pkl")
                
                modelo_carregado = carregar_modelo("modelo_incremental.pkl")
                
                # Avalia usando os dados reamostrados (treinamento)
                avaliar_modelo(modelo_carregado, X_resampled, y_resampled) 
                
                # Exemplo de Previsão
                print("\n[KEISY - ML] Testando previsão com novos dados:")
                # Estes dados devem ter 4 features, pois o Pipeline cuida do SelectKBest
                novos_dados = np.array([[5.1, 3.5, 1.4, 0.2], [6.2, 3.4, 5.4, 2.3]]) 
                realizar_previsao(modelo_carregado, novos_dados)
            else:
                print("[KEISY - ML] Falha ao treinar o modelo.")
        else:
            print("[KEISY - ML] Falha ao pré-processar os dados.")
    else:
        print("[KEISY - ML] Falha ao carregar os dados.")
    print("[KEISY - ML] Pipeline de ML finalizado.")

def loop_interacao(device_index=0):
    """Loop principal que gerencia a interação por voz e texto."""
    
    while True:
        # Opção de entrada por voz (reconhecer_voz) ou por texto (input)
        print("\n--- Modo de Interação ---")
        entrada_raw = input("Modo (1-Voz / 2-Texto): ").strip()
        
        if entrada_raw == '1':
            mensagem = reconhecer_voz()
        elif entrada_raw == '2':
            mensagem = input("Você: ").strip()
        else:
            print("Opção inválida. Digite '1' ou '2'.")
            continue

        if mensagem is None:
            # Não houve fala ou erro, reinicia o loop
            continue
            
        mensagem_formatada = mensagem.strip().lower()

        if 'sair' in mensagem_formatada or 'tchau' in mensagem_formatada:
            print("Encerrando o robô. Até mais!")
            responder_mensagem_com_audio("Tchau, tchau! Até a próxima!", device_index)
            break
        elif 'ml' in mensagem_formatada or 'machine learning' in mensagem_formatada:
            responder_mensagem_com_audio("Certo, vou iniciar o pipeline de machine learning para o dataset Iris. Aguarde...", device_index)
            main_ml() 
        elif 'dispositivos' in mensagem_formatada:
            listar_dispositivos_audio()
        else:
            # Resposta normal com o modelo de diálogo
            responder_mensagem_com_audio(mensagem, device_index)

if __name__ == "__main__":
    listar_dispositivos_audio()
    
    # Define o dispositivo de saída (Pode ser alterado pelo usuário)
    try:
        device_input = input("Digite o índice do dispositivo de OUTPUT de áudio (padrão 0): ").strip()
        DEVICE_AUDIO_SAIDA = int(device_input) if device_input.isdigit() else 0
    except Exception:
        DEVICE_AUDIO_SAIDA = 0

    print(f"\nDispositivo de saída selecionado: Índice {DEVICE_AUDIO_SAIDA}. Iniciando KEISY...")
    
    # Inicializa a conversa com áudio
    responder_mensagem_com_audio("Olá! Eu sou a Keisy. Estou pronta para conversar com você! Como você se chama?", DEVICE_AUDIO_SAIDA)

    # Inicia o loop de interação
    loop_interacao(DEVICE_AUDIO_SAIDA)
