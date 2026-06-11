import os
import sys
import subprocess
import re
import platform
import threading
import sqlite3
from datetime import datetime

# ==============================================================================
# 1. SCRIPT DE AUTO-INSTALAÇÃO MULTIPLATAFORMA
# ==============================================================================
def verificar_e_instalar_dependencias():
    import shutil
    import time
    sistema_operacional = platform.system().lower()

    if sistema_operacional == "linux":
        deps_sistema = {"gcc": "build-essential", "g++": "build-essential"}
        if not shutil.which("espeak") and not shutil.which("espeak-ng"):
            deps_sistema["espeak"] = "espeak-ng"

        tk_instalado = False
        try:
            import tkinter
            tk_instalado = True
        except ImportError: pass

        faltando_sistema = [pacote for cmd, pacote in deps_sistema.items() if not shutil.which(cmd)]
        if not tk_instalado: faltando_sistema.append("python3-tk")
        faltando_sistema = list(set(faltando_sistema))

        if faltando_sistema:
            try:
                subprocess.check_call(["sudo", "apt", "update", "-y"], stdout=subprocess.DEVNULL)
                subprocess.check_call(["sudo", "apt", "install", "-y"] + faltando_sistema)
            except: pass

    elif sistema_operacional == "windows":
        try:
            import tkinter
            tk_pronto = True
        except ImportError: tk_pronto = False

        if not tk_pronto:
            import urllib.request
            versao_python = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            arquitetura = "amd64" if sys.maxsize > 2**32 else "win32"
            url_instalador = f"https://www.python.org/ftp/python/{versao_python}/python-{versao_python}-{arquitetura}.exe"
            nome_arquivo = "instalador_python_suporte.exe"
            try:
                if not os.path.exists(nome_arquivo):
                    urllib.request.urlretrieve(url_instalador, nome_arquivo)
                subprocess.Popen([nome_arquivo])
                while True:
                    time.sleep(5)
                    try:
                        import tkinter
                        if os.path.exists(nome_arquivo): os.remove(nome_arquivo)
                        break
                    except ImportError: pass
            except: sys.exit(1)

    libs_obrigatorias = ["llama_cpp", "psutil", "yt_dlp", "gtts", "pygame"]
    libs_para_instalar = []
    for lib in libs_obrigatorias:
        try: __import__("llama_cpp" if lib == "llama_cpp" else lib)
        except ImportError:
            if lib == "llama_cpp": libs_para_instalar.append("llama-cpp-python")
            elif lib == "gtts": libs_para_instalar.append("gTTS")
            else: libs_para_instalar.append(lib)

    if libs_para_instalar:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install"] + libs_para_instalar)
        except: sys.exit(1)

verificar_e_instalar_dependencias()

import tkinter as tk
from tkinter import scrolledtext
import psutil
from gtts import gTTS
import pygame
from llama_cpp import Llama

# ==============================================================================
# 2. MOTOR DE ÁUDIO ASSÍNCRONO
# ==============================================================================
class SistemaAudioEControle:
    def __init__(self):
        try: pygame.mixer.init()
        except: pass
        self.audio_file = os.path.join("/tmp" if platform.system().lower() == "linux" else os.getcwd(), "keisy_voice.mp3")
        self.os = platform.system()

    def open_app(self, name: str) -> bool:
        try:
            if self.os.lower() == "windows": os.startfile(name); return True
            elif self.os.lower() == "darwin": return subprocess.call(["open", name]) == 0
            else: return subprocess.call(["xdg-open", name]) == 0
        except: return False

    def speak(self, text, voz_ativa=True, callback_fim=None):
        if not voz_ativa:
            if callback_fim: callback_fim()
            return

        def _run():
            try:
                texto_limpo = re.sub(r'\*.*?\*', '', text).strip()
                if not texto_limpo:
                    if callback_fim: callback_fim()
                    return

                if self.os.lower() == "windows":
                    from win32com.client import Dispatch
                    speaker = Dispatch("SAPI.SpVoice")
                    speaker.Speak(texto_limpo)
                elif self.os.lower() == "darwin":
                    subprocess.call(["say", texto_limpo])
                else:
                    if os.path.exists(self.audio_file):
                        try: pygame.mixer.music.unload()
                        except: pass
                        try: os.remove(self.audio_file)
                        except: pass
                    
                    tts = gTTS(text=texto_limpo, lang='pt', tld='com.br')
                    tts.save(self.audio_file)
                    
                    pygame.mixer.music.load(self.audio_file)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
            except Exception as e:
                print(f"Erro na voz: {e}")
            finally:
                if callback_fim: callback_fim()

        threading.Thread(target=_run, daemon=True).start()

system = SistemaAudioEControle()

# ==============================================================================
# 3. INTERFACE VISUAL CUSTOMIZÁVEL E CORE
# ==============================================================================
class KeisyApp:
    def __init__(self, root_tk):
        self.root = root_tk
        self.root.title("Keisy IA")
        self.root.geometry("500x730")
        
        # Estados internos configuráveis
        self.voz_ativa = False  # Voz desativada por padrão
        self.tema_atual = "dracula"
        
        # Paletas de cores (Três combinações completas)
        self.temas = {
            "dracula": {"bg": "#1e1e2e", "chat": "#020617", "texto": "#cdd6f4", "input": "#313244", "user": "#89b4fa", "keisy": "#cba6f7", "btn": "#0284c7", "bar": "#11111b"},
            "light": {"bg": "#f4f4f5", "chat": "#ffffff", "texto": "#18181b", "input": "#e4e4e7", "user": "#2563eb", "keisy": "#7c3aed", "btn": "#0f766e", "bar": "#e4e4e7"},
            "cyberpunk": {"bg": "#000000", "chat": "#0c0014", "texto": "#00ffcc", "input": "#1a0033", "user": "#ff007f", "keisy": "#00ffcc", "btn": "#9900ff", "bar": "#1a0033"}
        }

        self.model_path = "Keisy.gguf"
        self.db_path = "./data/keisy.db"
        self.download_dir = "./downloads"
        self.llm = None
        self.versao_software = "1.0"
        
        cpu_count = os.cpu_count() or 1
        self.n_threads = max(1, min(cpu_count, 4))
        self.n_ctx = 2048

        self.sistema_prompt = (
            "Tu és a Keisy, uma inteligência artificial avançada, elegante e de alta eficiência criada por Deivyson. "
            "Tu interages com diferentes usuários de forma educada, prestativa e natural. "
            "Responde sempre em português de forma direta, adaptando o tratamento de acordo com o interlocutor. "
            "Usa ações sutis na terceira pessoa entre asteriscos, como *sorri*, *analisa dados* ou *presta atenção*."
        )

        self.setup_ui()
        self.init_pastas_e_banco()
        self.aplicar_tema("dracula")
        
        threading.Thread(target=self.carregar_modelo_ia, daemon=True).start()

    def setup_ui(self):
        # 🟢 PAINEL SUPERIOR: Controles Rápidos de Configuração
        self.top_bar = tk.Frame(self.root, pady=5)
        self.top_bar.pack(fill=tk.X, side=tk.TOP)

        # Botão de Ativar/Desativar Voz
        self.btn_voz = tk.Button(self.top_bar, text="🔇 VOZ: DESATIVADA", font=("Arial", 9, "bold"), fg="white", bg="#f38ba8", command=self.alternar_voz, borderwidth=0, padx=8, pady=3)
        self.btn_voz.pack(side=tk.LEFT, padx=10)

        # Menus Rápidos de Configuração (Cores e Tamanhos)
        self.lbl_config = tk.Label(self.top_bar, text="⚙️ Ajustes:", font=("Arial", 9))
        self.lbl_config.pack(side=tk.LEFT, padx=(10, 2))

        # Dropdowns nativos do Tkinter para personalização rápida
        self.opt_tema = tk.StringVar(value="Dracula")
        self.menu_tema = tk.OptionMenu(self.top_bar, self.opt_tema, "Dracula", "Light", "Cyberpunk", command=lambda t: self.aplicar_tema(t.lower()))
        self.menu_tema.config(font=("Arial", 8), borderwidth=0, highlightthickness=0)
        self.menu_tema.pack(side=tk.LEFT, padx=2)

        self.opt_tamanho = tk.StringVar(value="Padrão")
        self.menu_tamanho = tk.OptionMenu(self.top_bar, self.opt_tamanho, "Compacto", "Padrão", "Expandido", command=self.redimensionar_janela)
        self.menu_tamanho.config(font=("Arial", 8), borderwidth=0, highlightthickness=0)
        self.menu_tamanho.pack(side=tk.LEFT, padx=2)

        # 🔵 ÁREA DE CONVERSA
        self.chat_area = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, font=("Courier", 11), borderwidth=0, padx=10, pady=10)
        self.chat_area.pack(padx=15, pady=10, fill=tk.BOTH, expand=True)
        self.chat_area.config(state=tk.DISABLED)
        self.chat_area.tag_config("bold", font=("Courier", 11, "bold"))
        
        # 🟡 INPUT DE TEXTO E ENVIO
        self.bottom_frame = tk.Frame(self.root)
        self.bottom_frame.pack(padx=15, pady=(0, 5), fill=tk.X)

        self.user_input = tk.Entry(self.bottom_frame, font=("Arial", 12), borderwidth=0)
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=8, padx=(0, 10))
        self.user_input.bind("<Return>", lambda event: self.send_message())

        self.send_btn = tk.Button(self.bottom_frame, text="ENVIAR", font=("Arial", 10, "bold"), borderwidth=0, command=self.send_message, padx=15, pady=5)
        self.send_btn.pack(side=tk.RIGHT)

        # 🔴 BARRA DE STATUS DINÂMICA NO RODAPÉ
        self.status_bar = tk.Label(self.root, text="🔄 Inicializando sistemas...", font=("Arial", 9, "italic"), anchor=tk.W, padx=15, pady=3)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    # ==============================================================================
    # 4. FUNÇÕES DE COSTUMIZAÇÃO DINÂMICA
    # ==============================================================================
    def alternar_voz(self):
        self.voz_ativa = not self.voz_ativa
        if self.voz_ativa:
            self.btn_voz.config(text="🔊 VOZ: ATIVADA", bg="#a6e3a1", fg="#11111b")
            self.atualizar_status("🔊 Saída de áudio habilitada.", self.temas[self.tema_atual]["texto"])
        else:
            self.btn_voz.config(text="🔇 VOZ: DESATIVADA", bg="#f38ba8", fg="white")
            self.atualizar_status("🔇 Modo silencioso ativo.", self.temas[self.tema_atual]["texto"])

    def aplicar_tema(self, nome_tema):
        if nome_tema not in self.temas: return
        self.tema_atual = nome_tema
        t = self.temas[nome_tema]

        # Atualiza as cores de fundo e frente dos widgets
        self.root.configure(bg=t["bg"])
        self.top_bar.configure(bg=t["bg"])
        self.bottom_frame.configure(bg=t["bg"])
        self.lbl_config.configure(bg=t["bg"], fg=t["texto"])
        
        self.chat_area.configure(bg=t["chat"], fg=t["texto"], insertbackground=t["texto"])
        self.user_input.configure(bg=t["input"], fg=t["texto"], insertbackground=t["texto"])
        self.send_btn.configure(bg=t["btn"], fg="white" if nome_tema != "cyberpunk" else "#000000")
        self.status_bar.configure(bg=t["bar"], fg=t["texto"])
        
        # Força reconfiguração de tags internas do histórico do chat
        self.chat_area.tag_config("Você", foreground=t["user"])
        self.chat_area.tag_config("Keisy", foreground=t["keisy"])
        self.chat_area.tag_config("SISTEMA", foreground=t["btn"])
        self.chat_area.tag_config("AVISO", foreground="#f38ba8")
        self.chat_area.tag_config("ATUALIZAÇÃO", foreground="#f5e0dc")

    def redimensionar_janela(self, escolha_tam=None):
        tamanho = self.opt_tamanho.get()
        if tamanho == "Compacto":
            self.root.geometry("400x550")
        elif tamanho == "Padrão":
            self.root.geometry("500x730")
        elif tamanho == "Expandido":
            self.root.geometry("700x850")

    def atualizar_status(self, texto, cor=None):
        cor_final = cor if cor else self.temas[self.tema_atual]["texto"]
        self.root.after(0, lambda: self.status_bar.config(text=texto, fg=cor_final))

    def append_message(self, sender, message, tag_name):
        self.root.after(0, self._safe_append_message, sender, message, tag_name)

    def _safe_append_message(self, sender, message, tag_name):
        self.chat_area.config(state=tk.NORMAL)
        self.chat_area.insert(tk.END, f"{sender}: ", ("bold", tag_name))
        self.chat_area.insert(tk.END, f"{message}\n\n", (tag_name,))
        self.chat_area.see(tk.END)
        self.chat_area.config(state=tk.DISABLED)

    def init_pastas_e_banco(self):
        os.makedirs("data", exist_ok=True)
        os.makedirs(self.download_dir, exist_ok=True)
        with sqlite3.connect(self.db_path) as con:
            con.execute("CREATE TABLE IF NOT EXISTS facts (key TEXT PRIMARY KEY, value TEXT)")

    def carregar_modelo_ia(self):
        self.atualizar_status("🔄 Carregando sistemas...")
        self.append_message("SISTEMA", f"Online ({system.os}). Matriz iniciada.", "SISTEMA")
        self.send_btn.config(state=tk.DISABLED)

        threading.Thread(target=self.verificar_atualizacao, daemon=True).start()

        if not os.path.exists(self.model_path):
            self.append_message("AVISO", f"Arquivo '{self.model_path}' oculto/ausente. Modo de automação local ativo.", "AVISO")
            self.atualizar_status("⚠️ Modo Automação Direta ativo.")
            self.root.after(0, lambda: self.send_btn.config(state=tk.NORMAL))
            return

        try:
            self.llm = Llama(model_path=self.model_path, n_ctx=self.n_ctx, n_threads=self.n_threads, n_batch=512, verbose=False)
            saudacao = "Sistemas integrados. Interface de comunicação ativa. Olá, em que posso ajudar hoje?"
            self.append_message("Keisy", saudacao, "Keisy")
            self.atualizar_status("🟢 Keisy está Pronta")
            system.speak(saudacao, voz_ativa=self.voz_ativa)
            self.root.after(0, lambda: self.send_btn.config(state=tk.NORMAL))
        except Exception as e:
            self.append_message("AVISO", f"Falha ao mapear GGUF: {str(e)}", "AVISO")
            self.atualizar_status("❌ Falha Crítica na IA")
            self.root.after(0, lambda: self.send_btn.config(state=tk.NORMAL))

    # ==============================================================================
    # 5. RECURSOS COMPLEMENTARES
    # ==============================================================================
    def memorizar_fato(self, chave, valor):
        try:
            with sqlite3.connect(self.db_path) as con:
                con.execute("INSERT OR REPLACE INTO facts (key, value) VALUES (?, ?)", (chave.lower().strip(), valor.strip()))
            return True
        except: return False

    def buscar_memoria(self, texto_usuario):
        try:
            with sqlite3.connect(self.db_path) as con:
                cursor = con.cursor()
                cursor.execute("SELECT key, value FROM facts")
                todos_os_fatos = cursor.fetchall()
            contexto = [f"Fato guardado sobre {c}: {v}" for c, v in todos_os_fatos if c in texto_usuario.lower()]
            return "\n[Fatos da Memória Local]:\n" + "\n".join(contexto) + "\n" if contexto else ""
        except: return ""

    def pesquisar_web(self, termo_busca):
        import urllib.request, urllib.parse
        self.atualizar_status("🌐 Pesquisando dados na internet...")
        try:
            url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(termo_busca)}"
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=6) as response:
                html = response.read().decode('utf-8')
            resumos = re.findall(r'<td class="result-snippet">(.*?)</td>', html, re.DOTALL)
            if resumos:
                texto = "".join([f"- {re.sub(r'<[^>]+>', '', r).strip()}\n" for r in resumos[:3]])
                return f"\n[Dados recentes da Web]:\n{texto}\n"
            return ""
        except: return ""

    def verificar_atualizacao(self):
        import urllib.request
        url_versao_remota = "https://raw.githubusercontent.com/seu-usuario/seu-repositorio/main/versao.txt"
        try:
            req = urllib.request.Request(url_versao_remota, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=3) as response:
                versao_remota = response.read().decode('utf-8').strip()
            if versao_remota > self.versao_software:
                self.append_message("ATUALIZAÇÃO", f"🔔 Versão {versao_remota} disponível na nuvem.", "ATUALIZAÇÃO")
        except: pass

    # ==============================================================================
    # 6. GERENCIADOR DE PROCESSAMENTO E RESPOSTAS
    # ==============================================================================
    def send_message(self):
        text = self.user_input.get().strip()
        if not text: return

        self.user_input.delete(0, tk.END)
        self.append_message("Você", text, "Você")
        self.send_btn.config(state=tk.DISABLED)
        
        self.atualizar_status("⚡ Executando triagem...")
        threading.Thread(target=self.processar_fluxo_geral, args=(text,), daemon=True).start()

    def processar_fluxo_geral(self, user_text):
        if user_text.lower().startswith("aprenda que"):
            padrao = user_text[11:].strip()
            chave, valor = padrao.split(" é ", 1) if " é " in padrao else (padrao, padrao)
            if self.memorizar_fato(chave, valor):
                resp = f"*memorizado* Entendido. Guardei que '{chave}' é '{valor}'."
            else: resp = "Erro ao acessar base SQLite."
            self.exibir_e_falar_resposta(resp)
            return

        cmd = user_text.lower()
        if "que horas são" in cmd:
            self.exibir_e_falar_resposta(f"Agora são {datetime.now().strftime('%H:%M')}.")
            return
            
        match = re.search(r"abra (?:o |a )?(.+)", cmd)
        if match:
            alvo = match.group(1).strip()
            if system.open_app(alvo): self.exibir_e_falar_resposta(f"Abrindo {alvo}. *processando*")
            else: self.exibir_e_falar_resposta(f"Não consegui iniciar o aplicativo {alvo}.")
            return

        if any(g in cmd for g in ["baixe", "baixar", "download"]):
            urls = re.findall(r'(https?://\S+)', user_text)
            termo = urls[0] if urls else re.sub(r'(baixe|baixar|download)\s*', '', user_text, flags=re.IGNORECASE).strip()
            if not termo:
                self.exibir_e_falar_resposta("Especifique um termo ou link para download.")
                return
            
            self.atualizar_status("📥 Baixando mídia...")
            def _dl():
                import yt_dlp
                try:
                    opts = {'outtmpl': f'{self.download_dir}/%(title)s.%(ext)s', 'format': 'best', 'quiet': True}
                    with yt_dlp.YoutubeDL(opts) as ydl: ydl.download([termo if urls else f"ytsearch1:{termo}"])
                    self.exibir_e_falar_resposta("✓ Download concluído com sucesso!")
                except: self.exibir_e_falar_resposta("❌ Erro ao baixar arquivo.")
            threading.Thread(target=_dl, daemon=True).start()
            return

        if not self.llm:
            self.exibir_e_falar_resposta("Matriz neural indisponível no momento.")
            return

        self.atualizar_status("🧠 Keisy está pensando...", "#f5c2e7")
        contexto_adicional = self.buscar_memoria(user_text)
        if any(p in cmd for p in ["pesquise", "busque", "quem é", "o que é"]):
            termo_pesquisa = re.sub(r'(pesquise|busque|na web|quem é|o que é)\s*', '', user_text, flags=re.IGNORECASE).strip()
            contexto_adicional += self.pesquisar_web(termo_pesquisa)

        prompt = (
            f"<|start_header_id|>system<|end_header_id|>\n\n{self.sistema_prompt}\n"
            f"{contexto_adicional}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_text}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        try:
            output = self.llm(prompt, max_tokens=128, temperature=0.5, stop=["<|eot_id|>", "\n\n"], verbose=False)
            resposta = output['choices'][0]['text'].strip() or "*reflete em silêncio*"
            self.exibir_e_falar_resposta(resposta)
        except Exception as e:
            self.exibir_e_falar_resposta(f"Erro na matriz neural: {str(e)}")

    def exibir_e_falar_resposta(self, resposta):
        self.append_message("Keisy", resposta, "Keisy")
        if self.voz_ativa:
            self.atualizar_status("🗣️ Keisy falando...")
        system.speak(resposta, voz_ativa=self.voz_ativa, callback_fim=lambda: self.finalizar_turno())

    def finalizar_turno(self):
        self.atualizar_status("🟢 Keisy está Pronta")
        self.root.after(0, lambda: self.send_btn.config(state=tk.NORMAL))

# ==============================================================================
# 7. INICIALIZAÇÃO DO PROGRAMA
# ==============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = KeisyApp(root)
    root.mainloop()
