"""O Fine-tuning demonstrativo de Tipos de veículo: dataset, curvas, prova e os seis veículos do antes e depois.

O treino em si (`modelo.train(..., freeze=11)`) fica à vista no notebook: é ele que
ilustra a teoria. Aqui fica o que só atrapalharia a leitura: descompactar o dataset,
desenhar as curvas e as grades, traduzir as métricas e recortar os seis veículos.
"""

import csv
import logging
import time
import warnings
import zipfile
from pathlib import Path

import cv2
import numpy as np

from . import desenho, estilo
from .ambiente import PASTA_SAIDAS
from .dados import baixar

ZIP = "dataset_tipos_de_veiculo.zip"
PASTA_DATASET = PASTA_SAIDAS / "tipos_de_veiculo"  # fora de dados/, que no Modo local é a pasta pública do Drive
PASTA_TREINOS = PASTA_SAIDAS / "treinos"
ESPECIALISTA = "yolo26n_tipos_de_veiculo.pt"
CURVAS_DO_ESPECIALISTA = "yolo26n_tipos_de_veiculo_treino.csv"
VIDEO = "bellevue-ne8th_2762-2802s.mp4"

DIVISOES = {"train": "treino", "val": "validação", "test": "teste"}


# Dataset ---------------------------------------------------------------------
def preparar_dataset():
    """Descompacta o dataset (uma vez só) e devolve o caminho do data.yaml, com o caminho absoluto."""
    yaml = PASTA_DATASET / "data.yaml"
    if not (PASTA_DATASET / "images" / "test").is_dir():
        with zipfile.ZipFile(baixar(ZIP)) as z:
            z.extractall(PASTA_DATASET.parent)
    texto = yaml.read_text(encoding="utf-8")
    linhas = [f"path: {PASTA_DATASET}" if l.startswith("path:") else l for l in texto.splitlines()]
    yaml.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return yaml


def classes_do_dataset(yaml):
    """{id: nome} lido do data.yaml."""
    nomes, dentro = {}, False
    for linha in Path(yaml).read_text(encoding="utf-8").splitlines():
        if linha.startswith("names:"):
            dentro = True
        elif dentro and ":" in linha and linha.startswith(" "):
            indice, nome = linha.split(":", 1)
            nomes[int(indice)] = nome.strip()
        elif dentro:
            break
    return nomes


def _rotulos(imagem):
    """[(classe, cx, cy, l, a)] do .txt YOLO de uma imagem do dataset."""
    txt = Path(str(imagem).replace("/images/", "/labels/")).with_suffix(".txt")
    if not txt.exists():
        return []
    return [(int(p[0]), *map(float, p[1:5])) for p in (l.split() for l in txt.read_text().splitlines()) if p]


def contar(yaml):
    """Imprime quadros e caixas por classe em cada divisão do dataset."""
    nomes = classes_do_dataset(yaml)
    cabecalho = ["Divisão", "Quadros", *[n.capitalize() for n in nomes.values()]]
    linhas = []
    for divisao, rotulo in DIVISOES.items():
        imagens = sorted((PASTA_DATASET / "images" / divisao).glob("*.jpg"))
        caixas = [0] * len(nomes)
        for imagem in imagens:
            for classe, *_ in _rotulos(imagem):
                caixas[classe] += 1
        linhas.append([rotulo.capitalize(), str(len(imagens)), *map(str, caixas)])
    larguras = [max(len(x) for x in coluna) for coluna in zip(cabecalho, *linhas)]
    formatar = lambda c: "  ".join(x.ljust(w) if k == 0 else x.rjust(w) for k, (x, w) in enumerate(zip(c, larguras)))  # noqa: E731
    print(formatar(cabecalho))
    print("  ".join("─" * w for w in larguras))
    for linha in linhas:
        print(formatar(linha))


def _mosaico(imagens, colunas=2, largura=1600):
    """Junta imagens BGR numa grade, cada uma redimensionada para a mesma largura."""
    lado = largura // colunas
    celulas = [cv2.resize(i, (lado, round(i.shape[0] * lado / i.shape[1])), interpolation=cv2.INTER_AREA) for i in imagens]
    altura = max(c.shape[0] for c in celulas)
    celulas = [np.pad(c, ((0, altura - c.shape[0]), (0, 0), (0, 0)), constant_values=252) for c in celulas]
    celulas += [np.full_like(celulas[0], 252)] * (-len(celulas) % colunas)
    margem = 8
    linhas = [
        np.hstack([np.pad(c, ((margem, margem), (margem, margem), (0, 0)), constant_values=252) for c in celulas[k : k + colunas]])
        for k in range(0, len(celulas), colunas)
    ]
    return np.vstack(linhas)


def _desenhar_gabarito(imagem, rotulos, nomes):
    import supervision as sv

    altura, largura = imagem.shape[:2]
    if not rotulos:
        return imagem.copy()
    xyxy = np.array([[(cx - l / 2) * largura, (cy - a / 2) * altura, (cx + l / 2) * largura, (cy + a / 2) * altura]
                     for _, cx, cy, l, a in rotulos])
    det = sv.Detections(xyxy=xyxy, class_id=np.array([c for c, *_ in rotulos]))
    return desenho.desenhar_caixas(imagem, det, nomes=nomes, rotulos=[nomes[c] for c, *_ in rotulos])


def foto_do_dataset(yaml, divisao="treino", numero=0):
    """(foto, arquivo de anotação, foto com as caixas desenhadas) de uma foto do dataset.

    `divisao` é treino, validação ou teste (ou train, val, test); `numero` dá a volta
    se passar do total da divisão.
    """
    pasta = {v: k for k, v in DIVISOES.items()}.get(divisao, divisao)
    imagens = sorted((PASTA_DATASET / "images" / pasta).glob("*.jpg"))
    foto = imagens[int(numero) % len(imagens)]
    anotacao = Path(str(foto).replace("/images/", "/labels/")).with_suffix(".txt")
    desenhada = _desenhar_gabarito(cv2.imread(str(foto)), _rotulos(foto), classes_do_dataset(yaml))
    return foto, anotacao, desenhada


# Treino ----------------------------------------------------------------------
def pasta_do_treino(nome="ao_vivo"):
    return PASTA_TREINOS / nome


def ler_curvas(origem):
    """Lê o results.csv da Ultralytics (de uma pasta de treino ou de um .csv) como {coluna: [valores]}."""
    origem = Path(origem)
    arquivo = origem / "results.csv" if origem.is_dir() else origem
    with arquivo.open(encoding="utf-8") as f:
        linhas = [{k.strip(): float(v) for k, v in linha.items() if v.strip()} for linha in csv.DictReader(f)]
    return {k: [l.get(k, np.nan) for l in linhas] for k in linhas[0]} if linhas else {}


def curvas(*origens, rotulos=None):
    """Erro × época e mAP50 × época, para um ou mais treinos (pasta do treino ou .csv).

    O erro mostrado é o de validação (caixa + classe), medido em fotos que o
    modelo não usa para aprender. Quando ele para de cair, o treino chegou ao vale.
    """
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    caminhos = [estilo.caminho_fonte(), estilo.caminho_fonte(negrito=True)]
    if all(c and "Raleway" in c for c in caminhos):
        for c in caminhos:
            font_manager.fontManager.addfont(c)
        plt.rcParams["font.family"] = "Raleway"
    rotulos = rotulos or [f"treino {k + 1}" for k in range(len(origens))]
    cores = [estilo.AZUL, estilo.VERMELHO, estilo.VERDE]
    figura, (erro, acerto) = plt.subplots(1, 2, figsize=(12, 4), facecolor=estilo.OFF_WHITE)
    for k, (origem, rotulo) in enumerate(zip(origens, rotulos)):
        c = ler_curvas(origem)
        if not c:
            continue
        epocas = c["epoch"]
        perda = np.add(c["val/box_loss"], c["val/cls_loss"])
        estilo_linha = {"color": cores[k % len(cores)], "lw": 2.5, "label": rotulo, "marker": "o" if len(epocas) <= 15 else None}
        erro.plot(epocas, perda, **estilo_linha)
        acerto.plot(epocas, c["metrics/mAP50(B)"], **estilo_linha)
    for eixo, titulo in ((erro, "Erro (loss) na validação"), (acerto, "Acerto (mAP50) na validação")):
        eixo.set_facecolor(estilo.OFF_WHITE)
        eixo.set_title(titulo, color=estilo.NAVY, fontsize=14, fontweight="bold", loc="left")
        eixo.set_xlabel("Época (epoch)", color=estilo.CINZA)
        eixo.grid(alpha=0.25)
        for lado in ("top", "right"):
            eixo.spines[lado].set_visible(False)
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    virgula = FuncFormatter(lambda x, _: f"{x:g}".replace(".", ","))
    for eixo in (erro, acerto):
        eixo.xaxis.set_major_locator(MaxNLocator(integer=True))
        eixo.yaxis.set_major_formatter(virgula)
    erro.set_ylim(bottom=0)
    acerto.set_ylim(0, 1)
    if len(origens) > 1:
        acerto.legend(frameon=False)
    plt.tight_layout()
    plt.show()


class _Silencio:
    """Esconde o log da Ultralytics (resumo do modelo, camadas congeladas, barras) dentro do bloco."""

    def __enter__(self):
        from ultralytics.utils import LOGGER

        self.logger, self.nivel = LOGGER, LOGGER.level
        LOGGER.setLevel(logging.WARNING)
        return self

    def __exit__(self, *erro):
        self.logger.setLevel(self.nivel)


class acompanhar(_Silencio):
    """`with acompanhar(modelo): modelo.train(...)` mostra uma linha por época, em português.

    O log da Ultralytics tem centenas de linhas (uma por camada congelada); aqui ficam
    só o erro e o acerto na validação a cada época, e o tempo total no fim.
    """

    def __init__(self, modelo):
        self.modelo = modelo

    def _epoca(self, treinador):
        # a validação final chama o callback de novo, com o contador já depois da última época
        if treinador.epoch + 1 in self.mostradas or treinador.epoch + 1 > treinador.epochs:
            return
        self.mostradas.add(treinador.epoch + 1)
        m = treinador.metrics or {}
        erro = m.get("val/box_loss", 0) + m.get("val/cls_loss", 0)
        virgula = lambda x: f"{x:.2f}".replace(".", ",")  # noqa: E731
        print(f"Época {treinador.epoch + 1}/{treinador.epochs}   erro {virgula(erro)}   "
              f"acerto (mAP50) {virgula(m.get('metrics/mAP50(B)', 0))}   ({self._tempo()})", flush=True)

    def _tempo(self):
        segundos = time.time() - self.inicio
        return f"{int(segundos // 60)} min {int(segundos % 60):02d} s"

    def __enter__(self):
        super().__enter__()
        self.inicio, self.mostradas = time.time(), set()
        warnings.filterwarnings("ignore", message=".*does not have a deterministic implementation.*")  # mps
        self.modelo.add_callback("on_fit_epoch_end", self._epoca)
        print("Preparando o treino (o primeiro resultado aparece no fim da primeira época)...", flush=True)
        return self

    def __exit__(self, *erro):
        super().__exit__(*erro)
        self.modelo.callbacks["on_fit_epoch_end"] = [
            c for c in self.modelo.callbacks.get("on_fit_epoch_end", []) if c != self._epoca
        ]
        if erro[0] is None:
            print(f"\nTreino concluído em {self._tempo()}.")


# O modelo treinado antes -----------------------------------------------------------
def especialista_pronto():
    """Baixa o Modelo especialista treinado antes da Oficina e devolve (modelo, caminho das curvas)."""
    from ultralytics import YOLO

    return YOLO(str(baixar(ESPECIALISTA))), baixar(CURVAS_DO_ESPECIALISTA)


# Prova -------------------------------------------------------------------------------
def provar(modelo, yaml, divisao="test", imgsz=960, dispositivo=None):
    """Avalia o modelo nas fotos de teste e imprime precisão, revocação e mAP50 por classe.

    Devolve o objeto de métricas da Ultralytics.
    """
    argumentos = {"data": str(yaml), "split": divisao, "imgsz": imgsz, "plots": False, "verbose": False,
                  "project": str(PASTA_TREINOS), "name": f"prova_{divisao}", "exist_ok": True}
    if dispositivo:
        argumentos["device"] = dispositivo
    with _Silencio():
        r = modelo.val(**argumentos)
    caixa = r.box
    nomes = modelo.names
    porcento = lambda x: f"{100 * x:.0f}%"  # noqa: E731
    linhas = [("Todas", porcento(caixa.mp), porcento(caixa.mr), porcento(caixa.map50))]
    for k, c in enumerate(caixa.ap_class_index):
        p, rev, m50, _ = caixa.class_result(k)
        linhas.append((nomes[int(c)].capitalize(), porcento(p), porcento(rev), porcento(m50)))
    cabecalho = ("Classe", "Precisão", "Revocação", "mAP50")
    larguras = [max(len(x) for x in coluna) for coluna in zip(cabecalho, *linhas)]
    formatar = lambda c: "  ".join(x.ljust(w) if k == 0 else x.rjust(w) for k, (x, w) in enumerate(zip(c, larguras)))  # noqa: E731
    print(f"Prova em {DIVISOES.get(divisao, divisao)}: fotos que o modelo nunca viu no treino\n")
    print(formatar(cabecalho))
    print("  ".join("─" * w for w in larguras))
    print(formatar(linhas[0]))
    print("  ".join("─" * w for w in larguras))
    for linha in linhas[1:]:
        print(formatar(linha))
    return r


def grade_de_teste(modelo, yaml, n=3, conf=0.25, imgsz=960, semente=11):
    """Lado a lado, em `n` fotos de teste: o gabarito (esquerda) e o que o modelo respondeu (direita)."""
    nomes = classes_do_dataset(yaml)
    imagens = sorted((PASTA_DATASET / "images" / "test").glob("*.jpg"))
    raras = [i for i in imagens if len({c for c, *_ in _rotulos(i)} - {0}) >= 2]
    escolhidas = np.random.default_rng(semente).choice(raras or imagens, size=min(n, len(raras or imagens)), replace=False)
    pares = []
    for imagem in escolhidas:
        quadro = cv2.imread(str(imagem))
        pares.append(_titulo(_desenhar_gabarito(quadro, _rotulos(imagem), nomes), "Gabarito"))
        resposta = modelo(quadro, conf=conf, imgsz=imgsz, verbose=False)
        pares.append(_titulo(desenho.desenhar_caixas(quadro, resposta), "Modelo"))
    desenho.mostrar(_mosaico(pares, colunas=2), largura=1100)


def _titulo(imagem, texto, altura=None):
    """Faixa azul com o texto em cima da imagem."""
    from PIL import Image, ImageDraw

    altura = altura or max(36, imagem.shape[0] // 14)
    faixa = Image.new("RGB", (imagem.shape[1], altura), estilo.hex_para_rgb(estilo.AZUL))
    ImageDraw.Draw(faixa).text((altura // 3, altura // 6), texto, font=estilo.fonte(int(altura * 0.6), negrito=True),
                               fill=estilo.hex_para_rgb(estilo.OFF_WHITE))
    return np.vstack([cv2.cvtColor(np.asarray(faixa), cv2.COLOR_RGB2BGR), imagem])


# Seis veículos (os recortes do antigo desafio "Que veículo é esse?") -------------------------------------------------------
# (rótulo, quadro do trecho de 40 s, caixa no quadro 1280×720, resposta certa). A Cena-âncora
# (quadro 623) não tem Pickup nem Caminhão: esses dois vêm de outros quadros dos mesmos 40 s.
# O Ônibus (F) é opcional: o pré-treinado acerta, e isso deixa o placar honesto.
DESAFIO = [
    ("A", 180, (226, 286, 316, 337), "pickup"),
    ("B", 623, (975, 310, 1056, 362), "carro"),  # o SUV: conta como Carro
    ("C", 623, (146, 483, 299, 567), "van"),
    ("D", 300, (815, 258, 961, 339), "caminhão"),  # o caminhão-plataforma, Norte → Leste
    ("E", 623, (792, 304, 911, 385), "carro"),
    ("F", 623, (555, 196, 661, 416), "ônibus"),
]


def quadros_do_video(numeros, video=None):
    """{número do quadro: imagem BGR} do trecho de 40 s, lendo o vídeo em ordem."""
    captura = cv2.VideoCapture(str(video or baixar(VIDEO)))
    quadros, numero = {}, 0
    try:
        while len(quadros) < len(set(numeros)):
            ok, quadro = captura.read()
            if not ok:
                break
            if numero in numeros:
                quadros[numero] = quadro
            numero += 1
    finally:
        captura.release()
    return quadros


def resposta(modelo, quadro, caixa, imgsz=960):
    """(classe, confiança) da detecção do modelo que mais se sobrepõe à caixa, no quadro inteiro, como no Case."""
    r = modelo(quadro, imgsz=imgsz, conf=0.1, verbose=False)[0]
    melhor, achado = 0.0, ("nada", None)
    for b, c, p in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
        x1, y1 = max(b[0], caixa[0]), max(b[1], caixa[1])
        x2, y2 = min(b[2], caixa[2]), min(b[3], caixa[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        uniao = (b[2] - b[0]) * (b[3] - b[1]) + (caixa[2] - caixa[0]) * (caixa[3] - caixa[1]) - inter
        iou = inter / uniao
        if iou > 0.5 and iou * p > melhor:
            melhor, achado = iou * p, (modelo.names[int(c)], p)
    return achado


def _cartao(recorte, largura=400, altura=300):
    """Recorte ampliado para caber em largura × altura, com margem off-white (todos os cartões do mesmo tamanho)."""
    escala = min(largura / recorte.shape[1], altura / recorte.shape[0])
    recorte = cv2.resize(recorte, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC)
    cartao = np.full((altura, largura, 3), estilo.hex_para_bgr(estilo.OFF_WHITE), np.uint8)
    y, x = (altura - recorte.shape[0]) // 2, (largura - recorte.shape[1]) // 2
    cartao[y : y + recorte.shape[0], x : x + recorte.shape[1]] = recorte
    return cartao


def _acertou(nome, certa):
    return estilo.nome_pt(nome) == certa or nome == certa


def seis_veiculos(modelos):
    """Seis veículos dos 40 s do Case: os recortes, a resposta de cada modelo e os acertos.

    `modelos` é {nome: modelo YOLO}, ex.: {"Pré-treinado": modelo, "Especialista": especialista}.
    São os recortes que eram do desafio "Que veículo é esse?".
    """
    quadros = quadros_do_video({n for _, n, _, _ in DESAFIO})
    placar = {nome: 0 for nome in modelos}
    cartoes = []
    linhas = []
    for rotulo, n, caixa, certa in DESAFIO:
        x1, y1, x2, y2 = caixa
        m = int(0.25 * max(x2 - x1, y2 - y1))
        recorte = quadros[n][max(0, y1 - m) : y2 + m, max(0, x1 - m) : x2 + m]
        recorte = _cartao(recorte)
        respostas = []
        for nome, modelo in modelos.items():
            classe, confianca = resposta(modelo, quadros[n], caixa)
            ok = _acertou(classe, certa)
            placar[nome] += ok
            texto = classe if confianca is None else f"{classe} {confianca:.2f}".replace(".", ",")
            respostas.append(f"{'✅' if ok else '❌'} {texto}")
        linhas.append((rotulo, certa, *respostas))
        cartoes.append(_titulo(recorte, f"{rotulo}  {certa}", altura=44))
    desenho.mostrar(_mosaico(cartoes, colunas=3, largura=1200), largura=900)
    cabecalho = ("", "Resposta certa", *modelos)
    larguras = [max(len(str(x)) for x in coluna) for coluna in zip(cabecalho, *linhas)]
    formatar = lambda c: "  ".join(str(x).ljust(w) for x, w in zip(c, larguras))  # noqa: E731
    print(formatar(cabecalho))
    print("  ".join("─" * w for w in larguras))
    for linha in linhas:
        print(formatar(linha))
    print("\nAcertos: " + " × ".join(f"{nome} {pontos}/{len(DESAFIO)}" for nome, pontos in placar.items()))
    return placar

