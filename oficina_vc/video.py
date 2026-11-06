"""O laço quadro a quadro do Case: detectar, rastrear, contar os Movimentos, desenhar e gravar o vídeo de saída."""

import copy
import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from . import desenho, estilo
from .ambiente import PASTA_SAIDAS
from .movimentos import ContadorDeMovimentos, sigla, tipo_de_movimento
from .painel import ALTURA as ALTURA_SAIDA
from .painel import LARGURA as LARGURA_PAINEL
from .painel import Painel

RASTRO_SEGUNDOS = 2
SUAVIZAR_SEGUNDOS = 0.64  # as Zonas usam a posição média dos pés nesse intervalo
LARGURA_VIDEO = 1280  # o vídeo entra em 1280×720; com o painel, a saída tem 1920×720

CREDITOS = {"bellevue": "Vídeo: City of Bellevue, Bellevue Traffic Video Dataset (2017)"}


@dataclass
class ResultadoCase:
    """O que o processamento dos 40 s (ou o Resultado pronto) devolve."""

    video: Path
    metricas: dict = field(default_factory=dict)

    @property
    def movimentos(self):
        """Total de Movimentos contados."""
        return self.metricas.get("movimentos", 0)

    @property
    def matriz(self):
        """{origem: {destino: n}}, todas as classes juntas."""
        return self.metricas.get("matriz", {})

    def resumo(self):
        """Imprime as Métricas de vídeo em português, com a Matriz de movimentos por classe."""
        m = self.metricas
        virgula = lambda x: f"{x:.1f}".replace(".", ",")  # noqa: E731
        print(f"Vídeo analisado: {m.get('duracao_s', 0):.0f} s ({m.get('quadros', 0)} quadros)")
        print(f"Movimentos: {self.movimentos}")
        print(f"Fluxo: {virgula(m.get('fluxo_por_minuto', 0))} veículos por minuto")
        print(f"Veículos no quadro: média {virgula(m.get('veiculos_media', 0))}, máxima {m.get('veiculos_maxima', 0)}")
        print(f"IDs diferentes atribuídos pelo rastreador: {m.get('ids_unicos', 0)}")
        print("\nPor movimento: " + ", ".join(f"{t} {n}" for t, n in m.get("por_tipo", {}).items()))
        print()
        print(self.tabela())

    def tabela(self):
        """A Matriz de movimentos como texto: uma linha por par origem → destino, uma coluna por classe."""
        classes = list(self.metricas.get("por_classe", {}))
        por_classe = self.metricas.get("matriz_por_classe", {})
        linhas = []
        for origem, destinos in self.matriz.items():
            for destino, n in destinos.items():
                if n:
                    contagens = [por_classe.get(c, {}).get(origem, {}).get(destino, 0) for c in classes]
                    linhas.append((f"{origem} → {destino}", tipo_de_movimento(origem, destino), contagens, n))
        cabecalho = ["Origem → Destino", "Movimento", *[c.capitalize() for c in classes], "Total"]
        totais = [sum(linha[2][k] for linha in linhas) for k in range(len(classes))]
        corpo = [[a, b, *map(str, c), str(n)] for a, b, c, n in linhas]
        corpo.append(["Total", "", *map(str, totais), str(self.movimentos)])
        larguras = [max(len(str(x)) for x in coluna) for coluna in zip(cabecalho, *corpo)]

        def formatar(celulas):
            return "  ".join(
                str(c).ljust(w) if k < 2 else str(c).rjust(w) for k, (c, w) in enumerate(zip(celulas, larguras))
            )

        regua = "  ".join("─" * w for w in larguras)
        return "\n".join([formatar(cabecalho), regua, *map(formatar, corpo[:-1]), regua, formatar(corpo[-1])])

    def salvar_metricas(self, caminho=None):
        caminho = Path(caminho or Path(self.video).with_suffix(".json"))
        caminho.write_text(json.dumps(self.metricas, ensure_ascii=False, indent=1), encoding="utf-8")
        return caminho

    @classmethod
    def carregar(cls, video, metricas_json):
        return cls(video=Path(video), metricas=json.loads(Path(metricas_json).read_text(encoding="utf-8")))


def resultado_pronto():
    """O Resultado pronto: o vídeo de saída e as métricas já processados antes da Oficina."""
    from .dados import baixar

    return ResultadoCase.carregar(baixar("case_resultado_pronto.mp4"), baixar("case_metricas.json"))


# Gravação em H.264 ---------------------------------------------------------
def _ffmpeg():
    caminho = shutil.which("ffmpeg")
    if caminho:
        return caminho
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as erro:
        raise RuntimeError("O ffmpeg não foi encontrado; ele é preciso para gravar o vídeo em H.264.") from erro


class GravadorH264:
    """Recebe quadros BGR e grava um MP4 H.264 que toca no navegador."""

    def __init__(self, caminho, largura, altura, fps):
        self.processo = subprocess.Popen(
            [
                _ffmpeg(), "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{largura}x{altura}", "-r", str(fps), "-i", "-",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(caminho),
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def gravar(self, quadro):
        self.processo.stdin.write(np.ascontiguousarray(quadro).tobytes())

    def fechar(self):
        self.processo.stdin.close()
        erro = self.processo.stderr.read().decode(errors="replace")
        if self.processo.wait() != 0:
            raise RuntimeError(f"O ffmpeg falhou ao gravar o vídeo:\n{erro}")


# O laço ----------------------------------------------------------------------
def _atualizar_rastreador(rastreador, deteccoes):
    if hasattr(rastreador, "update"):
        return rastreador.update(deteccoes)
    return rastreador.update_with_detections(deteccoes)  # sv.ByteTrack, de versões antigas


def processar_video(
    video,
    modelo,
    rastreador,
    zonas,
    confianca=0.25,
    classes=None,
    saida=None,
    dispositivo=None,
    limite_quadros=None,
    imgsz=None,
    passo=1,
):
    """Detecta, rastreia e conta os Movimentos quadro a quadro, e grava o vídeo de saída com o painel.

    - `modelo`: um YOLO da Ultralytics (pré-treinado ou ajustado).
    - `rastreador`: um ByteTrackTracker (do pacote `trackers`).
    - `zonas`: {nome da Aproximação: sv.PolygonZone}, como devolve `criar_zonas`.
    - `confianca`, `classes`: repassados ao modelo (ex.: classes=[2, 3, 5, 7] para
      car, motorcycle, bus e truck do COCO). Para rastrear, vale passar uma confiança
      baixa: o rastreador só abre rastros com caixas fortes e usa as fracas apenas
      para continuar um rastro que já existe.
    - `imgsz`: lado da imagem que entra no modelo (padrão do modelo: 640). Maior acha
      mais veículos pequenos ou encobertos, e custa mais tempo.
    - `passo`: analisa 1 quadro a cada `passo` (2 = metade do trabalho). O vídeo de
      saída fica com fps / passo quadros por segundo; o rastreador deve ser criado
      com essa mesma taxa (`frame_rate`).
    - `saida`: caminho do MP4 de saída (padrão: saidas/<nome do vídeo>_saida.mp4).
    - `limite_quadros`: processa só os primeiros N quadros (para testes rápidos).

    Devolve um ResultadoCase com o caminho do vídeo de saída e as Métricas de vídeo,
    também gravadas num .json ao lado do vídeo.
    """
    from tqdm.auto import tqdm

    video = Path(video)
    captura = cv2.VideoCapture(str(video))
    if not captura.isOpened():
        raise FileNotFoundError(f"Não consegui abrir o vídeo {video}.")
    fps_video = captura.get(cv2.CAP_PROP_FPS) or 30
    total = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
    if limite_quadros:
        total = min(total, int(limite_quadros))
    passo = max(int(passo), 1)
    fps = fps_video / passo  # taxa do vídeo de saída
    analisados = range(0, total, passo)
    largura = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    altura_video = ALTURA_SAIDA
    largura_video = round(largura * altura_video / altura / 2) * 2

    # Cópias: rodar a célula de novo não soma contagens às Zonas do notebook.
    contador = ContadorDeMovimentos(copy.deepcopy(zonas))
    if hasattr(rastreador, "reset"):
        rastreador.reset()
    # Média das últimas posições de cada ID: o pé que treme na borda de uma Zona não troca de Zona.
    suavizador = sv.DetectionsSmoother(length=max(round(SUAVIZAR_SEGUNDOS * fps), 1))

    nomes = desenho.nomes_das_classes(modelo)
    ids_contados = sorted(classes) if classes is not None else sorted(nomes)
    nomes_pt = {i: estilo.nome_pt(nomes[i]) for i in nomes}
    nomes_contados = [nomes_pt[i] for i in ids_contados]
    paleta, paleta_texto = desenho.paletas(nomes)
    caixas = sv.BoxAnnotator(color=paleta, thickness=desenho.ESPESSURA_CAIXA)
    rotulos = sv.LabelAnnotator(
        color=paleta, text_color=paleta_texto, text_scale=0.55, text_thickness=2, text_padding=4
    )
    rastros = sv.TraceAnnotator(
        color=paleta,
        position=sv.Position.BOTTOM_CENTER,
        trace_length=int(RASTRO_SEGUNDOS * fps),
        thickness=2,
    )
    camada = desenho.camada_zonas((altura, largura), zonas)
    credito = next((c for chave, c in CREDITOS.items() if chave in video.name.lower()), None)
    painel = Painel(
        duracao=total / fps_video, aproximacoes=list(zonas), classes=nomes_contados, credito=credito, fps=fps
    )

    saida = Path(saida) if saida else PASTA_SAIDAS / f"{video.stem}_saida.mp4"
    saida.parent.mkdir(parents=True, exist_ok=True)
    gravador = GravadorH264(saida, largura_video + LARGURA_PAINEL, altura_video, fps)

    serie_quadro, ids = [], set()
    votos = {}  # tracker_id → {classe: confiança somada}
    destinos = {}  # tracker_id → Aproximação de destino, para a etiqueta
    ultimo = None  # (índice do quadro, origem, destino) do último Movimento contado
    argumentos = {"conf": confianca, "verbose": False}
    if classes is not None:
        argumentos["classes"] = list(classes)
    if dispositivo:
        argumentos["device"] = dispositivo
    if imgsz:
        argumentos["imgsz"] = int(imgsz)

    try:
        for indice, numero in enumerate(tqdm(analisados, desc="Processando", unit="quadro")):
            for _ in range(passo - 1 if indice else 0):
                captura.grab()  # pula os quadros entre dois analisados, sem decodificar
            ok, quadro = captura.read()
            if not ok:
                break
            t = (numero + 1) / fps_video
            deteccoes = sv.Detections.from_ultralytics(modelo(quadro, **argumentos)[0])
            # Uma caixa por veículo: o modelo às vezes dá duas, uma para cada classe (car e truck).
            deteccoes = deteccoes.with_nms(threshold=0.6, class_agnostic=True)
            deteccoes = _atualizar_rastreador(rastreador, deteccoes)
            deteccoes = deteccoes[deteccoes.tracker_id >= 0]  # só rastros confirmados
            deteccoes = _votar_classe(deteccoes, votos)
            ids.update(int(i) for i in deteccoes.tracker_id)
            suavizadas = suavizador.update_with_detections(deteccoes)

            for movimento in contador.atualizar(suavizadas, t, nomes_pt):
                destinos[movimento.rastro] = movimento.destino
                ultimo = (indice, movimento.origem, movimento.destino)
            serie_quadro.append(len(deteccoes))

            desenho.aplicar_camada(quadro, camada)
            quadro = rastros.annotate(quadro, suavizadas)  # rastro e Zonas usam a mesma posição
            quadro = caixas.annotate(quadro, deteccoes)
            etiquetas = [_etiqueta(i, contador.de_onde(i), destinos.get(int(i))) for i in deteccoes.tracker_id]
            quadro = rotulos.annotate(quadro, deteccoes, labels=etiquetas)
            quadro = cv2.resize(quadro, (largura_video, altura_video), interpolation=cv2.INTER_AREA)

            imagem_painel = painel.desenhar(
                t=t,
                matriz=contador.matriz(),
                por_tipo=contador.por_tipo(),
                por_classe=contador.por_classe(nomes_contados),
                ultimo=ultimo,
                quadro=indice,
            )
            gravador.gravar(np.hstack([quadro, imagem_painel]))
    finally:
        captura.release()
        gravador.fechar()

    n = len(serie_quadro)
    duracao = n / fps
    total_movimentos = len(contador.movimentos)
    metricas = {
        "video_de_entrada": video.name,
        "fps": fps,
        "quadros": n,
        "passo": passo,
        "imgsz": int(imgsz) if imgsz else None,
        "duracao_s": round(duracao, 3),
        "confianca": confianca,
        "classes": {str(i): nomes_contados[k] for k, i in enumerate(ids_contados)},
        "zonas": {nome: [list(map(int, p)) for p in zona.polygon] for nome, zona in zonas.items()},
        "movimentos": total_movimentos,
        "fluxo_por_minuto": round(total_movimentos / duracao * 60, 2) if duracao else 0,
        "matriz": contador.matriz(),
        "matriz_por_classe": {c: contador.matriz(c) for c in nomes_contados},
        "por_tipo": contador.por_tipo(),
        "por_classe": contador.por_classe(nomes_contados),
        "veiculos_media": round(float(np.mean(serie_quadro)), 2) if n else 0,
        "veiculos_maxima": int(max(serie_quadro, default=0)),
        "ids_unicos": len(ids),
        "lista_de_movimentos": [
            {"id": m.rastro, "origem": m.origem, "destino": m.destino, "movimento": m.tipo, "classe": m.classe, "t": m.tempo}
            for m in contador.movimentos
        ],
        "veiculos_por_quadro": serie_quadro,
    }
    resultado = ResultadoCase(video=saida, metricas=metricas)
    resultado.salvar_metricas()
    return resultado


def _etiqueta(rastro, origem, destino):
    """'#12', '#12 N→' (já tem origem) ou '#12 N→L' (Movimento contado)."""
    if origem is None:
        return f"#{rastro}"
    return f"#{rastro} {sigla(origem)}→{sigla(destino) if destino else ''}"


def _votar_classe(deteccoes, votos):
    """Cada rastro fica com a classe que mais somou confiança até agora.

    Sem isso, o mesmo veículo pode ser Pickup num quadro e Carro no seguinte, e a
    Matriz por classe depende do quadro em que ele chega à Zona de destino.
    """
    if len(deteccoes) == 0:
        return deteccoes
    confiancas = deteccoes.confidence if deteccoes.confidence is not None else np.ones(len(deteccoes))
    classes = deteccoes.class_id.copy()
    for k, (rastro, classe, confianca) in enumerate(zip(deteccoes.tracker_id, deteccoes.class_id, confiancas)):
        soma = votos.setdefault(int(rastro), {})
        soma[int(classe)] = soma.get(int(classe), 0.0) + float(confianca)
        classes[k] = max(soma, key=soma.get)
    deteccoes.class_id = classes
    return deteccoes
