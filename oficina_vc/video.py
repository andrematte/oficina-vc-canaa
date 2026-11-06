"""O laço quadro a quadro do Case: detectar, rastrear, contar, desenhar e gravar o vídeo de saída."""

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
from .painel import ALTURA as ALTURA_SAIDA
from .painel import LARGURA as LARGURA_PAINEL
from .painel import Painel

RASTRO_SEGUNDOS = 2
LARGURA_VIDEO = 1280  # o vídeo de 1080p entra em 1280×720; com o painel, a saída tem 1920×720

CREDITOS = {"mot20": "Vídeo: MOT20-02, MOTChallenge, CC BY-NC-SA 3.0"}


@dataclass
class ResultadoCase:
    """O que o processamento dos 40 s (ou o Resultado pronto) devolve."""

    video: Path
    metricas: dict = field(default_factory=dict)

    @property
    def entradas(self):
        return self.metricas.get("entradas", 0)

    @property
    def saidas(self):
        return self.metricas.get("saidas", 0)

    def resumo(self):
        """Imprime as Métricas de vídeo em português."""
        m = self.metricas
        virgula = lambda x: f"{x:.1f}".replace(".", ",")  # noqa: E731
        print(f"Vídeo analisado: {m.get('duracao_s', 0):.0f} s ({m.get('quadros', 0)} quadros)")
        print(f"Entradas: {self.entradas}")
        print(f"Saídas: {self.saidas}")
        print(f"Fluxo pela linha: {virgula(m.get('fluxo_por_minuto', 0))} pessoas por minuto")
        print(f"Ocupação da Zona: média {virgula(m.get('ocupacao_media', 0))}, máxima {m.get('ocupacao_maxima', 0)}")
        print(f"Pessoas no quadro: média {virgula(m.get('pessoas_media', 0))}, máxima {m.get('pessoas_maxima', 0)}")
        print(f"IDs diferentes atribuídos pelo rastreador: {m.get('ids_unicos', 0)}")
        if len(m.get("classes", {})) > 1:
            print("Passagens pela linha, por classe:")
            for nome, n in m.get("passagens_por_classe", {}).items():
                print(f"  {nome}: {n}")

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
    linha=None,
    zona=None,
    confianca=0.25,
    classes=None,
    saida=None,
    dispositivo=None,
    limite_quadros=None,
):
    """Detecta, rastreia e conta quadro a quadro, e grava o vídeo de saída com o painel.

    - `modelo`: um YOLO da Ultralytics (pré-treinado ou ajustado).
    - `rastreador`: um ByteTrackTracker (do pacote `trackers`).
    - `linha`, `zona`: um sv.LineZone e um sv.PolygonZone (opcionais).
    - `confianca`, `classes`: repassados ao modelo (ex.: classes=[0] para só pessoas).
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
    fps = captura.get(cv2.CAP_PROP_FPS) or 25
    total = int(captura.get(cv2.CAP_PROP_FRAME_COUNT))
    if limite_quadros:
        total = min(total, int(limite_quadros))
    largura = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    altura_video = ALTURA_SAIDA
    largura_video = round(largura * altura_video / altura / 2) * 2

    # Cópias: rodar a célula de novo não soma contagens à linha e à zona do notebook.
    linha = copy.deepcopy(linha)
    zona = copy.deepcopy(zona)
    if hasattr(rastreador, "reset"):
        rastreador.reset()

    nomes = desenho.nomes_das_classes(modelo)
    ids_contados = sorted(classes) if classes is not None else sorted(nomes)
    nomes_contados = [estilo.nome_pt(nomes[i]) for i in ids_contados]
    paleta, paleta_texto = desenho.paletas(nomes)
    caixas = sv.BoxAnnotator(color=paleta, thickness=desenho.ESPESSURA_CAIXA)
    rotulos = sv.LabelAnnotator(
        color=paleta, text_color=paleta_texto, text_scale=0.6, text_thickness=2, text_padding=4
    )
    rastros = sv.TraceAnnotator(
        color=paleta,
        position=sv.Position.BOTTOM_CENTER,
        trace_length=int(RASTRO_SEGUNDOS * fps),
        thickness=2,
    )
    camada = desenho.camada_linha_e_zona((altura, largura), linha, zona)
    credito = next((c for chave, c in CREDITOS.items() if chave in video.name.lower()), None)
    painel = Painel(duracao=total / fps, quadros=total, classes=nomes_contados, credito=credito)

    saida = Path(saida) if saida else PASTA_SAIDAS / f"{video.stem}_saida.mp4"
    saida.parent.mkdir(parents=True, exist_ok=True)
    gravador = GravadorH264(saida, largura_video + LARGURA_PAINEL, altura_video, fps)

    serie_zona, serie_quadro, ids = [], [], set()
    pulsos = [None, None]
    entradas_antes = saidas_antes = 0
    argumentos = {"conf": confianca, "verbose": False}
    if classes is not None:
        argumentos["classes"] = list(classes)
    if dispositivo:
        argumentos["device"] = dispositivo

    try:
        for indice in tqdm(range(total), desc="Processando", unit="quadro"):
            ok, quadro = captura.read()
            if not ok:
                break
            deteccoes = sv.Detections.from_ultralytics(modelo(quadro, **argumentos)[0])
            deteccoes = _atualizar_rastreador(rastreador, deteccoes)
            deteccoes = deteccoes[deteccoes.tracker_id >= 0]  # só rastros confirmados
            ids.update(int(i) for i in deteccoes.tracker_id)

            entradas = saidas = 0
            if linha is not None:
                linha.trigger(deteccoes)
                entradas, saidas = linha.in_count, linha.out_count
            na_zona = int(zona.trigger(deteccoes).sum()) if zona is not None else 0
            serie_zona.append(na_zona)
            serie_quadro.append(len(deteccoes))
            if entradas > entradas_antes:
                pulsos[0] = indice
            if saidas > saidas_antes:
                pulsos[1] = indice
            entradas_antes, saidas_antes = entradas, saidas

            desenho.aplicar_camada(quadro, camada)
            quadro = rastros.annotate(quadro, deteccoes)
            quadro = caixas.annotate(quadro, deteccoes)
            quadro = rotulos.annotate(quadro, deteccoes, labels=[f"#{i}" for i in deteccoes.tracker_id])
            quadro = cv2.resize(quadro, (largura_video, altura_video), interpolation=cv2.INTER_AREA)

            imagem_painel = painel.desenhar(
                t=(indice + 1) / fps,
                entradas=entradas,
                saidas=saidas,
                na_zona=na_zona,
                no_quadro=len(deteccoes),
                serie_zona=serie_zona,
                passagens_por_classe=_passagens_por_classe(linha, nomes),
                pulsos=pulsos,
                quadro=indice,
            )
            gravador.gravar(np.hstack([quadro, imagem_painel]))
    finally:
        captura.release()
        gravador.fechar()

    n = len(serie_quadro)
    duracao = n / fps
    entradas = linha.in_count if linha is not None else 0
    saidas = linha.out_count if linha is not None else 0
    metricas = {
        "video_de_entrada": video.name,
        "fps": fps,
        "quadros": n,
        "duracao_s": round(duracao, 3),
        "confianca": confianca,
        "classes": {str(i): nomes_contados[k] for k, i in enumerate(ids_contados)},
        "entradas": entradas,
        "saidas": saidas,
        "fluxo_por_minuto": round((entradas + saidas) / duracao * 60, 2) if duracao else 0,
        "passagens_por_classe": _passagens_por_classe(linha, nomes),
        "ocupacao_media": round(float(np.mean(serie_zona)), 2) if n else 0,
        "ocupacao_maxima": int(max(serie_zona, default=0)),
        "pessoas_media": round(float(np.mean(serie_quadro)), 2) if n else 0,
        "pessoas_maxima": int(max(serie_quadro, default=0)),
        "ids_unicos": len(ids),
        "ocupacao_por_quadro": serie_zona,
        "pessoas_por_quadro": serie_quadro,
    }
    resultado = ResultadoCase(video=saida, metricas=metricas)
    resultado.salvar_metricas()
    return resultado


def _passagens_por_classe(linha, nomes):
    """{nome da classe: entradas + saídas} pela Linha virtual."""
    if linha is None:
        return {}
    passagens = {}
    for contagem in (linha.in_count_per_class, linha.out_count_per_class):
        for classe, n in contagem.items():
            nome = estilo.nome_pt(nomes.get(int(classe), str(classe)))
            passagens[nome] = passagens.get(nome, 0) + int(n)
    return passagens
