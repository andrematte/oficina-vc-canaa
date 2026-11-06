"""Desenho sobre os quadros (caixas, Linha virtual, Zona, Rastros) e exibição no notebook."""

import base64
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from PIL import Image, ImageDraw

from . import estilo

ESPESSURA_CAIXA = 3  # ~2,25 pt num quadro 1080p


# Classes e cores -----------------------------------------------------------
def nomes_das_classes(origem):
    """{id: nome} a partir de um modelo YOLO, de um resultado do YOLO ou de um dict."""
    if isinstance(origem, dict):
        return dict(origem)
    nomes = getattr(origem, "names", None)
    if nomes is None and isinstance(origem, (list, tuple)) and origem:
        nomes = getattr(origem[0], "names", None)
    return dict(nomes) if nomes else {0: "pessoa"}


def paletas(nomes):
    """Paletas do supervision (caixa e texto) indexadas pelo id da classe."""
    maior = max(nomes) if nomes else 0
    caixas, textos = [], []
    for indice in range(maior + 1):
        cor, cor_texto = estilo.cores_da_classe(nomes.get(indice, ""), indice)
        caixas.append(sv.Color.from_hex(cor))
        textos.append(sv.Color.from_hex(cor_texto))
    return sv.ColorPalette(caixas), sv.ColorPalette(textos)


def _para_deteccoes(deteccoes):
    """Aceita sv.Detections ou o que `modelo(quadro)` devolve."""
    if isinstance(deteccoes, sv.Detections):
        return deteccoes
    if isinstance(deteccoes, (list, tuple)):
        deteccoes = deteccoes[0]
    return sv.Detections.from_ultralytics(deteccoes)


# Caixas --------------------------------------------------------------------
def desenhar_caixas(imagem, deteccoes, nomes=None, rotulos=None):
    """Devolve uma cópia de `imagem` (BGR) com as caixas e etiquetas na cor de cada classe.

    `deteccoes` pode ser o resultado de `modelo(quadro, ...)` ou um `sv.Detections`.
    Sem `rotulos`, cada etiqueta mostra a classe e a confiança (ex.: "pessoa 0,87").
    """
    if nomes is None:
        nomes = nomes_das_classes(deteccoes)
    det = _para_deteccoes(deteccoes)
    paleta, paleta_texto = paletas(nomes)
    if rotulos is None:
        rotulos = []
        for classe, confianca in zip(det.class_id, det.confidence if det.confidence is not None else [None] * len(det)):
            nome = estilo.nome_pt(nomes.get(int(classe), str(classe)))
            rotulos.append(nome if confianca is None else f"{nome} {confianca:.2f}".replace(".", ","))
    saida = sv.BoxAnnotator(color=paleta, thickness=ESPESSURA_CAIXA).annotate(imagem.copy(), det)
    rotulador = sv.RichLabelAnnotator(
        color=paleta,
        text_color=paleta_texto,
        font_path=estilo.caminho_fonte(negrito=True),
        font_size=20,
        text_padding=5,
        smart_position=True,
    )
    return rotulador.annotate(saida, det, labels=rotulos)


# Linha virtual e Zona ------------------------------------------------------
def _pontos_da_linha(linha):
    inicio, fim = linha.vector.start, linha.vector.end
    return (int(inicio.x), int(inicio.y)), (int(fim.x), int(fim.y))


def _etiqueta(texto, tamanho=26):
    """Etiqueta azul com texto off-white (RGBA)."""
    fonte = estilo.fonte(tamanho, negrito=True)
    x0, y0, x1, y1 = fonte.getbbox(texto)
    margem = tamanho // 3
    largura, altura = x1 - x0 + 2 * margem, y1 + 2 * margem
    imagem = Image.new("RGBA", (largura, altura), (0, 0, 0, 0))
    traco = ImageDraw.Draw(imagem)
    traco.rounded_rectangle((0, 0, largura - 1, altura - 1), radius=altura // 2, fill=estilo.hex_para_rgb(estilo.AZUL) + (235,))
    traco.text((margem - x0, margem), texto, font=fonte, fill=estilo.hex_para_rgb(estilo.OFF_WHITE) + (255,))
    return imagem


def camada_linha_e_zona(forma, linha=None, zona=None):
    """Pré-desenha Linha virtual e Zona numa camada (cor BGR + alfa 0–1) do tamanho do quadro.

    É desenhada uma vez só e aplicada em cada quadro do vídeo com `aplicar_camada`.
    """
    altura, largura = forma[:2]
    camada = Image.new("RGBA", (largura, altura), (0, 0, 0, 0))
    azul = estilo.hex_para_rgb(estilo.COR_LINHA_ZONA)

    if zona is not None:
        poligono = [tuple(map(int, p)) for p in zona.polygon]
        preenchimento = Image.new("RGBA", (largura, altura), (0, 0, 0, 0))
        ImageDraw.Draw(preenchimento).polygon(poligono, fill=azul + (int(255 * estilo.ALFA_ZONA),))
        camada = Image.alpha_composite(camada, preenchimento)
        ImageDraw.Draw(camada).line(poligono + [poligono[0]], fill=azul + (230,), width=4, joint="curve")
        etiqueta = _etiqueta("Zona")
        x = min(p[0] for p in poligono)
        y = min(p[1] for p in poligono) - etiqueta.height - 8
        camada.alpha_composite(etiqueta, (max(x, 0), max(y, 0)))

    if linha is not None:
        inicio, fim = _pontos_da_linha(linha)
        traco = ImageDraw.Draw(camada)
        traco.line([inicio, fim], fill=azul + (int(255 * estilo.ALFA_LINHA),), width=8)
        for ponto in (inicio, fim):
            traco.ellipse((ponto[0] - 9, ponto[1] - 9, ponto[0] + 9, ponto[1] + 9), fill=azul + (255,))
        etiqueta = _etiqueta("Linha virtual")
        x = min(max(min(inicio[0], fim[0]), 0), largura - etiqueta.width)
        y = min(inicio[1], fim[1]) - etiqueta.height - 14
        camada.alpha_composite(etiqueta, (x, max(y, 0)))

    rgba = np.asarray(camada)
    cor = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2BGR).astype(np.float32)
    alfa = rgba[..., 3:4].astype(np.float32) / 255.0
    linhas, colunas = np.nonzero(rgba[..., 3])
    caixa = (linhas.min(), linhas.max() + 1, colunas.min(), colunas.max() + 1) if len(linhas) else None
    return cor, alfa, caixa


def aplicar_camada(imagem, camada):
    """Aplica uma camada de `camada_linha_e_zona` sobre a imagem (no lugar) e a devolve."""
    cor, alfa, caixa = camada
    if caixa is None:
        return imagem
    y0, y1, x0, x1 = caixa
    recorte = imagem[y0:y1, x0:x1].astype(np.float32)
    a = alfa[y0:y1, x0:x1]
    imagem[y0:y1, x0:x1] = (recorte * (1 - a) + cor[y0:y1, x0:x1] * a).astype(np.uint8)
    return imagem


def desenhar_linha(imagem, linha):
    """Cópia de `imagem` com a Linha virtual (um sv.LineZone)."""
    return aplicar_camada(imagem.copy(), camada_linha_e_zona(imagem.shape, linha=linha))


def desenhar_zona(imagem, zona):
    """Cópia de `imagem` com a Zona (um sv.PolygonZone)."""
    return aplicar_camada(imagem.copy(), camada_linha_e_zona(imagem.shape, zona=zona))


def desenhar_linha_e_zona(imagem, linha, zona):
    """Cópia de `imagem` com a Linha virtual e a Zona."""
    return aplicar_camada(imagem.copy(), camada_linha_e_zona(imagem.shape, linha, zona))


# Exibição ------------------------------------------------------------------
def mostrar(imagem, largura=960):
    """Mostra uma imagem BGR (ou um caminho de imagem) no notebook."""
    if isinstance(imagem, (str, Path)):
        imagem = cv2.imread(str(imagem))
    from IPython.display import Image as ImagemIPython
    from IPython.display import display

    ok, jpeg = cv2.imencode(".jpg", imagem, [cv2.IMWRITE_JPEG_QUALITY, 90])
    display(ImagemIPython(data=jpeg.tobytes(), format="jpeg", width=largura))


def mostrar_video(video, largura=960):
    """Mostra um MP4 no notebook. Aceita um caminho ou um ResultadoCase."""
    from IPython.display import HTML, display

    caminho = Path(getattr(video, "video", video))
    dados = base64.b64encode(caminho.read_bytes()).decode("ascii")
    display(
        HTML(
            f'<video width="{largura}" controls loop muted playsinline>'
            f'<source src="data:video/mp4;base64,{dados}" type="video/mp4"></video>'
        )
    )
