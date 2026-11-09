"""Desenho sobre os quadros (caixas, Zonas por Aproximação) e exibição no notebook."""

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
    return dict(nomes) if nomes else {}


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
    Sem `rotulos`, cada etiqueta mostra a classe e a confiança (ex.: "carro 0,87").
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


def tabela_de_caixas(deteccoes, n=5, nomes=None):
    """Escreve as `n` primeiras caixas como números: classe, confiança e os dois cantos."""
    if nomes is None:
        nomes = nomes_das_classes(deteccoes)
    det = _para_deteccoes(deteccoes)
    print(f"As {min(n, len(det))} primeiras caixas, em números:")
    print("classe      confiança     x1    y1    x2    y2")
    for (x1, y1, x2, y2), classe, confianca in list(zip(det.xyxy, det.class_id, det.confidence))[:n]:
        nome = estilo.nome_pt(nomes.get(int(classe), str(classe)))
        print(f"{nome:<10}  {confianca:9.2f}  {x1:5.0f} {y1:5.0f} {x2:5.0f} {y2:5.0f}".replace(".", ","))


# Zonas por Aproximação -------------------------------------------------------
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


def _vertices(zona):
    """Vértices de um sv.PolygonZone ou de uma lista de pontos (x, y)."""
    return [tuple(map(int, p)) for p in getattr(zona, "polygon", zona)]


def _lugar_da_etiqueta(poligono, tamanho, quadro):
    """Canto superior esquerdo da etiqueta de uma Zona.

    Na metade de cima do quadro, a etiqueta fica logo acima da Zona (sobre a calçada,
    onde não passa veículo e as caixas não a cobrem); na de baixo, no centro da Zona.
    """
    (w, h), (largura, altura) = tamanho, quadro
    topo = min(p[1] for p in poligono)
    if topo < altura / 2 and topo - h - 6 >= 0:
        de_cima = [p[0] for p in poligono if p[1] <= topo + 40]
        x, y = sum(de_cima) / len(de_cima) - w / 2, topo - h - 6
    else:
        x = sum(p[0] for p in poligono) / len(poligono) - w / 2
        y = sum(p[1] for p in poligono) / len(poligono) - h / 2
    return int(min(max(x, 0), largura - w)), int(min(max(y, 0), altura - h))


def camada_zonas(forma, zonas):
    """Pré-desenha as Zonas numa camada (cor BGR + alfa 0–1) do tamanho do quadro.

    `zonas` é {nome: sv.PolygonZone ou lista de pontos}. A camada é desenhada uma
    vez só e aplicada em cada quadro do vídeo com `aplicar_camada`.
    """
    altura, largura = forma[:2]
    camada = Image.new("RGBA", (largura, altura), (0, 0, 0, 0))
    azul = estilo.hex_para_rgb(estilo.COR_ZONA)
    poligonos = {nome: _vertices(zona) for nome, zona in (zonas or {}).items()}

    preenchimento = Image.new("RGBA", (largura, altura), (0, 0, 0, 0))
    for poligono in poligonos.values():
        ImageDraw.Draw(preenchimento).polygon(poligono, fill=azul + (int(255 * estilo.ALFA_ZONA),))
    camada = Image.alpha_composite(camada, preenchimento)
    traco = ImageDraw.Draw(camada)
    for poligono in poligonos.values():
        traco.line(poligono + [poligono[0]], fill=azul + (230,), width=4, joint="curve")
    for nome, poligono in poligonos.items():
        etiqueta = _etiqueta(nome)
        x, y = _lugar_da_etiqueta(poligono, etiqueta.size, (largura, altura))
        camada.alpha_composite(etiqueta, (x, y))

    rgba = np.asarray(camada)
    cor = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2BGR).astype(np.float32)
    alfa = rgba[..., 3:4].astype(np.float32) / 255.0
    linhas, colunas = np.nonzero(rgba[..., 3])
    caixa = (linhas.min(), linhas.max() + 1, colunas.min(), colunas.max() + 1) if len(linhas) else None
    return cor, alfa, caixa


def aplicar_camada(imagem, camada):
    """Aplica uma camada de `camada_zonas` sobre a imagem (no lugar) e a devolve."""
    cor, alfa, caixa = camada
    if caixa is None:
        return imagem
    y0, y1, x0, x1 = caixa
    recorte = imagem[y0:y1, x0:x1].astype(np.float32)
    a = alfa[y0:y1, x0:x1]
    imagem[y0:y1, x0:x1] = (recorte * (1 - a) + cor[y0:y1, x0:x1] * a).astype(np.uint8)
    return imagem


def desenhar_zonas(imagem, zonas):
    """Cópia de `imagem` com as Zonas por Aproximação ({nome: sv.PolygonZone ou pontos})."""
    return aplicar_camada(imagem.copy(), camada_zonas(imagem.shape, zonas))


# Exibição ------------------------------------------------------------------
def mostrar(imagem, largura=960):
    """Mostra uma imagem BGR (ou um caminho de imagem) no notebook."""
    if isinstance(imagem, (str, Path)):
        imagem = cv2.imread(str(imagem))
    from IPython.display import Image as ImagemIPython
    from IPython.display import display

    ok, jpeg = cv2.imencode(".jpg", imagem, [cv2.IMWRITE_JPEG_QUALITY, 90])
    display(ImagemIPython(data=jpeg.tobytes(), format="jpeg", width=largura))


def mostrar_recorte(quadro, recorte, linha, coluna, altura=360):
    """Mostra o quadro com o retângulo do recorte e, ao lado, o recorte ampliado.

    A ampliação repete cada pixel (sem suavizar): num recorte pequeno, os
    quadradinhos de cor aparecem.
    """
    h, w = recorte.shape[:2]
    if h == 0 or w == 0:
        print("O recorte ficou vazio: diminua a linha ou a coluna.")
        return
    marcado = quadro.copy()
    cv2.rectangle(marcado, (coluna, linha), (coluna + w - 1, linha + h - 1), estilo.hex_para_bgr(estilo.AMARELO), 4)
    escala = altura / quadro.shape[0]
    marcado = cv2.resize(marcado, (round(quadro.shape[1] * escala), altura), interpolation=cv2.INTER_AREA)
    fator = altura / max(h, w)
    ampliado = cv2.resize(recorte, (max(1, round(w * fator)), max(1, round(h * fator))), interpolation=cv2.INTER_NEAREST)
    lado = np.full((altura, altura, 3), estilo.hex_para_bgr(estilo.OFF_WHITE), np.uint8)
    lado[: ampliado.shape[0], : ampliado.shape[1]] = ampliado
    vao = np.full((altura, 16, 3), estilo.hex_para_bgr(estilo.OFF_WHITE), np.uint8)
    mostrar(np.hstack([marcado, vao, lado]), largura=960)


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
