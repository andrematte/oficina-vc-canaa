"""Gera as imagens didáticas do `1_case_trafego.ipynb` e as embute no notebook.

Na prática o projetor fica no Colab, sem slide de apoio; estas imagens trazem a
teoria para dentro do notebook ("Conteúdo didático dentro dos notebooks"):

    mapa-do-case       tira com o quadro, as caixas, os Rastros e as Zonas da Cena-âncora
    pixel-rgb          o último quadro da animação 01b (o mesmo pixel, três números)
    movimentos-norte   as três saídas de quem vem do Norte: reto, direita e esquerda

As imagens vão embutidas no próprio `.ipynb` (JPEG em base64), para funcionar no
Colab e no Modo local sem internet. Cada uma entra num `<img alt="NOME" src="...">`
de uma célula de texto: o script troca o `src` pela imagem nova, sem mexer no resto.

Uso (de dentro do repositório, com os arquivos da Oficina em `dados/`):

    python scripts/imagens_do_notebook.py \\
        --instrutor ../drive/oficina-vc-instrutor \\
        --notebook 1_case_trafego.ipynb
"""

import argparse
import base64
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oficina_vc import estilo  # noqa: E402
from oficina_vc.desenho import camada_zonas  # noqa: E402

QUALIDADE = 82


# Tira da abertura ------------------------------------------------------------
def mapa_do_case(cena):
    """Quatro miniaturas da Cena-âncora com o passo de cada uma embaixo."""
    base = Image.open(cena / "00-quadro.png").convert("RGBA")
    passos = [
        (None, "Vídeo"),
        ("02-deteccao.png", "Detectar"),
        ("04-rastreamento.png", "Rastrear"),
        ("05-zonas.png", "Contar por Zona"),
    ]
    larg, alt, margem, legenda = 400, 225, 20, 56
    tira = Image.new("RGB", (4 * larg + 5 * margem, alt + legenda + 2 * margem), estilo.OFF_WHITE)
    traco = ImageDraw.Draw(tira)
    fonte = estilo.fonte(28, negrito=True)
    for i, (camada, texto) in enumerate(passos):
        quadro = base.copy()
        if camada:
            quadro.alpha_composite(Image.open(cena / camada).convert("RGBA"))
        x = margem + i * (larg + margem)
        tira.paste(quadro.convert("RGB").resize((larg, alt), Image.LANCZOS), (x, margem))
        w = traco.textlength(texto, font=fonte)
        traco.text((x + (larg - w) / 2, margem + alt + 12), texto, font=fonte, fill=estilo.AZUL if i else estilo.CINZA)
        if i:  # seta entre as miniaturas
            meio = margem + alt / 2
            traco.polygon([(x - margem + 4, meio - 10), (x - 4, meio), (x - margem + 4, meio + 10)], fill=estilo.AMARELO)
    return tira


# Imagem como números ---------------------------------------------------------
def pixel_rgb(animacoes):
    """Último quadro do passo 3 da animação 01b, sem a margem vazia de cima e de baixo."""
    with tempfile.TemporaryDirectory() as tmp:
        saida = Path(tmp) / "fim.png"
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-sseof", "-0.1",
             "-i", str(animacoes / "01b-imagem-colorida_passo03.mp4"), "-frames:v", "1", str(saida)],
            check=True,
        )
        imagem = Image.open(saida).convert("RGB")
    w, h = imagem.size
    return imagem.crop((0, int(h * 0.12), w, int(h * 0.97))).resize((1200, int(1200 * h * 0.85 / w)), Image.LANCZOS)


# Movimentos de quem vem do Norte -------------------------------------------------
CORES_MOVIMENTO = {  # as mesmas do painel
    "Reto": estilo.OFF_WHITE,
    "Conversão à direita": estilo.VERDE,
    "Conversão à esquerda": "#5B70F4",
}


def _bezier(p0, p1, p2, n=60):
    return [
        ((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t**2 * p2[0],
         (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t**2 * p2[1])
        for t in (i / n for i in range(n + 1))
    ]


def _seta(traco, pontos, cor, largura=11):
    """Curva com contorno navy e ponta de seta no último ponto."""
    (xa, ya), (xb, yb) = pontos[-6], pontos[-1]
    ang = math.atan2(yb - ya, xb - xa)
    ponta = 30
    corpo = pontos[:-4]
    cabeca = [
        (xb + 6 * math.cos(ang), yb + 6 * math.sin(ang)),
        (xb - ponta * math.cos(ang - 0.45), yb - ponta * math.sin(ang - 0.45)),
        (xb - ponta * math.cos(ang + 0.45), yb - ponta * math.sin(ang + 0.45)),
    ]
    traco.line(corpo, fill=estilo.NAVY, width=largura + 6, joint="curve")
    traco.polygon(cabeca, fill=cor, outline=estilo.NAVY, width=3)
    traco.line(corpo, fill=cor, width=largura, joint="curve")


def _rotulo(traco, xy, texto, cor):
    fonte = estilo.fonte(26, negrito=True)
    x0, y0, x1, y1 = fonte.getbbox(texto)
    m = 10
    x, y = xy
    traco.rounded_rectangle((x, y, x + x1 - x0 + 2 * m, y + y1 + 2 * m), radius=(y1 + 2 * m) // 2, fill=cor, outline=estilo.NAVY, width=2)
    claro = cor in (estilo.OFF_WHITE, estilo.VERDE)
    traco.text((x + m - x0, y + m), texto, font=fonte, fill=estilo.NAVY if claro else estilo.OFF_WHITE)


def movimentos_norte(dados):
    """O quadro com as Zonas e as três saídas de um veículo que chega pelo Norte."""
    import cv2
    import numpy as np

    from oficina_vc.case import criar_zonas

    zonas = {  # as mesmas da etapa 4 do notebook
        "Norte": [(495, 55), (725, 55), (805, 235), (470, 235)],
        "Leste": [(890, 215), (1275, 225), (1275, 410), (1120, 420), (990, 330)],
        "Sul": [(540, 625), (1130, 560), (1200, 695), (560, 695)],
        "Oeste": [(0, 290), (300, 278), (445, 252), (455, 345), (310, 360), (310, 575), (0, 580)],
    }
    quadro = cv2.imread(str(dados / "cena_ancora_bellevue_623.png"))
    cor, alfa, _ = camada_zonas(quadro.shape, criar_zonas(zonas))
    quadro = (quadro * (1 - alfa) + cor * alfa).astype(np.uint8)
    imagem = Image.fromarray(cv2.cvtColor(quadro, cv2.COLOR_BGR2RGB)).convert("RGBA")
    # escurece um pouco o fundo para as setas saltarem
    imagem = Image.alpha_composite(imagem, Image.new("RGBA", imagem.size, (20, 25, 61, 70)))
    traco = ImageDraw.Draw(imagem)
    saida = (640, 215)
    _seta(traco, _bezier(saida, (720, 420), (830, 610)), CORES_MOVIMENTO["Reto"])
    _seta(traco, _bezier(saida, (560, 330), (370, 320)), CORES_MOVIMENTO["Conversão à direita"])
    _seta(traco, _bezier(saida, (720, 300), (1010, 290)), CORES_MOVIMENTO["Conversão à esquerda"])
    # rótulos sem "→": a Raleway não tem a seta
    _rotulo(traco, (760, 470), "Reto: para o Sul", CORES_MOVIMENTO["Reto"])
    _rotulo(traco, (95, 375), "À direita: para o Oeste", CORES_MOVIMENTO["Conversão à direita"])
    _rotulo(traco, (900, 335), "À esquerda: para o Leste", CORES_MOVIMENTO["Conversão à esquerda"])
    return imagem.convert("RGB")


# Embutir no notebook -------------------------------------------------------------
def data_uri(imagem):
    import io

    buf = io.BytesIO()
    imagem.save(buf, "JPEG", quality=QUALIDADE, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def embutir(notebook, imagens):
    nb = json.loads(notebook.read_text())
    faltando = set(imagens)
    for celula in nb["cells"]:
        if celula["cell_type"] != "markdown":
            continue
        texto = "".join(celula["source"])
        for nome, uri in imagens.items():
            padrao = re.compile(rf'(<img alt="{re.escape(nome)}"[^>]*?src=")[^"]*(")')
            if padrao.search(texto):
                texto = padrao.sub(lambda m: m.group(1) + uri + m.group(2), texto)
                faltando.discard(nome)
        celula["source"] = texto.splitlines(keepends=True)
    if faltando:
        sys.exit(f"Sem <img alt=...> no notebook para: {', '.join(sorted(faltando))}")
    notebook.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n")


def main():
    raiz = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--instrutor", type=Path, default=raiz.parent / "drive" / "oficina-vc-instrutor")
    p.add_argument("--dados", type=Path, default=raiz / "dados")
    p.add_argument("--notebook", type=Path, default=raiz / "1_case_trafego.ipynb")
    p.add_argument("--previa", type=Path, help="pasta onde gravar as imagens em JPEG, para conferir")
    a = p.parse_args()

    imagens = {
        "mapa-do-case": mapa_do_case(a.instrutor / "deck" / "cena-ancora"),
        "pixel-rgb": pixel_rgb(a.instrutor / "animacoes"),
        "movimentos-norte": movimentos_norte(a.dados),
    }
    if a.previa:
        a.previa.mkdir(parents=True, exist_ok=True)
        for nome, imagem in imagens.items():
            imagem.save(a.previa / f"{nome}.jpg", quality=QUALIDADE)
    uris = {nome: data_uri(imagem) for nome, imagem in imagens.items()}
    embutir(a.notebook, uris)
    for nome, uri in uris.items():
        print(f"{nome}: {len(uri) * 3 // 4 // 1024} KB")


if __name__ == "__main__":
    main()
