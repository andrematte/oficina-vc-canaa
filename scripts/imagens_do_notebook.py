"""Gera as imagens didáticas dos dois notebooks e as embute neles.

Na prática o projetor fica no Colab, sem slide de apoio; estas imagens trazem a
teoria para dentro do notebook ("Conteúdo didático dentro dos notebooks"). No `1_case_trafego.ipynb`:

    mapa-do-case       tira com o quadro, as caixas, os Rastros e as Zonas (de oficina_vc/case.py)
    pixel-rgb          o último quadro da animação 01b (o mesmo pixel, três números)
    coco-lote          um lote de treino do COCO, com as caixas e o número de cada classe
    movimentos-norte   as três saídas de quem vem do Norte: reto, direita e esquerda

e, no `2_fine_tuning_tipos_de_veiculo.ipynb`, o último quadro de quatro animações do deck:
corpo-e-cabeca (05, passo 3), do-zero-x-fine-tuning (05, passo 4),
divisao-do-dataset (02, passo 4) e epoca-e-erro (04c, passo 2).

As imagens vão embutidas no próprio `.ipynb` (JPEG em base64), para funcionar no
Colab e no Modo local sem internet. Cada uma entra num `<img alt="NOME" src="...">`
de uma célula de texto: o script troca o `src` pela imagem nova, sem mexer no resto.

Uso (de dentro do repositório, com os arquivos da Oficina em `dados/`):

    python scripts/imagens_do_notebook.py \\
        --instrutor ../drive/oficina-vc-instrutor  # os dois notebooks
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
def mapa_do_case(cena, dados):
    """Quatro miniaturas da Cena-âncora com o passo de cada uma embaixo.

    Caixas e Rastros vêm das camadas do deck; as Zonas são desenhadas a partir de
    `oficina_vc/case.py`, para seguirem o notebook mesmo se mudarem depois do deck.
    """
    base = Image.open(cena / "00-quadro.png").convert("RGBA")
    passos = [
        (None, "Vídeo"),
        ("02-deteccao.png", "Detectar"),
        ("04-rastreamento.png", "Rastrear"),
        ("zonas", "Contar por Zona"),
    ]
    larg, alt, margem, legenda = 400, 225, 20, 56
    tira = Image.new("RGB", (4 * larg + 5 * margem, alt + legenda + 2 * margem), estilo.OFF_WHITE)
    traco = ImageDraw.Draw(tira)
    fonte = estilo.fonte(28, negrito=True)
    for i, (camada, texto) in enumerate(passos):
        quadro = base.copy()
        if camada == "zonas":
            quadro = quadro_com_zonas(dados).convert("RGBA")
        elif camada:
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
def quadro_final(animacoes, nome, corte=(0.0, 1.0), largura=1200):
    """Último quadro de uma animação do deck (MP4), com corte vertical (fração de cima, fração de baixo)."""
    with tempfile.TemporaryDirectory() as tmp:
        saida = Path(tmp) / "fim.png"
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-sseof", "-0.1",
             "-i", str(animacoes / f"{nome}.mp4"), "-frames:v", "1", str(saida)],
            check=True,
        )
        imagem = Image.open(saida).convert("RGB")
    w, h = imagem.size
    imagem = imagem.crop((0, int(h * corte[0]), w, int(h * corte[1])))
    return imagem.resize((largura, round(largura * imagem.height / w)), Image.LANCZOS)


def pixel_rgb(animacoes):
    """Último quadro do passo 3 da animação 01b, sem a margem vazia de cima e de baixo."""
    return quadro_final(animacoes, "01b-imagem-colorida_passo03", (0.12, 0.97))


# Um lote de treino do COCO -----------------------------------------------------------
# Página do COCO na documentação da Ultralytics (AGPL-3.0): docs.ultralytics.com/datasets/detect/coco
URL_COCO = "https://cdn.ul.run/i/f9843f887d8186a8b43fdec6f570dcba.avif"


def coco_lote():
    """As duas primeiras fileiras do mosaico de treino do COCO: fotos, caixas e o número da classe."""
    import urllib.request

    with tempfile.TemporaryDirectory() as tmp:
        avif, png = Path(tmp) / "coco.avif", Path(tmp) / "coco.png"
        urllib.request.urlretrieve(URL_COCO, avif)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(avif), str(png)], check=True)
        imagem = Image.open(png).convert("RGB")
    w, h = imagem.size
    alto = round(h * 0.48)  # as duas primeiras fileiras, sem as etiquetas que sobem da terceira
    return imagem.crop((0, 0, w, alto)).resize((1200, round(1200 * alto / w)), Image.LANCZOS)


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


def quadro_com_zonas(dados):
    """A Cena-âncora com as Zonas de `oficina_vc/case.py` (RGB)."""
    import cv2
    import numpy as np

    from oficina_vc.case import ZONAS, criar_zonas

    quadro = cv2.imread(str(dados / "cena_ancora_bellevue_623.png"))
    cor, alfa, _ = camada_zonas(quadro.shape, criar_zonas(ZONAS))
    quadro = (quadro * (1 - alfa) + cor * alfa).astype(np.uint8)
    return Image.fromarray(cv2.cvtColor(quadro, cv2.COLOR_BGR2RGB))


def movimentos_norte(dados):
    """O quadro com as Zonas e as três saídas de um veículo que chega pelo Norte."""
    imagem = quadro_com_zonas(dados).convert("RGBA")
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


def imagens_por_notebook(instrutor, dados):
    """{notebook: {alt da imagem: função que gera a imagem}}."""
    animacoes = instrutor / "animacoes"
    return {
        "1_case_trafego.ipynb": {
            "mapa-do-case": lambda: mapa_do_case(instrutor / "deck" / "cena-ancora", dados),
            "pixel-rgb": lambda: pixel_rgb(animacoes),
            "coco-lote": coco_lote,
            "movimentos-norte": lambda: movimentos_norte(dados),
        },
        "2_fine_tuning_tipos_de_veiculo.ipynb": {
            # corpo congelado e classificador novo; do zero × fine-tuning; treino, validação e teste; epoch e loss
            "corpo-e-cabeca": lambda: quadro_final(animacoes, "05-fine-tuning_passo03", (0.0, 0.93)),
            "do-zero-x-fine-tuning": lambda: quadro_final(animacoes, "05-fine-tuning_passo04", (0.0, 0.93)),
            "divisao-do-dataset": lambda: quadro_final(animacoes, "02-divisao-do-dataset_passo04", (0.0, 0.93)),
            "epoca-e-erro": lambda: quadro_final(animacoes, "04c-epocas_passo02", (0.08, 0.97)),
        },
    }


def main():
    raiz = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--instrutor", type=Path, default=raiz.parent / "drive" / "oficina-vc-instrutor")
    p.add_argument("--dados", type=Path, default=raiz / "dados")
    p.add_argument("--notebook", nargs="*", help="só estes notebooks (padrão: os dois)")
    p.add_argument("--previa", type=Path, help="pasta onde gravar as imagens em JPEG, para conferir")
    a = p.parse_args()

    for notebook, geradores in imagens_por_notebook(a.instrutor, a.dados).items():
        if a.notebook and notebook not in a.notebook:
            continue
        imagens = {nome: gerar() for nome, gerar in geradores.items()}
        if a.previa:
            a.previa.mkdir(parents=True, exist_ok=True)
            for nome, imagem in imagens.items():
                imagem.save(a.previa / f"{nome}.jpg", quality=QUALIDADE)
        uris = {nome: data_uri(imagem) for nome, imagem in imagens.items()}
        embutir(raiz / notebook, uris)
        print(notebook + ": " + ", ".join(f"{nome} {len(uri) * 3 // 4 // 1024} KB" for nome, uri in uris.items()))


if __name__ == "__main__":
    main()
