"""Gera as camadas da Cena-âncora para a leitura tarefa a tarefa no deck.

A Cena-âncora é o quadro 623 do vídeo do Case (20,77 s, com o ônibus no centro do
cruzamento). Cada tarefa vira um PNG transparente do tamanho do quadro, para empilhar
no PowerPoint com animação de entrada, na ordem da Parte teórica:

    00-quadro.png          o quadro 623 (fundo opaco)
    01-classificacao.png   uma classe para a imagem inteira (YOLO26n-cls, ImageNet)
    02-deteccao.png        caixas e classes dos veículos (YOLO26n, o modelo do Case)
    03-segmentacao.png     máscaras dos veículos (YOLO26n-seg)
    04-rastreamento.png    Rastros de 2 s e o ID de cada veículo
    05-zonas.png           as Zonas por Aproximação do Case
    06-painel.png          o painel com a Matriz de movimentos em 20,77 s (outro tamanho:
                           a mesma altura do quadro, para ficar ao lado dele)
    06b-painel-final.png   o mesmo painel no fim dos 40 s (38 Movimentos)
    previa.jpg             folha com os passos, para conferir

Uso recomendado no deck: cada tarefa entra sozinha sobre o quadro e a anterior sai
(empilhadas todas, caixas, máscaras e Rastros se cobrem); só as Zonas entram por cima
dos Rastros, e o painel entra ao lado, no passo das Métricas de vídeo.

As camadas 00–05 têm 2560×1440 (o dobro do quadro, para traços e textos nítidos na
projeção) e o painel, 1280×1440. Todas as camadas 00–05 se alinham quando inseridas
no mesmo lugar e no mesmo tamanho.

Rastros, IDs e Matriz saem do mesmo processamento do Case (`oficina_vc/case.py`,
1 quadro a cada 2), parado no quadro 622, o último analisado antes do 623. Assim os
IDs e a contagem batem com o Resultado pronto naquele instante. As caixas do quadro
622 são trocadas pelas do 623 (a que mais se sobrepõe), para encostarem nos veículos.

Uso (de dentro do repositório, com os arquivos da Oficina em `dados/`):

    python scripts/gerar_cena_ancora.py \\
        --destino ../drive/oficina-vc-instrutor/deck/cena-ancora \\
        --modelos ../drive/oficina-vc-instrutor/modelos

Os pesos `yolo26n-seg.pt` e `yolo26n-cls.pt` são baixados pela Ultralytics para a
pasta `--modelos` se ainda não estiverem lá.
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oficina_vc import case, dados, desenho, estilo
from oficina_vc.movimentos import ContadorDeMovimentos
from oficina_vc.painel import Painel
from oficina_vc.video import (
    RASTRO_SEGUNDOS,
    SUAVIZAR_SEGUNDOS,
    _atualizar_rastreador,
    _votar_classe,
)

QUADRO = 623
E = 2  # escala das camadas em relação ao quadro de 1280×720
CONFIANCA_EXIBIDA = 0.25  # caixas e máscaras mostradas no slide (o rastreador usa a do Case, 0,05)

# As três primeiras classes do ImageNet que o classificador costuma dar para este quadro.
NOMES_IMAGENET_PT = {
    "streetcar": "bonde",
    "aircraft_carrier": "porta-aviões",
    "racer": "carro de corrida",
    "trolleybus": "ônibus elétrico",
    "passenger_car": "vagão de passageiros",
    "amphibian": "veículo anfíbio",
    "solar_dish": "antena solar",
}


# Utilidades ------------------------------------------------------------------
def rgba(cor, alfa=255):
    return estilo.hex_para_rgb(cor) + (alfa,)


def numero(valor):
    return f"{valor:.2f}".replace(".", ",")


def camada_vazia(largura, altura):
    return Image.new("RGBA", (largura * E, altura * E), (0, 0, 0, 0))


def pilula(texto, fundo, cor_texto, tamanho, alfa=255):
    """Etiqueta arredondada (RGBA), no estilo das etiquetas das Zonas."""
    fonte = estilo.fonte(tamanho, negrito=True)
    x0, _, x1, y1 = fonte.getbbox(texto)
    margem = tamanho // 3
    largura, altura = x1 - x0 + 2 * margem, y1 + 2 * margem
    imagem = Image.new("RGBA", (largura, altura), (0, 0, 0, 0))
    traco = ImageDraw.Draw(imagem)
    traco.rounded_rectangle((0, 0, largura - 1, altura - 1), radius=altura // 2, fill=rgba(fundo, alfa))
    traco.text((margem - x0, margem), texto, font=fonte, fill=rgba(cor_texto))
    return imagem


def bgr_para_rgba(desenho_bgr):
    """Desenho do supervision feito sobre fundo preto → RGBA (o preto vira transparente)."""
    alfa = (desenho_bgr.max(axis=2) > 0).astype(np.uint8) * 255
    rgb = cv2.cvtColor(desenho_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(np.dstack([rgb, alfa]), "RGBA")


def escalar(deteccoes):
    """Cópia das detecções com as caixas (e máscaras) no tamanho das camadas."""
    det = copy.deepcopy(deteccoes)
    det.xyxy = det.xyxy * E
    if det.mask is not None:
        det.mask = np.stack(
            [cv2.resize(m.astype(np.float32), None, fx=E, fy=E, interpolation=cv2.INTER_LINEAR) > 0.5 for m in det.mask]
        ) if len(det) else det.mask
    return det


# 1. Classificação --------------------------------------------------------------
def camada_classificacao(quadro, modelo_cls):
    altura, largura = quadro.shape[:2]
    resultado = modelo_cls(quadro, verbose=False)[0]
    probs = resultado.probs
    top3 = [(modelo_cls.names[int(i)], float(probs.data[int(i)])) for i in probs.top5[:3]]

    camada = camada_vazia(largura, altura)
    traco = ImageDraw.Draw(camada)
    # A moldura amarela: a classificação olha para a imagem inteira, sem lugar.
    espessura = 4 * E
    traco.rectangle((0, 0, largura * E - 1, altura * E - 1), outline=rgba(estilo.AMARELO), width=espessura)

    # Cartão com as três classes mais prováveis
    fonte = estilo.fonte(22 * E, negrito=True)
    fonte_fraca = estilo.fonte(22 * E)
    margem, linha, barra = 18 * E, 40 * E, 150 * E
    nomes = [NOMES_IMAGENET_PT.get(n, n.replace("_", " ")) for n, _ in top3]
    coluna_nome = max(traco.textlength(n, font=fonte) for n in nomes)
    largura_cartao = margem * 3 + coluna_nome + barra + traco.textlength("0,00", font=fonte) + 10 * E
    altura_cartao = margem * 2 + linha * len(top3) - 8 * E
    x, y = 24 * E, 24 * E
    cartao = Image.new("RGBA", camada.size, (0, 0, 0, 0))
    ImageDraw.Draw(cartao).rounded_rectangle(
        (x, y, x + largura_cartao, y + altura_cartao), radius=14 * E, fill=rgba(estilo.NAVY, 225)
    )
    camada.alpha_composite(cartao)
    for k, (nome, (_, p)) in enumerate(zip(nomes, top3)):
        yk = y + margem + k * linha
        cor = estilo.AMARELO if k == 0 else estilo.OFF_WHITE
        traco.text((x + margem, yk), nome, font=fonte if k == 0 else fonte_fraca, fill=rgba(cor))
        xb = x + margem * 2 + coluna_nome
        traco.rounded_rectangle((xb, yk + 7 * E, xb + barra, yk + 21 * E), radius=7 * E, fill=rgba(estilo.CINZA))
        traco.rounded_rectangle((xb, yk + 7 * E, xb + max(barra * p, 14 * E), yk + 21 * E), radius=7 * E, fill=rgba(cor))
        traco.text((xb + barra + 10 * E, yk), numero(p), font=fonte if k == 0 else fonte_fraca, fill=rgba(cor))
    return camada, top3


# 2. Detecção ---------------------------------------------------------------------
def rotulos_de(deteccoes, nomes):
    return [
        f"{estilo.nome_pt(nomes[int(c)])} {numero(p)}" for c, p in zip(deteccoes.class_id, deteccoes.confidence)
    ]


def camada_deteccao(quadro, deteccoes, nomes):
    altura, largura = quadro.shape[:2]
    det = escalar(deteccoes)
    paleta, paleta_texto = desenho.paletas(nomes)
    tela = np.zeros((altura * E, largura * E, 3), np.uint8)
    tela = sv.BoxAnnotator(color=paleta, thickness=desenho.ESPESSURA_CAIXA * E).annotate(tela, det)
    tela = sv.RichLabelAnnotator(
        color=paleta,
        text_color=paleta_texto,
        font_path=estilo.caminho_fonte(negrito=True),
        font_size=17 * E,
        text_padding=4 * E,
        smart_position=True,
    ).annotate(tela, det, labels=rotulos_de(deteccoes, nomes))
    return bgr_para_rgba(tela)


# 3. Segmentação ------------------------------------------------------------------
def camada_segmentacao(quadro, deteccoes, nomes, alfa=0.5):
    altura, largura = quadro.shape[:2]
    det = escalar(deteccoes)
    cor = np.zeros((altura * E, largura * E, 3), np.uint8)
    opaco = np.zeros((altura * E, largura * E), np.uint8)
    contornos = np.zeros_like(opaco)
    for mascara, classe in zip(det.mask, det.class_id):
        rgb = estilo.hex_para_rgb(estilo.cores_da_classe(nomes[int(classe)], int(classe))[0])
        cor[mascara] = rgb
        opaco[mascara] = int(255 * alfa)
        linhas, _ = cv2.findContours(mascara.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(cor, linhas, -1, rgb, 2 * E, lineType=cv2.LINE_AA)
        cv2.drawContours(contornos, linhas, -1, 255, 2 * E, lineType=cv2.LINE_AA)
    alfa_final = np.maximum(opaco, contornos)
    return Image.fromarray(np.dstack([cor, alfa_final]), "RGBA")


# 4. Rastreamento: o processamento do Case até o quadro 622 -------------------------
def rastrear_ate(video, modelo, ultimo_quadro):
    """Repete o laço de `oficina_vc.video.processar_video` (sem desenhar) até `ultimo_quadro`.

    Devolve as detecções rastreadas e suavizadas do último quadro analisado, os
    pontos dos pés de cada ID (para os Rastros), o contador de Movimentos e o
    índice do último quadro analisado.
    """
    captura = cv2.VideoCapture(str(video))
    fps_video = captura.get(cv2.CAP_PROP_FPS) or 30
    passo = case.PASSO
    fps = fps_video / passo
    zonas = case.criar_zonas()
    contador = ContadorDeMovimentos(copy.deepcopy(zonas))
    rastreador = case.criar_rastreador()
    suavizador = sv.DetectionsSmoother(length=max(round(SUAVIZAR_SEGUNDOS * fps), 1))
    nomes = desenho.nomes_das_classes(modelo)
    nomes_pt = {i: estilo.nome_pt(nomes[i]) for i in nomes}
    votos, pontos = {}, {}
    argumentos = {"conf": case.CONFIANCA, "classes": case.CLASSES, "imgsz": case.IMGSZ, "verbose": False}

    analisados = list(range(0, ultimo_quadro + 1, passo))
    for indice, numero_quadro in enumerate(analisados):
        for _ in range(passo - 1 if indice else 0):
            captura.grab()
        ok, quadro = captura.read()
        if not ok:
            break
        t = (numero_quadro + 1) / fps_video
        deteccoes = sv.Detections.from_ultralytics(modelo(quadro, **argumentos)[0])
        deteccoes = deteccoes.with_nms(threshold=0.6, class_agnostic=True)
        deteccoes = _atualizar_rastreador(rastreador, deteccoes)
        deteccoes = deteccoes[deteccoes.tracker_id >= 0]
        deteccoes = _votar_classe(deteccoes, votos)
        suavizadas = suavizador.update_with_detections(deteccoes)
        contador.atualizar(suavizadas, t, nomes_pt)
        pes = suavizadas.get_anchors_coordinates(sv.Position.BOTTOM_CENTER) if len(suavizadas) else []
        for rastro, ponto in zip(suavizadas.tracker_id, pes):
            pontos.setdefault(int(rastro), []).append((indice, tuple(map(float, ponto))))
    captura.release()
    return suavizadas, pontos, contador, indice, fps


def trocar_pelas_caixas_do_quadro(rastreadas, deteccoes, limiar=0.3):
    """Para cada caixa rastreada, a caixa do quadro 623 que mais se sobrepõe a ela (ou a própria)."""
    caixas = rastreadas.xyxy.copy()
    if len(deteccoes) and len(rastreadas):
        iou = sv.box_iou_batch(rastreadas.xyxy, deteccoes.xyxy)
        for k in range(len(rastreadas)):
            j = int(iou[k].argmax())
            if iou[k, j] >= limiar:
                caixas[k] = deteccoes.xyxy[j]
    return caixas


def camada_rastreamento(quadro, rastreadas, caixas, pontos, ultimo_indice, fps):
    altura, largura = quadro.shape[:2]
    camada = camada_vazia(largura, altura)
    traco = ImageDraw.Draw(camada)
    janela = int(RASTRO_SEGUNDOS * fps)
    amarelo, navy = rgba(estilo.AMARELO), rgba(estilo.NAVY, 200)
    cabecas = []
    for rastro, caixa in zip(rastreadas.tracker_id, caixas):
        cabeca = ((caixa[0] + caixa[2]) / 2 * E, caixa[3] * E)
        caminho = [(x * E, y * E) for i, (x, y) in pontos.get(int(rastro), []) if i > ultimo_indice - janela]
        caminho.append(cabeca)
        if len(caminho) > 1 and np.hypot(*np.subtract(caminho[-1], caminho[0])) > 6 * E:
            traco.line(caminho, fill=navy, width=9 * E, joint="curve")
            traco.line(caminho, fill=amarelo, width=4 * E, joint="curve")
        r = 6 * E
        traco.ellipse((cabeca[0] - r, cabeca[1] - r, cabeca[0] + r, cabeca[1] + r), fill=amarelo, outline=navy, width=2 * E)
        cabecas.append((int(rastro), cabeca))
    # IDs, depois de todos os Rastros, para ficarem por cima
    for rastro, (x, y) in cabecas:
        etiqueta = pilula(f"#{rastro}", estilo.NAVY, estilo.AMARELO, 15 * E, alfa=235)
        w, h = etiqueta.size
        camada.alpha_composite(etiqueta, (int(min(max(x - w / 2, 0), camada.width - w)), int(min(y + 9 * E, camada.height - h))))
    return camada


# 5. Zonas e painel -----------------------------------------------------------------
def camada_zonas(quadro, zonas):
    """As Zonas do Case, no desenho de `oficina_vc.desenho`, no dobro do tamanho."""
    altura, largura = quadro.shape[:2]
    camada = camada_vazia(largura, altura)
    poligonos = {nome: [(x * E, y * E) for x, y in pontos] for nome, pontos in zonas.items()}
    preenchimento = Image.new("RGBA", camada.size, (0, 0, 0, 0))
    for poligono in poligonos.values():
        ImageDraw.Draw(preenchimento).polygon(poligono, fill=rgba(estilo.COR_ZONA, int(255 * estilo.ALFA_ZONA)))
    camada.alpha_composite(preenchimento)
    traco = ImageDraw.Draw(camada)
    for poligono in poligonos.values():
        traco.line(poligono + [poligono[0]], fill=rgba(estilo.COR_ZONA, 230), width=4 * E, joint="curve")
    for nome, poligono in poligonos.items():
        etiqueta = desenho._etiqueta(nome, tamanho=26 * E)
        x, y = desenho._lugar_da_etiqueta(poligono, etiqueta.size, camada.size)
        camada.alpha_composite(etiqueta, (x, y))
    return camada


def painel(t, matriz, por_tipo, por_classe, aproximacoes, classes, cores):
    imagem = Painel(duracao=40, aproximacoes=aproximacoes, classes=classes, cores=cores, fps=15).desenhar(
        t=t, matriz=matriz, por_tipo=por_tipo, por_classe=por_classe, ultimo=None
    )
    rgb = Image.fromarray(cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB))
    return rgb.resize((rgb.width * E, rgb.height * E), Image.LANCZOS)


# Prévia ----------------------------------------------------------------------------
def previa(base, camadas, painel_img, caminho):
    """Folha 2×3 com o uso recomendado no deck: cada tarefa sozinha sobre o quadro (a camada
    anterior sai quando a nova entra), menos as Zonas, que entram por cima dos Rastros; o
    painel fica ao lado do último passo."""
    classificacao, deteccao, segmentacao, rastreamento, zonas = camadas
    passos = [
        ("Quadro 623", []),
        ("Classificação", [classificacao]),
        ("Detecção", [deteccao]),
        ("Segmentação", [segmentacao]),
        ("Rastreamento", [rastreamento]),
        ("Métricas de vídeo", [zonas, rastreamento]),
    ]
    largura_mini = 960
    escala = largura_mini / base.width
    altura_mini = round(base.height * escala)
    largura_painel = round(painel_img.width * altura_mini / painel_img.height)
    folha = Image.new("RGB", (2 * largura_mini + largura_painel + 60, 3 * (altura_mini + 50) + 20), rgba(estilo.OFF_WHITE)[:3])
    traco = ImageDraw.Draw(folha)
    for k, (titulo, pilha_do_passo) in enumerate(passos):
        pilha = base.convert("RGBA")
        for camada in pilha_do_passo:
            pilha = Image.alpha_composite(pilha, camada)
        x, y = 20 + (k % 2) * (largura_mini + 20), 20 + (k // 2) * (altura_mini + 50)
        traco.text((x, y), titulo, font=estilo.fonte(24, negrito=True), fill=rgba(estilo.NAVY)[:3])
        folha.paste(pilha.resize((largura_mini, altura_mini), Image.LANCZOS).convert("RGB"), (x, y + 32))
    x, y = 20 + 2 * (largura_mini + 20), 20 + 2 * (altura_mini + 50) + 32
    folha.paste(painel_img.resize((largura_painel, altura_mini), Image.LANCZOS), (x, y))
    folha.save(caminho, quality=88)


def main():
    argumentos = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    argumentos.add_argument("--destino", type=Path, required=True, help="pasta dos PNGs")
    argumentos.add_argument("--modelos", type=Path, required=True, help="pasta dos pesos -seg e -cls")
    opcoes = argumentos.parse_args()

    from ultralytics import YOLO

    opcoes.destino.mkdir(parents=True, exist_ok=True)
    opcoes.modelos.mkdir(parents=True, exist_ok=True)
    video = dados.caminho(case.VIDEO)
    modelo = YOLO(str(dados.caminho("yolo26n.pt")))
    nomes = desenho.nomes_das_classes(modelo)
    pasta = Path.cwd()
    os.chdir(opcoes.modelos)  # a Ultralytics baixa os pesos que faltam na pasta atual
    modelo_seg, modelo_cls = YOLO("yolo26n-seg.pt"), YOLO("yolo26n-cls.pt")
    os.chdir(pasta)

    # O quadro 623, lido do PNG da Cena-âncora (é o mesmo quadro do vídeo, sem perda)
    quadro = cv2.imread(str(dados.caminho("cena_ancora_bellevue_623.png")))
    altura, largura = quadro.shape[:2]
    base = Image.fromarray(cv2.cvtColor(quadro, cv2.COLOR_BGR2RGB)).resize((largura * E, altura * E), Image.LANCZOS)

    print("Classificação…")
    classificacao, top3 = camada_classificacao(quadro, modelo_cls)

    print("Detecção e segmentação…")
    args = {"imgsz": case.IMGSZ, "classes": case.CLASSES, "verbose": False}
    exibidas = sv.Detections.from_ultralytics(modelo(quadro, conf=CONFIANCA_EXIBIDA, **args)[0])
    exibidas = exibidas.with_nms(threshold=0.6, class_agnostic=True)
    deteccao = camada_deteccao(quadro, exibidas, nomes)
    mascaras = sv.Detections.from_ultralytics(modelo_seg(quadro, conf=CONFIANCA_EXIBIDA, **args)[0])
    mascaras = mascaras.with_nms(threshold=0.6, class_agnostic=True)
    segmentacao = camada_segmentacao(quadro, mascaras, desenho.nomes_das_classes(modelo_seg))

    print(f"Rastreamento: o Case até o quadro {QUADRO - 1}…")
    rastreadas, pontos, contador, ultimo_indice, fps = rastrear_ate(video, modelo, QUADRO - 1)
    todas = sv.Detections.from_ultralytics(modelo(quadro, conf=case.CONFIANCA, **args)[0])
    todas = todas.with_nms(threshold=0.6, class_agnostic=True)
    caixas = trocar_pelas_caixas_do_quadro(rastreadas, todas)
    rastreamento = camada_rastreamento(quadro, rastreadas, caixas, pontos, ultimo_indice, fps)

    zonas = camada_zonas(quadro, case.ZONAS)
    ids_contados = sorted(case.CLASSES)
    classes = [estilo.nome_pt(nomes[i]) for i in ids_contados]
    cores = [estilo.cores_da_classe(nomes[i], i)[0] for i in ids_contados]
    t = QUADRO / 30
    painel_agora = painel(t, contador.matriz(), contador.por_tipo(), contador.por_classe(classes), list(case.ZONAS), classes, cores)
    final = json.loads(dados.caminho("case_metricas.json").read_text(encoding="utf-8"))
    painel_final = painel(40, final["matriz"], final["por_tipo"], final["por_classe"], list(case.ZONAS), classes, cores)

    camadas = [classificacao, deteccao, segmentacao, rastreamento, zonas]
    nomes_arquivos = ["01-classificacao", "02-deteccao", "03-segmentacao", "04-rastreamento", "05-zonas"]
    base.save(opcoes.destino / "00-quadro.png", optimize=True)
    for nome, camada in zip(nomes_arquivos, camadas):
        camada.save(opcoes.destino / f"{nome}.png", optimize=True)
    painel_agora.save(opcoes.destino / "06-painel.png", optimize=True)
    painel_final.save(opcoes.destino / "06b-painel-final.png", optimize=True)
    previa(base, camadas, painel_agora, opcoes.destino / "previa.jpg")

    # Resumo para o ticket
    no_json = [m for m in final["lista_de_movimentos"] if m["t"] <= t]
    print()
    print("Classificação: " + ", ".join(f"{n} {numero(p)}" for n, p in top3))
    print(f"Detecção: {len(exibidas)} veículos ({', '.join(rotulos_de(exibidas, nomes))})")
    print(f"Segmentação: {len(mascaras)} máscaras")
    print(f"Rastreamento: {len(rastreadas)} IDs no quadro {QUADRO - 1}")
    print(f"Movimentos até {numero(t)} s: {len(contador.movimentos)} (Resultado pronto: {len(no_json)})")
    print(f"Arquivos em {opcoes.destino}")


if __name__ == "__main__":
    main()
