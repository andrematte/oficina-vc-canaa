"""Materiais do Fine-tuning demonstrativo: anotação ao vivo e desafio "Que veículo é esse?".

    ao-vivo   as 5 imagens da anotação ao vivo no makesense.ai: recortes do campo
              próximo de quadros do Bellevue que não estão no dataset (nem nos 40 s),
              ampliados para 640×640, com o gabarito (caixas do pré-treinado, classes
              conferidas à vista) em YOLO e desenhado
    desafio   os recortes do "Que veículo é esse?" (Pickup, SUV, Van, Caminhão, Carro e,
              opcional, Ônibus), da Cena-âncora e de outros quadros dos 40 s, com a
              resposta do pré-treinado e a do especialista

Uso (depois de `dataset_tipos_de_veiculo.py amostrar` e de `treinar_tipos_de_veiculo.py treinar`):

    python scripts/materiais_fine_tuning.py ao-vivo
    python scripts/materiais_fine_tuning.py desafio
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
sys.path.insert(0, str(RAIZ / "scripts"))

from dataset_tipos_de_veiculo import CACHE, CLASSES

DRIVE = RAIZ.parent / "drive"
DESTINO = DRIVE / "oficina-vc-instrutor" / "fine-tuning"
PUBLICO = DRIVE / "oficina-vc-publico"

# Anotação ao vivo ----------------------------------------------------------------
# (arquivo horário, instante do quadro-chave, x0, y0, lado do recorte na fonte, {caixa: classe},
#  caixas a mais em pixels do recorte já em 640×640). As caixas vêm do pré-treinado
# (deteccoes.json do cache de candidatos); veículos cortados na borda entram à mão.
AO_VIVO = [
    ("0911-1408", "00895.0", 0, 0, 544, {4: "caminhão", 2: "caminhão", 0: "carro", 5: "carro"}, []),
    ("0911-1108", "02705.7", 0, 8, 480, {2: "van", 3: "carro", 17: "carro", 4: "carro", 13: "carro", 10: "carro"}, []),
    ("0911-1408", "02858.1", 536, 184, 512, {15: "pickup", 4: "van", 7: "carro", 6: "carro", 11: "carro", 8: "carro"},
     [("carro", (115, 500, 245, 640))]),
    ("0911-1108", "01749.7", 320, 136, 480, {12: "van", 8: "caminhão", 3: "carro", 5: "carro"}, []),
    ("0911-1608", "00392.6", 232, 0, 480, {0: "ônibus", 7: "carro", 3: "carro", 11: "carro", 15: "carro"},
     [("carro", (0, 565, 140, 640)), ("carro", (588, 195, 632, 232))]),
]


def _desenhar(imagem, caixas):
    """caixas: [(classe, (x1, y1, x2, y2))] → cópia com as caixas nas cores dos Tipos de veículo."""
    from oficina_vc import estilo

    imagem = imagem.copy()
    for classe, (x1, y1, x2, y2) in caixas:
        cor, texto = estilo.cores_da_classe(classe)
        cv2.rectangle(imagem, (int(x1), int(y1)), (int(x2), int(y2)), estilo.hex_para_bgr(cor), 3)
        (w, h), _ = cv2.getTextSize(classe, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(imagem, (int(x1), int(y1) - h - 8), (int(x1) + w + 8, int(y1)), estilo.hex_para_bgr(cor), -1)
        cv2.putText(imagem, classe, (int(x1) + 4, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, estilo.hex_para_bgr(texto), 2)
    return imagem


def ao_vivo(opcoes):
    pasta = DESTINO / "anotacao-ao-vivo"
    (pasta / "gabarito").mkdir(parents=True, exist_ok=True)
    resumo = []
    for n, (horario, t, x0, y0, lado, classes, extras) in enumerate(AO_VIVO, 1):
        quadro = cv2.imread(str(CACHE / horario / f"{t}.jpg"))
        escala = 640 / lado
        imagem = cv2.resize(quadro[y0 : y0 + lado, x0 : x0 + lado], (640, 640), interpolation=cv2.INTER_CUBIC)
        deteccoes = json.loads((CACHE / horario / "deteccoes.json").read_text())[t]["caixas"]
        caixas = []
        for k, classe in classes.items():
            x1, y1, x2, y2 = deteccoes[k]["xyxy"]
            assert x0 <= x1 and x2 <= x0 + lado and y0 <= y1 and y2 <= y0 + lado, f"caixa {k} fora do recorte {n}"
            caixas.append((classe, ((x1 - x0) * escala, (y1 - y0) * escala, (x2 - x0) * escala, (y2 - y0) * escala)))
        caixas += extras
        nome = f"anotacao_{n}"
        cv2.imwrite(str(pasta / f"{nome}.png"), imagem)
        cv2.imwrite(str(pasta / "gabarito" / f"{nome}_gabarito.png"), _desenhar(imagem, caixas))
        linhas = [
            f"{CLASSES.index(c)} {(a + b) / 1280:.6f} {(d + e) / 1280:.6f} {(b - a) / 640:.6f} {(e - d) / 640:.6f}"
            for c, (a, d, b, e) in caixas
        ]
        (pasta / "gabarito" / f"{nome}.txt").write_text("\n".join(linhas) + "\n", encoding="utf-8")
        contagem = {c: sum(1 for x, _ in caixas if x == c) for c in CLASSES}
        resumo.append(f"| {nome}.png | {horario[2:4]}/{horario[:2]} {horario[5:7]}h, {float(t):.0f} s | "
                      + " | ".join(str(contagem[c]) for c in CLASSES) + " |")
    (pasta / "gabarito" / "classes.txt").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    (pasta / "LEIA-ME.md").write_text(
        "# Anotação ao vivo (makesense.ai)\n\n"
        "Cinco recortes do campo próximo do cruzamento Bellevue Way × NE 8th, de quadros que não estão no dataset\n"
        "nem nos 40 s do Case, ampliados para 640×640. Classes: " + ", ".join(CLASSES) + ".\n\n"
        "Gabarito em `gabarito/`: caixas do YOLO26n pré-treinado e classes conferidas à vista, em formato YOLO\n"
        "(com `classes.txt`, que o makesense.ai importa) e desenhado (`*_gabarito.png`).\n\n"
        "| Imagem | Origem | " + " | ".join(c.capitalize() for c in CLASSES) + " |\n|---|---|" + "---|" * len(CLASSES) + "\n"
        + "\n".join(resumo) + "\n",
        encoding="utf-8",
    )
    print("\n".join(resumo))
    print(f"→ {pasta}")


# Desafio "Que veículo é esse?" -----------------------------------------------------
# Os recortes ficam em oficina_vc/fine_tuning.py: o notebook do fine-tuning refaz o desafio ao vivo.
from oficina_vc.fine_tuning import DESAFIO  # noqa: E402


def _quadro_do_trecho(n):
    video = PUBLICO / "bellevue-ne8th_2762-2802s.mp4"
    saida = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(video), "-vf", f"select=eq(n\\,{n})", "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:"],
        capture_output=True, check=True,
    ).stdout
    return np.frombuffer(saida, np.uint8).reshape(720, 1280, 3).copy()


def _resposta(modelo, quadro, caixa):
    """'classe (0,87)' da detecção que mais se sobrepõe à caixa, como no notebook (oficina_vc.fine_tuning.resposta)."""
    from oficina_vc.fine_tuning import resposta

    classe, confianca = resposta(modelo, quadro, caixa)
    return classe if confianca is None else f"{classe} ({confianca:.2f})".replace(".", ",")


def desafio(opcoes):
    from ultralytics import YOLO

    pre = YOLO(str(PUBLICO / "yolo26n.pt"))
    especialista = YOLO(str(PUBLICO / "yolo26n_tipos_de_veiculo.pt"))
    pasta = DESTINO / "desafio-que-veiculo"
    pasta.mkdir(parents=True, exist_ok=True)
    quadros = {n: _quadro_do_trecho(n) for n in {d[1] for d in DESAFIO}}
    linhas = []
    for rotulo, n, (x1, y1, x2, y2), certa in DESAFIO:
        m = int(0.25 * max(x2 - x1, y2 - y1))
        recorte = quadros[n][max(0, y1 - m) : y2 + m, max(0, x1 - m) : x2 + m]
        escala = 600 / max(recorte.shape[:2])
        cv2.imwrite(str(pasta / f"veiculo_{rotulo}.png"),
                    cv2.resize(recorte, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC))
        p, e = _resposta(pre, quadros[n], (x1, y1, x2, y2)), _resposta(especialista, quadros[n], (x1, y1, x2, y2))
        linhas.append(f"| {rotulo} | quadro {n} | {certa} | {p} | {e} |")
    texto = (
        "# Desafio \"Que veículo é esse?\"\n\n"
        "Recortes do trecho de 40 s do Case (`bellevue-ne8th_2762-2802s.mp4`): SUV, Van, Carro e Ônibus da Cena-âncora\n"
        "(quadro 623); a Pickup (quadro 180) e o caminhão-plataforma (quadro 300) de outros quadros, porque a\n"
        "Cena-âncora não tem nenhum dos dois. O Ônibus (F) é opcional. Ampliados para 600 px.\n"
        "As respostas dos modelos vêm do quadro inteiro (imgsz 960, como no Case): a detecção que mais se sobrepõe\n"
        "ao veículo, com a confiança.\n\n"
        "| Recorte | Origem | Resposta certa | Pré-treinado | Especialista |\n|---|---|---|---|---|\n"
        + "\n".join(linhas) + "\n"
    )
    (pasta / "gabarito.md").write_text(texto, encoding="utf-8")
    print(texto)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="etapa", required=True)
    sub.add_parser("ao-vivo")
    sub.add_parser("desafio")
    opcoes = p.parse_args()
    {"ao-vivo": ao_vivo, "desafio": desafio}[opcoes.etapa](opcoes)


if __name__ == "__main__":
    main()
