"""Soma quadros do MIO-TCD ao treino do dataset de Tipos de veículo (só para o modelo treinado antes).

O MIO-TCD (localização) tem 137.743 quadros de câmeras de trânsito do Canadá e dos EUA,
anotados à mão em 11 classes (Luo et al., "MIO-TCD: A new benchmark dataset for vehicle
classification and localization", IEEE TIP, 2018; CC BY-NC-SA 4.0;
https://tcd.miovision.com/challenge/dataset.html). Aqui:

- as classes viram as 5 do dataset de Bellevue: car → carro, pickup_truck → pickup,
  work_van → van, single_unit_truck e articulated_truck → caminhão, bus → ônibus;
  pedestres, bicicletas, motos e veículos não motorizados ficam sem caixa (o dataset de
  Bellevue também não os anota);
- quadros com `motorized_vehicle` (veículo pequeno demais para dizer o tipo) ficam de
  fora: sem caixa, eles ensinariam o modelo a ignorar veículos;
- são sorteados `--n` quadros, com mais chance para os que têm classes raras (peso de
  cada caixa = 1 / frequência da classe);
- os quadros entram só em `images/train` (nome `mio_<id>.jpg`); validação e teste
  continuam só de Bellevue, para a prova seguir sendo nas fotos do cruzamento.

Grava um zip com a mesma estrutura do `dataset_tipos_de_veiculo.zip`, para o
`treinar_tipos_de_veiculo.py treinar --zip ...`.

Uso:

    python scripts/mio_tcd.py --tar ~/.cache/oficina-vc/mio-tcd/MIO-TCD-Localization.tar \\
        --saida ~/.cache/oficina-vc/mio-tcd/dataset_tipos_de_veiculo_mio.zip --n 1500
"""

import argparse
import csv
import random
import shutil
import subprocess
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

RAIZ = Path(__file__).resolve().parent.parent
BELLEVUE = RAIZ.parent / "drive" / "oficina-vc-publico" / "dataset_tipos_de_veiculo.zip"

PARA_NOSSAS = {"car": 0, "pickup_truck": 1, "work_van": 2, "single_unit_truck": 3, "articulated_truck": 3, "bus": 4}
SEM_CAIXA = {"pedestrian", "bicycle", "motorcycle", "non-motorized_vehicle"}
DESCARTA_QUADRO = "motorized_vehicle"
NOMES = ["carro", "pickup", "van", "caminhão", "ônibus"]


def extrair(tar, pasta):
    """Extrai o tar uma vez só e devolve (csv do gabarito de treino, pasta das imagens de treino)."""
    if not any(pasta.rglob("gt_train.csv")):
        pasta.mkdir(parents=True, exist_ok=True)
        print(f"Extraindo {tar.name} (uma vez só)...", flush=True)
        subprocess.run(["tar", "-xf", str(tar), "-C", str(pasta)], check=True)
    gabarito = next(pasta.rglob("gt_train.csv"))
    return gabarito, gabarito.parent / "train"


def ler_gabarito(gabarito):
    """{id do quadro: [(rótulo, x1, y1, x2, y2)]}."""
    quadros = defaultdict(list)
    with open(gabarito, newline="") as f:
        for linha in csv.reader(f):
            if len(linha) < 6 or not linha[2].strip().lstrip("-").isdigit():
                continue  # cabeçalho, se houver
            quadros[linha[0]].append((linha[1], *map(float, linha[2:6])))
    return quadros


def sortear(quadros, n, semente):
    """Quadros válidos sorteados com peso para as classes raras."""
    validos = {}
    for qid, caixas in quadros.items():
        if any(r == DESCARTA_QUADRO for r, *_ in caixas):
            continue
        nossas = [(PARA_NOSSAS[r], *c) for r, *c in caixas if r in PARA_NOSSAS]
        if nossas and all(r in PARA_NOSSAS or r in SEM_CAIXA for r, *_ in caixas):
            validos[qid] = nossas
    freq = Counter(c for caixas in validos.values() for c, *_ in caixas)
    pesos = {qid: sum(1 / freq[c] for c, *_ in caixas) for qid, caixas in validos.items()}
    rng = random.Random(semente)
    # sorteio ponderado sem reposição (Efraimidis–Spirakis)
    chaves = sorted(validos, key=lambda q: rng.random() ** (1 / pesos[q]), reverse=True)
    return {q: validos[q] for q in chaves[:n]}, len(validos)


def montar(escolhidos, imagens, bellevue, saida):
    trabalho = saida.parent / "montagem"
    if trabalho.exists():
        shutil.rmtree(trabalho)
    with zipfile.ZipFile(bellevue) as z:
        z.extractall(trabalho)
    raiz = trabalho / "tipos_de_veiculo"
    contagem = Counter()
    for qid, caixas in escolhidos.items():
        origem = imagens / f"{qid}.jpg"
        destino = raiz / "images" / "train" / f"mio_{qid}.jpg"
        shutil.copy(origem, destino)
        largura, altura = Image.open(origem).size
        linhas = []
        for c, x1, y1, x2, y2 in caixas:
            x1, x2 = max(0, x1), min(largura, x2)
            y1, y2 = max(0, y1), min(altura, y2)
            linhas.append(f"{c} {(x1 + x2) / 2 / largura:.6f} {(y1 + y2) / 2 / altura:.6f} "
                          f"{(x2 - x1) / largura:.6f} {(y2 - y1) / altura:.6f}")
            contagem[c] += 1
        (raiz / "labels" / "train" / f"mio_{qid}.txt").write_text("\n".join(linhas) + "\n")
    yaml = raiz / "data.yaml"
    yaml.write_text(yaml.read_text(encoding="utf-8").replace(
        "# Caixas do YOLO26n", "# Treino com quadros do MIO-TCD (Luo et al., IEEE TIP 2018; CC BY-NC-SA 4.0).\n# Caixas do YOLO26n"), encoding="utf-8")
    saida.unlink(missing_ok=True)
    with zipfile.ZipFile(saida, "w", zipfile.ZIP_STORED) as z:
        for arquivo in sorted(raiz.rglob("*")):
            if arquivo.is_file():
                z.write(arquivo, arquivo.relative_to(trabalho))
    shutil.rmtree(trabalho)
    return contagem


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tar", type=Path, required=True)
    p.add_argument("--saida", type=Path, required=True)
    p.add_argument("--bellevue", type=Path, default=BELLEVUE)
    p.add_argument("--n", type=int, default=1500)
    p.add_argument("--semente", type=int, default=0)
    a = p.parse_args()

    gabarito, imagens = extrair(a.tar, a.tar.parent / "extraido")
    quadros = ler_gabarito(gabarito)
    escolhidos, validos = sortear(quadros, a.n, a.semente)
    print(f"{len(quadros)} quadros no gabarito, {validos} válidos, {len(escolhidos)} sorteados")
    contagem = montar(escolhidos, imagens, a.bellevue, a.saida)
    print("Caixas do MIO-TCD no treino: " + ", ".join(f"{NOMES[c]} {contagem[c]}" for c in range(5)))
    print(f"{a.saida} ({a.saida.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    sys.exit(main())
