"""Monta o dataset de Tipos de veículo (Carro, Pickup, Van, Caminhão, Ônibus) do Fine-tuning demonstrativo.

O dataset sai de 366 quadros de outros arquivos horários do cruzamento Bellevue Way ×
NE 8th (nunca o das 12:08, que contém os 40 s do Case), em duas rodadas: 204 sorteados
(com reforço de truck/bus) e 162 ricos em classes raras, apontados pelo especialista
treinado na 1ª rodada. A Anotação automática:

- as **caixas** vêm do YOLO26n pré-treinado, na união das detecções com imgsz 640 e 960
  (no 960 ele quebra veículos grandes em pedaços; no 640 perde os pequenos do fundo);
- a **classe** de cada caixa vem de um Modelo fundacional local, que recebe o recorte do
  veículo com duas perguntas fechadas (ver `anotar_com_moondream.py`);
- as caixas ruins que o pré-treinado dá (carro com a sombra longa, grupo de carros, pedaço
  de um veículo grande) saem por `caixas_descartadas.csv` ou pela classe "x" da revisão;
- o instrutor revisa só as classes, na folha de contato `revisao.html`.

Etapas (cada uma retoma de onde parou):

    amostrar    baixa ~420 quadros-chave candidatos por arquivo horário, só os bytes de
                cada quadro (ver quadros_remotos.py), detecta os veículos e sorteia os
                quadros do dataset, com reforço de quadros que o pré-treinado marca com
                "truck"/"bus" e intervalo mínimo entre eles
    reforcar    2ª rodada: até 30 quadros por arquivo horário ricos em classes raras,
                apontados pelo especialista da 1ª rodada, entre os candidatos que ficaram
                de fora (grava rodada2.csv com as caixas novas)
    recortar    grava os recortes de cada caixa, para o Modelo fundacional e a revisão
    folhas      a folha de contato `revisao.html` (recortes por classe) para a revisão
    exportar    o dataset em formato YOLO (zip), com a divisão por arquivo horário

Uso (no Mac do instrutor, num ambiente com ultralytics e supervision; ffmpeg no PATH):

    python scripts/dataset_tipos_de_veiculo.py amostrar
    python scripts/dataset_tipos_de_veiculo.py recortar
    python scripts/anotar_com_moondream.py ...            # classes por Modelo fundacional
    python scripts/dataset_tipos_de_veiculo.py folhas --classes classes_moondream3.csv
    # o instrutor abre revisao.html, corrige e exporta classes_revisadas.csv
    python scripts/treinar_tipos_de_veiculo.py treinar    # especialista da 1ª rodada
    python scripts/dataset_tipos_de_veiculo.py reforcar
    python scripts/dataset_tipos_de_veiculo.py recortar
    python scripts/anotar_com_moondream.py --lista rodada2.csv ...
    python scripts/dataset_tipos_de_veiculo.py folhas --classes classes_moondream3_rodada2.csv \
        --so rodada2.csv --saida revisao_rodada2.html
    python scripts/dataset_tipos_de_veiculo.py exportar --classes classes_revisadas.csv classes_revisadas_rodada2.csv

Fica fora do `requirements.txt` do Colab: é preparação, feita uma vez antes da Oficina.
"""

import argparse
import csv
import json
import random
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quadros_remotos import MP4Remoto

RAIZ = Path(__file__).resolve().parent.parent
TRABALHO = RAIZ.parent / "drive" / "oficina-vc-instrutor" / "tipos-de-veiculo"
CACHE = Path.home() / ".cache" / "oficina-vc" / "candidatos"  # ~500 MB, fora do Drive
PUBLICO = RAIZ.parent / "drive" / "oficina-vc-publico"
PESOS = PUBLICO / "yolo26n.pt"

URL_DRIVE = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"

# Arquivos horários diurnos do NE 8th (pasta 17bn7l7Qm5s-r5DYoFQPhviFZ0jWY9qk5 da prefeitura),
# com a divisão do dataset por arquivo: o teste nunca vê um quadro da mesma hora do treino.
# O das 12:08 (2017-09-11_12-08-31), que contém os 40 s do Case, fica de fora.
HORARIOS = {
    "0910-1808": ("11zD3GffsvHar97p5Zofa7Pc8z0JRrC6k", "train"),  # domingo, fim de tarde
    "0911-0708": ("12bA-fApjQPRB-l0Pd-57-TtOLAOzQkLJ", "train"),  # segunda, manhã cedo
    "0911-0908": ("1BWwZHoqBiouS8aWyKtQZlS91r7hmLMZP", "val"),
    "0911-1108": ("1AhkpHfIbqwArljS7MkZzeJGHnhbtk5b9", "train"),
    "0911-1408": ("1BkcUT9o_mdc93cFi27WIdRjOMK2Dv2nH", "test"),
    "0911-1608": ("1BP8QDOFOGDBSQifxK2zkLM5prUjLRpfv", "train"),
}

CLASSES = ["carro", "pickup", "van", "caminhão", "ônibus"]  # ordem dos ids no data.yaml
DESCARTE = "x"  # na revisão: não é um veículo inteiro (pedaço, sombra, grupo, duplicata)
COCO_VEICULOS = [2, 3, 5, 7]  # car, motorcycle, bus, truck
COCO_NOMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

CANDIDATO_A_CADA = 8  # quadros-chave (~8,5 s)
QUADROS_POR_HORARIO = 34  # 6 × 34 = 204
MAXIMO_COM_CAMINHAO = 14  # por arquivo horário, sorteados entre os que o pré-treinado marca com truck/bus
INTERVALO_MINIMO_S = 25
CONFIANCA_CAIXA = 0.30  # caixas do dataset
CONFIANCA_CAMINHAO = 0.30  # para o reforço de quadros com truck/bus
MINIMO_VEICULOS = 3  # quadros quase vazios não entram no sorteio
SEMENTE = 40


# Detecção --------------------------------------------------------------------
def _iou_e_contencao(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9), inter / (min(area_a, area_b) + 1e-9)


def unir_caixas(caixas, iou=0.5, contida=0.8):
    """NMS entre classes e escalas: fica a mais confiante; some quem estiver ≥80% dentro dela.

    A contenção apaga os pedaços de um veículo grande (o 960 às vezes dá uma caixa para a
    cabine e outra para a carroceria), mas só quando a caixa grande é mais confiante.
    """
    ficam = []
    for c in sorted(caixas, key=lambda c: -c["conf"]):
        sobra = True
        for f in ficam:  # f é mais confiante que c
            r_iou, r_contida = _iou_e_contencao(c["xyxy"], f["xyxy"])
            if r_iou >= iou or (r_contida >= contida and _area(c["xyxy"]) < _area(f["xyxy"])):
                sobra = False
                break
        if sobra:
            ficam.append(c)
    return ficam


def _area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


class Detector:
    def __init__(self, dispositivo="mps"):
        from ultralytics import YOLO

        self.modelo = YOLO(str(PESOS))
        self.dispositivo = dispositivo

    def __call__(self, quadro):
        """{'caixas': [...] unidas, 'caminhao': maior confiança de truck/bus em qualquer escala}."""
        brutas, caminhao = [], 0.0
        for imgsz in (640, 960):
            r = self.modelo(quadro, imgsz=imgsz, conf=0.1, classes=COCO_VEICULOS, device=self.dispositivo, verbose=False)[0]
            for b, c, p in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                c = int(c)
                if c in (5, 7):
                    caminhao = max(caminhao, p)
                brutas.append({"xyxy": [round(v, 1) for v in b], "conf": round(p, 3), "coco": COCO_NOMES[c], "imgsz": imgsz})
        unidas = unir_caixas([c for c in brutas if c["conf"] >= CONFIANCA_CAIXA])
        return {"caixas": unidas, "caminhao": round(caminhao, 3)}


# Amostragem -------------------------------------------------------------------
def baixar_candidatos(nome, id_drive, detector, conexoes=10):
    pasta = CACHE / nome
    pasta.mkdir(parents=True, exist_ok=True)
    indice = pasta / "deteccoes.json"
    deteccoes = json.loads(indice.read_text()) if indice.exists() else {}
    arquivo = MP4Remoto(URL_DRIVE.format(id_drive))
    chaves = [k for k in range(0, len(arquivo.chaves), CANDIDATO_A_CADA) if f"{arquivo.tempo_da_chave(k):07.1f}" not in deteccoes]
    print(f"{nome}: {len(arquivo.chaves)} quadros-chave, {len(chaves)} candidatos a baixar", flush=True)
    with ThreadPoolExecutor(conexoes) as fila:
        futuros = [fila.submit(arquivo.quadro_chave, k) for k in chaves]
        for n, futuro in enumerate(as_completed(futuros), 1):
            t, quadro = futuro.result()
            chave = f"{t:07.1f}"
            cv2.imwrite(str(pasta / f"{chave}.jpg"), quadro, [cv2.IMWRITE_JPEG_QUALITY, 95])
            deteccoes[chave] = detector(quadro)
            if n % 50 == 0 or n == len(futuros):
                indice.write_text(json.dumps(deteccoes))
                print(f"  {n}/{len(futuros)}", flush=True)
    indice.write_text(json.dumps(deteccoes))
    return deteccoes


def sortear(deteccoes, rng):
    """Quadros de um arquivo horário: primeiro os com truck/bus, depois sorteio, sempre com intervalo mínimo."""
    tempos = sorted(deteccoes, key=float)
    elegiveis = [t for t in tempos if len(deteccoes[t]["caixas"]) >= MINIMO_VEICULOS]
    escolhidos, motivos = [], {}

    def cabe(t):
        return all(abs(float(t) - float(e)) >= INTERVALO_MINIMO_S for e in escolhidos)

    com_caminhao = [t for t in elegiveis if deteccoes[t]["caminhao"] >= CONFIANCA_CAMINHAO]
    rng.shuffle(com_caminhao)
    for t in com_caminhao:
        if len(escolhidos) >= MAXIMO_COM_CAMINHAO:
            break
        if cabe(t):
            escolhidos.append(t)
            motivos[t] = "truck"
    resto = [t for t in elegiveis if t not in motivos]
    rng.shuffle(resto)
    for t in resto:
        if len(escolhidos) >= QUADROS_POR_HORARIO:
            break
        if cabe(t):
            escolhidos.append(t)
            motivos[t] = "sorteio"
    return sorted(escolhidos, key=float), motivos


def amostrar(opcoes):
    detector = Detector(opcoes.dispositivo)
    rng = random.Random(SEMENTE)
    pasta_quadros = TRABALHO / "quadros"
    pasta_quadros.mkdir(parents=True, exist_ok=True)
    linhas_amostra, linhas_caixas = [], []
    for nome, (id_drive, divisao) in HORARIOS.items():
        deteccoes = baixar_candidatos(nome, id_drive, detector, opcoes.conexoes)
        escolhidos, motivos = sortear(deteccoes, rng)
        for t in escolhidos:
            quadro_id = f"{nome}_{t}"
            destino = pasta_quadros / f"{quadro_id}.jpg"
            if not destino.exists():
                destino.write_bytes((CACHE / nome / f"{t}.jpg").read_bytes())
            linhas_amostra.append({"quadro": quadro_id, "horario": nome, "t": t, "divisao": divisao, "motivo": motivos[t]})
            for k, c in enumerate(deteccoes[t]["caixas"]):
                x1, y1, x2, y2 = c["xyxy"]
                linhas_caixas.append({
                    "caixa": f"{quadro_id}_{k:02d}", "quadro": quadro_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "conf": c["conf"], "coco": c["coco"], "imgsz": c["imgsz"],
                })
        print(f"{nome}: {len(escolhidos)} quadros ({sum(m == 'truck' for m in motivos.values())} pelo reforço de truck)")
    _gravar_csv(TRABALHO / "amostra.csv", linhas_amostra)
    _gravar_csv(TRABALHO / "caixas.csv", linhas_caixas)
    print(f"\n{len(linhas_amostra)} quadros e {len(linhas_caixas)} caixas em {TRABALHO}")


# Segunda rodada: reforço das classes raras ------------------------------------------
QUADROS_REFORCO = 30  # por arquivo horário
# Quadros da anotação ao vivo (materiais_fine_tuning.py): nunca entram no dataset.
FORA_DO_DATASET = {"0911-1408_00895.0", "0911-1108_02705.7", "0911-1408_02858.1", "0911-1108_01749.7", "0911-1608_00392.6"}


def _contem_mais_confiante(caixas):
    """Índices das caixas que contêm (≥80%) uma caixa mais confiante 1,5× menor: carro com a
    sombra longa ou grupo de carros, quase sempre (na 1ª rodada, 25 de 31 casos)."""
    suspeitas = set()
    for i, b in enumerate(caixas):
        for j, a in enumerate(caixas):
            if i != j and a["conf"] > b["conf"] and _iou_e_contencao(a["xyxy"], b["xyxy"])[1] >= 0.8 \
                    and _area(b["xyxy"]) > 1.5 * _area(a["xyxy"]):
                suspeitas.add(i)
    return suspeitas


def reforcar(opcoes):
    """Acrescenta quadros ricos em Pickup, Van, Caminhão e Ônibus, apontados pelo especialista
    da 1ª rodada e pelo truck/bus do pré-treinado, entre os candidatos que ficaram de fora."""
    from ultralytics import YOLO

    especialista = YOLO(str(opcoes.especialista))
    amostra = _ler_csv(TRABALHO / "amostra.csv")
    caixas = _ler_csv(TRABALHO / "caixas.csv")
    descartes = _ler_csv(TRABALHO / "caixas_descartadas.csv")
    ja = {q["quadro"] for q in amostra}
    novas_caixas, novos_quadros = [], []
    for nome, (_, divisao) in HORARIOS.items():
        deteccoes = json.loads((CACHE / nome / "deteccoes.json").read_text())
        usados = [float(q["t"]) for q in amostra if q["horario"] == nome]
        pontos = {}
        for t in sorted(deteccoes, key=float):
            if f"{nome}_{t}" in ja | FORA_DO_DATASET or len(deteccoes[t]["caixas"]) < MINIMO_VEICULOS:
                continue
            if any(abs(float(t) - u) < INTERVALO_MINIMO_S for u in usados):
                continue
            r = especialista(str(CACHE / nome / f"{t}.jpg"), imgsz=960, conf=0.3, device=opcoes.dispositivo, verbose=False)[0]
            raras = sum(
                1 for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist())
                if especialista.names[int(c)] != "carro" and max(b[2] - b[0], b[3] - b[1]) >= 40
            )
            pontos[t] = raras + (deteccoes[t]["caminhao"] >= CONFIANCA_CAMINHAO)
        escolhidos = []
        for t in sorted(pontos, key=lambda t: -pontos[t]):
            if len(escolhidos) >= QUADROS_REFORCO or pontos[t] < 1:
                break
            if all(abs(float(t) - float(e)) >= INTERVALO_MINIMO_S for e in escolhidos):
                escolhidos.append(t)
        for t in sorted(escolhidos, key=float):
            quadro_id = f"{nome}_{t}"
            (TRABALHO / "quadros" / f"{quadro_id}.jpg").write_bytes((CACHE / nome / f"{t}.jpg").read_bytes())
            novos_quadros.append({"quadro": quadro_id, "horario": nome, "t": t, "divisao": divisao, "motivo": "reforço"})
            cx = deteccoes[t]["caixas"]
            suspeitas = _contem_mais_confiante(cx)
            for k, c in enumerate(cx):
                x1, y1, x2, y2 = c["xyxy"]
                caixa = f"{quadro_id}_{k:02d}"
                novas_caixas.append({
                    "caixa": caixa, "quadro": quadro_id, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "conf": c["conf"], "coco": c["coco"], "imgsz": c["imgsz"],
                })
                if k in suspeitas:
                    descartes.append({"caixa": caixa, "motivo": "automático: contém uma caixa mais confiante"})
        print(f"{nome}: +{len(escolhidos)} quadros ({sum(pontos[t] for t in escolhidos)} pontos de classe rara)", flush=True)
    _gravar_csv(TRABALHO / "amostra.csv", amostra + novos_quadros)
    _gravar_csv(TRABALHO / "caixas.csv", caixas + novas_caixas)
    _gravar_csv(TRABALHO / "caixas_descartadas.csv", descartes)
    _gravar_csv(TRABALHO / "rodada2.csv", [{"caixa": c["caixa"]} for c in novas_caixas])
    print(f"\n+{len(novos_quadros)} quadros e +{len(novas_caixas)} caixas (lista em rodada2.csv)")


# Recortes ----------------------------------------------------------------------
def recorte(quadro, caixa, margem=0.15, minimo=8):
    """Recorte com margem em volta da caixa (o Modelo fundacional precisa de um pouco de contexto)."""
    x1, y1, x2, y2 = (float(caixa[k]) for k in ("x1", "y1", "x2", "y2"))
    m = margem * max(x2 - x1, y2 - y1) + minimo
    a, b = int(max(0, x1 - m)), int(max(0, y1 - m))
    c, d = int(min(quadro.shape[1], x2 + m)), int(min(quadro.shape[0], y2 + m))
    return quadro[b:d, a:c]


def recortar(opcoes):
    pasta = TRABALHO / "recortes"
    pasta.mkdir(exist_ok=True)
    caixas = _ler_csv(TRABALHO / "caixas.csv")
    quadros = {}
    for c in caixas:
        destino = pasta / f"{c['caixa']}.png"
        if destino.exists():
            continue
        if c["quadro"] not in quadros:
            quadros = {c["quadro"]: cv2.imread(str(TRABALHO / "quadros" / f"{c['quadro']}.jpg"))}
        cv2.imwrite(str(destino), recorte(quadros[c["quadro"]], c))
    print(f"{len(caixas)} recortes em {pasta}")


# Folha de contato (revisão das classes) ---------------------------------------------
def folhas(opcoes):
    """Gera `revisao.html`: os recortes agrupados por classe, para o instrutor corrigir à vista.

    A página abre direto do disco (os recortes vêm de `recortes/`, ao lado). Um clique
    (ou uma tecla com o mouse em cima) troca a classe; o progresso fica no navegador; o
    botão "Exportar" baixa `classes_revisadas.csv`, que vai para `exportar --classes`.
    """
    sys.path.insert(0, str(RAIZ))
    from oficina_vc import estilo

    classes = _classes_finais(opcoes.classes)
    respostas = {linha["caixa"]: linha for caminho in opcoes.classes for linha in _ler_csv(caminho)}
    caixas = _ler_csv(TRABALHO / "caixas.csv")
    if opcoes.so:
        so = {linha["caixa"] for linha in _ler_csv(opcoes.so)}
        caixas = [c for c in caixas if c["caixa"] in so]
    itens = []
    for c in caixas:
        b = [float(c[k]) for k in ("x1", "y1", "x2", "y2")]
        r = respostas.get(c["caixa"], {})
        itens.append({
            "id": c["caixa"], "c": classes.get(c["caixa"], "?"), "lado": round(max(b[2] - b[0], b[3] - b[1])),
            "md": f"{r.get('curta', '')} · {r.get('fina', '')}".strip(" ·"), "coco": c["coco"],
        })
    cores = {c: estilo.cores_da_classe(c)[0] for c in CLASSES}
    cores[DESCARTE] = estilo.CINZA_CLARO
    cores["?"] = estilo.CINZA_CLARO
    html = (Path(__file__).with_name("revisao_modelo.html").read_text(encoding="utf-8")
            .replace("__ITENS__", json.dumps(itens, ensure_ascii=False))
            .replace("__CORES__", json.dumps(cores, ensure_ascii=False))
            .replace("__ARQUIVO__", opcoes.exporta))
    destino = TRABALHO / (opcoes.saida or "revisao.html")
    destino.write_text(html, encoding="utf-8")
    contagem = {c: sum(1 for i in itens if i["c"] == c) for c in [*CLASSES, DESCARTE, "?"]}
    print(f"{destino}\n" + ", ".join(f"{c} {n}" for c, n in contagem.items()))


# Exportação -------------------------------------------------------------------------
def exportar(opcoes):
    classes = _classes_finais(opcoes.classes)
    caixas = _ler_csv(TRABALHO / "caixas.csv")
    amostra = _ler_csv(TRABALHO / "amostra.csv")
    por_quadro = {}
    for c in caixas:
        por_quadro.setdefault(c["quadro"], []).append(c)
    destino = Path(opcoes.destino)
    zip_ = destino / "dataset_tipos_de_veiculo.zip"
    contagem = {d: {c: 0 for c in CLASSES} for d in ("train", "val", "test")}
    quadros_por_divisao = {d: 0 for d in contagem}
    with zipfile.ZipFile(zip_, "w", zipfile.ZIP_STORED) as z:
        for q in amostra:
            d = q["divisao"]
            linhas = []
            for c in por_quadro.get(q["quadro"], []):
                classe = classes.get(c["caixa"])
                if classe not in CLASSES:  # descartada ou "x"
                    continue
                x1, y1, x2, y2 = (float(c[k]) for k in ("x1", "y1", "x2", "y2"))
                linhas.append(f"{CLASSES.index(classe)} {(x1 + x2) / 2560:.6f} {(y1 + y2) / 1440:.6f} {(x2 - x1) / 1280:.6f} {(y2 - y1) / 720:.6f}")
                contagem[d][classe] += 1
            quadros_por_divisao[d] += 1
            z.write(TRABALHO / "quadros" / f"{q['quadro']}.jpg", f"tipos_de_veiculo/images/{d}/{q['quadro']}.jpg")
            z.writestr(f"tipos_de_veiculo/labels/{d}/{q['quadro']}.txt", "\n".join(linhas) + ("\n" if linhas else ""))
        nomes = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
        z.writestr(
            "tipos_de_veiculo/data.yaml",
            "# Tipos de veículo: cruzamento Bellevue Way × NE 8th (City of Bellevue Traffic Video Dataset, 2017)\n"
            "# Caixas do YOLO26n pré-treinado; classes por Modelo fundacional local, revisadas pelo instrutor.\n"
            f"path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{nomes}\n",
        )
        z.writestr("tipos_de_veiculo/LEIA-ME.md", _leia_me(contagem, quadros_por_divisao))
    print(f"{zip_} ({zip_.stat().st_size:_} bytes)")
    for d in contagem:
        print(f"  {d}: {quadros_por_divisao[d]} quadros, " + ", ".join(f"{c} {n}" for c, n in contagem[d].items()))
    total = {c: sum(contagem[d][c] for d in contagem) for c in CLASSES}
    print("  total: " + ", ".join(f"{c} {n}" for c, n in total.items()))


def _leia_me(contagem, quadros):
    linhas = [
        "# Dataset de Tipos de veículo",
        "",
        "Quadros do cruzamento Bellevue Way × NE 8th, do City of Bellevue Traffic Video Dataset",
        "(github.com/City-of-Bellevue/TrafficVideoDataset), \"made available for research purposes\".",
        "Nenhum quadro vem do arquivo horário das 12:08, que contém o trecho de 40 s do Case de tráfego.",
        "",
        "- Caixas: YOLO26n pré-treinado (Ultralytics), união das detecções com imgsz 640 e 960.",
        "- Classes: Modelo fundacional local sobre o recorte de cada caixa, revisadas pelo instrutor por folhas de contato.",
        "- Divisão por arquivo horário: treino 10/09 18h e 11/09 7h, 11h e 16h; validação 11/09 9h; teste 11/09 14h.",
        "",
        "| Divisão | Quadros | " + " | ".join(c.capitalize() for c in CLASSES) + " |",
        "|---|---|" + "---|" * len(CLASSES),
    ]
    for d in contagem:
        linhas.append(f"| {d} | {quadros[d]} | " + " | ".join(str(contagem[d][c]) for c in CLASSES) + " |")
    return "\n".join(linhas) + "\n"


def _classes_finais(caminhos):
    """{caixa: classe} juntando os CSVs em ordem (o último vence). Num CSV do Moondream 3
    (com a coluna `curta`), as caixas de caixas_descartadas.csv viram "x"; num CSV
    revisado, vale o que o instrutor deixou."""
    classes = {}
    descartes = TRABALHO / "caixas_descartadas.csv"
    descartadas = {linha["caixa"] for linha in _ler_csv(descartes)} if descartes.exists() else set()
    for caminho in caminhos:
        linhas = _ler_csv(caminho)
        do_moondream = bool(linhas) and "curta" in linhas[0]
        for linha in linhas:
            classes[linha["caixa"]] = DESCARTE if do_moondream and linha["caixa"] in descartadas else linha["classe"]
    return classes


# CSV --------------------------------------------------------------------------------
def _gravar_csv(caminho, linhas):
    caminho = Path(caminho)
    with caminho.open("w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(linhas[0]))
        escritor.writeheader()
        escritor.writerows(linhas)


def _ler_csv(caminho):
    with Path(caminho).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="etapa", required=True)
    a = sub.add_parser("amostrar")
    a.add_argument("--dispositivo", default="mps")
    a.add_argument("--conexoes", type=int, default=10)
    sub.add_parser("recortar")
    f = sub.add_parser("folhas")
    f.add_argument("--classes", type=Path, nargs="+", required=True, help="CSVs de classes; o último vence")
    f.add_argument("--so", type=Path, help="CSV com a coluna caixa: só essas na página")
    f.add_argument("--saida", help="nome da página (padrão: revisao.html)")
    f.add_argument("--exporta", default="classes_revisadas.csv", help="nome do CSV que a página exporta")
    r2 = sub.add_parser("reforcar")
    r2.add_argument("--especialista", type=Path, default=RAIZ.parent / "drive" / "oficina-vc-instrutor" / "modelos" / "yolo26n_tipos_de_veiculo.pt")
    r2.add_argument("--dispositivo", default="mps")
    e = sub.add_parser("exportar")
    e.add_argument("--classes", type=Path, nargs="+", required=True, help="CSVs de classes; o último vence")
    e.add_argument("--destino", type=Path, default=PUBLICO)
    opcoes = p.parse_args()
    {"amostrar": amostrar, "reforcar": reforcar, "recortar": recortar, "folhas": folhas, "exportar": exportar}[opcoes.etapa](opcoes)


if __name__ == "__main__":
    main()
