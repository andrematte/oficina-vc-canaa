"""Treina o Modelo especialista de Tipos de veículo e gera o render de reserva nos 40 s do Case.

É o "modelo treinado antes" do Fine-tuning demonstrativo: o mesmo YOLO26n do Case,
ajustado no dataset de `dataset_tipos_de_veiculo.py` (Carro, Pickup, Van, Caminhão,
Ônibus). Na Oficina, o treino ao vivo roda poucas épocas; este roda até convergir.

Etapas:

    treinar    YOLO26n pré-treinado → especialista (mps), com imgsz 960 como o Case;
               grava `yolo26n_tipos_de_veiculo.pt` e o mAP50 do teste em
               `yolo26n_tipos_de_veiculo.json`, na pasta do instrutor
    render     processa os 40 s do Case com o especialista (mesmas Zonas, rastreador,
               confiança, imgsz e passo do `oficina_vc/case.py`), com o NMS entre
               classes e o voto de classe por rastro do `processar_video`: a Matriz de
               movimentos sai por Tipo de veículo

Uso (no Mac do instrutor, no ambiente do repositório):

    python scripts/treinar_tipos_de_veiculo.py treinar --epocas 120
    python scripts/treinar_tipos_de_veiculo.py render
"""

import argparse
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

DRIVE = RAIZ.parent / "drive"
ZIP = DRIVE / "oficina-vc-publico" / "dataset_tipos_de_veiculo.zip"
MODELOS = DRIVE / "oficina-vc-instrutor" / "modelos"
PESOS = MODELOS / "yolo26n_tipos_de_veiculo.pt"
TREINOS = Path.home() / ".cache" / "oficina-vc" / "treinos"


def _dataset(zip_=ZIP):
    """Descompacta o zip num diretório de trabalho e devolve o data.yaml com caminho absoluto."""
    pasta = TREINOS / "dataset"
    if pasta.exists():
        shutil.rmtree(pasta)
    with zipfile.ZipFile(zip_) as z:
        z.extractall(pasta)
    raiz = pasta / "tipos_de_veiculo"
    yaml = raiz / "data.yaml"
    yaml.write_text(yaml.read_text(encoding="utf-8").replace("path: .", f"path: {raiz}"), encoding="utf-8")
    return yaml


def treinar(opcoes):
    from ultralytics import YOLO

    from oficina_vc import case

    yaml = _dataset(opcoes.zip)
    inicio = time.time()
    modelo = YOLO(str(DRIVE / "oficina-vc-publico" / "yolo26n.pt"))
    modelo.train(
        data=str(yaml), epochs=opcoes.epocas, imgsz=case.IMGSZ, batch=opcoes.lote, device=opcoes.dispositivo,
        project=str(TREINOS), name="especialista", exist_ok=True, seed=0, patience=opcoes.paciencia,
        workers=0, plots=True, verbose=False,
    )
    minutos = (time.time() - inicio) / 60
    melhor = TREINOS / "especialista" / "weights" / "best.pt"
    pesos = opcoes.pesos
    pesos.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(melhor, pesos)

    especialista = YOLO(str(pesos))
    metricas = {"epocas_pedidas": opcoes.epocas, "minutos_de_treino": round(minutos, 1), "imgsz": case.IMGSZ}
    for divisao in ("val", "test"):
        r = especialista.val(data=str(yaml), split=divisao, imgsz=case.IMGSZ, device=opcoes.dispositivo,
                             project=str(TREINOS), name=f"aval_{divisao}", exist_ok=True, plots=True, verbose=False)
        por_classe = {especialista.names[int(c)]: round(float(r.box.ap50[k]), 3) for k, c in enumerate(r.box.ap_class_index)}
        metricas[divisao] = {"mAP50": round(float(r.box.map50), 3), "mAP50-95": round(float(r.box.map), 3), "mAP50_por_classe": por_classe}
    pesos.with_suffix(".json").write_text(json.dumps(metricas, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(metricas, ensure_ascii=False, indent=1))
    print(f"\n{pesos} ({pesos.stat().st_size:_} bytes), {minutos:.1f} min")


def render(opcoes):
    from ultralytics import YOLO

    from oficina_vc import case, dados, video

    especialista = YOLO(str(opcoes.pesos))
    mp4 = opcoes.saida
    inicio = time.time()
    resultado = video.processar_video(
        dados.caminho(case.VIDEO),
        especialista,
        case.criar_rastreador(),
        case.criar_zonas(),
        confianca=case.CONFIANCA,
        classes=None,  # todos os Tipos de veículo
        imgsz=case.IMGSZ,
        passo=case.PASSO,
        saida=mp4,
        dispositivo=opcoes.dispositivo,
        limite_quadros=opcoes.limite,
    )
    print()
    resultado.resumo()
    print(f"\nProcessado em {time.time() - inicio:.0f} s ({opcoes.dispositivo}) → {mp4}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="etapa", required=True)
    t = sub.add_parser("treinar")
    t.add_argument("--epocas", type=int, default=120)
    t.add_argument("--lote", type=int, default=8)
    t.add_argument("--paciencia", type=int, default=40)
    t.add_argument("--dispositivo", default="mps")
    t.add_argument("--zip", type=Path, default=ZIP)
    t.add_argument("--pesos", type=Path, default=PESOS, help="onde gravar o especialista")
    r = sub.add_parser("render")
    r.add_argument("--dispositivo", default="cpu")
    r.add_argument("--pesos", type=Path, default=PESOS)
    r.add_argument("--saida", type=Path, default=MODELOS / "especialista_resultado.mp4")
    r.add_argument("--limite", type=int, help="só os primeiros N quadros (teste)")
    opcoes = p.parse_args()
    {"treinar": treinar, "render": render}[opcoes.etapa](opcoes)


if __name__ == "__main__":
    main()
