"""Gera o Resultado pronto do Case de tráfego.

O Resultado pronto é a saída do Case já processada antes da Oficina: o mesmo vídeo
de saída (1920×720, com o painel) e as mesmas Métricas de vídeo que cada participante
gera ao vivo no notebook `1_case_trafego.ipynb`. Quem travar, ou ficar para trás,
carrega este resultado e segue a aula.

Usa os mesmos valores do notebook, que ficam em `oficina_vc/case.py` (Zonas por
Aproximação, rastreador, classes, confiança, `imgsz` e `passo`), e grava:

- `case_resultado_pronto.mp4`: o vídeo de saída;
- `case_metricas.json`: as Métricas de vídeo, com a Matriz de movimentos.

Por fim, imprime o tempo de processamento e os tamanhos em bytes para a tabela
`ARQUIVOS` de `oficina_vc/dados.py`.

Uso (de dentro do repositório, com os arquivos da Oficina em `dados/`):

    python scripts/gerar_resultado_pronto.py --destino ../drive/oficina-vc-publico
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from oficina_vc import ambiente, case, dados, video


def main():
    argumentos = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    argumentos.add_argument("--destino", type=Path, default=ambiente.PASTA_SAIDAS,
                            help="pasta onde gravar o MP4 e o JSON (padrão: saidas/)")
    argumentos.add_argument("--dispositivo", default="cpu", help="cpu, mps ou cuda (padrão: cpu)")
    opcoes = argumentos.parse_args()

    from ultralytics import YOLO

    opcoes.destino.mkdir(parents=True, exist_ok=True)
    mp4 = opcoes.destino / "case_resultado_pronto.mp4"
    modelo = YOLO(str(dados.caminho("yolo26n.pt")))

    inicio = time.time()
    resultado = video.processar_video(
        dados.caminho(case.VIDEO),
        modelo,
        case.criar_rastreador(),
        case.criar_zonas(),
        confianca=case.CONFIANCA,
        classes=case.CLASSES,
        imgsz=case.IMGSZ,
        passo=case.PASSO,
        saida=mp4,
        dispositivo=opcoes.dispositivo,
    )
    segundos = time.time() - inicio
    json_ = resultado.salvar_metricas(opcoes.destino / "case_metricas.json")
    mp4.with_suffix(".json").unlink(missing_ok=True)  # o processar_video grava um .json com o nome do MP4

    print()
    resultado.resumo()
    print(f"\nProcessado em {segundos:.0f} s ({opcoes.dispositivo}).")
    print("\nTamanhos para oficina_vc/dados.py:")
    for caminho in (mp4, json_):
        print(f'    "{caminho.name}": {caminho.stat().st_size:_}')


if __name__ == "__main__":
    main()
