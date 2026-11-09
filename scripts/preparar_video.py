"""Prepara os arquivos do Case de tráfego a partir do City of Bellevue Traffic Video Dataset.

Gera:

- `Bellevue_Bellevue_NE8th__2017-09-11_12-08-31.mp4`: o arquivo horário de origem
  (60 min, 1280×720, 30 fps), fonte do instrutor, baixado sem recodificar;
- `bellevue-ne8th_2762-2802s.mp4`: o trecho de 40 s usado na Oficina (12:54:33–12:55:13
  no relógio sobreposto; 1200 quadros, 1280×720, 30 fps constantes, H.264);
- `Bellevue_Bellevue_NE8th__2017-09-11_14-08-31.mp4`: o arquivo horário das 14:08, a hora
  da prova do dataset de Tipos de veículo (957 MB);
- `bellevue-ne8th_1408_2335-2375s.mp4`: 40 s das 14:08 (14:47:26–14:48:06 no relógio),
  para a etapa 9 do notebook do fine-tuning: o especialista nunca treinou com quadros
  dessa hora, e nela passam vans e um caminhão que ele reconhece;
- `cena_ancora_bellevue_623.png`: o quadro 623 do trecho (20,77 s, 12:54:53), a
  Cena-âncora, sem perda;
- `yolo26n.pt`: os pesos pré-treinados, espelhados do release v8.4.0 da Ultralytics.

O vídeo vem do cruzamento Bellevue Way × NE 8th, do City of Bellevue Traffic Video
Dataset (github.com/City-of-Bellevue/TrafficVideoDataset). A base não tem licença
formal: o README diz só que ela foi "made available for research purposes". Ver
`LICENCA-VIDEO.md`.

Detalhes que o script resolve:

1. A pasta do cruzamento no Google Drive da prefeitura baixa sem login; o arquivo
   horário (986 MB) vem numa conexão só, com retomada se cair.
2. A câmera perde alguns quadros (a hora tem 107 974, não 108 000). O corte usa o filtro
   `fps=30`, que preenche a lacuna repetindo o quadro anterior: o trecho tem 1200 quadros
   e o quadro n fica em n/30 s. Na janela escolhida falta um quadro só, por volta de 27,5 s.
3. O corte começa em 2761,99 s, no quadro da fonte de 2761,993 s. Começar em 2762 s
   duplicaria o primeiro quadro. O das 14:08 começa em 2334,99 s (quadro de 2334,9945 s).
4. A fonte é `yuvj420p` (faixa completa). O corte converte para `yuv420p` de faixa padrão
   (BT.709); sem isso, alguns players mostram o contraste alterado. A Cena-âncora sai da
   fonte, pela mesma seleção de quadros do corte, direto para PNG.
5. A faixa do relógio (24 px embaixo) fica, de propósito: mantém o quadro em 1280×720.

Uso (Python 3.10+, ffmpeg e curl no PATH; só biblioteca padrão):

    python scripts/preparar_video.py --publico ../drive/oficina-vc-publico \\
                                     --instrutor ../drive/oficina-vc-instrutor/fontes/bellevue-ne8th

    # só o trecho das 14:08 (o arquivo horário pode ficar fora do Drive, ex.: no cache)
    python scripts/preparar_video.py --trechos 14h --publico ../drive/oficina-vc-publico \\
                                     --instrutor ~/.cache/oficina-vc/horarios
"""

import argparse
import hashlib
import subprocess
import urllib.request
from pathlib import Path

URL_DRIVE = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
URL_YOLO = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt"

HORARIO = "Bellevue_Bellevue_NE8th__2017-09-11_12-08-31.mp4"
TRECHO = "bellevue-ne8th_2762-2802s.mp4"
INICIO_S, DURACAO_S = "2761.99", "40"

# Arquivos da pasta do NE 8th no Drive da prefeitura (17bn7l7Qm5s-r5DYoFQPhviFZ0jWY9qk5):
# {apelido: (arquivo horário, ID no Drive, tamanho em bytes, trecho, início do corte em s)}
TRECHOS = {
    "12h": (HORARIO, "1YobDWnMBM6w_DHKGEaeMGO8JCZGg9h8j", 986_047_389, TRECHO, INICIO_S),
    "14h": ("Bellevue_Bellevue_NE8th__2017-09-11_14-08-31.mp4", "1BkcUT9o_mdc93cFi27WIdRjOMK2Dv2nH", 957_407_779,
            "bellevue-ne8th_1408_2335-2375s.mp4", "2334.99"),
}
QUADRO_ANCORA = 623  # do trecho; 20,77 s, 12:54:53 no relógio
ANCORA = f"cena_ancora_bellevue_{QUADRO_ANCORA}.png"

# SHA-256 dos arquivos gerados em 06/11/2026 ("Preparar o vídeo do Case de tráfego").
# O trecho pode mudar de bytes com outra versão do ffmpeg/x264; o arquivo horário, a
# Cena-âncora e o yolo26n.pt não.
SHA256 = {
    HORARIO: "15c9b8478b0897f935ecd0d05d5e5407f771e05c7d2f70801180ff24a4d56a60",
    TRECHO: "5ad0edd06a81277f66fd8f00ad0cf4df1920dcc13a6e1b307024677c7a01bd97",
    ANCORA: "db4abe3c74421a39c10508b92cb015f748486161c41d7212f5af729adee17355",
    "yolo26n.pt": "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef",
    # o das 14:08, gerado em 09/11/2026
    "Bellevue_Bellevue_NE8th__2017-09-11_14-08-31.mp4": "28e5a8b81e26f15581c86783fa1f71f1cec1b14c834ead44b06da520d9bbebdc",
    "bellevue-ne8th_1408_2335-2375s.mp4": "caade273d3e7974e4b2af08767819a1ba7f3c2881d7032365000cc4e6b0aa906",
}

COR = ["-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]


def ffmpeg(*argumentos):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *argumentos], check=True)


def baixar_horario(destino, id_drive, tamanho):
    """Baixa o arquivo horário do Drive da prefeitura numa conexão só, retomando se interrompido."""
    if destino.exists() and destino.stat().st_size == tamanho:
        return
    subprocess.run(["curl", "-fL", "--retry", "5", "-C", "-", "-o", str(destino), URL_DRIVE.format(id_drive)], check=True)


def cortar_trecho(horario, destino, inicio=INICIO_S):
    ffmpeg(
        "-ss", inicio, "-i", str(horario), "-t", DURACAO_S,
        "-vf", "fps=30,scale=in_range=full:out_range=tv,format=yuv420p", *COR,
        "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-movflags", "+faststart", "-an", str(destino),
    )


def extrair_cena_ancora(horario, destino):
    ffmpeg(
        "-ss", INICIO_S, "-i", str(horario), "-t", DURACAO_S,
        "-vf", f"fps=30,select=eq(n\\,{QUADRO_ANCORA})", "-frames:v", "1", str(destino),
    )


def sha256(caminho):
    resumo = hashlib.sha256()
    with open(caminho, "rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1 << 20), b""):
            resumo.update(bloco)
    return resumo.hexdigest()


def conferir(caminho):
    igual = SHA256.get(caminho.name) == sha256(caminho)
    print(f"{'✅' if igual else '⚠️ '} {caminho.name}" + ("" if igual else " (SHA-256 diferente do registrado)"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--publico", type=Path, default=Path("dados"), help="pasta da oficina-vc-publico")
    parser.add_argument("--instrutor", type=Path, default=Path("fontes"), help="pasta dos arquivos horários")
    parser.add_argument("--trechos", nargs="+", choices=sorted(TRECHOS), default=sorted(TRECHOS))
    args = parser.parse_args()
    for pasta in (args.publico, args.instrutor):
        pasta.mkdir(parents=True, exist_ok=True)

    gerados = []
    for apelido in args.trechos:
        nome_horario, id_drive, tamanho, nome_trecho, inicio = TRECHOS[apelido]
        horario = args.instrutor / nome_horario
        trecho = args.publico / nome_trecho
        print(f"Baixando o arquivo horário das {apelido} ({tamanho / 1e6:.0f} MB)...")
        baixar_horario(horario, id_drive, tamanho)
        print(f"Cortando o trecho de 40 s das {apelido}...")
        cortar_trecho(horario, trecho, inicio)
        gerados += [horario, trecho]
        if apelido == "12h":
            ancora = args.publico / ANCORA
            extrair_cena_ancora(horario, ancora)
            gerados.append(ancora)
    pesos = args.publico / "yolo26n.pt"
    if not pesos.exists():
        print("Baixando o yolo26n.pt...")
        urllib.request.urlretrieve(URL_YOLO, pesos)
    gerados.append(pesos)

    print()
    for caminho in gerados:
        conferir(caminho)


if __name__ == "__main__":
    main()
