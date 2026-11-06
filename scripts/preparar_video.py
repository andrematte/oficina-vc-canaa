"""Prepara os arquivos do Case de monitoramento urbano a partir do MOT20 (MOTChallenge).

Gera, sem guardar nenhum JPG no fim:

- `mot20-02_completo.mp4`: o MOT20-02 inteiro (1:51, 25 fps, 1920×1080, H.264), fonte do instrutor;
- `cena_ancora_2301.png`: o quadro 2301, a Cena-âncora, sem perda;
- `mot20-02_65-105s.mp4`: o trecho de 40 s (65–105 s, quadros 1626–2625) usado na Oficina;
- `yolo26n.pt`: os pesos pré-treinados, espelhados do release v8.4.0 da Ultralytics.

O MOTChallenge não publica o MOT20-02 em vídeo 1080p: os quadros limpos só existem
como JPGs dentro do `MOT20.zip` (5 GB). O MOT20-02 é a primeira sequência do zip e
ocupa um bloco contínuo no começo do arquivo (~1,13 GB), então o script:

1. lê só o índice central do zip, por requisições HTTP com faixa de bytes;
2. baixa, numa única conexão contínua (várias conexões travam no servidor), só a faixa
   de bytes com os JPGs do MOT20-02;
3. extrai cada JPG direto dessa faixa, pelos cabeçalhos locais do zip;
4. codifica, corta e extrai a Cena-âncora com o ffmpeg; apaga os JPGs.

Uso (Python 3.10+, ffmpeg e curl no PATH; só biblioteca padrão):

    python scripts/preparar_video.py --publico ../drive/oficina-vc-publico \\
                                     --instrutor ../drive/oficina-vc-instrutor/fontes

    python scripts/preparar_video.py --so-listar   # só confere o índice do zip
"""

import argparse
import hashlib
import io
import shutil
import struct
import subprocess
import urllib.request
import zipfile
import zlib
from pathlib import Path

URL_MOT20 = "https://motchallenge.net/data/MOT20.zip"
PREFIXO = "MOT20/train/MOT20-02/img1/"
URL_YOLO = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt"

QUADRO_ANCORA = 2301
INICIO_S, DURACAO_S = 65, 40

# SHA-256 dos arquivos gerados em 05/11/2026 ("Preparar o vídeo do Case").
# Os vídeos podem mudar de bytes com outra versão do ffmpeg/x264; o yolo26n.pt não.
SHA256 = {
    "mot20-02_completo.mp4": "a6c8578f0fdb971242ffa36fb08ae612a6a7a08840e868f85f6e076218e2c5f9",
    "mot20-02_65-105s.mp4": "5ee57b02896a9b3499720c0c69f7269bdccb1b22a95ad81551936d2ded687bbd",
    "cena_ancora_2301.png": "2a84e1d02629ebd1598b2636a527ba6f542028b607f2e7a79577ad9b4a0e8ae1",
    "yolo26n.pt": "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef",
}

# Cor: os JPGs viram yuv420p de faixa padrão (BT.709). Sem isso sai yuvj420p, que
# alguns players mostram com o contraste alterado.
COR = ["-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]


class ArquivoRemoto(io.RawIOBase):
    """Arquivo somente leitura sobre HTTP, lido por faixas de bytes (o bastante para o zipfile ler o índice)."""

    def __init__(self, url):
        self.url, self.posicao = url, 0
        pedido = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(pedido) as resposta:
            self.tamanho = int(resposta.headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.posicao

    def seek(self, deslocamento, origem=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.posicao, io.SEEK_END: self.tamanho}[origem]
        self.posicao = base + deslocamento
        return self.posicao

    def read(self, n=-1):
        if n < 0:
            n = self.tamanho - self.posicao
        if n == 0 or self.posicao >= self.tamanho:
            return b""
        fim = min(self.posicao + n, self.tamanho) - 1
        pedido = urllib.request.Request(self.url, headers={"Range": f"bytes={self.posicao}-{fim}"})
        with urllib.request.urlopen(pedido) as resposta:
            dados = resposta.read()
        self.posicao += len(dados)
        return dados


def listar_quadros():
    """Entradas do zip com os JPGs do MOT20-02, em ordem, e o último byte a baixar."""
    with zipfile.ZipFile(ArquivoRemoto(URL_MOT20)) as zip_remoto:
        entradas = sorted(
            (e for e in zip_remoto.infolist() if e.filename.startswith(PREFIXO) and e.filename.endswith(".jpg")),
            key=lambda e: e.filename,
        )
    # O cabeçalho local tem 30 bytes + nome + extra; o extra local pode ser maior que
    # o do índice central, então sobra uma folga de 64 KB no fim da faixa.
    ultimo_byte = max(e.header_offset + 30 + len(e.filename.encode()) + len(e.extra) + e.compress_size for e in entradas)
    return entradas, ultimo_byte + 65_536


def baixar_faixa(destino, ultimo_byte):
    """Baixa os bytes 0..ultimo_byte do MOT20.zip numa conexão só, retomando se interrompido."""
    subprocess.run(
        ["curl", "-fL", "--retry", "5", "-C", "-", "-r", f"0-{ultimo_byte}", "-o", str(destino), URL_MOT20],
        check=True,
    )


def extrair_quadros(faixa, entradas, pasta):
    """Extrai os JPGs da faixa baixada, lendo o cabeçalho local de cada entrada."""
    pasta.mkdir(parents=True, exist_ok=True)
    with open(faixa, "rb") as arquivo:
        for entrada in entradas:
            arquivo.seek(entrada.header_offset)
            cabecalho = arquivo.read(30)
            if cabecalho[:4] != b"PK\x03\x04":
                raise ValueError(f"Cabeçalho local inválido em {entrada.filename}.")
            tamanho_nome, tamanho_extra = struct.unpack("<HH", cabecalho[26:30])
            arquivo.seek(entrada.header_offset + 30 + tamanho_nome + tamanho_extra)
            dados = arquivo.read(entrada.compress_size)
            if entrada.compress_type == zipfile.ZIP_DEFLATED:
                dados = zlib.decompressobj(-15).decompress(dados)
            elif entrada.compress_type != zipfile.ZIP_STORED:
                raise ValueError(f"Compressão não suportada em {entrada.filename}.")
            if zlib.crc32(dados) != entrada.CRC:
                raise ValueError(f"CRC não confere em {entrada.filename}.")
            (pasta / Path(entrada.filename).name).write_bytes(dados)


def ffmpeg(*argumentos):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *argumentos], check=True)


def codificar_completo(pasta_quadros, destino):
    ffmpeg(
        "-framerate", "25", "-start_number", "1", "-i", str(pasta_quadros / "%06d.jpg"),
        "-vf", "scale=in_range=full:out_range=tv,format=yuv420p", *COR,
        "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-movflags", "+faststart", str(destino),
    )


def extrair_cena_ancora(pasta_quadros, destino):
    ffmpeg("-i", str(pasta_quadros / f"{QUADRO_ANCORA:06d}.jpg"), str(destino))


def cortar_trecho(completo, destino):
    ffmpeg(
        "-ss", str(INICIO_S), "-i", str(completo), "-t", str(DURACAO_S),
        "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", *COR,
        "-movflags", "+faststart", "-an", str(destino),
    )


def sha256(caminho):
    resumo = hashlib.sha256()
    with open(caminho, "rb") as arquivo:
        for bloco in iter(lambda: arquivo.read(1 << 20), b""):
            resumo.update(bloco)
    return resumo.hexdigest()


def conferir(caminho):
    esperado = SHA256.get(caminho.name)
    igual = esperado == sha256(caminho)
    print(f"{'✅' if igual else '⚠️ '} {caminho.name}" + ("" if igual else " (SHA-256 diferente do de 05/11/2026)"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--publico", type=Path, default=Path("dados"), help="pasta da oficina-vc-publico")
    parser.add_argument("--instrutor", type=Path, default=Path("fontes"), help="pasta do vídeo completo")
    parser.add_argument("--trabalho", type=Path, default=Path("trabalho_video"), help="pasta temporária")
    parser.add_argument("--so-listar", action="store_true", help="só lê o índice do zip e para")
    args = parser.parse_args()

    print("Lendo o índice do MOT20.zip...")
    entradas, ultimo_byte = listar_quadros()
    print(f"MOT20-02: {len(entradas)} quadros nos bytes 0–{ultimo_byte} ({ultimo_byte / 1e9:.2f} GB).")
    if args.so_listar:
        return

    for pasta in (args.publico, args.instrutor, args.trabalho):
        pasta.mkdir(parents=True, exist_ok=True)
    faixa = args.trabalho / "MOT20-inicio.zip.parcial"
    quadros = args.trabalho / "img1"

    print("Baixando a faixa do zip (uma conexão, ~10 MB/s)...")
    baixar_faixa(faixa, ultimo_byte)
    print("Extraindo os JPGs...")
    extrair_quadros(faixa, entradas, quadros)

    completo = args.instrutor / "mot20-02_completo.mp4"
    ancora = args.publico / "cena_ancora_2301.png"
    trecho = args.publico / "mot20-02_65-105s.mp4"
    pesos = args.publico / "yolo26n.pt"

    print("Codificando o MOT20-02 inteiro...")
    codificar_completo(quadros, completo)
    extrair_cena_ancora(quadros, ancora)
    print("Cortando o trecho de 40 s...")
    cortar_trecho(completo, trecho)
    print("Baixando o yolo26n.pt...")
    urllib.request.urlretrieve(URL_YOLO, pesos)

    shutil.rmtree(args.trabalho)
    print()
    for caminho in (completo, ancora, trecho, pesos):
        conferir(caminho)


if __name__ == "__main__":
    main()
