"""Quadros-chave de um MP4 remoto sem baixar o arquivo inteiro.

Os arquivos horários do Bellevue têm ~1 GB cada, com o índice (`moov`) no fim e um
quadro-chave a cada ~1,07 s. Para amostrar quadros de uma hora inteira, basta baixar
o índice (~1 MB) e, por pedidos HTTP com `Range`, só os bytes de cada quadro-chave
escolhido (~150 KB). Cada quadro-chave é um IDR do H.264: decodifica sozinho, com o
SPS e o PPS que vêm no índice.

Uso:

    arquivo = MP4Remoto(url)              # ou um caminho local
    t, quadro = arquivo.quadro_chave(k)   # k-ésimo quadro-chave → (segundos, BGR)

Só biblioteca padrão, numpy e o ffmpeg no PATH.
"""

import struct
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np

CONTEINERES = {b"moov", b"trak", b"mdia", b"minf", b"stbl"}


def _caixas(dados, inicio=0, fim=None):
    """(tipo, início do conteúdo, fim) das caixas MP4 em dados[inicio:fim]."""
    fim = len(dados) if fim is None else fim
    pos = inicio
    while pos + 8 <= fim:
        tamanho, tipo = struct.unpack(">I4s", dados[pos : pos + 8])
        cabecalho = 8
        if tamanho == 1:
            tamanho = struct.unpack(">Q", dados[pos + 8 : pos + 16])[0]
            cabecalho = 16
        elif tamanho == 0:
            tamanho = fim - pos
        yield tipo, pos + cabecalho, pos + tamanho
        pos += tamanho


def _achar(dados, caminho, inicio=0, fim=None):
    """Conteúdo (início, fim) da primeira caixa no caminho, ex. [b'mdia', b'minf']."""
    for tipo, a, b in _caixas(dados, inicio, fim):
        if tipo == caminho[0]:
            return (a, b) if len(caminho) == 1 else _achar(dados, caminho[1:], a, b)
    return None


class MP4Remoto:
    def __init__(self, origem, tentativas=6):
        self.origem = str(origem)
        self.local = not self.origem.startswith("http")
        self.tentativas = tentativas
        self._ler_indice()

    # Leitura por faixa de bytes ------------------------------------------------
    def _ler(self, inicio, tamanho):
        if self.local:
            with open(self.origem, "rb") as f:
                f.seek(inicio)
                return f.read(tamanho)
        pedido = urllib.request.Request(
            self.origem, headers={"Range": f"bytes={inicio}-{inicio + tamanho - 1}", "User-Agent": "oficina-vc"}
        )
        for tentativa in range(self.tentativas):
            try:
                with urllib.request.urlopen(pedido, timeout=60) as resposta:
                    dados = resposta.read()
                if len(dados) == tamanho:
                    return dados
            except OSError:
                pass
            time.sleep(2**tentativa)
        raise OSError(f"Falhei ao ler {tamanho} bytes em {inicio} de {self.origem}")

    def _ler_indice(self):
        # Caixas do nível de cima: só os cabeçalhos, até achar o moov.
        pos = 0
        while True:
            cabecalho = self._ler(pos, 16)
            tamanho, tipo = struct.unpack(">I4s", cabecalho[:8])
            if tamanho == 1:
                tamanho = struct.unpack(">Q", cabecalho[8:16])[0]
            if tipo == b"moov":
                moov = self._ler(pos, tamanho)
                break
            pos += tamanho
        self._indexar(moov)

    def _indexar(self, moov):
        _, _, fim = next(_caixas(moov))
        # A trilha de vídeo é a que tem stsd com avc1.
        for tipo, a, b in _caixas(moov, 8, fim):
            if tipo != b"trak":
                continue
            stbl = _achar(moov, [b"mdia", b"minf", b"stbl"], a, b)
            stsd = _achar(moov, [b"stsd"], *stbl)
            if moov[stsd[0] + 12 : stsd[0] + 16] == b"avc1":
                mdhd = _achar(moov, [b"mdia", b"mdhd"], a, b)
                break
        else:
            raise ValueError("O MP4 não tem trilha H.264 (avc1).")

        versao = moov[mdhd[0]]
        self.escala = struct.unpack(">I", moov[mdhd[0] + (20 if versao else 12) : mdhd[0] + (24 if versao else 16)])[0]

        # avcC: SPS e PPS, para montar um fluxo Annex B decodificável.
        avc1 = stsd[0] + 8  # depois de versão/flags e contagem de entradas
        avc1_fim = avc1 + struct.unpack(">I", moov[avc1 : avc1 + 4])[0]
        avcc = _achar(moov, [b"avcC"], avc1 + 8 + 78, avc1_fim)
        c = moov[avcc[0] : avcc[1]]
        self.tamanho_nal = (c[4] & 3) + 1
        pos, cabecalhos = 6, []
        for _ in range(c[5] & 0x1F):
            n = struct.unpack(">H", c[pos : pos + 2])[0]
            cabecalhos.append(c[pos + 2 : pos + 2 + n])
            pos += 2 + n
        for _ in range(c[pos]):
            n = struct.unpack(">H", c[pos + 1 : pos + 3])[0]
            cabecalhos.append(c[pos + 3 : pos + 3 + n])
            pos += 2 + n
        self.annexb_cabecalho = b"".join(b"\x00\x00\x00\x01" + h for h in cabecalhos)

        def tabela(nome):
            caixa = _achar(moov, [nome], *stbl)
            return None if caixa is None else moov[caixa[0] : caixa[1]]

        # Tempos (stts), tamanhos (stsz), pedaços (stsc + stco/co64) e quadros-chave (stss).
        stts = tabela(b"stts")
        duracoes = []
        for k in range(struct.unpack(">I", stts[4:8])[0]):
            n, d = struct.unpack(">II", stts[8 + 8 * k : 16 + 8 * k])
            duracoes.append(np.full(n, d, dtype=np.int64))
        duracoes = np.concatenate(duracoes)
        self.tempos = np.concatenate([[0], np.cumsum(duracoes)[:-1]]) / self.escala

        stsz = tabela(b"stsz")
        fixo, n = struct.unpack(">II", stsz[4:12])
        self.tamanhos = np.full(n, fixo, dtype=np.int64) if fixo else np.frombuffer(stsz[12 : 12 + 4 * n], ">u4").astype(np.int64)

        if (co := tabela(b"stco")) is not None:
            m = struct.unpack(">I", co[4:8])[0]
            pedacos = np.frombuffer(co[8 : 8 + 4 * m], ">u4").astype(np.int64)
        else:
            co = tabela(b"co64")
            m = struct.unpack(">I", co[4:8])[0]
            pedacos = np.frombuffer(co[8 : 8 + 8 * m], ">u8").astype(np.int64)

        stsc = tabela(b"stsc")
        regras = [struct.unpack(">III", stsc[8 + 12 * k : 20 + 12 * k]) for k in range(struct.unpack(">I", stsc[4:8])[0])]
        self.posicoes = np.zeros(n, dtype=np.int64)
        amostra = 0
        for r, (primeiro, por_pedaco, _) in enumerate(regras):
            ultimo = regras[r + 1][0] - 1 if r + 1 < len(regras) else len(pedacos)
            for pedaco in range(primeiro - 1, ultimo):
                pos = pedacos[pedaco]
                for _ in range(por_pedaco):
                    if amostra >= n:
                        break
                    self.posicoes[amostra] = pos
                    pos += self.tamanhos[amostra]
                    amostra += 1

        stss = tabela(b"stss")
        self.chaves = np.frombuffer(stss[8 : 8 + 4 * struct.unpack(">I", stss[4:8])[0]], ">u4").astype(np.int64) - 1
        self.duracao = float(self.tempos[-1])

    # Quadros ---------------------------------------------------------------------
    def tempo_da_chave(self, k):
        return float(self.tempos[self.chaves[k]])

    def quadro_chave(self, k):
        """(t em segundos, quadro BGR) do k-ésimo quadro-chave."""
        amostra = self.chaves[k]
        dados = self._ler(int(self.posicoes[amostra]), int(self.tamanhos[amostra]))
        annexb, pos = [self.annexb_cabecalho], 0
        while pos < len(dados):
            n = int.from_bytes(dados[pos : pos + self.tamanho_nal], "big")
            annexb.append(b"\x00\x00\x00\x01" + dados[pos + self.tamanho_nal : pos + self.tamanho_nal + n])
            pos += self.tamanho_nal + n
        saida = subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-f", "h264", "-i", "pipe:", "-frames:v", "1",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:"],
            input=b"".join(annexb), capture_output=True, check=True,
        ).stdout
        largura, altura = 1280, 720
        quadro = np.frombuffer(saida, np.uint8)
        if quadro.size != largura * altura * 3:
            raise ValueError(f"Quadro com {quadro.size} bytes; esperava 1280×720×3.")
        return self.tempo_da_chave(k), quadro.reshape(altura, largura, 3).copy()


if __name__ == "__main__":
    import sys

    arquivo = MP4Remoto(sys.argv[1])
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    t, quadro = arquivo.quadro_chave(k)
    import cv2

    destino = Path(sys.argv[3] if len(sys.argv) > 3 else "quadro.png")
    cv2.imwrite(str(destino), quadro)
    print(f"{len(arquivo.chaves)} quadros-chave em {arquivo.duracao:.0f} s; chave {k} em {t:.3f} s → {destino}")
