"""Painel de métricas animado, desenhado ao lado do vídeo de saída: a Matriz de movimentos ao vivo."""

import numpy as np
from PIL import Image, ImageDraw

from . import estilo
from .movimentos import APROXIMACOES, TIPOS, tipo_de_movimento

LARGURA, ALTURA = 640, 720
MARGEM = 40
PULSO_SEGUNDOS = 0.6  # duração do "+1" e do brilho da célula quando um Movimento é contado


def _misturar(cor, fundo, quanto):
    """Mistura duas cores hex; quanto=1 dá `cor`, quanto=0 dá `fundo`."""
    a, b = estilo.hex_para_rgb(cor), estilo.hex_para_rgb(fundo)
    return tuple(round(x * quanto + y * (1 - quanto)) for x, y in zip(a, b))


def _entre(a, b, quanto):
    return tuple(round(x * quanto + y * (1 - quanto)) for x, y in zip(a, b))


FUNDO = estilo.hex_para_rgb(estilo.NAVY)
BRANCO = estilo.hex_para_rgb(estilo.OFF_WHITE)
AMARELO = estilo.hex_para_rgb(estilo.AMARELO)
LEGENDA = _misturar(estilo.OFF_WHITE, estilo.NAVY, 0.68)
APAGADO = _misturar(estilo.OFF_WHITE, estilo.NAVY, 0.35)
TRILHO = _misturar(estilo.OFF_WHITE, estilo.NAVY, 0.16)
CELULA = _misturar(estilo.OFF_WHITE, estilo.NAVY, 0.09)

# Uma cor por tipo de Movimento: a marca embaixo de cada célula da Matriz e a bolinha da legenda.
CORES_TIPOS = {
    "Reto": estilo.hex_para_rgb(estilo.OFF_WHITE),
    "Conversão à direita": estilo.hex_para_rgb(estilo.VERDE),
    "Conversão à esquerda": estilo.hex_para_rgb("#5B70F4"),
    "Retorno": estilo.hex_para_rgb("#F01414"),
}

# Geometria da Matriz
Y_MATRIZ = 290  # linha dos títulos das colunas
ALTURA_CELULA = 40
X_CELULAS = 150  # onde começa a primeira coluna
LARGURA_CELULA = (LARGURA - MARGEM - X_CELULAS) // 4


def _tempo(segundos):
    segundos = int(segundos)
    return f"{segundos // 60:02d}:{segundos % 60:02d}"


def _numero(valor, casas=0):
    return f"{valor:.{casas}f}".replace(".", ",")


class Painel:
    """Desenha um quadro do painel por chamada de `desenhar`.

    - `aproximacoes`: nomes das Zonas, na ordem das linhas e colunas da Matriz;
    - `classes`: nomes (em português) das classes contadas;
    - `cores`: cor (hex) de cada classe; `processar_video` passa as cores buscadas pelo nome cru
      do modelo, as mesmas das caixas;
    - `duracao` em segundos; `fps` é o do vídeo de saída (com `passo=2`, metade do original).
    """

    def __init__(self, duracao, aproximacoes, classes, credito=None, fps=25, cores=None):
        self.duracao = max(duracao, 1e-6)
        self.fps = fps
        self.pulso_quadros = max(round(PULSO_SEGUNDOS * fps), 1)
        self.aproximacoes = list(aproximacoes)[:4]
        self.classes = list(classes)[:5]  # o especialista tem 5 Tipos de veículo
        self.passo_classe = 32 if len(self.classes) <= 4 else 28  # 5 linhas cabem antes do crédito
        # Cor de cada classe, em hex; o padrão busca pelo nome, que aqui já vem em português.
        self.cores = list(cores) if cores else [estilo.cores_da_classe(n, k)[0] for k, n in enumerate(self.classes)]
        self.credito = credito
        self.fontes = {
            "titulo": estilo.fonte(30, negrito=True),
            "rotulo": estilo.fonte(22),
            "rotulo_negrito": estilo.fonte(22, negrito=True),
            "grande": estilo.fonte(88, negrito=True),
            "medio": estilo.fonte(52, negrito=True),
            "celula": estilo.fonte(26, negrito=True),
            "pequeno": estilo.fonte(19),
            "pequeno_negrito": estilo.fonte(19, negrito=True),
            "pulso": estilo.fonte(30, negrito=True),
            "credito": estilo.fonte(15),
        }
        self.fundo = self._fundo()

    # Partes fixas, desenhadas uma vez -----------------------------------
    def _x_coluna(self, j):
        return X_CELULAS + j * LARGURA_CELULA

    def _y_linha(self, i):
        return Y_MATRIZ + 30 + i * ALTURA_CELULA

    def _fundo(self):
        imagem = Image.new("RGB", (LARGURA, ALTURA), FUNDO)
        traco = ImageDraw.Draw(imagem)
        f = self.fontes
        traco.ellipse((MARGEM, 44, MARGEM + 14, 58), fill=AMARELO)
        traco.text((MARGEM + 26, 36), "Métricas de vídeo", font=f["titulo"], fill=BRANCO)
        traco.text((MARGEM, 106), "Movimentos", font=f["rotulo"], fill=LEGENDA)
        traco.text((LARGURA // 2 + 10, 106), "Fluxo", font=f["rotulo"], fill=LEGENDA)
        traco.text((LARGURA // 2 + 10, 196), "veículos por minuto", font=f["rotulo"], fill=LEGENDA)

        # Matriz: títulos, origens e células vazias
        traco.text((MARGEM, 246), "Matriz de movimentos", font=f["rotulo_negrito"], fill=BRANCO)
        dica = "de (linha) → para (coluna)"
        traco.text((LARGURA - MARGEM - traco.textlength(dica, font=f["pequeno"]), 249), dica, font=f["pequeno"], fill=LEGENDA)
        for j, nome in enumerate(self.aproximacoes):
            x = self._x_coluna(j) + LARGURA_CELULA / 2
            traco.text((x - traco.textlength(nome, font=f["pequeno"]) / 2, Y_MATRIZ), nome, font=f["pequeno"], fill=LEGENDA)
        for i, origem in enumerate(self.aproximacoes):
            y = self._y_linha(i)
            traco.text((MARGEM, y + 8), origem, font=f["rotulo"], fill=LEGENDA)
            for j, destino in enumerate(self.aproximacoes):
                x = self._x_coluna(j)
                traco.rounded_rectangle((x + 3, y + 3, x + LARGURA_CELULA - 3, y + ALTURA_CELULA - 3), radius=6, fill=CELULA)
                cor = CORES_TIPOS[tipo_de_movimento(origem, destino)] if self._cardinais() else TRILHO
                traco.rectangle((x + 14, y + ALTURA_CELULA - 7, x + LARGURA_CELULA - 14, y + ALTURA_CELULA - 4), fill=_entre(cor, FUNDO, 0.8))

        # Legendas de baixo
        y = 506
        traco.text((MARGEM, y), "Por movimento", font=f["rotulo"], fill=LEGENDA)
        traco.text((LARGURA // 2 + 10, y), "Por tipo de veículo", font=f["rotulo"], fill=LEGENDA)
        for k, tipo in enumerate(TIPOS):
            yk = y + 36 + k * 32
            traco.ellipse((MARGEM, yk + 7, MARGEM + 12, yk + 19), fill=CORES_TIPOS[tipo])
            traco.text((MARGEM + 22, yk), tipo, font=f["pequeno"], fill=BRANCO)
        for k, nome in enumerate(self.classes):
            yk = y + 36 + k * self.passo_classe
            traco.ellipse((LARGURA // 2 + 10, yk + 7, LARGURA // 2 + 22, yk + 19), fill=estilo.hex_para_rgb(self.cores[k]))
            traco.text((LARGURA // 2 + 32, yk), nome.capitalize(), font=f["pequeno"], fill=BRANCO)
        if self.credito:
            traco.text((MARGEM, ALTURA - 30), self.credito, font=f["credito"], fill=LEGENDA)
        return imagem

    def _cardinais(self):
        return all(nome in APROXIMACOES for nome in self.aproximacoes)

    # Um quadro ---------------------------------------------------------------
    def desenhar(self, t, matriz, por_tipo, por_classe, ultimo=None, quadro=0):
        """Devolve o painel do instante `t` (em segundos) como imagem BGR (numpy).

        - `matriz`: {origem: {destino: n}}; `por_tipo` e `por_classe`: {nome: n};
        - `ultimo`: (quadro em que o último Movimento foi contado, origem, destino),
          para o "+1" e o brilho da célula.
        """
        imagem = self.fundo.copy()
        traco = ImageDraw.Draw(imagem)
        f = self.fontes

        # Relógio e barra de tempo
        relogio = f"{_tempo(t)} / {_tempo(self.duracao)}"
        largura_relogio = traco.textlength(relogio, font=f["rotulo"])
        traco.text((LARGURA - MARGEM - largura_relogio, 40), relogio, font=f["rotulo"], fill=LEGENDA)
        y = 84
        traco.rounded_rectangle((MARGEM, y, LARGURA - MARGEM, y + 6), radius=3, fill=TRILHO)
        fim = MARGEM + (LARGURA - 2 * MARGEM) * min(t / self.duracao, 1)
        traco.rounded_rectangle((MARGEM, y, max(fim, MARGEM + 6), y + 6), radius=3, fill=AMARELO)

        quanto = 0.0  # 1 logo depois de um Movimento contado, 0 depois de PULSO_SEGUNDOS
        if ultimo is not None and 0 <= quadro - ultimo[0] < self.pulso_quadros:
            quanto = 1 - (quadro - ultimo[0]) / self.pulso_quadros

        # Total e fluxo
        total = sum(sum(linha.values()) for linha in matriz.values())
        texto = str(total)
        traco.text((MARGEM - 4, 122), texto, font=f["grande"], fill=AMARELO)
        if quanto:
            largura = traco.textlength(texto, font=f["grande"])
            traco.text((MARGEM + largura + 6, 140 - 10 * (1 - quanto)), "+1", font=f["pulso"], fill=_entre(AMARELO, FUNDO, quanto))
        fluxo = total / t * 60 if t >= 1 else 0
        traco.text((LARGURA // 2 + 8, 134), _numero(fluxo), font=f["medio"], fill=BRANCO)

        # Células da Matriz
        for i, origem in enumerate(self.aproximacoes):
            for j, destino in enumerate(self.aproximacoes):
                n = matriz.get(origem, {}).get(destino, 0)
                x, yc = self._x_coluna(j), self._y_linha(i)
                acesa = quanto and ultimo[1] == origem and ultimo[2] == destino
                if acesa:
                    fundo = _entre(AMARELO, CELULA, 0.85 * quanto)
                    traco.rounded_rectangle((x + 3, yc + 3, x + LARGURA_CELULA - 3, yc + ALTURA_CELULA - 9), radius=6, fill=fundo)
                cor = (FUNDO if quanto > 0.5 else BRANCO) if acesa else (BRANCO if n else APAGADO)
                texto = str(n) if n else "·"
                largura = traco.textlength(texto, font=f["celula"])
                traco.text((x + (LARGURA_CELULA - largura) / 2, yc + 4), texto, font=f["celula"], fill=cor)

        # Totais por tipo de Movimento e por classe
        y = 506
        for k, tipo in enumerate(TIPOS):
            self._valor(traco, LARGURA // 2 - 24, y + 36 + k * 32, por_tipo.get(tipo, 0))
        for k, nome in enumerate(self.classes):
            self._valor(traco, LARGURA - MARGEM, y + 36 + k * self.passo_classe, por_classe.get(nome, 0))

        return np.asarray(imagem)[..., ::-1].copy()

    def _valor(self, traco, direita, y, n):
        texto = str(n)
        fonte = self.fontes["pequeno_negrito"]
        traco.text((direita - traco.textlength(texto, font=fonte), y), texto, font=fonte, fill=BRANCO if n else APAGADO)

