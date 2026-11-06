"""Painel de métricas animado, desenhado ao lado do vídeo de saída."""

import numpy as np
from PIL import Image, ImageDraw

from . import estilo

LARGURA, ALTURA = 640, 720
MARGEM = 40
PULSO_QUADROS = 15  # duração do "+1" que aparece quando a contagem sobe


def _misturar(cor, fundo, quanto):
    """Mistura duas cores hex; quanto=1 dá `cor`, quanto=0 dá `fundo`."""
    a, b = estilo.hex_para_rgb(cor), estilo.hex_para_rgb(fundo)
    return tuple(int(round(x * quanto + y * (1 - quanto))) for x, y in zip(a, b))


FUNDO = estilo.hex_para_rgb(estilo.NAVY)
BRANCO = estilo.hex_para_rgb(estilo.OFF_WHITE)
AMARELO = estilo.hex_para_rgb(estilo.AMARELO)
LEGENDA = _misturar(estilo.OFF_WHITE, estilo.NAVY, 0.68)
TRILHO = _misturar(estilo.OFF_WHITE, estilo.NAVY, 0.16)


def _tempo(segundos):
    segundos = int(segundos)
    return f"{segundos // 60:02d}:{segundos % 60:02d}"


def _numero(valor, casas=0):
    return f"{valor:.{casas}f}".replace(".", ",")


class Painel:
    """Desenha um quadro do painel por chamada de `desenhar`.

    `classes` é a lista de nomes (em português) das classes contadas. Com uma
    classe, o painel mostra fluxo e o gráfico de ocupação da Zona; com mais de
    uma, troca esses dois pelas passagens por classe.
    """

    def __init__(self, duracao, quadros, classes, credito=None):
        self.duracao = max(duracao, 1e-6)
        self.quadros = max(int(quadros), 2)  # o gráfico cresce da esquerda para a direita até este total
        self.classes = list(classes)
        self.por_classe = len(self.classes) > 1
        self.credito = credito
        self.fontes = {
            "titulo": estilo.fonte(30, negrito=True),
            "rotulo": estilo.fonte(22),
            "grande": estilo.fonte(104, negrito=True),
            "medio": estilo.fonte(60, negrito=True),
            "pequeno": estilo.fonte(36, negrito=True),
            "pulso": estilo.fonte(32, negrito=True),
            "credito": estilo.fonte(16),
        }
        self.fundo = self._fundo()

    # Partes fixas, desenhadas uma vez -----------------------------------
    def _fundo(self):
        imagem = Image.new("RGB", (LARGURA, ALTURA), FUNDO)
        traco = ImageDraw.Draw(imagem)
        f = self.fontes
        traco.ellipse((MARGEM, 44, MARGEM + 14, 58), fill=AMARELO)
        traco.text((MARGEM + 26, 36), "Métricas de vídeo", font=f["titulo"], fill=BRANCO)
        traco.text((MARGEM, 116), "Entradas", font=f["rotulo"], fill=LEGENDA)
        traco.text((LARGURA // 2 + 10, 116), "Saídas", font=f["rotulo"], fill=LEGENDA)
        traco.text((MARGEM, 290), "Na Zona agora", font=f["rotulo"], fill=LEGENDA)
        traco.text((LARGURA // 2 + 10, 290), "Pessoas no quadro", font=f["rotulo"], fill=LEGENDA)
        if self.por_classe:
            traco.text((MARGEM, 420), "Passagens pela linha, por classe", font=f["rotulo"], fill=LEGENDA)
        else:
            traco.text((MARGEM, 420), "Fluxo pela linha", font=f["rotulo"], fill=LEGENDA)
            traco.text((MARGEM, 520), "Ocupação da Zona", font=f["rotulo"], fill=LEGENDA)
        if self.credito:
            traco.text((MARGEM, ALTURA - 36), self.credito, font=f["credito"], fill=LEGENDA)
        return imagem

    # Um quadro ---------------------------------------------------------------
    def desenhar(self, t, entradas, saidas, na_zona, no_quadro, serie_zona, passagens_por_classe=None, pulsos=(None, None), quadro=0):
        """Devolve o painel do instante `t` (em segundos) como imagem BGR (numpy)."""
        imagem = self.fundo.copy()
        traco = ImageDraw.Draw(imagem)
        f = self.fontes

        # Relógio e barra de tempo
        relogio = f"{_tempo(t)} / {_tempo(self.duracao)}"
        largura_relogio = traco.textlength(relogio, font=f["rotulo"])
        traco.text((LARGURA - MARGEM - largura_relogio, 40), relogio, font=f["rotulo"], fill=LEGENDA)
        y = 88
        traco.rounded_rectangle((MARGEM, y, LARGURA - MARGEM, y + 6), radius=3, fill=TRILHO)
        fim = MARGEM + (LARGURA - 2 * MARGEM) * min(t / self.duracao, 1)
        traco.rounded_rectangle((MARGEM, y, max(fim, MARGEM + 6), y + 6), radius=3, fill=AMARELO)

        # Entradas e Saídas
        for x, valor, cor, pulso in (
            (MARGEM, entradas, AMARELO, pulsos[0]),
            (LARGURA // 2 + 10, saidas, BRANCO, pulsos[1]),
        ):
            texto = str(valor)
            traco.text((x - 4, 138), texto, font=f["grande"], fill=cor)
            if pulso is not None and 0 <= quadro - pulso < PULSO_QUADROS:
                quanto = 1 - (quadro - pulso) / PULSO_QUADROS
                cor_pulso = tuple(int(c * quanto + b * (1 - quanto)) for c, b in zip(AMARELO, FUNDO))
                largura = traco.textlength(texto, font=f["grande"])
                traco.text((x + largura + 6, 160 - 10 * (1 - quanto)), "+1", font=f["pulso"], fill=cor_pulso)

        # Zona e quadro
        traco.text((MARGEM - 2, 318), str(na_zona), font=f["medio"], fill=AMARELO)
        traco.text((LARGURA // 2 + 8, 318), str(no_quadro), font=f["medio"], fill=BRANCO)

        if self.por_classe:
            self._classes(traco, passagens_por_classe or {})
        else:
            fluxo = (entradas + saidas) / t * 60 if t >= 1 else 0
            texto = _numero(fluxo, 0)
            traco.text((MARGEM - 2, 446), texto, font=f["pequeno"], fill=BRANCO)
            largura = traco.textlength(texto, font=f["pequeno"])
            traco.text((MARGEM + largura + 10, 460), "pessoas por minuto", font=f["rotulo"], fill=LEGENDA)
            self._grafico(traco, serie_zona)

        return np.asarray(imagem)[..., ::-1].copy()

    def _grafico(self, traco, serie):
        x0, x1 = MARGEM, LARGURA - MARGEM
        y0, y1 = 556, ALTURA - 60  # topo e base
        traco.line((x0, y1, x1, y1), fill=TRILHO, width=2)
        if len(serie) < 2:
            return
        # Média móvel de 1 s só no desenho: a contagem quadro a quadro oscila demais para ler.
        janela = 25
        acumulado = np.cumsum(np.insert(np.asarray(serie, dtype=float), 0, 0))
        inicio = np.maximum(np.arange(1, len(serie) + 1) - janela, 0)
        suave = (acumulado[1:] - acumulado[inicio]) / (np.arange(1, len(serie) + 1) - inicio)
        maximo = max(max(serie), 1)
        passo = (x1 - x0) / (self.quadros - 1)
        pontos = [(x0 + i * passo, y1 - (y1 - y0) * v / maximo) for i, v in enumerate(suave)]
        area = [(x0, y1)] + pontos + [(pontos[-1][0], y1)]
        traco.polygon(area, fill=_misturar(estilo.AMARELO, estilo.NAVY, 0.22))
        traco.line(pontos, fill=AMARELO, width=3, joint="curve")
        xu, yu = pontos[-1]
        traco.ellipse((xu - 6, yu - 6, xu + 6, yu + 6), fill=AMARELO)
        rotulo = f"máx. {maximo}"
        largura = traco.textlength(rotulo, font=self.fontes["rotulo"])
        traco.text((x1 - largura, 520), rotulo, font=self.fontes["rotulo"], fill=LEGENDA)

    def _classes(self, traco, passagens):
        y = 456
        for indice, nome in enumerate(self.classes[:5]):
            cor, _ = estilo.cores_da_classe(nome, indice)
            traco.ellipse((MARGEM, y + 8, MARGEM + 18, y + 26), fill=estilo.hex_para_rgb(cor))
            traco.text((MARGEM + 30, y), nome.capitalize(), font=self.fontes["rotulo"], fill=BRANCO)
            valor = str(passagens.get(nome, 0))
            largura = traco.textlength(valor, font=self.fontes["pequeno"])
            traco.text((LARGURA - MARGEM - largura, y - 8), valor, font=self.fontes["pequeno"], fill=BRANCO)
            y += 46
