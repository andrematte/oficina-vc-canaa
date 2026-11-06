"""Movimentos num cruzamento: de que Zona por Aproximação cada veículo veio e para qual foi.

Cada Aproximação (braço do cruzamento) tem uma Zona. Um veículo rastreado ganha
uma **origem** na primeira Zona em que aparece; quando ele chega a outra Zona,
o par origem → destino é o seu **Movimento**, contado uma vez só. A **Matriz de
movimentos** soma esses pares, separados por classe do veículo.

Os nomes dos Movimentos supõem um cruzamento de 4 braços chamados pelos pontos
cardeais (Norte, Leste, Sul, Oeste) e mão direita, como no Brasil e nos EUA.
"""

from dataclasses import dataclass

import numpy as np

APROXIMACOES = ("Norte", "Leste", "Sul", "Oeste")  # em sentido horário, vistas de cima
TIPOS = ("Reto", "Conversão à direita", "Conversão à esquerda", "Retorno")
RETORNO_PIXELS = 150  # quanto o veículo anda fora das Zonas antes de voltar à de origem, num Retorno


def sigla(aproximacao):
    """'Norte' → 'N', 'Leste' → 'L', 'Sul' → 'S', 'Oeste' → 'O'."""
    return aproximacao[0].upper()


def tipo_de_movimento(origem, destino):
    """Nome do Movimento entre duas Aproximações, com mão direita.

    Quem vem do Norte anda rumo ao sul: segue reto para o Sul, tem o Oeste à
    direita e o Leste à esquerda.
    """
    passos = (APROXIMACOES.index(destino) - APROXIMACOES.index(origem)) % 4
    return {0: "Retorno", 1: "Conversão à esquerda", 2: "Reto", 3: "Conversão à direita"}[passos]


@dataclass
class Movimento:
    rastro: int  # o ID do rastreador
    origem: str
    destino: str
    classe: str
    tempo: float  # segundos do vídeo em que o veículo chegou ao destino

    @property
    def tipo(self):
        return tipo_de_movimento(self.origem, self.destino)


class ContadorDeMovimentos:
    """Acompanha cada ID pelas Zonas e registra o Movimento de cada um.

    - `zonas`: {nome da Aproximação: sv.PolygonZone}.
    - `atualizar(deteccoes, t, nomes)` recebe as detecções rastreadas de um quadro
      (com `tracker_id`) e devolve os Movimentos que terminaram nesse quadro.

    Regras:
    - a origem é a primeira Zona em que o ID aparece. Um ID que nasce no meio do
      cruzamento (fora de todas as Zonas) só ganha origem quando entra numa Zona;
      se essa Zona for, na verdade, o destino, o Movimento se perde;
    - o Movimento termina quando o ID entra numa Zona diferente da origem;
    - Retorno: o ID sai da origem, anda pelo menos `RETORNO_PIXELS` fora de todas
      as Zonas e volta para a mesma Zona. A distância evita que um carro parado na
      borda da Zona, cuja caixa treme um pixel para fora e para dentro, vire Retorno;
    - cada ID conta um Movimento, no máximo.
    """

    def __init__(self, zonas, retorno_pixels=RETORNO_PIXELS):
        self.zonas = dict(zonas)
        self.retorno_pixels = retorno_pixels
        # O ponto do veículo que as Zonas usam (os pés, por padrão).
        primeira = next(iter(self.zonas.values()), None)
        self.ancora = next(iter(primeira.triggering_anchors)) if primeira is not None else None
        self.reiniciar()

    def reiniciar(self):
        self.origem = {}  # ID → Aproximação de origem
        self.saida = {}  # ID → [ponto em que saiu de todas as Zonas, maior distância dele até agora]
        self.concluidos = set()
        self.movimentos = []

    def zona_de_cada(self, deteccoes):
        """Lista com o nome da Zona de cada detecção (ou None, fora de todas)."""
        onde = [None] * len(deteccoes)
        if len(deteccoes) == 0:
            return onde
        for nome, zona in self.zonas.items():
            dentro = zona.trigger(deteccoes)
            for k in np.flatnonzero(dentro):
                if onde[k] is None:
                    onde[k] = nome
        return onde

    def atualizar(self, deteccoes, t, nomes):
        """Processa um quadro. `t` em segundos; `nomes`: {id da classe: nome}."""
        novos = []
        pontos = deteccoes.get_anchors_coordinates(self.ancora) if len(deteccoes) else []
        for k, zona in enumerate(self.zona_de_cada(deteccoes)):
            rastro = int(deteccoes.tracker_id[k])
            if rastro in self.concluidos:
                continue
            origem = self.origem.get(rastro)
            if zona is None:
                if origem is not None:
                    saida = self.saida.setdefault(rastro, [pontos[k], 0.0])
                    saida[1] = max(saida[1], float(np.hypot(*(pontos[k] - saida[0]))))
                continue
            if origem is None:
                self.origem[rastro] = zona
                continue
            saida = self.saida.pop(rastro, None)
            retorno = zona == origem and saida is not None and saida[1] >= self.retorno_pixels
            if zona != origem or retorno:
                classe = nomes.get(int(deteccoes.class_id[k]), str(deteccoes.class_id[k]))
                movimento = Movimento(rastro, origem, zona, classe, round(t, 2))
                self.movimentos.append(movimento)
                self.concluidos.add(rastro)
                novos.append(movimento)
        return novos

    def de_onde(self, rastro):
        """Origem já registrada de um ID (ou None)."""
        return self.origem.get(int(rastro))

    # Resumos -----------------------------------------------------------------
    def matriz(self, classe=None):
        """{origem: {destino: n}} com todas as Aproximações (zeros incluídos)."""
        nomes = list(self.zonas)
        tabela = {o: {d: 0 for d in nomes} for o in nomes}
        for m in self.movimentos:
            if classe is None or m.classe == classe:
                tabela[m.origem][m.destino] += 1
        return tabela

    def por_tipo(self):
        contagem = {tipo: 0 for tipo in TIPOS}
        for m in self.movimentos:
            contagem[m.tipo] += 1
        return contagem

    def por_classe(self, classes=()):
        contagem = {c: 0 for c in classes}
        for m in self.movimentos:
            contagem[m.classe] = contagem.get(m.classe, 0) + 1
        return contagem
