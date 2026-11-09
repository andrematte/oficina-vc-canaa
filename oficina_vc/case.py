"""Zonas por Aproximação e rastreador do Case de tráfego, já calibrados.

Os valores foram ajustados no trecho de 40 s do cruzamento Bellevue Way × NE 8th
(12:54:33–12:55:13) contra a Matriz de movimentos contada à vista: 38 Movimentos
completos, mais 3 na borda do corte (faixa aceitável de 38 a 41). Com estes valores,
o Case dá 38 Movimentos e erra uma célula só: acha 5 das 6 conversões Norte → Oeste
(o carro que falta já está na curva no primeiro quadro, com caixas fracas).

O notebook do Case e o script do Resultado pronto (`scripts/gerar_resultado_pronto.py`)
usam este módulo, e por isso os dois chegam ao mesmo resultado.
"""

import numpy as np
import supervision as sv

# Como o vídeo é analisado ------------------------------------------------------
VIDEO = "bellevue-ne8th_2762-2802s.mp4"
CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus e truck do COCO
CONFIANCA = 0.05  # o rastreador usa as caixas fracas para continuar rastros (veja criar_rastreador)
IMGSZ = 960  # com o padrão (640), o rastreador perde mais conversões
PASSO = 2  # 1 quadro a cada 2: metade do tempo em CPU, a mesma Matriz de movimentos
FPS_DO_VIDEO = 30

# Uma Zona por Aproximação, em pixels (x, y) do quadro de 1280×720.
ZONAS = {
    # as quatro faixas da Bellevue Way ao norte, até a faixa de pedestres
    "Norte": [(495, 55), (725, 55), (805, 235), (470, 235)],
    # a NE 8th a leste, com a fila parada no semáforo
    "Leste": [(890, 215), (1275, 225), (1275, 410), (1120, 420), (990, 330)],
    # a saída para o sul, entre a faixa branca curva e a faixa do relógio
    "Sul": [(540, 625), (1130, 560), (1200, 695), (560, 695)],
    # a NE 8th a oeste, da borda do quadro até a esquina: quem vem do Norte e converte à
    # direita costuma trocar de ID na curva, e assim entra na Zona antes da troca
    "Oeste": [(0, 290), (445, 252), (455, 580), (0, 580)],
}


def criar_zonas(zonas=ZONAS):
    """{nome da Aproximação: sv.PolygonZone}, a partir dos vértices (x, y) de cada Zona.

    Os nomes precisam ser Norte, Leste, Sul e Oeste: é deles que sai o nome de
    cada Movimento (Reto, Conversão à direita, Conversão à esquerda, Retorno).
    A Zona conta pelos pés de cada veículo (o centro da base da caixa); com o
    padrão do supervision, ela exigiria os 4 cantos da caixa dentro do polígono.
    """
    return {
        nome: sv.PolygonZone(
            polygon=np.array(pontos, dtype=np.int32),
            triggering_anchors=[sv.Position.BOTTOM_CENTER],
        )
        for nome, pontos in zonas.items()
    }


def criar_rastreador(fps=FPS_DO_VIDEO / PASSO):
    """ByteTrack ajustado para o cruzamento.

    - Só caixas com confiança de pelo menos 0,45 abrem um rastro novo ou são ligadas
      primeiro aos rastros que já existem;
    - as mais fracas entram numa segunda rodada, só para continuar um rastro (é o que
      segura o ID do carro meio escondido atrás do ônibus ou na curva);
    - um rastro perdido espera até 1 s por uma caixa nova;
    - `fps` é a taxa dos quadros analisados: a do vídeo dividida pelo `passo`.
    """
    from trackers import ByteTrackTracker

    return ByteTrackTracker(
        frame_rate=fps,
        lost_track_buffer=30,  # em quadros de um vídeo de 30 fps: 1 s
        track_activation_threshold=0.45,
        high_conf_det_threshold=0.45,
        minimum_iou_threshold=0.15,
        minimum_consecutive_frames=2,
    )
