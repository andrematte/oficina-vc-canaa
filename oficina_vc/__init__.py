"""Biblioteca local da Oficina "Visão Computacional Aplicada à Cidade" (Canaã dos Carajás, 11/11/2026).

Guarda o que não cabe explicar numa oficina curta: o laço de vídeo, o painel de
métricas, os desenhos, os downloads e a preparação do ambiente. O código que
ilustra a teoria fica visível nos notebooks.

As importações são preguiçosas: `from oficina_vc import preparar` funciona antes
de os pacotes (ultralytics, supervision, trackers) estarem instalados.
"""

from importlib import import_module

_ONDE = {
    "EM_COLAB": "ambiente",
    "PASTA_DADOS": "ambiente",
    "PASTA_SAIDAS": "ambiente",
    "preparar": "ambiente",
    "dispositivo": "ambiente",
    "pasta_instrutor": "ambiente",
    "caminho": "dados",
    "baixar": "dados",
    "mostrar": "desenho",
    "mostrar_video": "desenho",
    "desenhar_caixas": "desenho",
    "desenhar_linha": "desenho",
    "desenhar_zona": "desenho",
    "desenhar_linha_e_zona": "desenho",
    "processar_video": "video",
    "resultado_pronto": "video",
    "ResultadoCase": "video",
}

__all__ = sorted(_ONDE)


def __getattr__(nome):
    if nome in _ONDE:
        return getattr(import_module(f".{_ONDE[nome]}", __name__), nome)
    raise AttributeError(f"module 'oficina_vc' has no attribute '{nome}'")
