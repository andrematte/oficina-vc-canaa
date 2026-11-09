"""Arquivos grandes da Oficina: a tabela `nome do arquivo → ID do Drive` e o download.

Os arquivos ficam na pasta pública `oficina-vc-publico/` do Drive do iSACI. Cada um
é baixado com o `gdown`, um por vez. Se o arquivo já existe com o tamanho certo,
ele é pulado (é isso que deixa o bootstrap rápido na segunda vez e no Modo local).
"""

from dataclasses import dataclass

from .ambiente import PASTA_DADOS


@dataclass(frozen=True)
class Arquivo:
    id_drive: str  # ID do arquivo na pasta pública do Drive
    tamanho: int | None  # em bytes; None = só confere se existe
    descricao: str


# IDs da pasta pública (https://drive.google.com/drive/folders/1jNzljo__R_z26QzzmlxexHj70MTcca6Q),
# preenchidos em 08/11/2026 em "Transferir as pastas para o Drive do iSACI".
ARQUIVOS = {
    "bellevue-ne8th_2762-2802s.mp4": Arquivo("1i9A0KF3kPFfU3DUpe5fWQ-30SKQ74Go1", 14_264_302, "vídeo"),
    "cena_ancora_bellevue_623.png": Arquivo("1v099NqG_Iab19DdfcZfzQoGZHPkdW6sD", 1_242_855, "Cena-âncora"),
    "yolo26n.pt": Arquivo("1Ben17d7u2o3t0kjPWCGfsJCH0YvN-OlR", 5_544_453, "modelo"),
    # Gerados por scripts/gerar_resultado_pronto.py
    "case_resultado_pronto.mp4": Arquivo("12vMS44385rEu63b5DWo2w6C7lZtJkLuF", 5_598_054, "Resultado pronto (vídeo)"),
    "case_metricas.json": Arquivo("1h8Aa2T9UaoO-Eo1P2RNcYTX-UKDnSGSP", 11_858, "Resultado pronto (métricas)"),
    # Gerado por scripts/dataset_tipos_de_veiculo.py exportar
    "dataset_tipos_de_veiculo.zip": Arquivo("1SoMhFyZgqxm2V_QT8DvGPc4S4TM1fnTw", 147_757_846, "dataset de Tipos de veículo (YOLO)"),
    # Gerados por scripts/treinar_tipos_de_veiculo.py treinar
    "yolo26n_tipos_de_veiculo.pt": Arquivo("1B20FEBFuBMk8_A-3g1AaQpryjF2SIiVS", 5_481_861, "Modelo especialista treinado antes"),
    "yolo26n_tipos_de_veiculo_treino.csv": Arquivo("1rqZGIPmvzWPaQt5LouyoG6n7oMStnM16", 16_577, "curvas do Modelo especialista"),
}

# O que cada notebook baixa no bootstrap.
CONJUNTOS = {
    "case": [
        "bellevue-ne8th_2762-2802s.mp4",
        "cena_ancora_bellevue_623.png",
        "yolo26n.pt",
        "case_resultado_pronto.mp4",
        "case_metricas.json",
    ],
    "fine_tuning": [
        "bellevue-ne8th_2762-2802s.mp4",
        "yolo26n.pt",
        "dataset_tipos_de_veiculo.zip",
        "yolo26n_tipos_de_veiculo.pt",
        "yolo26n_tipos_de_veiculo_treino.csv",
    ],
}


class ArquivoIndisponivel(RuntimeError):
    pass


def caminho(nome):
    """Caminho local de um arquivo da Oficina (pode ainda não ter sido baixado)."""
    return PASTA_DADOS / nome


def _ja_existe(destino, arquivo):
    if not destino.exists():
        return False
    return arquivo.tamanho is None or destino.stat().st_size == arquivo.tamanho


def baixar(nome, silencioso=True):
    """Baixa um arquivo da tabela, se ainda não estiver na pasta de dados. Devolve o caminho."""
    if nome not in ARQUIVOS:
        raise KeyError(f"'{nome}' não está na tabela de arquivos da Oficina.")
    arquivo = ARQUIVOS[nome]
    destino = caminho(nome)
    if _ja_existe(destino, arquivo):
        return destino
    if not arquivo.id_drive:
        raise ArquivoIndisponivel(
            f"{nome}: o ID do Drive ainda não foi preenchido em oficina_vc/dados.py "
            f"e o arquivo não está em {PASTA_DADOS}."
        )
    import gdown

    destino.parent.mkdir(parents=True, exist_ok=True)
    temporario = destino.with_name(destino.name + ".parcial")
    try:
        gdown.download(id=arquivo.id_drive, output=str(temporario), quiet=silencioso)
    except Exception as erro:
        temporario.unlink(missing_ok=True)
        raise ArquivoIndisponivel(f"{nome}: falha no download do Drive ({erro}).") from erro
    if not temporario.exists():
        raise ArquivoIndisponivel(f"{nome}: o Drive não devolveu o arquivo.")
    if arquivo.tamanho is not None and temporario.stat().st_size != arquivo.tamanho:
        tamanho = temporario.stat().st_size
        temporario.unlink()
        raise ArquivoIndisponivel(
            f"{nome}: chegou com {tamanho} bytes, esperado {arquivo.tamanho}. Rode a célula de novo."
        )
    temporario.replace(destino)
    return destino


def baixar_conjunto(conjunto):
    """Baixa os arquivos de um notebook. Devolve {nome: caminho ou mensagem de erro}."""
    situacao = {}
    for nome in CONJUNTOS[conjunto]:
        try:
            situacao[nome] = baixar(nome)
        except ArquivoIndisponivel as erro:
            situacao[nome] = str(erro)
    return situacao

