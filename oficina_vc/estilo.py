"""Identidade visual da Oficina: cores, cores das classes e fonte.

Os hex vêm do template do deck ("Aprovar a identidade visual do deck").
"""

from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

# Cores do deck (hex) -------------------------------------------------------
AZUL = "#2541F2"
AMARELO = "#FFD000"
OFF_WHITE = "#FCFCFA"
NAVY = "#14193D"
CINZA = "#3D4363"
CINZA_CLARO = "#5A6080"
VERMELHO = "#C8101A"
VERDE = "#1FBF4F"

# Cores fixas por classe nas caixas de detecção -----------------------------
# Chave: nome cru da classe no modelo (o de `modelo.names`), em minúsculas e sem acento.
# Valor: (cor da caixa, cor do texto da etiqueta).
#
# - Modelo pré-treinado (COCO, nomes em inglês): tudo em amarelo.
# - Modelo especialista de Tipos de veículo (nomes em português): uma cor por classe;
#   o Ônibus, que entrou depois da paleta, fica em off-white (o azul do deck se confundia
#   com o azul-claro da Van e sumia no fundo navy do painel).
#   O Carro fica no amarelo do pré-treinado, que marcava quase tudo como car; os
#   outros três são os que o pré-treinado não separava.
#
# As chaves não se cruzam: o pré-treinado diz "car"/"truck", o especialista diz
# "carro"/"caminhão". Por isso a cor é sempre buscada pelo nome cru, nunca pelo nome
# traduzido de `nome_pt` (senão o "caminhão" do pré-treinado ficaria vermelho).
CORES_PRE_TREINADO = {
    nome: (AMARELO, NAVY) for nome in ("car", "motorcycle", "bus", "truck", "person", "bicycle")
}
CORES_TIPOS_DE_VEICULO = {
    "carro": (AMARELO, NAVY),
    "pickup": (VERDE, NAVY),
    "van": ("#5B70F4", OFF_WHITE),
    "caminhao": ("#F01414", OFF_WHITE),
    "onibus": (OFF_WHITE, NAVY),  # 5ª classe, fora das quatro cores de classe do deck
}
CORES_CLASSES = {**CORES_PRE_TREINADO, **CORES_TIPOS_DE_VEICULO}
# Para classes fora da tabela, em ordem.
CORES_RESERVA = list(CORES_TIPOS_DE_VEICULO.values())

COR_ZONA = AZUL
ALFA_ZONA = 0.22  # preenchimento translúcido das Zonas

# Nomes das classes do COCO que aparecem na Oficina, em português.
NOMES_PT = {
    "person": "pessoa",
    "bicycle": "bicicleta",
    "car": "carro",
    "motorcycle": "moto",
    "bus": "ônibus",
    "truck": "caminhão",
}


def _sem_acento(texto):
    import unicodedata

    texto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in texto if not unicodedata.combining(c)).lower().replace("_", " ").strip()


def nome_pt(nome):
    """Nome da classe em português (o COCO vem em inglês)."""
    return NOMES_PT.get(nome, nome)


def cores_da_classe(nome, indice=0):
    """(cor da caixa, cor do texto) em hex para uma classe, pelo nome cru do modelo."""
    chave = _sem_acento(nome)
    if chave in CORES_CLASSES:
        return CORES_CLASSES[chave]
    return CORES_RESERVA[indice % len(CORES_RESERVA)]


def hex_para_bgr(cor):
    cor = cor.lstrip("#")
    r, g, b = (int(cor[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def hex_para_rgb(cor):
    b, g, r = hex_para_bgr(cor)
    return (r, g, b)


# Fonte ---------------------------------------------------------------------
PASTA_FONTES = Path(__file__).parent / "fontes"


def caminho_fonte(negrito=False):
    """Caminho do TTF da Raleway; se não existir, o da DejaVu Sans (vem com o matplotlib)."""
    raleway = PASTA_FONTES / ("Raleway-Bold.ttf" if negrito else "Raleway-Regular.ttf")
    if raleway.exists():
        return str(raleway)
    try:
        from matplotlib import font_manager

        return font_manager.findfont("DejaVu Sans:bold" if negrito else "DejaVu Sans")
    except Exception:
        return None


@lru_cache(maxsize=64)
def fonte(tamanho, negrito=False):
    """Fonte do PIL: Raleway, com DejaVu Sans como reserva."""
    caminho = caminho_fonte(negrito)
    if caminho:
        try:
            return ImageFont.truetype(caminho, tamanho)
        except OSError:
            pass
    return ImageFont.load_default(tamanho)
