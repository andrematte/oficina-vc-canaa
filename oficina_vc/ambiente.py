"""Onde o notebook está rodando, onde ficam os arquivos e a preparação do ambiente.

A primeira célula dos dois notebooks (o bootstrap) é esta:

    #@title 0. Preparar o ambiente
    import os, sys, subprocess
    if "google.colab" in sys.modules:
        PASTA = "/content/oficina-vc-canaa"
        if not os.path.isdir(PASTA + "/.git"):
            subprocess.run(["git", "clone", "-q", "--depth", "1", "--branch", "v2026.11.11",
                            "https://github.com/andrematte/oficina-vc-canaa", PASTA], check=True)
    else:
        PASTA = os.path.abspath(".")  # Modo local: o notebook roda de dentro do repositório
    if PASTA not in sys.path:
        sys.path.insert(0, PASTA)
    from oficina_vc import preparar
    preparar("case")  # ou "fine_tuning"

O clone fica na célula porque a biblioteca só existe depois dele. Todo o resto
(instalação, downloads, checklist) está em `preparar`.

**Modo local** (sem Colab, no computador do instrutor): o clone e o `pip install`
são pulados, os arquivos vêm de uma pasta local e um caminho local substitui o
`drive.mount`. Dois caminhos podem ser trocados por variáveis de ambiente:

- `OFICINA_VC_DADOS`: pasta com os arquivos da `oficina-vc-publico/` (padrão: `dados/` no repositório);
- `OFICINA_VC_INSTRUTOR`: cópia local da `oficina-vc-instrutor/` (padrão: `../drive/oficina-vc-instrutor`).

Este módulo só usa a biblioteca padrão: ele roda antes do `pip install`.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

EM_COLAB = "google.colab" in sys.modules

PASTA_REPO = Path(__file__).resolve().parent.parent

if EM_COLAB:
    PASTA_DADOS = Path("/content/dados")
    PASTA_SAIDAS = Path("/content/saidas")
else:
    PASTA_DADOS = Path(os.environ.get("OFICINA_VC_DADOS", PASTA_REPO / "dados"))
    PASTA_SAIDAS = PASTA_REPO / "saidas"

# Pasta do instrutor no "Meu Drive" da conta do iSACI, vista pelo drive.mount.
PASTA_INSTRUTOR_COLAB = Path(
    "/content/drive/MyDrive/Palestras/20261111 - Semana do IDURB - Visao Computacional/oficina-vc-instrutor"
)
PASTA_INSTRUTOR_LOCAL = Path(
    os.environ.get("OFICINA_VC_INSTRUTOR", PASTA_REPO.parent / "drive" / "oficina-vc-instrutor")
)

# Versões fixadas (as mesmas de requirements.txt).
PACOTES = {
    "ultralytics": "8.4.201",
    "supervision": "0.31.4",
    "trackers": "2.6.1",
    "gdown": "6.5.1",
}


def _versao(pacote):
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(pacote)
    except PackageNotFoundError:
        return None


def pacotes_ok():
    return all(_versao(p) == v for p, v in PACOTES.items())


def instalar_pacotes():
    """Instala as versões fixadas, com a saída escondida. No Modo local, não faz nada."""
    if not EM_COLAB or pacotes_ok():
        return True
    exigidos = [f"{p}=={v}" for p, v in PACOTES.items()]
    processo = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *exigidos],
        capture_output=True,
        text=True,
    )
    if processo.returncode != 0:
        print(processo.stderr[-2000:])
        return False
    importlib.invalidate_caches()
    return pacotes_ok()


def dispositivo():
    """'cuda', 'mps' ou 'cpu', com um nome legível."""
    try:
        import torch
    except ImportError:
        return "cpu", "CPU"
    if torch.cuda.is_available():
        return "cuda", f"GPU ({torch.cuda.get_device_name(0)})"
    if torch.backends.mps.is_available():
        return "mps", "GPU do Mac (mps)"
    return "cpu", "CPU"


def pasta_instrutor():
    """Pasta do instrutor, se esta conta tiver acesso a ela; senão, None.

    No Colab, monta o Drive de quem roda o notebook (só o instrutor encontra a pasta).
    No Modo local, usa a cópia local.
    """
    if EM_COLAB:
        from google.colab import drive

        if not Path("/content/drive/MyDrive").exists():
            drive.mount("/content/drive")
        pasta = PASTA_INSTRUTOR_COLAB
    else:
        pasta = PASTA_INSTRUTOR_LOCAL
    return pasta if pasta.is_dir() else None


def preparar(notebook):
    """Instala os pacotes (no Colab), baixa os arquivos do notebook e mostra a checklist.

    `notebook` é "case" ou "fine_tuning".
    """
    from . import dados

    if notebook not in dados.CONJUNTOS:
        raise ValueError(f"notebook deve ser um de {sorted(dados.CONJUNTOS)}.")

    linhas = []
    if EM_COLAB:
        print("Instalando os pacotes (cerca de 1 minuto)...")
    ok = instalar_pacotes()
    linhas.append(("pacotes", ok, "" if ok else "a instalação falhou (veja a mensagem acima)"))

    print("Baixando os arquivos da Oficina...")
    situacao = dados.baixar_conjunto(notebook)
    for nome in dados.CONJUNTOS[notebook]:
        resultado = situacao[nome]
        ok = isinstance(resultado, Path)
        linhas.append((dados.ARQUIVOS[nome].descricao, ok, "" if ok else resultado))

    _, nome_dispositivo = dispositivo()
    modo = "Colab" if EM_COLAB else "Modo local"

    print()
    for rotulo, ok, detalhe in linhas:
        print(f"{'✅' if ok else '❌'} {rotulo}" + (f": {detalhe}" if detalhe else ""))
    print(f"🖥️  {nome_dispositivo} ({modo})")
    if all(ok for _, ok, _ in linhas):
        print("\nTudo pronto. Pode seguir para a próxima célula.")
    else:
        print("\nAlgo falhou. Rode esta célula de novo; se persistir, avise o instrutor.")
