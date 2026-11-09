"""Resumo da arquitetura de um modelo YOLO, em português, para os notebooks do Case e do fine-tuning."""

# O que cada tipo de bloco faz, em poucas palavras
BLOCOS = {
    "Conv": "convolução: filtros que varrem a imagem",
    "C3k2": "várias convoluções em sequência",
    "SPPF": "junta o que foi visto em várias escalas",
    "C2PSA": "atenção: pesa as regiões mais importantes",
    "Upsample": "amplia os mapas (sem pesos)",
    "Concat": "junta mapas de camadas diferentes (sem pesos)",
    "Detect": "saída: caixas, classes e confianças",
}
FIM_DO_CORPO = 10  # no YOLO26, os blocos 0 a 10 são o corpo (backbone); é o que o freeze=11 congela


def _milhar(n):
    return f"{n:,}".replace(",", ".")


def resumo_do_modelo(modelo, congelar=None):
    """Escreve os blocos do modelo, divididos em Corpo e Cabeça, com o número de pesos de cada um.

    Com `congelar=11` (o `freeze` do treino), marca os blocos 0 a 10 como congelados e
    soma os pesos que ficam parados e os que aprendem.
    """
    blocos = modelo.model.model
    total = sum(p.numel() for p in modelo.model.parameters())
    print(f"{len(blocos)} blocos, {_milhar(total)} pesos ao todo\n")
    parados = 0
    for i, bloco in enumerate(blocos):
        if i == 0:
            print("CORPO: transforma a imagem em mapas de características")
        elif i == FIM_DO_CORPO + 1:
            print("\nCABEÇA: transforma os mapas em caixas e classes")
        tipo = type(bloco).__name__
        pesos = sum(p.numel() for p in bloco.parameters())
        marca = ""
        if congelar is not None:
            congelado = i < congelar
            parados += pesos if congelado else 0
            marca = "🔒 " if congelado else "✏️ "
        print(f"  {marca}{i:>2}  {tipo:<9}{_milhar(pesos):>10} pesos   {BLOCOS.get(tipo, '')}")
    if congelar is not None:
        print(f"\n🔒 congelados (não mudam): {_milhar(parados)} pesos ({parados / total:.0%})")
        print(f"✏️ aprendem no treino:      {_milhar(total - parados)} pesos ({(total - parados) / total:.0%})")
