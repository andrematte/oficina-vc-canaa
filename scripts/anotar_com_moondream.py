"""Classe de cada recorte de veículo por um Modelo fundacional local (Anotação automática).

Recebe a pasta de recortes de `dataset_tipos_de_veiculo.py recortar` e grava um CSV
`caixa,curta,fina,classe`, com a classe entre carro, pickup, van, caminhão e ônibus (ou "?" quando
a resposta não for nenhuma das cinco).

Modelo: Moondream 3 preview, `beshkenadze/moondream3-preview-mlx-4bit` (MLX, 4 bits, ~5 GB,
licença BSL 1.1), rodando localmente no Mac do instrutor pelo `mlx-vlm`, sem nuvem.
Ambiente: `pip install "mlx-vlm>=0.7" pillow`.

Uso:

    python scripts/anotar_com_moondream.py --recortes .../recortes --saida classes_moondream3.csv
    python scripts/anotar_com_moondream.py --recortes .../recortes --lista piloto.csv --saida piloto.csv

Fica fora do `requirements.txt` do Colab.
"""

import argparse
import csv
import re
import time
from pathlib import Path

# Duas perguntas por recorte. No piloto (50 recortes com gabarito à vista), cada uma
# sozinha acertou 45; juntas, com a regra de `combinar`, 46, e o erro que sobra vai
# quase sempre para uma classe rara (fácil de achar na revisão), não para Carro.
PERGUNTAS = {
    "curta": "What type of vehicle is this? Answer with one word: car, pickup, van, truck or bus.",
    "fina": (
        "What kind of vehicle is in the center of the image? Choose one: sedan, hatchback, SUV, minivan, "
        "pickup truck, cargo van, box truck, semi truck, dump truck, school bus, city bus, coach bus, shuttle bus. "
        "Answer with the category only."
    ),
}
# Palavra da resposta → Tipo de veículo. SUV e minivan contam como Carro; micro-ônibus, como Ônibus.
PARA_CLASSE = {
    **dict.fromkeys(["car", "sedan", "hatchback", "suv", "minivan", "wagon", "coupe", "crossover"], "carro"),
    "pickup": "pickup",
    **dict.fromkeys(["van", "cargo"], "van"),
    **dict.fromkeys(["truck", "box", "semi", "dump", "tractor", "flatbed", "delivery", "lorry"], "caminhão"),
    **dict.fromkeys(["bus", "school", "city", "coach", "shuttle", "minibus"], "ônibus"),
}

# Pasta local com os arquivos de huggingface.co/beshkenadze/moondream3-preview-mlx-4bit
MODELO = Path.home() / ".cache" / "oficina-vc" / "moondream3-preview-mlx-4bit"


def classe_da_resposta(resposta):
    """Primeira palavra reconhecida da resposta (o Moondream 3 às vezes começa com um token reservado)."""
    for palavra in re.findall(r"[a-z]+", resposta.replace("<|md_reserved_4|>", "").lower()):
        if palavra in PARA_CLASSE:
            return PARA_CLASSE[palavra]
    return "?"


def combinar(curta, fina):
    """Uma classe rara (não Carro) dita por qualquer das duas perguntas vence; se as duas
    dizem classes raras diferentes, fica a da pergunta fina."""
    raras = [c for c in (fina, curta) if c not in ("carro", "?")]
    if raras:
        return raras[0]
    return "carro" if "carro" in (curta, fina) else "?"


class Moondream3:
    def __init__(self, caminho=MODELO):
        from mlx_vlm import load
        from mlx_vlm.prompt_utils import apply_chat_template

        self.modelo, self.processador = load(str(caminho), trust_remote_code=True)
        self.modelo_template = apply_chat_template

    def __call__(self, imagem, pergunta):
        from mlx_vlm import generate

        prompt = self.modelo_template(self.processador, self.modelo.config, pergunta, num_images=1)
        saida = generate(self.modelo, self.processador, prompt, [imagem], max_tokens=8, temperature=0.0, verbose=False)
        return getattr(saida, "text", saida)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--recortes", type=Path, required=True, help="pasta com <caixa>.png")
    p.add_argument("--lista", type=Path, help="CSV com a coluna caixa (só essas); padrão: todos os recortes")
    p.add_argument("--saida", type=Path, required=True)
    opcoes = p.parse_args()

    if opcoes.lista:
        with opcoes.lista.open(encoding="utf-8") as f:
            caixas = [linha["caixa"] for linha in csv.DictReader(f)]
    else:
        caixas = sorted(c.stem for c in opcoes.recortes.glob("*.png"))

    feitas = {}
    if opcoes.saida.exists():
        with opcoes.saida.open(encoding="utf-8") as f:
            feitas = {linha["caixa"]: linha for linha in csv.DictReader(f)}

    inicio = time.time()
    modelo = Moondream3()
    print(f"Moondream 3 carregado em {time.time() - inicio:.0f} s", flush=True)
    inicio = time.time()
    novas = 0
    with opcoes.saida.open("a", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=["caixa", "curta", "fina", "classe"])
        if not feitas:
            escritor.writeheader()
        for k, caixa in enumerate(caixas, 1):
            if caixa in feitas:
                continue
            caminho = opcoes.recortes / f"{caixa}.png"
            curta, fina = (modelo(str(caminho), p).replace("<|md_reserved_4|>", "").strip() for p in PERGUNTAS.values())
            classe = combinar(classe_da_resposta(curta), classe_da_resposta(fina))
            escritor.writerow({"caixa": caixa, "curta": curta, "fina": fina, "classe": classe})
            novas += 1
            if k % 100 == 0:
                f.flush()
                print(f"  {k}/{len(caixas)}", flush=True)
    if novas:
        print(f"{novas} recortes em {time.time() - inicio:.0f} s ({(time.time() - inicio) / novas:.2f} s cada) → {opcoes.saida}")


if __name__ == "__main__":
    main()
