# Visão Computacional Aplicada à Cidade

**Como imagens viram dados de cidade**

Oficina da Semana do Urbanismo de Canaã dos Carajás (IDURB)
11 de novembro de 2026, das 14h às 17h

Dr. Carlos André de Mattos Teixeira
Pesquisador em Engenharia e Tecnologia
Instituto Sustentabilidade da Amazônia com Ciência e Inovação (iSACI)

## Materiais

| | |
|---|---|
| **Slides da Oficina (PDF)** | [baixar o PDF](LINK_DO_PDF) |
| **1. Case de tráfego**<br>Detectar, rastrear e contar os movimentos dos veículos num cruzamento | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andrematte/oficina-vc-canaa/blob/v2026.11.11/1_case_trafego.ipynb) |
| **2. Fine-tuning de Tipos de veículo**<br>Ensinar o modelo a separar Carro, Pickup, Van, Caminhão e Ônibus | [![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/andrematte/oficina-vc-canaa/blob/v2026.11.11/2_fine_tuning_tipos_de_veiculo.ipynb) |

## Como usar no Colab

1. Clique em **Abrir no Colab** e entre com uma conta Google.
2. Rode a primeira célula, **0. Preparar o ambiente**. Ela instala os pacotes, baixa o vídeo e o modelo e termina com uma checklist. Na segunda vez, é bem mais rápida.
3. Siga as células na ordem, de cima para baixo.
4. Para guardar suas mudanças: **Arquivo › Salvar uma cópia no Drive**.

O notebook 1 roda em CPU. O notebook 2 treina um modelo: escolha uma GPU em **Ambiente de execução › Alterar o tipo de ambiente de execução › T4 GPU**. Em casa, aumente o número de épocas (20 ou mais) para um modelo melhor.

## No seu computador

Também funciona fora do Colab, com Python 3.13:

```bash
git clone --branch v2026.11.11 https://github.com/andrematte/oficina-vc-canaa
cd oficina-vc-canaa
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Fora do Colab, a célula de preparação não clona nem instala nada: ela só baixa os arquivos que faltam para a pasta `dados/`. Os controles deslizantes do Colab viram atribuições comuns; troque o valor direto no código.

## O que tem aqui

```
1_case_trafego.ipynb                   notebook do Case
2_fine_tuning_tipos_de_veiculo.ipynb   notebook do fine-tuning
oficina_vc/        biblioteca local: laço de vídeo, painel de métricas, desenhos, downloads
scripts/           como os arquivos da Oficina foram preparados (vídeo, Cena-âncora, resultados)
requirements.txt   versões fixadas
```

O código curto que ilustra a teoria fica nos notebooks. A pasta `oficina_vc/` guarda o que não cabe explicar numa oficina de três horas; o código está aberto para quem quiser ler.

Vídeos e modelos não ficam no repositório: a célula de preparação os baixa de uma pasta pública do Google Drive.

## Licenças

- **Código**: [GNU AGPL-3.0](LICENSE). Os notebooks usam o [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), também AGPL-3.0. **Uso comercial** do Ultralytics, inclusive de modelos treinados com ele, exige a [licença Enterprise da Ultralytics](https://www.ultralytics.com/license). Este aviso não é parecer jurídico.
- **Vídeo e imagens do Case**: cruzamento Bellevue Way × NE 8th, do [Bellevue Traffic Video Dataset](https://github.com/City-of-Bellevue/TrafficVideoDataset) da City of Bellevue, sem licença formal, usado para ensino e pesquisa. Detalhes e crédito em [LICENCA-VIDEO.md](LICENCA-VIDEO.md).
- **Fonte Raleway** (em `oficina_vc/fontes/`): SIL Open Font License 1.1, em [OFL.txt](oficina_vc/fontes/OFL.txt).
- **Slides**: © André Mattos / iSACI, CC BY-NC-SA 4.0, exceto onde indicado.
