# TSCor - Projeto e Análise de Algoritmo
Programa de Pos-Graduacao em informática - PUC Minas

Aluno: Victor Gabriel Mendes Sundermann

## Dependências principais e ambiente de execução
Dependências Principais:
- PySide6
- Numba
- Numpy
- MatplotLib
- Networks

Para executar o programa:

* python -m venv env
* source env/bin/activate
* pip install -r requirement.txt
* python3 main.py

# Arquitetura Desenvolvida e uso
A implementação do trabalho seguiu as arquiteturas MVP/MCP, onde um componente extra age como uma camada de abstração entre a Interface Gráfica e processamento das séries temporais. A GUI é encarregada de selecionar Bancos de Dados, Séries Temporais e parâmetros variados que são passados para o componente CTSPU, esse componente realiza chamadas de execução para os algoritmos de casamento de séries, classificação, spearman e grafos. Após finalizar os resultados são enviados de volta para o componente GUI para serem repassados ao usuário através de gráficos e métricas.

1) Abrir diretório contendo banco de dados
2) Selecionar diretório desejado (Dois Clicks)
3) Escolher duas séries temporais, sendo possivel misturar treino e teste
4) Definir parâmetros no painel de controle
5) Iniciar execucao
6) Visualizar resultados no widget central

# Resultados preliminares

|       |   Original   |   Numba   |   AeonTK   |
|-------|----------|----------|----------|
|  DTW  |  446     |    136   |   0.51   |
| CDTW  |  1317    |   9.19   |    -     |
| LCSS  |   441    |   138    |   0.52   |
| SDTW  |   1.55   |  1.55    |   1.55   |
| ADTW  |   1.37   |  1.37    |   1.37   |


# Referencias para Time Warping
## Algoritmo DTW
Referencias:
https://github.com/pollen-robotics/dtw
https://github.com/DynamicTimeWarping/dtw-python
https://github.com/markdregan/K-Nearest-Neighbors-with-Dynamic-Time-Warping

## Algoritmo 1 - CDTW
https://www.mathworks.com/matlabcentral/fileexchange/16350-continuous-dynamic-time-warping  (2022)

## Algoritmo 3 - LCS
https://github.com/bguillouet/traj-dist
https://github.com/tslearn-team/tslearn/blob/main/tslearn/metrics/dtw_variants.py

## Algoritmo 6 - Soft-DTW
https://github.com/mblondel/soft-dtw
https://rtavenar.github.io/ml4ts_ensai/contents/align/softdtw.html

## Algoritmo 8 - ADTW
https://github.com/HerrmannM/paper-2021-ADTW
https://www.aeon-toolkit.org/en/latest/api_reference/distances.html