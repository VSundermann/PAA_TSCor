# TSCor - Projeto e Analise de Alogirtmo
Programa de Pos-Graduacao em informatica - PUC Minas
Aluno: Victor Gabriel Mendes Sundermann

Dependencias Principais

PySide - Qt Framework for Python

Executando o programa:

O TSCor foi desenvolvido utilizando um ambiente virtual para guardar as dependencias, para tanto é necessario rodar os seguintes comandos:

python -m venv env

source env/bin/activate

pip install -r requirement.txt

# Algoritmo DTW
Referencias:
https://github.com/pollen-robotics/dtw
https://github.com/DynamicTimeWarping/dtw-python
https://github.com/markdregan/K-Nearest-Neighbors-with-Dynamic-Time-Warping

# Algoritmo 1 - CDTW
https://www.mathworks.com/matlabcentral/fileexchange/16350-continuous-dynamic-time-warping  (2022)

# Algoritmo 3 - LCS
https://github.com/bguillouet/traj-dist
https://github.com/tslearn-team/tslearn/blob/main/tslearn/metrics/dtw_variants.py

# Algoritmo 6 - Soft-DTW
https://github.com/mblondel/soft-dtw
https://rtavenar.github.io/ml4ts_ensai/contents/align/softdtw.html

# Algoritmo 8 - ADTW
https://github.com/HerrmannM/paper-2021-ADTW
https://www.aeon-toolkit.org/en/latest/api_reference/distances.html

## Arquitetura Desevolvida

A implementação do trabalho seguiu as arquiteturas MVP/MCP, onde um componente extra age como uma camada de abstração entre a Interface Grafica e processamento das series temporais