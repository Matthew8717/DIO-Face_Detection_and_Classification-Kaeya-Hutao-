# Sistema de Reconhecimento e Detecção Facial: Projeto DIO

Este projeto foi desenvolvido como parte de um desafio prático na plataforma [DIO (Digital Innovation One)](www.dio.me). O objetivo era criar um sistema completo de reconhecimento facial do zero, utilizando as bibliotecas `TensorFlow`, `Keras`, `OpenCV` e `ultralytics` (YOLOv8).

O sistema é capaz de detectar várias faces em uma imagem e classificá-las entre duas classes, referidos neste projeto como **Kaeya** e **Hutao**, escolhi esses dois personagens de **Genshin Impact** por eu já ter uma base de dados já construida e pronta para uso.

O código e a estrutura do projeto foram desenvolvidos por mim, **Matheus**, com assistência e orientação de um modelo de IA (Gemini).

## 🚀 Arquitetura do Projeto

O pipeline do sistema segue a abordagem de duas etapas descrita no desafio:

1.  **Detecção de Faces (YOLOv8):** Um modelo YOLOv8 foi treinado/ajustado (`fine-tuning`) para identificar as coordenadas exatas dos rostos nas imagens de entrada.
2.  **Classificação de Indivíduos (TensorFlow/Keras):** Os rostos detectados são recortados e redimensionados, e em seguida, um modelo de Rede Neural Convolucional (CNN) baseado em Keras classifica a identidade do indivíduo.

## 🛠️ Tecnologias Utilizadas

*   **Python 3.x**
*   **TensorFlow / Keras**
*   **Ultralytics YOLOv8**
*   **OpenCV**
*   **NumPy**

## 📂 Estrutura de Pastas

O repositório está organizado da seguinte forma:
```
/Seu_Repositorio/
├── DetectorDataset/                   # Conjunto de dados original para detecção (imagens + .txt)
├── ClassifierDataset/                 # Conjunto de dados de rostos recortados (para o classificador TF)
├── runs/                              # Pasta de saída dos resultados do treinamento YOLO
├── kaeya_hutao_classifier_model.h5    # O modelo final do TensorFlow (classificador)
├── yolov8n.pt                         # O modelo base YOLO usado (baixado manualmente)
├── face_detection_config.yaml         # Configuração do treinamento YOLO
├── train_detector.py                  # Script para treinar o detector YOLO
├── prepare_classification_data.py     # Script para recortar rostos e organizar ClassifierDataset
├── train_classifier.py                # Script para treinar o classificador Keras/TF
├── recognize_faces.py                 # Script final de integração (detecta E classifica)
├── teste_imagem.jpg                   # Exemplo de imagem para teste
├── README.md                          # Este arquivo
└── requirements.txt                   # Dependências do projeto
```
## ⚙️ Como Executar o Projeto

### Pré-requisitos

1. Clone este repositório para sua máquina local.
2. Instale as dependências listadas no `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Passos para Rodar
1. **Treinar o Detector:** Execute train_detector.py.

2. **Preparar Dados de Classificação:** Execute prepare_classification_data.py (após o treino do detector).

3. **Treinar o Classificador:** Execute train_classifier.py (após a preparação dos dados).

4. **Testar o Sistema Completo:** Coloque uma imagem de teste na pasta e execute recognize_faces.py.

## 🧑‍💻 Autor
Matheus (Matthew)

[Meu Perfil da DIO](https://web.dio.me/users/87md_matthew)

