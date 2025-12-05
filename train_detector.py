# Arquivo: train_detector.py
import os
from ultralytics import YOLO

# --- Configurações ---
CONFIG_PATH = 'face_detection_config.yaml'
# Usaremos um modelo pré-treinado otimizado para rostos para acelerar o treino
MODEL_BASE = 'yolov8n.pt' 
EPOCHS = 50 
IMG_SIZE = 640
BATCH_SIZE = 8 # Reduza se sua GPU tiver pouca memória (VRAM)

# 1. Carregar o modelo base
print(f"Carregando modelo base: {MODEL_BASE}")
model = YOLO(MODEL_BASE)

# 2. Iniciar o treinamento com seus dados personalizados
print(f"Iniciando treinamento com a configuração: {CONFIG_PATH}")
results = model.train(
    data=CONFIG_PATH, 
    epochs=EPOCHS, 
    imgsz=IMG_SIZE, 
    batch=BATCH_SIZE, 
    name='yolo_face_detector_run' # Nome da pasta de saída dos resultados
)

print("\nTreinamento concluído!")
# O modelo treinado será salvo em 'runs/detect/yolo_face_detector_run/weights/best.pt'
