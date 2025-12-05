# Arquivo: prepare_classification_data.py
import os
import shutil
import cv2
from ultralytics import YOLO

# --- Configurações ---
# Caminho para o modelo detector que você treinou na etapa anterior
DETECTOR_MODEL_PATH = 'runs/detect/yolo_face_detector_run/weights/best.pt'

# Caminho para o seu conjunto de dados original (onde estão as imagens grandes)
# Use a pasta 'DetectorDataset' que você organizou
SOURCE_DATASET_DIR = './DetectorDataset/' 

# Caminho para o novo conjunto de dados de classificação (onde os rostos recortados serão salvos)
DEST_CLASSIFIER_DIR = './ClassifierDataset/'

# 1. Carregar o modelo detector treinado
model = YOLO(DETECTOR_MODEL_PATH)

# 2. Garantir que a pasta de destino esteja limpa e pronta
if os.path.exists(DEST_CLASSIFIER_DIR):
    shutil.rmtree(DEST_CLASSIFIER_DIR)
os.makedirs(DEST_CLASSIFIER_DIR, exist_ok=True)

print(f"Diretório de destino {DEST_CLASSIFIER_DIR} preparado.")

# 3. Iterar sobre os dados (train e val)
for split in ['train', 'val']:
    # Criar subpastas para classes (Kaeya, Hutao) dentro de train/val
    os.makedirs(os.path.join(DEST_CLASSIFIER_DIR, split, 'Kaeya'), exist_ok=True)
    os.makedirs(os.path.join(DEST_CLASSIFIER_DIR, split, 'Hutao'), exist_ok=True)
    
    images_dir = os.path.join(SOURCE_DATASET_DIR, 'images', split)
    
    for image_name in os.listdir(images_dir):
        if image_name.endswith('.jpg') or image_name.endswith('.png'):
            image_path = os.path.join(images_dir, image_name)
            
            # Descobrir a classe original pelo nome do arquivo (assume que começa com 'kaeya_' ou 'hutao_')
            if image_name.lower().startswith('image_kaeya'):
                character_name = 'Kaeya'
            elif image_name.lower().startswith('image_hutao'):
                character_name = 'Hutao'
            else:
                print(f"Aviso: Nome de arquivo desconhecido '{image_name}'. Ignorando.")
                continue

            # 4. Executar a detecção usando o modelo treinado
            results = model(image_path, verbose=False)

            # 5. Processar resultados e recortar rostos
            img = cv2.imread(image_path)
            for i, result in enumerate(results):
                boxes = result.boxes
                for box in boxes:
                    # Obter coordenadas da bounding box (formato xyxy: x_min, y_min, x_max, y_max)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Recortar o rosto da imagem original
                    # Adiciona um pequeno padding para garantir que o recorte pegue toda a face
                    padding = 10
                    cropped_face = img[max(0, y1-padding):min(img.shape[0], y2+padding), 
                                       max(0, x1-padding):min(img.shape[1], x2+padding)]
                    
                    if cropped_face.size == 0:
                        continue
                        
                    # 6. Salvar o rosto recortado na nova estrutura de pastas
                    # Nome do arquivo de saída: originalName_faceIndex.jpg
                    output_filename = f"{os.path.splitext(image_name)[0]}_face{i}.jpg"
                    save_path = os.path.join(DEST_CLASSIFIER_DIR, split, character_name, output_filename)
                    
                    cv2.imwrite(save_path, cropped_face)
                    print(f"Salvo: {save_path}")

print("\nProcesso de recorte concluído. Seus dados estão prontos para o treinamento do classificador.")
