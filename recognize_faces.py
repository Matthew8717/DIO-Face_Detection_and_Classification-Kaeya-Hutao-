# Arquivo: recognize_faces.py
import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
import os

# --- Configurações dos Modelos ---
YOLO_DETECTOR_PATH = 'runs/detect/yolo_face_detector_run/weights/best.pt'
TF_CLASSIFIER_PATH = 'kaeya_hutao_classifier_model.h5'
IMAGE_TO_TEST = 'teste_imgs/ML_teste2.png' # << SUBSTITUA PELO NOME DA IMAGEM QUE VOCE QUER TESTAR

# As classes em ordem alfabética (como o Keras carrega por padrão)
# Verifique a ordem exata que foi impressa no console do train_classifier.py
CLASS_NAMES = ['Hutao', 'Kaeya'] 
IMG_HEIGHT, IMG_WIDTH = 150, 150 # Tamanho usado no treinamento do classificador

# 1. Carregar os modelos
print(f"Carregando detector YOLO de: {YOLO_DETECTOR_PATH}")
detector = YOLO(YOLO_DETECTOR_PATH)

print(f"Carregando classificador TF de: {TF_CLASSIFIER_PATH}")
classifier = tf.keras.models.load_model(TF_CLASSIFIER_PATH)

# 2. Carregar a imagem de teste
if not os.path.exists(IMAGE_TO_TEST):
    print(f"Erro: Imagem de teste não encontrada em {IMAGE_TO_TEST}")
    exit()

img = cv2.imread(IMAGE_TO_TEST)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 3. Executar a detecção de rostos com o YOLO
print("Detectando rostos na imagem...")
results = detector(img_rgb, verbose=False)

# 4. Processar cada rosto detectado
for result in results:
    boxes = result.boxes
    for box in boxes:

        # --- A LINHA QUE PRECISA DE CORREÇÃO ---
        # ERRO: x1, y1, x2, y2 = map(int, box.xyxy)
        # CORREÇÃO: Pegar o primeiro tensor da lista (usando .cpu().numpy() para garantir compatibilidade)
        coordinates = box.xyxy.cpu().numpy()[0] 
        x1, y1, x2, y2 = map(int, coordinates)
 

        # Recortar o rosto
        cropped_face = img[y1:y2, x1:x2]
        if cropped_face.size == 0:
            continue

        # 5. Preparar o rosto recortado para o classificador TF
        # Redimensionar para o tamanho esperado (150x150)
        face_resized = cv2.resize(cropped_face, (IMG_WIDTH, IMG_HEIGHT))
        # Converter de BGR para RGB (Keras/TF espera RGB)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        # Normalizar e adicionar dimensão de batch (TF espera [1, 150, 150, 3])
        face_input = np.expand_dims(face_rgb, axis=0) / 255.0

        # 6. Executar a classificação com o modelo TF
        predictions = classifier.predict(face_input, verbose=0)
        score = tf.nn.softmax(predictions[0])
        predicted_class_index = np.argmax(score)
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = 100 * np.max(score)

        # 7. Desenhar a bounding box e o rótulo na imagem original
        label = f'{predicted_class_name}: {confidence:.2f}%'
        color = (0, 255, 0) if predicted_class_name == 'Kaeya' else (255, 0, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

# 8. Mostrar a imagem resultante (requer que o OpenCV consiga abrir uma janela)
cv2.imshow('Face Recognition Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Opcional: Salvar a imagem com as caixas desenhadas
output_path = 'teste_imgs/result_recognized.jpg'
cv2.imwrite(output_path, img)
print(f"Imagem final salva como {output_path}")

