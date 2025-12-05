# Arquivo: train_classifier.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os

# --- Configurações ---
DATASET_DIR = './ClassifierDataset/'
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
EPOCHS_CLASSIFIER = 20 # 20 épocas geralmente é um bom começo para classificação

# 1. Carregar os dados usando Keras (ele infere rótulos dos nomes das pastas)
print(f"Carregando dados de classificação de: {DATASET_DIR}")
train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, 'train'),
    labels='inferred',
    label_mode='categorical',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, 'val'),
    labels='inferred',
    label_mode='categorical',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE
)

# Obter os nomes das classes (serão ['Hutao', 'Kaeya'] ou similar)
class_names = train_ds.class_names
num_classes = len(class_names)
print(f"Classes encontradas: {class_names}")

# 2. Definir a arquitetura do modelo de classificação (CNN simples)
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    # Camada final com 2 neurônios (um para cada personagem) e ativação softmax
    Dense(num_classes, activation='softmax')
])

# 3. Compilar o modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 4. Treinar o modelo
print("\nIniciando treinamento do classificador...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_CLASSIFIER
)

# 5. Salvar o modelo final
MODEL_SAVE_PATH = 'kaeya_hutao_classifier_model.h5'
model.save(MODEL_SAVE_PATH)
print(f"\nTreinamento do classificador concluído. Modelo salvo como {MODEL_SAVE_PATH}")
