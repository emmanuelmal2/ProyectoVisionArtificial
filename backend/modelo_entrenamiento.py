# backend/modelo_entrenamiento.py

import numpy as np

# ——— Monkey-patch para compatibilidad con tensorflowjs en NumPy ≥1.24 ———
if not hasattr(np, 'object'):
    np.object = object
if not hasattr(np, 'bool'):
    np.bool = bool
# —————————————————————————————————————————————————————————

from tensorflow.keras import layers, models
from preprocesamiento import cargar_datos_generador
import pickle
import tensorflowjs as tfjs
import json
from pathlib import Path

def crear_modelo(input_shape, num_classes):
    """Crea un modelo de red neuronal convolucional simple."""
    modelo = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return modelo

if __name__ == "__main__":
    # 1) Entrenamiento
    # Calculamos la ruta absoluta al dataset
    project_root = Path(__file__).resolve().parent.parent
    base_path = project_root / "datasets" / "fruits-360" / \
                "fruits-360_dataset_100x100" / "fruits-360" / "Training"

    train_gen, val_gen = cargar_datos_generador(str(base_path))
    num_classes = len(train_gen.class_indices)

    modelo = crear_modelo((128, 128, 3), num_classes)
    modelo.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Entrenamos solo 1 epoch
    historial = modelo.fit(
        train_gen,
        epochs=1,
        validation_data=val_gen
    )

    # 2) Guarda historial y .h5
    with open(project_root / "historial_entrenamiento.pkl", "wb") as f:
        pickle.dump(historial.history, f)
    model_h5_path = project_root / "backend" / "modelo_entrenado.h5"
    modelo.save(model_h5_path)

    # 3) Convierte a TF.js Layers format
    tfjs_out = project_root / "frontend" / "carpeta_modelo_tfjs"
    tfjs.converters.save_keras_model(
        modelo,
        str(tfjs_out)
    )

    # 4) Guarda labels.json para mapear índice → nombre de fruta
    class_indices = train_gen.class_indices
    labels = [None] * len(class_indices)
    for name, idx in class_indices.items():
        labels[idx] = name

    with open(tfjs_out / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False, indent=2)

    print(f"Entrenado 1 epoch, convertido a TF.js en {tfjs_out} y labels.json generado")
