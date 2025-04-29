import tensorflow as tf
import cv2
import numpy as np
from preprocesamiento import cargar_datos_generador

def clasificar_imagen(modelo_path, imagen_path, img_size=(128, 128)):
    # Cargar el modelo entrenado
    modelo = tf.keras.models.load_model(modelo_path)

    # Leer y procesar la imagen
    img = cv2.imread(imagen_path)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen en la ruta: {imagen_path}")
    img = cv2.resize(img, img_size) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predecir la categoría
    prediccion = modelo.predict(img).argmax(axis=1)[0]

    # Obtener categorías
    base_path = "datasets/fruits-360/fruits-360_dataset_100x100/fruits-360/Training"
    train_generator, _ = cargar_datos_generador(base_path)
    categorias = list(train_generator.class_indices.keys())

    # Mostrar resultado
    print(f"La imagen pertenece a la categoría: {categorias[prediccion]}")

if __name__ == "__main__":
    modelo_path = "modelo_entrenado.h5"
    imagen_path = "datasets/fruits-360/fruits-360_dataset_100x100/fruits-360/Test/Apple Braeburn 1/7_100.jpg"  # Cambia esto a la ruta de tu imagen
    clasificar_imagen(modelo_path, imagen_path)
