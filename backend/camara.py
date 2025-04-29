import tensorflow as tf
import cv2
import numpy as np
from preprocesamiento import cargar_datos_generador

def clasificar_camara(modelo_path, base_path, img_size=(128, 128)):
    # Cargar el modelo entrenado
    modelo = tf.keras.models.load_model(modelo_path)

    # Cargar el generador para obtener las categorías
    train_generator, _ = cargar_datos_generador(base_path)
    categorias = list(train_generator.class_indices.keys())  # Obtener nombres de las categorías

    # Iniciar la cámara
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("No se pudo acceder a la cámara.")
        return

    print("Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se pudo capturar la imagen desde la cámara.")
            break

        # Preprocesar el cuadro capturado
        # Aplicar balance de blancos automático
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(frame)
        l = cv2.equalizeHist(l)
        frame = cv2.merge((l, a, b))
        frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)

        # Aplicar suavizado para reducir ruido
        frame = cv2.GaussianBlur(frame, (5, 5), 0)

        # Redimensionar y normalizar
        img = cv2.resize(frame, img_size) / 255.0
        img = np.expand_dims(img, axis=0)  # Añadir dimensión para el batch

        # Realizar múltiples predicciones para estabilizar resultados
        predicciones = []
        for _ in range(5):  # Realizar 5 predicciones
            prediccion = modelo.predict(img).argmax(axis=1)[0]
            predicciones.append(prediccion)
        categoria_predicha = categorias[max(set(predicciones), key=predicciones.count)]

        # Calcular confianza (probabilidad más alta)
        probabilidades = modelo.predict(img)[0]
        confianza = np.max(probabilidades)

        # Mostrar la categoría predicha y la confianza en el cuadro
        texto = f"Fruta: {categoria_predicha} ({confianza:.2f})"
        cv2.putText(frame, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Clasificador de Frutas", frame)

        # Salir con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Ruta al modelo y al dataset
    modelo_path = "modelo_entrenado.h5"
    base_path = "datasets/fruits-360/fruits-360_dataset_100x100/fruits-360/Training"

    # Ejecutar clasificación en tiempo real con la cámara
    clasificar_camara(modelo_path, base_path)
