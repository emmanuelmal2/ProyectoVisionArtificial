import tensorflow as tf
from sklearn.metrics import classification_report
from preprocesamiento import cargar_datos_generador

def evaluar_modelo(modelo_path, validation_generator):
    # Cargar el modelo entrenado
    modelo = tf.keras.models.load_model(modelo_path)

    # Obtener predicciones
    y_pred = modelo.predict(validation_generator).argmax(axis=1)
    y_true = validation_generator.classes  # Etiquetas verdaderas

    # Obtener las clases del generador
    categorias = list(validation_generator.class_indices.keys())

    # Generar y mostrar reporte de clasificación
    print(classification_report(y_true, y_pred, target_names=categorias))

if __name__ == "__main__":
    # Ruta al modelo y dataset
    modelo_path = "modelo_entrenado.h5"
    base_path = "datasets/fruits-360/fruits-360_dataset_100x100/fruits-360/Test"

    # Cargar datos de validación
    _, validation_generator = cargar_datos_generador(base_path)

    # Evaluar el modelo
    evaluar_modelo(modelo_path, validation_generator)
