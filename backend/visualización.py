import matplotlib.pyplot as plt
import pickle

# Cargar historial desde archivo
with open("historial_entrenamiento.pkl", "rb") as f:
    historial = pickle.load(f)

# Graficar precisión
plt.plot(historial['accuracy'], label='Precisión Entrenamiento')
plt.plot(historial['val_accuracy'], label='Precisión Validación')
plt.title('Precisión del Modelo')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.show()

# Graficar pérdida
plt.plot(historial['loss'], label='Pérdida Entrenamiento')
plt.plot(historial['val_loss'], label='Pérdida Validación')
plt.title('Pérdida del Modelo')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.show()
