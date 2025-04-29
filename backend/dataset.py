import os

# Ruta para guardar el dataset
output_dir = "datasets/fruits-360"

# Asegúrate de que el directorio exista
os.makedirs(output_dir, exist_ok=True)

# Comando para descargar el dataset
os.system(f"kaggle datasets download -d moltean/fruits -p {output_dir} --unzip")

print(f"Dataset descargado y extraído en: {output_dir}")
