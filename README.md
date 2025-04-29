
# 🍎 Proyecto de Visión Artificial - Clasificador de Frutas

Este proyecto implementa un sistema de clasificación de frutas utilizando aprendizaje profundo. Se entrena un modelo con el dataset [Fruits 360](https://www.kaggle.com/moltean/fruits) y se despliega una interfaz web para detectar frutas en tiempo real con la cámara.

## 🚀 Demo en línea

🔗 [Ver aplicación en Netlify](https://graceful-brigadeiros-626428.netlify.app)

## 🧠 Modelo

El modelo fue entrenado con TensorFlow y luego convertido a TensorFlow.js para su uso en el navegador.

📂 El modelo no está incluido en el repositorio por su tamaño, pero puedes descargarlo desde el siguiente enlace:

🔗 [Modelo entrenado (Google Drive) (Aun en proceso de subirse)](https://drive.google.com/)

## 📁 Estructura del proyecto

```  
ProyectoVisionArtificial/
│
├── backend/                  # Scripts de entrenamiento (Python).  
│   
│
├── frontend/                 # Código para el despliegue web. 
│
├── datasets/ # Dataset (ignorado en el repositorio)
│
├── .gitignore
├── README.md
└── 
```

## 🧪 Requisitos

```bash
pip install -r requirements.txt
```

> Asegúrate de tener TensorFlow, OpenCV y demás dependencias instaladas.

## 📸 Cómo usar

1. Inicia la app en Netlify o localmente con un servidor (por ejemplo `Live Server` de VS Code).
2. Permite acceso a tu cámara.
3. Haz clic en “Detectar Fruta”.
4. La interfaz mostrará la fruta detectada y su nivel de confianza.

## 📝 Notas

- El modelo `.h5` no está subido a GitHub por límites de tamaño.
- Las imágenes del dataset están ignoradas para mantener liviano el repositorio.

## 🧑‍💻 Autor

- Emmanuel Maldonado – [@emmanuelmal2](https://github.com/emmanuelmal2)

