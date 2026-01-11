# Traffic CNN - Vehicle Detection System

Sistema de detección de vehículos basado en CenterNet para análisis de flujo de tráfico.

## 🎯 Características

- Detección de 7 tipos de vehículos (bicycle, bus, car, motorbike, rickshaw, truck, van)
- Basado en arquitectura CenterNet con grid detection
- Soporte para entrenamiento en Google Colab con GPU
- Checkpoints automáticos guardados en Google Drive
- Compatible con dataset en formato YOLO

## 🚀 Uso en Google Colab

### Opción 1: Usar el Notebook Interactivo (Recomendado)

1. **Preparar dataset:**
   - Sube `vehicleDataset.zip` a tu Google Drive (carpeta raíz "Mi unidad")

2. **Abrir notebook:**
   - Haz clic aquí: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/carloscabani/Traffic_CNN-/blob/main/notebooks/Traffic_CNN_Colab.ipynb)

3. **Ejecutar todas las celdas** en orden (Runtime → Run all)

4. **Los modelos se guardarán automáticamente en:** `/content/drive/MyDrive/Traffic_CNN_Models/`

### Opción 2: Setup Manual

```python
# 1. Montar Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Clonar repositorio
!git clone https://github.com/carloscabani/Traffic_CNN-.git
%cd Traffic_CNN-

# 3. Extraer dataset
!unzip -q /content/drive/MyDrive/vehicleDataset.zip -d /content/

# 4. Instalar dependencias
!pip install -q torch torchvision pillow numpy opencv-python tqdm

# 5. Ejecutar setup automático
!python setup_colab.py

# 6. Entrenar
%cd src/train
!python train_grid_detector.py
```

## 📊 Dataset

Este proyecto usa el dataset **Vehicle Detection** de Roboflow con las siguientes características:

### Estructura del Dataset
```
/content/
├── train/
│   ├── images/  → 7556 imágenes JPG
│   └── labels/  → 7556 archivos .txt (formato YOLO)
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### Clases de Vehículos
```
0: bicycle   - Bicicletas
1: bus       - Autobuses
2: car       - Automóviles
3: motorbike - Motocicletas
4: rickshaw  - Rickshaws
5: truck     - Camiones
6: van       - Vans
```

### Formato de Labels
Formato YOLO con coordenadas normalizadas (0-1):
```
class_id center_x center_y width height
```

Ejemplo:
```
2 0.5 0.5 0.1 0.2
```

## 🏗️ Arquitectura

El modelo utiliza una arquitectura basada en **CenterNet**:

- **Backbone:** CNN personalizada
- **Output:**
  - Heatmap (1 canal): Ubicación de centros de objetos
  - Size (2 canales): Ancho y alto de bounding boxes
  - Offset (2 canales): Refinamiento sub-píxel de centros

### Pipeline de Entrenamiento

```
Imagen 512x512 
    ↓
[Backbone CNN]
    ↓
Grid 128x128 (stride=4)
    ↓
[Detection Head]
    ↓
├─ Heatmap: [B, 1, 128, 128]
├─ Size:    [B, 2, 128, 128]
└─ Offset:  [B, 2, 128, 128]
```

### Función de Pérdida

Multi-task loss combinando:
- **Focal Loss** para heatmap
- **L1 Loss** para regresión de tamaño
- **L1 Loss** para offset

## 📁 Estructura del Proyecto

```
Traffic_CNN-/
├── src/
│   ├── dataset/
│   │   └── dtset.py              # Dataset loader (formato YOLO)
│   ├── models/
│   │   ├── architecture.py       # TrafficQuantizerNet (CenterNet)
│   │   └── loss.py              # Multi-task loss
│   ├── train/
│   │   └── train_grid_detector.py # Script de entrenamiento
│   ├── infer/
│   │   └── evaluate_model.py     # Inferencia
│   └── utils/
│       └── gaussian.py           # Funciones para heatmap gaussiano
├── notebooks/
│   ├── Traffic_CNN_Colab.ipynb   # Notebook para Google Colab
│   └── visualize_grid_targets.py # Visualización de targets
├── setup_colab.py                # Setup automático para Colab
└── environment.yml               # Dependencias conda
```

## ⚙️ Configuración

### Hiperparámetros por defecto

```python
BATCH_SIZE = 8
LEARNING_RATE = 1.25e-4
NUM_EPOCHS = 50
INPUT_SIZE = 512
GRID_STRIDE = 4
OUTPUT_SIZE = 128  # 512 / 4
```

### Hardware Soportado

- ✅ Google Colab (GPU T4/P100/V100)
- ✅ NVIDIA CUDA
- ✅ Apple Silicon (M1/M2/M3) con MPS
- ✅ CPU (entrenamiento lento)

## 💾 Modelos Guardados

Los checkpoints se guardan automáticamente:

- **Cada 5 épocas:** `traffic_model_ep{N}.pth`
- **Modelo final:** `traffic_model_final.pth`

**Ubicación en Colab:**
```
/content/drive/MyDrive/Traffic_CNN_Models/
├── traffic_model_ep5.pth
├── traffic_model_ep10.pth
├── ...
└── traffic_model_final.pth
```

**Ubicación local:**
```
models/
├── traffic_model_ep5.pth
├── traffic_model_ep10.pth
├── ...
└── traffic_model_final.pth
```

## 🔄 Reanudar Entrenamiento

Si el entrenamiento se interrumpe, puedes reanudarlo:

```bash
# Desde el último checkpoint
python train_grid_detector.py --resume

# Desde un checkpoint específico
python train_grid_detector.py --resume --checkpoint models/traffic_model_ep25.pth
```

## 🛠️ Instalación Local

Para ejecutar localmente (sin Colab):

```bash
# Clonar repositorio
git clone https://github.com/carloscabani/Traffic_CNN-.git
cd Traffic_CNN-

# Crear ambiente conda
conda env create -f environment.yml
conda activate traffic_cnn

# O instalar con pip
pip install torch torchvision pillow numpy opencv-python tqdm matplotlib

# Entrenar
cd src/train
python train_grid_detector.py
```

**Nota:** Debes ajustar las rutas del dataset en `train_grid_detector.py` para uso local.

## 📈 Monitoreo

Durante el entrenamiento, observa las métricas:

```
Epoch [25/50] Step [ 50/945] Loss: 2.1234 (HM:0.523 WH:0.412 Off:0.188)
─────────────────────────────────────────────────────────────
📈 Epoch 25 Completada
   Loss Promedio: 2.0145
   - Heatmap:  0.5123
   - Size:     0.4012
   - Offset:   0.1810
─────────────────────────────────────────────────────────────
💾 Checkpoint guardado: traffic_model_ep25.pth
```

**Señales de buen entrenamiento:**
- Loss total disminuye consistentemente
- Heatmap loss < 1.0 indica buena localización
- Size y Offset loss < 0.5 indica buena regresión

## 🎓 Conceptos Clave

### CenterNet vs YOLO

| Aspecto | CenterNet (Este proyecto) | YOLO |
|---------|---------------------------|------|
| Detección | Centros de objetos | Bounding boxes |
| Anchors | No usa | Sí (v2+) |
| Heatmap | Gaussiano | No |
| Refinamiento | Offset sub-píxel | Sí |

### Ventajas de CenterNet

1. **Sin anchors:** Más simple, menos hiperparámetros
2. **Mejor para objetos pequeños:** Heatmap gaussiano reduce ambigüedad
3. **Una sola pasada:** Eficiente en inferencia

## 🐛 Troubleshooting

**Problema:** Error "No se encuentra la carpeta de imágenes"
- **Solución:** Verifica que el dataset esté extraído en `/content/train/`

**Problema:** Out of Memory (OOM)
- **Solución:** Reduce BATCH_SIZE a 4 o 2

**Problema:** Loss no disminuye
- **Solución:** Verifica que el dataset tenga labels correctos en formato YOLO

**Problema:** Checkpoints no se guardan en Drive
- **Solución:** Verifica que Google Drive esté montado correctamente

## 📚 Referencias

- [CenterNet Paper](https://arxiv.org/abs/1904.07850)
- [Objects as Points](https://arxiv.org/abs/1904.07850)
- [Roboflow - Vehicle Detection Dataset](https://universe.roboflow.com/vehicles-fydp9/vehicles-jl95q)

## 🤝 Contribuir

Las contribuciones son bienvenidas:

1. Fork el repositorio
2. Crea una rama (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la licencia especificada en el archivo LICENSE.

## ✨ Créditos

- **Dataset:** [Vehicle Detection - Roboflow](https://universe.roboflow.com/)
- **Arquitectura:** Basada en CenterNet (Objects as Points)
- **Autor:** Carlos Cabani

---

**¿Necesitas ayuda?** Abre un [issue](https://github.com/carloscabani/Traffic_CNN-/issues) en GitHub.
