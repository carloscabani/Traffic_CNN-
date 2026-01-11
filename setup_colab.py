"""
Setup automático para Google Colab
Uso: !python setup_colab.py

Este script configura automáticamente el ambiente de Google Colab para 
entrenar Traffic CNN con el dataset vehicleDataset.
"""

import os
import sys


def setup_colab():
    """
    Configura el ambiente de Google Colab para Traffic CNN.
    
    Pasos:
    1. Verifica que estamos en Google Colab
    2. Monta Google Drive
    3. Crea directorios necesarios
    4. Verifica que el dataset esté presente
    5. Configura el path de Python
    """
    print("🔧 Configurando ambiente para Google Colab...")
    print("="*60)
    
    # Verificar que estamos en Colab
    try:
        from google.colab import drive
        print("✅ Google Colab detectado")
    except ImportError:
        print("⚠️  Este script está diseñado para Google Colab")
        print("    Si estás ejecutando localmente, no es necesario usar este script.")
        return False
    
    # Montar Drive si no está montado
    if not os.path.exists('/content/drive'):
        print("\n📂 Montando Google Drive...")
        drive.mount('/content/drive')
        print("✅ Google Drive montado")
    else:
        print("\n✅ Google Drive ya está montado")
    
    # Crear directorio de modelos
    model_dir = '/content/drive/MyDrive/Traffic_CNN_Models'
    os.makedirs(model_dir, exist_ok=True)
    print(f"✅ Directorio de modelos creado: {model_dir}")
    
    # Verificar dataset
    print("\n📊 Verificando dataset...")
    dataset_paths = {
        'train_images': '/content/train/images',
        'train_labels': '/content/train/labels',
        'valid_images': '/content/valid/images',
        'valid_labels': '/content/valid/labels',
        'test_images': '/content/test/images',
        'test_labels': '/content/test/labels'
    }
    
    all_found = True
    for name, path in dataset_paths.items():
        if os.path.exists(path):
            count = len([f for f in os.listdir(path) if not f.startswith('.')])
            print(f"   ✅ {name}: {count} archivos")
        else:
            print(f"   ❌ {name}: No encontrado")
            all_found = False
    
    if not all_found:
        print("\n⚠️  Dataset no encontrado o incompleto.")
        print("   Ejecuta estos comandos para extraer el dataset:")
        print("   !unzip -q /content/drive/MyDrive/vehicleDataset.zip -d /content/")
        print("\n   O si el archivo está en otra ubicación, ajusta la ruta.")
        return False
    
    # Agregar src al path
    repo_src = '/content/Traffic_CNN-/src'
    if os.path.exists(repo_src):
        sys.path.insert(0, repo_src)
        print(f"\n✅ Path configurado: {repo_src}")
    else:
        print(f"\n⚠️  Directorio src no encontrado: {repo_src}")
        print("   Asegúrate de haber clonado el repositorio:")
        print("   !git clone https://github.com/carloscabani/Traffic_CNN-.git")
        return False
    
    # Resumen final
    print("\n" + "="*60)
    print("🎉 Setup completado exitosamente!")
    print("="*60)
    print("\n📝 Próximos pasos:")
    print("   1. Importa los módulos necesarios:")
    print("      from dataset.dtset import TrafficFlowDataset")
    print("      from models.architecture import TrafficQuantizerNet")
    print("      from models.loss import TrafficLoss")
    print("\n   2. Inicia el entrenamiento:")
    print("      %cd /content/Traffic_CNN-/src/train")
    print("      !python train_grid_detector.py")
    print("\n💾 Los modelos se guardarán en:")
    print(f"   {model_dir}")
    
    return True


def verify_dependencies():
    """
    Verifica que las dependencias necesarias estén instaladas.
    """
    print("\n🔍 Verificando dependencias...")
    
    dependencies = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'cv2': 'OpenCV'
    }
    
    missing = []
    for module, name in dependencies.items():
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name}")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️  Faltan dependencias: {', '.join(missing)}")
        print("   Instala con:")
        print("   !pip install torch torchvision pillow numpy opencv-python")
        return False
    
    print("✅ Todas las dependencias están instaladas")
    return True


def print_dataset_info():
    """
    Muestra información sobre el dataset vehicleDataset.
    """
    print("\n" + "="*60)
    print("📊 INFORMACIÓN DEL DATASET: vehicleDataset")
    print("="*60)
    print("\n🏷️  Clases (7 tipos de vehículos):")
    classes = [
        "0: bicycle   - Bicicletas",
        "1: bus       - Autobuses",
        "2: car       - Automóviles",
        "3: motorbike - Motocicletas",
        "4: rickshaw  - Rickshaws",
        "5: truck     - Camiones",
        "6: van       - Vans"
    ]
    for cls in classes:
        print(f"   {cls}")
    
    print("\n📁 Estructura esperada:")
    print("   /content/")
    print("   ├── train/")
    print("   │   ├── images/  → Imágenes JPG")
    print("   │   └── labels/  → Archivos .txt (formato YOLO)")
    print("   ├── valid/")
    print("   │   ├── images/")
    print("   │   └── labels/")
    print("   └── test/")
    print("       ├── images/")
    print("       └── labels/")
    
    print("\n📝 Formato de labels:")
    print("   class_id center_x center_y width height")
    print("   Coordenadas normalizadas (0-1)")
    print("   Ejemplo: 2 0.5 0.5 0.1 0.2")


if __name__ == "__main__":
    print("\n" + "🚀 TRAFFIC CNN - SETUP PARA GOOGLE COLAB" + "\n")
    
    # Información del dataset
    print_dataset_info()
    
    # Setup del ambiente
    success = setup_colab()
    
    if success:
        # Verificar dependencias
        verify_dependencies()
    
    print("\n" + "="*60)
