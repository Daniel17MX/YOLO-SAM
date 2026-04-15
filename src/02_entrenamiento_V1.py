import os
from ultralytics import YOLO

# --- CONFIGURACIÓN DE RUTAS ---
PROJECT_ROOT = r"D:\skin-lesion-detector"
YAML_PATH = os.path.join(PROJECT_ROOT, "data", "yolo_dataset", "dataset.yaml")

def main():
    print("Cargando modelo YOLO11 nano...")
    # Puedes cambiar a 'yolo11n-seg.pt' si decides hacer segmentación en lugar de cajas
    model = YOLO('yolo11n.pt') 

    print("Iniciando entrenamiento...")
    results = model.train(
        data=YAML_PATH,
        epochs=50,                  
        imgsz=640,                  
        batch=16,                   
        device=0,                   # GPU
        project=os.path.join(PROJECT_ROOT, "runs"), # Guarda los resultados en D:\skin-lesion-detector\runs
        name="detector_lesiones_v1"
    )
    print("¡Entrenamiento finalizado!")

if __name__ == "__main__":
    main()