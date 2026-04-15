import os
from ultralytics import YOLO

PROJECT_ROOT = r"D:\skin-lesion-detector"
YAML_PATH = os.path.join(PROJECT_ROOT, "data", "yolo_dataset", "dataset.yaml")

def main():
    print("=== Iniciando Entrenamiento YOLOv11 (Fase 2: Optimización Médica) ===")
    
    print("Cargando modelo YOLO11 SMALL (Mayor capacidad de aprendizaje)...")
    model = YOLO('yolo11s.pt') 

    print("\nConfigurando hiperparámetros para RTX 3050 (4GB VRAM)...")
    print("El entrenamiento ha comenzado. ¡Asegúrate de ventilar bien la laptop!\n")

    results = model.train(
        # --- Datos y Hardware ---
        data=YAML_PATH,
        epochs=200,                  # Aumentamos el límite máximo de épocas
        patience=30,                 # Early Stopping: se detendrá si pasa 30 épocas sin mejorar
        imgsz=640,                   # Resolución de entrada
        batch=8,                     # Reducido a 8 para no asfixiar los 4GB de VRAM
        device=0,                    # Usar la tarjeta gráfica NVIDIA
        workers=4,                   # Hilos de procesamiento de la CPU
        
        # --- Guardado de Resultados ---
        project=os.path.join(PROJECT_ROOT, "runs"), 
        name="detector_lesiones_v2", # Todo se guardará en esta nueva carpeta
        
        degrees=90.0,                # Rota las imágenes hasta 90 grados
        fliplr=0.5,                  # 50% de probabilidad de volteo horizontal (espejo)
        flipud=0.5,                  # 50% de probabilidad de volteo vertical
        hsv_h=0.015,                 # Ligera variación en el tono (color)
        hsv_s=0.7,                   # Variación en la saturación
        hsv_v=0.4,                   # Variación en el brillo (simula distintas luces de consultorio)
        
        mosaic=1.0,                  # Combina 4 imágenes en 1 para dar contexto al modelo
        close_mosaic=10              # Desactiva el mosaico en las últimas 10 épocas para afinar detalles
    )
    
    print("\n=== ¡Entrenamiento Finalizado Exitosamente! ===")
    print(r"Tus nuevos pesos médicos están guardados en: D:\skin-lesion-detector\runs\detector_lesiones_v2\weights\best.pt")

if __name__ == "__main__":
    main()
