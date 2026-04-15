import os
from ultralytics import YOLO

# --- 1. CONFIGURACIÓN DE RUTAS ---
PROJECT_ROOT = r"D:\prueba 2\skin-lesion-detector"
# APUNTAMOS AL DATASET SIN SAM (Bounding Box Completo)
YAML_PATH = os.path.join(PROJECT_ROOT, "data", "yolo_dataset_sin_sam", "dataset_sin_sam.yaml")

def main():
    print("=== Iniciando Estudio Comparativo: YOLOv11 (SIN SAM) ===")
    
    # 2. CARGA DEL MODELO
    print("Cargando modelo YOLO11 SMALL (Para una comparación justa con el FINAL)...")
    model = YOLO('yolo11s.pt') 

    print("\nIniciando entrenamiento...")

    # 3. EJECUCIÓN DEL ENTRENAMIENTO (MISMOS PARÁMETROS QUE EL FINAL)
    results = model.train(
        # --- Datos y Hardware ---
        data=YAML_PATH,
        epochs=150,                  # Igual que el modelo Con SAM
        patience=25,                 
        imgsz=640,                   
        batch=8,                     # Protege tu RTX 3050
        device=0,                    
        workers=4,                   
        
        # --- Guardado de Resultados ---
        project=os.path.join(PROJECT_ROOT, "runs"), 
        name="detector_lesiones_SIN_SAM", # Carpeta separada para comparar mañana
        
        # --- CORRECCIÓN DE DATA AUGMENTATION (Igual que el modelo FINAL) ---
        degrees=0.0,                 
        flipud=0.0,                  
        fliplr=0.5,                  
        hsv_h=0.01,                  
        hsv_s=0.6,                   
        hsv_v=0.3,                   
        
        # --- NUEVAS OPTIMIZACIONES DE ALGORITMO ---
        cos_lr=True,                 
        mosaic=1.0,                  
        close_mosaic=10              
    )
    
    print("\n=== ¡Entrenamiento Comparativo Finalizado! ===")
    print(r"Revisa los resultados en: D:\skin-lesion-detector\runs\detector_lesiones_SIN_SAM")

if __name__ == "__main__":
    main()