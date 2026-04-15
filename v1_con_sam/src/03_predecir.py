import os
import random
from ultralytics import YOLO

# --- 1. CONFIGURACIÓN DE RUTAS ---
PROJECT_ROOT = r"D:\skin-lesion-detector"

# Apuntamos al modelo v2 (el que ya terminó). 
# Si el modelo "FINAL" ya terminó de entrenar, cambia "detector_lesiones_v2" por "detector_lesiones_FINAL"
MODEL_PATH = os.path.join(PROJECT_ROOT, "runs", "detector_lesiones_v2", "weights", "best.pt")
TEST_IMAGES_DIR = os.path.join(PROJECT_ROOT, "data", "yolo_dataset", "images", "test")

def main():
    print("=== Generador de Predicciones Visuales (Fase Final) ===")

    if not os.path.exists(MODEL_PATH):
        print(f"[-] Error: No se encontró el modelo en {MODEL_PATH}")
        return

    # 2. CARGAR EL MODELO ENTRENADO
    print("[+] Cargando el 'cerebro' de la IA (YOLO11s)...")
    model = YOLO(MODEL_PATH)

    # 3. SELECCIONAR IMÁGENES DE PRUEBA
    # Buscamos en la carpeta 'test' (imágenes que el modelo nunca ha visto)
    imagenes_prueba = [f for f in os.listdir(TEST_IMAGES_DIR) if f.endswith('.jpg')]
    
    if not imagenes_prueba:
        print("[-] Error: No se encontraron imágenes en la carpeta test.")
        return

    # Elegimos 3 imágenes al azar para la demostración
    # Puedes cambiar el 3 por el número de fotos que quieras generar
    imagenes_elegidas = random.sample(imagenes_prueba, min(3, len(imagenes_prueba)))
    rutas_imagenes = [os.path.join(TEST_IMAGES_DIR, img) for img in imagenes_elegidas]

    # 4. HACER LA PREDICCIÓN Y DIBUJAR CAJAS
    print(f"\n[+] Analizando {len(rutas_imagenes)} imágenes clínicamente...")
    results = model.predict(
        source=rutas_imagenes,
        conf=0.25,      # Umbral de confianza: Solo muestra detecciones con más del 25% de seguridad
        save=True,      # Le dice a YOLO que dibuje la caja y guarde la foto
        line_width=2,   # Grosor de la línea de la caja
        project=os.path.join(PROJECT_ROOT, "runs"),
        name="predicciones_presentacion" # Nombre de la carpeta de salida
    )

    print("\n=== ¡Misión Cumplida! ===")
    print(r"Revisa tus imágenes con las predicciones dibujadas en:")
    print(r"D:\skin-lesion-detector\runs\predicciones_presentacion")

if __name__ == "__main__":
    main()